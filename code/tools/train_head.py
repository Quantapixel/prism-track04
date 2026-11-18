#!/usr/bin/env python3
"""Train the temporal action-detection head on cached EPIC-100 *train* features.

  python code/tools/build_train_features.py --hours 25     # first
  python code/tools/train_head.py                          # ~10-30 min on an RTX 3060

A held-out set of train videos (--dev-frac) picks the best epoch and is compared with the
training-free detector, so you can see whether the head actually helps before using it.
The best checkpoint is written to weights/head.pt, which the pipeline picks up automatically.
Validation (manifest) videos are refused.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "tools"))

from evaluate import action_map  # noqa: E402
from prism import vocab  # noqa: E402
from prism.actions import detect_training_free, to_json_actions, window_probs  # noqa: E402
from prism.config import Config, weights_path  # noqa: E402
from prism.features import load_features  # noqa: E402
from prism.head import TemporalHead, build_targets, giou_1d, head_detect, save_head, sigmoid_focal  # noqa: E402


def _ts(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def load_train_gt(path):
    gt = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            gt[r["video_id"]].append((_ts(r["start_timestamp"]), _ts(r["stop_timestamp"]),
                                      int(r["verb_class"]), int(r["noun_class"])))
    return gt


class Video:
    def __init__(self, vid, feats, gt, stride):
        self.vid = vid
        self.x = feats["omni_feat"]  # fp16 [N,1024]
        c = feats["win_center"].astype(np.float64)
        self.c0 = float(c[0])
        self.N = len(self.x)
        g = np.array(gt, dtype=np.float64).reshape(-1, 4)
        if len(g):
            g[:, 0] = (g[:, 0] - self.c0) / stride
            g[:, 1] = (g[:, 1] - self.c0) / stride
            g = g[g[:, 1] > g[:, 0]]
        self.g = g
        self.feats = feats


def make_batch(videos, max_len, n_levels, n_verbs, n_cls, rng, train=True):
    xs, gts = [], []
    for v in videos:
        if train and v.N > max_len:
            s = rng.randint(0, v.N - max_len)
        else:
            s = 0
        e = min(v.N, s + max_len)
        g = v.g.copy()
        if len(g):
            g[:, :2] -= s
            L = e - s
            ln = g[:, 1] - g[:, 0]
            keep = (np.minimum(g[:, 1], L - 1) - np.maximum(g[:, 0], 0)) >= 0.5 * ln
            g = g[keep]
            g[:, 0] = np.clip(g[:, 0], 0, L - 1)
            g[:, 1] = np.clip(g[:, 1], 0, L - 1)
        xs.append(v.x[s:e].astype(np.float32))
        gts.append(g)
    T = max(len(x) for x in xs)
    T = int(math.ceil(T / 2 ** (n_levels - 1)) * 2 ** (n_levels - 1))
    B = len(xs)
    x = np.zeros((B, T, xs[0].shape[1]), np.float32)
    m = np.zeros((B, T), bool)
    for i, xi in enumerate(xs):
        x[i, :len(xi)] = xi
        m[i, :len(xi)] = True
    tg = [build_targets(T, g, n_levels, n_cls, n_verbs) for g in gts]
    cls_t = [np.stack([tg[b][l][0] for b in range(B)]) for l in range(n_levels)]
    reg_t = [np.stack([tg[b][l][1] for b in range(B)]) for l in range(n_levels)]
    pos = [np.stack([tg[b][l][2] for b in range(B)]) for l in range(n_levels)]
    return x, m, cls_t, reg_t, pos


def evaluate_dev(model, dev, cfg, device):
    model.eval()
    preds, gt = {}, {}
    V, N = vocab.verbs(), vocab.nouns()
    for v in dev:
        probs = window_probs(v.feats, cfg)
        dets = head_detect(model, v.feats, cfg, device, omni_probs=probs)
        preds[v.vid] = to_json_actions(dets, float(v.feats["duration"]))
        gt[v.vid] = v.gt_raw
    model.train()
    return action_map(preds, gt)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "cache" / "train"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "annotations" / "EPIC_100_train.csv"))
    ap.add_argument("--out", default=str(weights_path("head.pt")))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=4, help="random crops per video per epoch")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = Config.load(a.config, device=a.device)
    device = cfg.resolve_device()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    with open(ROOT / "package" / "manifest.csv") as fh:
        forbidden = {r["video_id"] for r in csv.DictReader(fh)}
    gt_all = load_train_gt(a.gt)
    V, N = vocab.verbs(), vocab.nouns()
    files = sorted(f for f in Path(a.cache).glob("*.npz") if f.stem in gt_all)
    for f in files:
        if f.stem in forbidden:
            raise SystemExit(f"refusing to train on manifest/test video {f.stem}")
    if len(files) < 3:
        raise SystemExit(f"only {len(files)} feature files in {a.cache}; run build_train_features.py")
    rng.shuffle(files)
    n_dev = max(1, int(round(a.dev_frac * len(files))))
    dev_ids = {f.stem for f in files[:n_dev]}
    dev, train = [], []
    for f in files:
        feats = load_features(f)
        if "omni_feat" not in feats or len(feats["omni_feat"]) < 8:
            continue
        if f.stem not in dev_ids:
            feats.pop("omni_logits", None)  # only dev needs logits; saves GBs of RAM
        v = Video(f.stem, feats, gt_all[f.stem], cfg.window_stride_s)
        v.gt_raw = [(s, e, V[vb], N[nn]) for s, e, vb, nn in gt_all[f.stem]]
        (dev if f.stem in dev_ids else train).append(v)
    print(f"train videos: {len(train)} ({sum(v.N for v in train) * cfg.window_stride_s / 3600:.1f} h), "
          f"dev videos: {len(dev)}")

    # training-free baseline on dev, for reference
    base_preds = {v.vid: to_json_actions(detect_training_free(v.feats, cfg), float(v.feats["duration"])) for v in dev}
    base = action_map(base_preds, {v.vid: v.gt_raw for v in dev})[0]
    print(f"dev mAP training-free detector: {100 * base:.2f}")

    model = TemporalHead(in_dim=vids[0].x.shape[1], d=a.d, n_layers=a.layers, n_levels=a.levels).to(device)
    nv, n_cls = model.n_verbs, model.n_verbs + model.n_nouns
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = max(1, len(train) * a.repeats // a.batch)
    total = a.epochs * steps_per_epoch
    warm = a.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return a.lr * (step + 1) / warm
        return a.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    norm = 100.0
    best, step = -1.0, 0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        pool = train * a.repeats
        rng.shuffle(pool)
        tot_c = tot_r = 0.0
        for bi in range(steps_per_epoch):
            batch = pool[bi * a.batch:(bi + 1) * a.batch]
            if not batch:
                break
            x, m, cls_t, reg_t, pos = make_batch(batch, a.max_len, a.levels, nv, n_cls, rng)
            x = torch.from_numpy(x).to(device)
            m = torch.from_numpy(m).to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                cls_o, reg_o, masks = model(x, m)
            cl = torch.cat([c.float().reshape(-1, n_cls) for c in cls_o])
            rg = torch.cat([r.float().reshape(-1, 2) for r in reg_o])
            mk = torch.cat([mm.reshape(-1) for mm in masks])
            ct = torch.cat([torch.from_numpy(c.reshape(-1, n_cls)) for c in cls_t]).to(device)
            rt = torch.cat([torch.from_numpy(r.reshape(-1, 2)) for r in reg_t]).to(device)
            ps = torch.cat([torch.from_numpy(p.reshape(-1)) for p in pos]).to(device) & mk
            npos = float(ps.sum())
            norm = 0.9 * norm + 0.1 * max(npos, 1.0)
            loss_c = sigmoid_focal(cl[mk], ct[mk]) / norm
            loss_r = giou_1d(rg[ps], rt[ps]) / norm if npos > 0 else rg.sum() * 0
            loss = loss_c + loss_r
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot_c += float(loss_c)
            tot_r += float(loss_r)
            step += 1
        msg = (f"epoch {ep + 1:3d}/{a.epochs}  cls {tot_c / steps_per_epoch:.4f}  reg {tot_r / steps_per_epoch:.4f}"
               f"  lr {lr_at(step):.2e}  {(time.time() - t0) / 60:.1f} min")
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            m_dev = evaluate_dev(model, dev, cfg, device)
            msg += f"  dev mAP {100 * m_dev:.2f} (training-free {100 * base:.2f})"
            if m_dev > best:
                best = m_dev
                save_head(model, a.out, extra={"epoch": ep + 1, "dev_map": m_dev, "baseline_dev_map": base,
                                               "train_videos": [v.vid for v in train],
                                               "dev_videos": [v.vid for v in dev], "args": vars(a)})
                msg += "  * saved"
        print(msg, flush=True)
    print(f"best dev mAP {100 * best:.2f} vs training-free {100 * base:.2f} -> {a.out}")
    if best <= base:
        print("NOTE: the head did not beat the training-free detector on dev. Delete weights/head.pt "
              "to keep using the training-free path, or train on more hours.")
    Path(a.out).with_suffix(".json").write_text(json.dumps({"best_dev_map": best, "baseline_dev_map": base}, indent=1))


if __name__ == "__main__":
    main()
