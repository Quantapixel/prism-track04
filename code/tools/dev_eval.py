#!/usr/bin/env python3
"""Score detector / highlight settings on held-out EPIC *train* videos (safe to tune on).

Uses cached train features (build_train_features.py) and EPIC_100_train.csv, so you can
compare configurations without touching the hidden test set:

  python code/tools/dev_eval.py                                  # training-free vs head
  python code/tools/dev_eval.py --set det_pad_s=0.8 --set cov_lengths=[4.02,6]
  python code/tools/dev_eval.py --sweep det_pad_s=0.3,0.6,0.9

Dev videos = the ones train_head.py held out (weights/head.pt metadata), else a seeded 10 %.
Caveat: Omnivore was fine-tuned on these videos, so its scores are more confident here
than on unseen (validation) videos. Prefer settings that win by a clear margin.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "tools"))

from evaluate import COV_RULES, action_map, coverage  # noqa: E402
from prism import vocab  # noqa: E402
from prism.actions import detect_training_free, to_json_actions, window_probs  # noqa: E402
from prism.config import Config, weights_path  # noqa: E402
from prism.features import load_features  # noqa: E402
from prism.highlights import coverage_highlights  # noqa: E402
from train_head import load_train_gt  # noqa: E402


def parse_val(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def run(cfg, dev, gt, head, device):
    preds, objs = {}, {}
    for vid, feats in dev.items():
        if head is not None:
            from prism.head import head_detect

            dets = head_detect(head, feats, cfg, device, omni_probs=window_probs(feats, cfg))
        else:
            dets = detect_training_free(feats, cfg)
        dur = float(feats["duration"])
        preds[vid] = to_json_actions(dets, dur)
        objs[vid] = {"duration_sec": dur, "highlights": coverage_highlights(dets, dur, cfg)}
    m = action_map(preds, gt)[0]
    cov = {r: coverage(objs, gt, r) for r in COV_RULES}
    return m, cov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "cache" / "train"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "annotations" / "EPIC_100_train.csv"))
    ap.add_argument("--set", action="append", default=[], help="key=value config override")
    ap.add_argument("--sweep", default=None, help="key=v1,v2,... (training-free + head)")
    ap.add_argument("--max-videos", type=int, default=40)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = Config.load(None, device=a.device)
    for kv in a.set:
        k, v = kv.split("=", 1)
        v = parse_val(v)
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    device = cfg.resolve_device()
    hp = weights_path(cfg.head_weights)
    head, dev_ids = None, None
    if hp.exists():
        import torch

        from prism.head import load_head

        head = load_head(hp, device)
        dev_ids = torch.load(hp, map_location="cpu", weights_only=False)["extra"].get("dev_videos")
    files = sorted(Path(a.cache).glob("*.npz"))
    if dev_ids is None:
        random.Random(0).shuffle(files)
        dev_ids = [f.stem for f in files[: max(1, len(files) // 10)]]
    dev_ids = dev_ids[: a.max_videos]
    gt_all = load_train_gt(a.gt)
    V, N = vocab.verbs(), vocab.nouns()
    gt = {v: [(s, e, V[b], N[n]) for s, e, b, n in gt_all[v]] for v in dev_ids if v in gt_all}
    dev = {v: load_features(Path(a.cache) / f"{v}.npz") for v in gt}
    print(f"{len(dev)} dev videos")
    sweeps = [(None, None)]
    if a.sweep:
        k, vals = a.sweep.split("=", 1)
        sweeps = [(k, parse_val(x)) for x in vals.split(",")]
    for k, v in sweeps:
        if k:
            setattr(cfg, k, v)
        for name, h in (("training-free", None), ("head", head)):
            if name == "head" and h is None:
                continue
            m, cov = run(cfg, dev, gt, h, device)
            tag = f"{k}={v} " if k else ""
            print(f"{tag}{name:13s} mAP {100 * m:5.2f}  coverage " +
                  "  ".join(f"{r}:{100 * c:.1f}" for r, c in cov.items()))


if __name__ == "__main__":
    main()
