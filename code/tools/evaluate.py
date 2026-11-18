#!/usr/bin/env python3
"""Local re-implementation of the four PRISM Track-04 metrics.

The organisers' scorer is not public, so this follows the README definitions:

  Action mAP (25 %)      EPIC: ActivityNet-style mAP over (verb, noun) action classes
                         present in GT, averaged over tIoU 0.1:0.1:0.5.
  Action Coverage (40 %) EPIC: share of distinct GT (verb, noun) pairs captured by
                         highlights (>= 4 s segments only); soft penalty budget/total
                         when highlights exceed 15 %. "Captured" is ambiguous, so it is
                         reported for several rules (--cov-rule picks the headline).
  Highlight F1 (25 %)    FPVSum: frame-level F1 vs the top-15 % consensus frames
                         (FPVSum's own Binary_score).
  Label accuracy (10 %)  FPVSum: category vs prediction.

Ground truth is read from data/annotations (fetch it with code/tools/fetch_annotations.sh).
Use it to sanity-check the pipeline. Do not tune on it: the EPIC validation split and
the FPVSum annotations are the hidden test set.

  python code/tools/evaluate.py --pred outputs/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from prism import vocab  # noqa: E402

TIOUS = (0.1, 0.2, 0.3, 0.4, 0.5)
COV_RULES = ("mid", "half", "any", "full", "mid_or_half")


def _ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def load_epic_gt(path: str | Path, video_ids=None) -> dict[str, list[tuple]]:
    """video_id -> [(start, end, verb_key, noun_key)]"""
    V, N = vocab.verbs(), vocab.nouns()
    out: dict[str, list] = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if video_ids is not None and r["video_id"] not in video_ids:
                continue
            out[r["video_id"]].append((_ts(r["start_timestamp"]), _ts(r["stop_timestamp"]),
                                       V[int(r["verb_class"])], N[int(r["noun_class"])]))
    return dict(out)


def load_fpv_gt(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import scipy.io as sio

    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["gt_data"]
    return {str(e.Name): (np.asarray(e.Score, float), np.asarray(e.Binary_score, bool)) for e in m}


# ---------------------------------------------------------------------------------------
def _interp_ap(prec: np.ndarray, rec: np.ndarray) -> float:
    mprec = np.concatenate([[0], prec, [0]])
    mrec = np.concatenate([[0], rec, [1]])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def action_map(preds: dict[str, list[dict]], gt: dict[str, list[tuple]], tious=TIOUS):
    """preds: video_id -> actions list (official JSON format)."""
    g_by_cls: dict[tuple, dict[str, np.ndarray]] = defaultdict(dict)
    tmp: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for vid, items in gt.items():
        for s, e, v, n in items:
            tmp[(v, n)][vid].append((s, e))
    for c, d in tmp.items():
        g_by_cls[c] = {vid: np.array(x, float) for vid, x in d.items()}
    p_by_cls: dict[tuple, list] = defaultdict(list)
    for vid, acts in preds.items():
        if vid not in gt:
            continue
        for a in acts:
            p_by_cls[(a["verb"], a["noun"])].append((a["score"], vid, a["segment"][0], a["segment"][1]))
    aps = np.zeros((len(tious), len(g_by_cls)))
    for ci, (c, gvids) in enumerate(g_by_cls.items()):
        npos = sum(len(x) for x in gvids.values())
        plist = sorted(p_by_cls.get(c, []), key=lambda x: -x[0])
        if not plist:
            continue
        for ti, th in enumerate(tious):
            used = {vid: np.zeros(len(x), bool) for vid, x in gvids.items()}
            tp = np.zeros(len(plist))
            for k, (_, vid, s, e) in enumerate(plist):
                g = gvids.get(vid)
                if g is None:
                    continue
                inter = np.clip(np.minimum(e, g[:, 1]) - np.maximum(s, g[:, 0]), 0, None)
                iou = inter / ((e - s) + (g[:, 1] - g[:, 0]) - inter + 1e-9)
                for j in np.argsort(-iou):
                    if iou[j] < th:
                        break
                    if used[vid][j]:
                        continue
                    used[vid][j] = True
                    tp[k] = 1
                    break
            ctp = np.cumsum(tp)
            rec = ctp / npos
            prec = ctp / np.arange(1, len(tp) + 1)
            aps[ti, ci] = _interp_ap(prec, rec)
    per = aps.mean(1) if aps.size else np.zeros(len(tious))
    return float(per.mean()), dict(zip(tious, per.tolist()))


def _valid_highlights(obj: dict) -> list[tuple[float, float]]:
    return [tuple(h["segment"]) for h in obj["highlights"] if h["segment"][1] - h["segment"][0] >= 4.0]


def _captured(gs, ge, hl, rule):
    mid, ln = (gs + ge) / 2, max(ge - gs, 1e-3)
    for a, b in hl:
        ov = max(0.0, min(b, ge) - max(a, gs))
        if rule == "mid" and a <= mid <= b:
            return True
        if rule == "half" and ov >= 0.5 * ln:
            return True
        if rule == "any" and ov > 0:
            return True
        if rule == "full" and ov >= ln - 1e-6:
            return True
        if rule == "mid_or_half" and (a <= mid <= b or ov >= 0.5 * ln):
            return True
    return False


def coverage(objs: dict[str, dict], gt: dict[str, list[tuple]], rule: str = "mid_or_half"):
    vals = []
    for vid, items in gt.items():
        obj = objs.get(vid)
        if obj is None:
            vals.append(0.0)
            continue
        hl = _valid_highlights(obj)
        pairs = defaultdict(list)
        for s, e, v, n in items:
            pairs[(v, n)].append((s, e))
        got = sum(any(_captured(s, e, hl, rule) for s, e in inst) for inst in pairs.values())
        cov = got / max(len(pairs), 1)
        dur = obj["duration_sec"]
        total = sum(b - a for a, b in hl)
        if total > 0.15 * dur:
            cov *= 0.15 * dur / total
        vals.append(cov)
    return float(np.mean(vals)) if vals else 0.0


def highlight_f1(objs: dict[str, dict], fpv_gt: dict, ids) -> float:
    f1s = []
    for vid in ids:
        if vid not in fpv_gt:
            continue
        _, binary = fpv_gt[vid]
        obj = objs.get(vid)
        if obj is None:
            f1s.append(0.0)
            continue
        n = len(binary)
        t = (np.arange(n) + 0.5) * obj["duration_sec"] / n
        pred = np.zeros(n, bool)
        for a, b in _valid_highlights(obj):
            pred |= (t >= a) & (t <= b)
        tp = float((pred & binary).sum())
        p = tp / max(pred.sum(), 1)
        r = tp / max(binary.sum(), 1)
        f1s.append(0.0 if tp == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def label_accuracy(objs: dict[str, dict], ids) -> float:
    ok = [int(objs.get(v, {}).get("label", {}).get("name") == v.split("_")[0].lower()) for v in ids]
    return float(np.mean(ok)) if ok else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="directory of <video_id>.json")
    ap.add_argument("--manifest", default=str(ROOT / "package" / "manifest.csv"))
    ap.add_argument("--epic-gt", default=str(ROOT / "data" / "annotations" / "EPIC_100_validation.csv"))
    ap.add_argument("--fpv-gt", default=str(ROOT / "data" / "annotations" / "fpvsum_gt_data.mat"))
    ap.add_argument("--cov-rule", default="mid_or_half", choices=COV_RULES)
    ap.add_argument("--json", default=None, help="write the scores here")
    a = ap.parse_args()

    with open(a.manifest) as fh:
        man = list(csv.DictReader(fh))
    epic_ids = [r["video_id"] for r in man if r["dataset"] == "epic_kitchens_100"]
    fpv_ids = [r["video_id"] for r in man if r["dataset"] == "fpvsum"]
    objs = {}
    for p in Path(a.pred).glob("*.json"):
        try:
            objs[p.stem] = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    have_epic = [v for v in epic_ids if v in objs]
    have_fpv = [v for v in fpv_ids if v in objs]
    print(f"predictions: {len(have_epic)}/{len(epic_ids)} EPIC, {len(have_fpv)}/{len(fpv_ids)} FPVSum")
    scores = {}
    if have_epic and Path(a.epic_gt).exists():
        gt = load_epic_gt(a.epic_gt, set(have_epic))
        m, per = action_map({v: objs[v]["actions"] for v in have_epic}, gt)
        scores["action_map"] = m
        scores["action_map_per_tiou"] = per
        scores["coverage"] = {r: coverage(objs, gt, r) for r in COV_RULES}
        print(f"Action mAP           {100 * m:6.2f}   " + "  ".join(f"@{k}:{100 * v:.1f}" for k, v in per.items()))
        print(f"Action coverage      {100 * scores['coverage'][a.cov_rule]:6.2f}   ("
              + ", ".join(f"{r}: {100 * v:.1f}" for r, v in scores["coverage"].items()) + ")")
    if have_fpv and Path(a.fpv_gt).exists():
        fg = load_fpv_gt(a.fpv_gt)
        missing = [v for v in have_fpv if v not in fg]
        if missing:
            print(f"note: no FPVSum GT for {missing}")
        scores["highlight_f1"] = highlight_f1(objs, fg, have_fpv)
        scores["label_acc"] = label_accuracy(objs, have_fpv)
        print(f"Highlight F1         {100 * scores['highlight_f1']:6.2f}")
        print(f"Label accuracy       {100 * scores['label_acc']:6.2f}")
    if {"action_map", "highlight_f1"} <= scores.keys():
        final = (0.25 * scores["action_map"] + 0.40 * scores["coverage"][a.cov_rule]
                 + 0.25 * scores["highlight_f1"] + 0.10 * scores["label_acc"])
        scores["final"] = final
        print(f"Weighted final       {100 * final:6.2f}   (only {len(have_epic) + len(have_fpv)}/{len(man)} videos)"
              if len(have_epic) + len(have_fpv) < len(man) else f"Weighted final       {100 * final:6.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
