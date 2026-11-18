"""Training-free temporal action detection from Omnivore sliding-window scores.

Omnivore classifies every 2.13 s window (0.5 s stride) into one of 3806 EPIC actions.
For each action class that is ever in a window's top-k we smooth its probability curve,
cut it at several relative thresholds (temporal grouping, as in TAG), turn each run into
a segment, score it, and soft-NMS per class. This needs no training and is the default
when no learned head is present.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import vocab
from .config import Config
from .segments import runs, soft_nms


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def window_probs(feats: dict, cfg: Config) -> np.ndarray:
    p = softmax(feats["omni_logits"])
    if cfg.det_smooth_sigma > 0 and len(p) > 1:
        p = gaussian_filter1d(p, sigma=cfg.det_smooth_sigma, axis=0, mode="nearest")
    return p


Detection = tuple  # (start, end, action_idx, score)


def detect_training_free(feats: dict, cfg: Config) -> list[Detection]:
    if "omni_logits" not in feats or len(feats["omni_logits"]) == 0:
        return []
    duration = float(feats["duration"])
    centers = feats["win_center"].astype(np.float64)
    stride = cfg.window_stride_s
    p = window_probs(feats, cfg)
    k = min(cfg.det_topk_per_window, p.shape[1])
    cand = np.unique(np.argpartition(-p, k - 1, axis=1)[:, :k])

    per_class: dict[int, list] = defaultdict(list)
    for c in cand.tolist():
        s = p[:, c]
        peak = float(s.max())
        if peak < cfg.det_min_peak:
            continue
        seen = set()
        for r in cfg.det_rel_thresholds:
            th = max(r * peak, cfg.det_abs_threshold)
            for i, j in runs(s >= th):
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                st = max(0.0, centers[i] - stride / 2 - cfg.det_pad_s)
                en = min(duration, centers[j] + stride / 2 + cfg.det_pad_s)
                if en - st < 0.1:
                    continue
                seg_s = s[i:j + 1]
                score = 0.5 * float(seg_s.max()) + 0.5 * float(seg_s.mean())
                per_class[c].append((st, en, score))

    dets: list[Detection] = []
    for c, items in per_class.items():
        arr = np.array(items, dtype=np.float64)
        segs, sc = soft_nms(arr[:, :2], arr[:, 2], sigma=cfg.det_nms_sigma, max_keep=200)
        dets.extend((float(a), float(b), c, float(v)) for (a, b), v in zip(segs, sc))
    dets.sort(key=lambda d: -d[3])
    return dets[: cfg.det_max_per_video]


def to_json_actions(dets: list[Detection], duration: float) -> list[dict]:
    acts = vocab.actions()
    out = []
    for st, en, a, sc in dets:
        st, en = round(max(0.0, st), 2), round(min(duration, en), 2)
        if en <= st:
            continue
        v, n = acts[a]
        out.append({"segment": [st, en], "verb": v, "noun": n,
                    "score": round(float(np.clip(sc, 0.0, 1.0)), 4)})
    return out
