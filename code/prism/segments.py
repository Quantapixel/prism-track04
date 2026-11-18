"""1-D segment helpers: tIoU, soft-NMS, contiguous runs."""
from __future__ import annotations

import numpy as np


def tiou(seg: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """tIoU between one segment [2] and many [N,2]."""
    inter = np.clip(np.minimum(seg[1], segs[:, 1]) - np.maximum(seg[0], segs[:, 0]), 0, None)
    union = (seg[1] - seg[0]) + (segs[:, 1] - segs[:, 0]) - inter
    return inter / np.maximum(union, 1e-9)


def tiou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.clip(np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def soft_nms(segs: np.ndarray, scores: np.ndarray, sigma: float = 0.5, min_score: float = 1e-4,
             max_keep: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian soft-NMS (Bodla et al.) for one class. Returns kept segs and decayed scores."""
    segs = segs.astype(np.float64).copy()
    scores = scores.astype(np.float64).copy()
    keep_s, keep_v = [], []
    while len(scores) and len(keep_s) < max_keep:
        i = int(scores.argmax())
        if scores[i] < min_score:
            break
        s, v = segs[i].copy(), scores[i]
        keep_s.append(s)
        keep_v.append(v)
        segs = np.delete(segs, i, axis=0)
        scores = np.delete(scores, i)
        if len(scores):
            iou = tiou(s, segs)
            scores *= np.exp(-(iou ** 2) / sigma)
    if not keep_s:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(keep_s), np.array(keep_v)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive [start, end] index pairs of contiguous True runs."""
    if mask.size == 0:
        return []
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))
