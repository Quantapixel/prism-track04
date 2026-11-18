"""Highlight selection under the official constraints.

Rules (TEAM_README section 3): segments sorted, strictly non-overlapping, at least one
segment >= 4.0 s, segments < 4.0 s are ignored when scoring, total <= 15 % of duration.

* EPIC  -> ``coverage_highlights``: Action Coverage (40 % of the score) counts the
  distinct GT (verb, noun) pairs captured, so we maximise the *expected number of
  distinct predicted pairs covered*:  sum_pair 1 - prod_{d in pair, covered}(1 - p_d),
  with a cost-benefit greedy over candidate windows (a budgeted submodular problem).
* FPV   -> ``importance_highlights``: Highlight F1 compares frames against the top 15 %
  consensus frames, so we fill the budget with the highest-importance time.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d

from .config import Config

EPS = 1e-6


# ---------------------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------------------
def _candidates(duration: float, lengths, step: float) -> np.ndarray:
    out = []
    for L in lengths:
        if L > duration + EPS:
            continue
        starts = np.arange(0.0, max(duration - L, 0.0) + EPS, step)
        if len(starts) == 0 or starts[-1] < duration - L - EPS:
            starts = np.append(starts, duration - L)
        out.append(np.stack([starts, starts + L], 1))
    return np.concatenate(out) if out else np.zeros((0, 2))


def _free(cands: np.ndarray, chosen: list[tuple[float, float]]) -> np.ndarray:
    ok = np.ones(len(cands), bool)
    for a, b in chosen:
        ok &= ~((cands[:, 0] < b - EPS) & (cands[:, 1] > a + EPS))
    return ok


def finalize(segs: list[tuple[float, float]], duration: float, min_seg: float) -> list[dict]:
    """Round, clip, merge touching/overlapping, sort; guarantee one segment >= 4 s."""
    segs = sorted((max(0.0, a), min(duration, b)) for a, b in segs if b - a > EPS)
    merged: list[list[float]] = []
    for a, b in segs:
        if merged and a <= merged[-1][1] + 1e-3:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        # round inwards (start up, end down) so rounding can never grow the total length
        a2, b2 = np.ceil(a * 100 - 1e-6) / 100, np.floor(b * 100 + 1e-6) / 100
        a2, b2 = float(round(a2, 2)), float(round(b2, 2))
        if b2 - a2 < 4.0 and (b - a) >= 4.0:  # rounding pushed it under 4 s
            a2 = round(max(0.0, b2 - min_seg), 2)
        if out and a2 <= out[-1][1]:
            a2 = round(out[-1][1] + 0.01, 2)
        if b2 > a2:
            out.append([a2, b2])
    if not any(b - a >= 4.0 for a, b in out):
        # must contain one >= 4 s segment: replace with the best-placed single window
        if out:
            mid = (out[0][0] + out[0][1]) / 2
        else:
            mid = duration / 2
        L = min(min_seg, duration)
        a = float(np.clip(mid - L / 2, 0.0, max(0.0, duration - L)))
        out = [[round(a, 2), round(min(duration, a + L), 2)]]
    return [{"segment": s} for s in out]


def _spread_fallback(duration: float, cfg: Config) -> list[tuple[float, float]]:
    budget = cfg.budget_frac * duration * cfg.budget_safety
    L = cfg.min_seg_s
    n = max(1, int(budget // L))
    if L > duration:
        return [(0.0, duration)]
    centers = (np.arange(n) + 0.5) * duration / n
    return [(float(np.clip(c - L / 2, 0, duration - L)), float(np.clip(c - L / 2, 0, duration - L) + L))
            for c in centers]


# ---------------------------------------------------------------------------------------
# EPIC: distinct-action coverage
# ---------------------------------------------------------------------------------------
def _captured(cands: np.ndarray, st: np.ndarray, en: np.ndarray, rule: str,
              chunk: int = 2048) -> sparse.csr_matrix:
    """Sparse [n_cands, n_dets]: does candidate window i capture detection j?"""
    st, en = st.astype(np.float32), en.astype(np.float32)
    mid = (st + en) / 2
    ln = np.maximum(en - st, 1e-3)
    blocks = []
    for i in range(0, len(cands), chunk):
        a = cands[i:i + chunk, :1].astype(np.float32)
        b = cands[i:i + chunk, 1:].astype(np.float32)
        ov = np.clip(np.minimum(b, en[None]) - np.maximum(a, st[None]), 0, None)
        if rule == "mid":
            m = (mid[None] >= a) & (mid[None] <= b)
        elif rule == "half":
            m = ov >= 0.5 * ln[None]
        elif rule == "any":
            m = ov > 0
        elif rule == "full":
            m = ov >= ln[None] - 1e-4
        else:  # mid_or_half
            m = ((mid[None] >= a) & (mid[None] <= b)) | (ov >= 0.5 * ln[None])
        blocks.append(sparse.csr_matrix(m, dtype=np.float32))
    return sparse.vstack(blocks).tocsr()


def coverage_highlights(dets: list[tuple], duration: float, cfg: Config) -> list[dict]:
    budget = cfg.budget_frac * duration * cfg.budget_safety
    if not dets:
        return finalize(_spread_fallback(duration, cfg), duration, cfg.min_seg_s)
    dets = sorted(dets, key=lambda d: -d[3])[: cfg.cov_max_dets]
    st = np.array([d[0] for d in dets])
    en = np.array([d[1] for d in dets])
    act = np.array([d[2] for d in dets])
    p = np.clip(np.array([d[3] for d in dets]) ** cfg.cov_score_power, 0, 1 - 1e-6)
    pairs, pid = np.unique(act, return_inverse=True)
    logq = np.log1p(-p)

    short = budget < cfg.min_seg_s
    lengths = (min(cfg.min_seg_s, duration),) if short else cfg.cov_lengths
    cands = _candidates(duration, lengths, cfg.cov_step_s)
    if len(cands) == 0:
        return finalize([(0.0, duration)], duration, cfg.min_seg_s)
    C = _captured(cands, st, en, cfg.cov_rule)
    clen = cands[:, 1] - cands[:, 0]
    covered = np.zeros(len(dets), bool)
    pair_q = np.ones(len(pairs))  # prod(1 - p) over covered detections of each pair
    chosen: list[tuple[float, float]] = []
    used = 0.0
    rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
    phase = 0  # 0 = distinct pairs, 1 = remaining instance mass (only after pairs saturate)
    while True:
        remaining = budget - used
        valid = _free(cands, chosen) & (clen <= remaining + EPS)
        if short and not chosen:
            valid = np.ones(len(cands), bool)
        if not valid.any():
            break
        if phase == 0:
            w = np.where(covered, 0.0, logq)
            D = sparse.csr_matrix((w, (np.arange(len(dets)), pid)), shape=(len(dets), len(pairs)))
            M = (C @ D).tocsr()
            mrows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
            vals = pair_q[M.indices] * (-np.expm1(M.data))
            gain = np.bincount(mrows, weights=vals, minlength=len(cands))
        else:
            gain = np.bincount(rows, weights=(C.data * np.where(covered, 0.0, p)[C.indices]),
                               minlength=len(cands))
        ratio = np.where(valid, gain / clen, -1.0)
        best = int(ratio.argmax())
        if ratio[best] <= 1e-9:
            if phase == 0:
                phase = 1
                continue
            break
        a, b = cands[best]
        chosen.append((float(a), float(b)))
        used += b - a
        newly = C.getrow(best).indices
        newly = newly[~covered[newly]]
        if len(newly):
            np.multiply.at(pair_q, pid[newly], 1 - p[newly])
            covered[newly] = True
        if short:
            break
    if not chosen:
        chosen = _spread_fallback(duration, cfg)
    elif budget - used >= cfg.min_seg_s:  # spend the rest of the budget evenly in the gaps
        extra = [s for s in _spread_fallback(duration, cfg)
                 if _free(np.array([s]), chosen)[0]]
        for s in extra:
            if used + (s[1] - s[0]) > budget:
                break
            chosen.append(s)
            used += s[1] - s[0]
    return finalize(chosen, duration, cfg.min_seg_s)


# ---------------------------------------------------------------------------------------
# FPV: importance
# ---------------------------------------------------------------------------------------
def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def importance_curve(feats: dict, pos_text: np.ndarray, neg_text: np.ndarray, cfg: Config,
                     res: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Importance on a fine grid (``res`` seconds). Returns (times, scores)."""
    duration = float(feats["duration"])
    grid = np.arange(0.0, duration, res) + res / 2
    emb = feats["clip_emb"].astype(np.float32)
    ct = feats["clip_t"].astype(np.float64)
    parts, weights = [], []
    if len(emb) >= 2 and emb.shape[1] == pos_text.shape[1]:
        clip_s = (emb @ pos_text.T).mean(1) - (emb @ neg_text.T).mean(1)
        mean_e = emb.mean(0)
        mean_e /= np.linalg.norm(mean_e) + 1e-9
        novelty = 1.0 - emb @ mean_e
        parts += [np.interp(grid, ct, _z(clip_s)), np.interp(grid, ct, _z(novelty))]
        weights += [cfg.fpv_w_clip, cfg.fpv_w_novelty]
    mot = feats["motion"].astype(np.float64)
    if len(mot) >= 2:
        mt = (np.arange(len(mot)) + 0.5) / float(feats["decode_fps"])
        mot = gaussian_filter1d(np.log1p(mot), sigma=max(1.0, float(feats["decode_fps"]) / 2))
        parts.append(np.interp(grid, mt, _z(mot)))
        weights.append(cfg.fpv_w_motion)
    if not parts:
        return grid, np.zeros_like(grid)
    s = sum(w * p for w, p in zip(weights, parts)) / max(sum(weights), 1e-9)
    if cfg.fpv_smooth_s > 0:
        s = gaussian_filter1d(s, sigma=cfg.fpv_smooth_s / res, mode="nearest")
    return grid, s


def importance_highlights(grid: np.ndarray, score: np.ndarray, duration: float, cfg: Config,
                          res: float = 0.25) -> list[dict]:
    budget = cfg.budget_frac * duration * cfg.budget_safety
    if budget < cfg.min_seg_s or len(score) == 0:
        L = min(cfg.min_seg_s, duration)
        if len(score) == 0:
            return finalize([(0.0, L)], duration, cfg.min_seg_s)
        cs = np.concatenate([[0], np.cumsum(score)])
        n = max(1, int(round(L / res)))
        means = (cs[n:] - cs[:-n]) / n
        i = int(means.argmax())
        a = min(i * res, duration - L)
        return finalize([(a, a + L)], duration, cfg.min_seg_s)
    cands = _candidates(duration, cfg.fpv_lengths, cfg.fpv_step_s)
    cs = np.concatenate([[0], np.cumsum(score)])
    i0 = np.clip((cands[:, 0] / res).astype(int), 0, len(score))
    i1 = np.clip(np.ceil(cands[:, 1] / res).astype(int), 0, len(score))
    means = (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1)
    clen = cands[:, 1] - cands[:, 0]
    chosen: list[tuple[float, float]] = []
    used = 0.0
    while True:
        valid = _free(cands, chosen) & (clen <= budget - used + EPS)
        if not valid.any():
            break
        best = int(np.where(valid, means, -np.inf).argmax())
        a, b = cands[best]
        chosen.append((float(a), float(b)))
        used += b - a
    # grow segments into the best adjacent time until the budget is used
    taken = np.zeros(len(score), bool)
    for a, b in chosen:
        taken[int(a / res):int(np.ceil(b / res))] = True
    remaining = budget - used
    segs = [list(s) for s in chosen]
    while remaining >= res and segs:
        best, best_v, best_side = -1, -np.inf, 0
        for k, (a, b) in enumerate(segs):
            li, ri = int(a / res) - 1, int(np.ceil(b / res))
            if li >= 0 and not taken[li] and score[li] > best_v:
                best, best_v, best_side = k, score[li], -1
            if ri < len(score) and not taken[ri] and score[ri] > best_v:
                best, best_v, best_side = k, score[ri], +1
        if best < 0:
            break
        a, b = segs[best]
        if best_side < 0:
            segs[best][0] = max(0.0, a - res)
            taken[int(a / res) - 1] = True
        else:
            segs[best][1] = min(duration, b + res)
            taken[int(np.ceil(b / res))] = True
        remaining -= res
    return finalize([tuple(s) for s in segs], duration, cfg.min_seg_s)
