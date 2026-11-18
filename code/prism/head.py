"""Learned temporal action-detection head (ActionFormer-style, compact).

Input : per-window Omnivore features [T, 1024] at a uniform 0.5 s stride.
Output: per pyramid level and position, verb/noun sigmoid scores and (left, right)
        boundary distances. Decoding pairs the top verbs and nouns into EPIC actions
        and optionally fuses the Omnivore action probability.

Trained by ``code/tools/train_head.py`` on EPIC-100 *train* videos only.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vocab
from .segments import soft_nms

REG_RANGES = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]


def _sinusoid(T: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _head(d: int, out: int, prior: float | None = None) -> nn.Sequential:
    last = nn.Conv1d(d, out, 3, padding=1)
    if prior is not None:
        nn.init.constant_(last.bias, -math.log((1 - prior) / prior))
    return nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.GroupNorm(16, d), nn.ReLU(),
                         nn.Conv1d(d, d, 3, padding=1), nn.GroupNorm(16, d), nn.ReLU(), last)


class TemporalHead(nn.Module):
    def __init__(self, in_dim: int = 1024, d: int = 256, n_layers: int = 4, n_heads: int = 4,
                 n_levels: int = 5, n_verbs: int = vocab.N_VERBS_EPIC,
                 n_nouns: int = vocab.N_NOUNS_EPIC, dropout: float = 0.1):
        super().__init__()
        self.hparams = dict(in_dim=in_dim, d=d, n_layers=n_layers, n_heads=n_heads,
                            n_levels=n_levels, n_verbs=n_verbs, n_nouns=n_nouns, dropout=dropout)
        self.n_levels, self.n_verbs, self.n_nouns = n_levels, n_verbs, n_nouns
        self.inp = nn.Sequential(nn.Conv1d(in_dim, d, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(d, d, 3, padding=1), nn.ReLU())
        self.drop = nn.Dropout(dropout)

        def layer():
            return nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True,
                                              norm_first=True)

        self.enc = nn.ModuleList([layer() for _ in range(n_layers)])
        self.down = nn.ModuleList([nn.Conv1d(d, d, 3, stride=2, padding=1) for _ in range(n_levels - 1)])
        self.lvl = nn.ModuleList([layer() for _ in range(n_levels - 1)])
        self.norm = nn.LayerNorm(d)
        self.cls = _head(d, n_verbs + n_nouns, prior=0.01)
        self.reg = _head(d, 2)
        self.scales = nn.Parameter(torch.ones(n_levels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x [B,T,C] float, mask [B,T] bool (True = valid)."""
        h = self.inp(x.transpose(1, 2)).transpose(1, 2)
        h = self.drop(h + _sinusoid(h.shape[1], h.shape[2], h.device)[None])
        for blk in self.enc:
            h = blk(h, src_key_padding_mask=~mask)
        feats, masks = [h], [mask]
        for down, blk in zip(self.down, self.lvl):
            h = down(h.transpose(1, 2)).transpose(1, 2)
            mask = mask[:, ::2]
            h = blk(h, src_key_padding_mask=~mask)
            feats.append(h)
            masks.append(mask)
        cls_out, reg_out = [], []
        for lvl, (f, m) in enumerate(zip(feats, masks)):
            f = self.norm(f).transpose(1, 2)
            cls_out.append(self.cls(f).transpose(1, 2))
            reg_out.append(F.relu(self.reg(f) * self.scales[lvl]).transpose(1, 2))
        return cls_out, reg_out, masks


# ---------------------------------------------------------------------------------------
# targets & losses
# ---------------------------------------------------------------------------------------
def build_targets(T: int, gts: np.ndarray, n_levels: int, n_cls: int, n_verbs: int,
                  radius: float = 1.5):
    """gts: [G,4] (start_idx, end_idx, verb, noun) in base-feature units.

    Returns per level (cls_t [T_l, n_cls], reg_t [T_l, 2], pos [T_l] bool).
    """
    out = []
    for lvl in range(n_levels):
        s = 2 ** lvl
        Tl = (T + s - 1) // s
        x = np.arange(Tl, dtype=np.float64) * s
        cls_t = np.zeros((Tl, n_cls), np.float32)
        reg_t = np.zeros((Tl, 2), np.float32)
        pos = np.zeros(Tl, bool)
        if len(gts):
            gs, ge = gts[:, 0][None], gts[:, 1][None]
            left, right = x[:, None] - gs, ge - x[:, None]
            inside = (left > 0) & (right > 0)
            c = (gs + ge) / 2
            cs = np.maximum(gs, c - radius * s)
            ce = np.minimum(ge, c + radius * s)
            center = (x[:, None] >= cs) & (x[:, None] <= ce)
            mx = np.maximum(left, right)
            lo, hi = REG_RANGES[lvl]
            if lvl == n_levels - 1:
                hi = np.inf
            ok = inside & center & (mx >= lo) & (mx < hi)
            length = np.where(ok, (ge - gs), np.inf)
            best = length.argmin(1)
            pos = np.isfinite(length.min(1))
            for p in np.flatnonzero(pos):
                g = best[p]
                cls_t[p, int(gts[g, 2])] = 1.0
                cls_t[p, n_verbs + int(gts[g, 3])] = 1.0
                reg_t[p] = (left[p, g] / s, right[p, g] / s)
        out.append((cls_t, reg_t, pos))
    return out


def sigmoid_focal(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * loss).sum()


def giou_1d(pred, target):
    pl, pr = pred[:, 0], pred[:, 1]
    tl, tr = target[:, 0], target[:, 1]
    inter = torch.minimum(pl, tl) + torch.minimum(pr, tr)
    union = (pl + pr) + (tl + tr) - inter
    enclose = torch.maximum(pl, tl) + torch.maximum(pr, tr)
    iou = inter / union.clamp(min=1e-6)
    giou = iou - (enclose - union) / enclose.clamp(min=1e-6)
    return (1 - giou).sum()


# ---------------------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def head_detect(model: TemporalHead, feats: dict, cfg, device: str,
                omni_probs: np.ndarray | None = None) -> list[tuple]:
    """Run the head over a video's features and return (start, end, action_idx, score)."""
    centers = feats["win_center"].astype(np.float64)
    x_all = feats["omni_feat"].astype(np.float32)
    N = len(x_all)
    if N == 0:
        return []
    stride = cfg.window_stride_s
    c0 = float(centers[0])
    duration = float(feats["duration"])
    pair2a = vocab.pair_to_action()
    nv = model.n_verbs
    chunk = max(64, int(cfg.head_chunk))
    step = max(32, chunk * 3 // 4)
    cand = []  # (start_t, end_t, action, head_score)
    for s0 in range(0, max(1, N - chunk // 4), step):
        s1 = min(N, s0 + chunk)
        x = torch.from_numpy(x_all[s0:s1])[None].to(device)
        m = torch.ones(1, s1 - s0, dtype=torch.bool, device=device)
        cls_out, reg_out, _ = model(x, m)
        for lvl, (cl, rg) in enumerate(zip(cls_out, reg_out)):
            s = 2 ** lvl
            prob = torch.sigmoid(cl[0].float())
            pv, iv = prob[:, :nv].topk(min(cfg.head_topk_verb, nv), dim=1)
            pn, ino = prob[:, nv:].topk(min(cfg.head_topk_noun, prob.shape[1] - nv), dim=1)
            pv, iv, pn, ino = (t.cpu().numpy() for t in (pv, iv, pn, ino))
            rg = rg[0].float().cpu().numpy()
            for p in range(len(rg)):
                xi = s0 + p * s
                st = c0 + (xi - rg[p, 0] * s) * stride
                en = c0 + (xi + rg[p, 1] * s) * stride
                if en - st < 0.1:
                    continue
                for a in range(pv.shape[1]):
                    for b in range(pn.shape[1]):
                        act = pair2a.get((int(iv[p, a]), int(ino[p, b])))
                        if act is not None:
                            cand.append((st, en, act, math.sqrt(pv[p, a] * pn[p, b])))
        if s1 == N:
            break
    if not cand:
        return []
    arr = np.array(cand, dtype=np.float64)
    order = np.argsort(-arr[:, 3])[: cfg.head_pre_nms]
    arr = arr[order]
    arr[:, 0] = np.clip(arr[:, 0], 0, duration)
    arr[:, 1] = np.clip(arr[:, 1], 0, duration)
    if omni_probs is not None and cfg.head_fuse_omni > 0:
        w = cfg.head_fuse_omni
        i0 = np.clip(np.searchsorted(centers, arr[:, 0]), 0, N - 1)
        i1 = np.clip(np.searchsorted(centers, arr[:, 1]), 0, N - 1)
        fused = np.empty(len(arr))
        for r in range(len(arr)):
            a, b = int(i0[r]), max(int(i0[r]), int(i1[r]))
            om = float(omni_probs[a:b + 1, int(arr[r, 2])].max())
            fused[r] = (arr[r, 3] ** (1 - w)) * (max(om, 1e-6) ** w)
        arr[:, 3] = fused
    dets = []
    for act in np.unique(arr[:, 2]).astype(int):
        sel = arr[arr[:, 2] == act]
        segs, sc = soft_nms(sel[:, :2], sel[:, 3], sigma=cfg.det_nms_sigma, max_keep=200)
        dets.extend((float(a), float(b), int(act), float(v)) for (a, b), v in zip(segs, sc))
    dets.sort(key=lambda d: -d[3])
    return dets[: cfg.det_max_per_video]


def save_head(model: TemporalHead, path, extra: dict | None = None) -> None:
    torch.save({"hparams": model.hparams, "state": model.state_dict(), "extra": extra or {}}, path)


def load_head(path, device: str) -> TemporalHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalHead(**ck["hparams"])
    model.load_state_dict(ck["state"])
    return model.eval().to(device)
