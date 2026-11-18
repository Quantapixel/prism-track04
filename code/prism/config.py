"""Central configuration. Every tunable number lives here so runs are reproducible.

Override at runtime with a JSON file (``--config my.json``) containing any subset of
these keys, or with ``PRISM_*`` environment variables for the few listed in ``ENV``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "weights"
PACKAGE_DIR = REPO_ROOT / "package"
DATA_DIR = REPO_ROOT / "data"


@dataclass
class Config:
    # ---- identity -------------------------------------------------------------------
    team_id: str = "TEAM_ID"  # set via PRISM_TEAM_ID or --team-id

    # ---- decoding -------------------------------------------------------------------
    decode_fps: float = 15.0  # Omnivore-EPIC was trained on 32 frames @ 15 fps (30 fps, rate 2)
    frame_size: int = 224  # short side resize + centre crop
    hwaccel: str = ""  # e.g. "cuda" to let ffmpeg decode on the GPU

    # ---- Omnivore sliding window ------------------------------------------------------
    omni_weights: str = "omnivore_swinB_epic.torch"
    clip_frames: int = 32  # frames per window (2.13 s at 15 fps)
    window_stride_s: float = 0.5  # stride between window starts
    batch_size: int = 8
    skip_omnivore_if_not_cooking: bool = True  # key-frame CLIP pre-check; FPV videos skip Omnivore

    # ---- CLIP (label + FPV importance) -------------------------------------------------
    clip_model: str = "clip-vit-large-patch14"  # directory name under weights/
    clip_hf_name: str = "openai/clip-vit-large-patch14"  # used only by tools/fetch_weights.py
    clip_fps: float = 1.0
    clip_batch: int = 32

    # ---- label -----------------------------------------------------------------------
    label_temperature: float = 100.0
    other_logit_bias: float = -2.0  # make "other" win only when nothing fits

    # ---- training-free action detection ----------------------------------------------
    det_smooth_sigma: float = 1.0  # in windows
    det_topk_per_window: int = 5
    det_rel_thresholds: tuple = (0.15, 0.3, 0.5, 0.7)
    det_abs_threshold: float = 0.01
    det_min_peak: float = 0.03
    det_pad_s: float = 0.6  # extend each run by this much on both sides
    det_nms_sigma: float = 0.5  # gaussian soft-NMS sigma (on tIoU)
    det_max_per_video: int = 2000

    # ---- learned temporal head (optional; used if weights/head.pt exists) ---------------
    head_weights: str = "head.pt"
    head_fuse_omni: float = 0.5  # final = head^(1-w) * omnivore^w
    head_chunk: int = 2048  # max windows per forward pass
    head_topk_verb: int = 3
    head_topk_noun: int = 5
    head_pre_nms: int = 6000

    # ---- highlights ------------------------------------------------------------------
    budget_frac: float = 0.15
    budget_safety: float = 0.995  # stay just under the budget after rounding
    min_seg_s: float = 4.02  # >= 4.0 s after 2-decimal rounding
    cov_lengths: tuple = (4.02, 5.0, 6.0, 8.0)
    cov_step_s: float = 0.5
    cov_max_dets: int = 1500
    cov_rule: str = "mid_or_half"  # how the proxy decides a detection is "captured"
    cov_score_power: float = 1.0  # p_d = score ** power
    fpv_lengths: tuple = (4.02, 6.0, 8.0, 10.0, 12.0)
    fpv_step_s: float = 1.0
    fpv_smooth_s: float = 1.5
    fpv_w_clip: float = 0.6
    fpv_w_motion: float = 0.2
    fpv_w_novelty: float = 0.2

    # ---- misc ------------------------------------------------------------------------
    device: str = ""  # "" = cuda if available else cpu
    fp16: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, **overrides) -> "Config":
        cfg = cls()
        names = {f.name for f in fields(cls)}
        if path:
            with open(path) as fh:
                data = json.load(fh)
            for k, v in data.items():
                if k not in names:
                    raise KeyError(f"unknown config key {k!r} in {path}")
                setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
        env_team = os.environ.get("PRISM_TEAM_ID")
        if env_team:
            cfg.team_id = env_team
        env_dev = os.environ.get("PRISM_DEVICE")
        if env_dev:
            cfg.device = env_dev
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


def weights_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else WEIGHTS_DIR / p
