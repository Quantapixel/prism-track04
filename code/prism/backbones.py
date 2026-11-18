"""Frozen backbones: Omnivore Swin-B (EPIC-100 fine-tuned) and CLIP.

Both load strictly from local files under ``weights/`` so ``run.sh`` works offline.
Fetch them once with ``python code/tools/fetch_weights.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Config, weights_path
from .third_party.omnivore_swin import SwinTransformer3D

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _amp(device: str, fp16: bool):
    if device.startswith("cuda") and fp16:
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


class OmnivoreEpic(nn.Module):
    """Swin-B video model fine-tuned on the EPIC-KITCHENS-100 *train* split.

    forward(uint8 [B,T,H,W,3]) -> (pooled feature [B,1024], action logits [B,3806])
    """

    def __init__(self, ckpt: str | Path):
        super().__init__()
        self.trunk = SwinTransformer3D(
            pretrained2d=False, patch_size=(2, 4, 4), embed_dim=128, depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32], window_size=(16, 7, 7), drop_path_rate=0.4,
            patch_norm=True,
        )
        self.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(1024, 3806, bias=True))
        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing - run: python code/tools/fetch_weights.py")
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
        except Exception:  # older checkpoints may hold non-tensor metadata
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.trunk.load_state_dict(state["trunk"])
        self.head.load_state_dict(state["heads"])
        self.register_buffer("mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMNET_STD).view(1, 3, 1, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor):
        x = frames.permute(0, 4, 1, 2, 3).float().div_(255.0)  # B,3,T,H,W
        x = (x - self.mean) / self.std
        feat = self.trunk(x)
        return feat, self.head(feat)


class ClipEncoder:
    """Thin wrapper over HF CLIP that takes our pre-cropped uint8 frames."""

    def __init__(self, cfg: Config, device: str):
        from transformers import CLIPModel, CLIPTokenizer

        path = weights_path(cfg.clip_model)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: python code/tools/fetch_weights.py")
        dtype = torch.float16 if (device.startswith("cuda") and cfg.fp16) else torch.float32
        self.model = CLIPModel.from_pretrained(str(path), local_files_only=True, torch_dtype=dtype)
        self.model.eval().to(device)
        self.tok = CLIPTokenizer.from_pretrained(str(path), local_files_only=True)
        self.device, self.dtype = device, dtype
        size = self.model.config.vision_config.image_size
        self.size = size
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    @staticmethod
    def _as_tensor(out):
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            if getattr(out, attr, None) is not None:
                return getattr(out, attr)
        return out[0]

    @torch.no_grad()
    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-1] != self.size:
            x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bicubic",
                                                align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        emb = self._as_tensor(self.model.get_image_features(pixel_values=x)).float()
        return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tok = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        emb = self._as_tensor(self.model.get_text_features(**tok)).float()
        return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
