"""End-to-end single-video inference: video file -> official JSON dict."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import actions as act_mod
from .config import Config, weights_path
from .features import FeatureExtractor, load_features, save_features
from .highlights import coverage_highlights, importance_curve, importance_highlights
from .schema import validate_obj


class Pipeline:
    def __init__(self, cfg: Config, log=print):
        import torch

        self.cfg, self.log = cfg, log
        self.device = cfg.resolve_device()
        threads = os.environ.get("PRISM_THREADS")
        if threads:  # cap CPU use so the machine stays responsive
            torch.set_num_threads(max(1, int(threads)))
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True  # fixed input shapes -> autotuned kernels
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._omni = self._clip = self._labeler = self._head = None
        self._head_checked = False

    # lazy model loading ------------------------------------------------------------
    @property
    def omni(self):
        if self._omni is None:
            from .backbones import OmnivoreEpic

            t = time.time()
            self._omni = OmnivoreEpic(weights_path(self.cfg.omni_weights)).eval().to(self.device)
            self.log(f"[pipeline] Omnivore loaded on {self.device} in {time.time() - t:.1f}s")
        return self._omni

    @property
    def clip(self):
        if self._clip is None:
            from .backbones import ClipEncoder

            self._clip = ClipEncoder(self.cfg, self.device)
        return self._clip

    @property
    def labeler(self):
        if self._labeler is None:
            from .label import LabelClassifier

            self._labeler = LabelClassifier(self.cfg, self.clip)
        return self._labeler

    @property
    def head(self):
        if not self._head_checked:
            self._head_checked = True
            p = weights_path(self.cfg.head_weights)
            if p.exists():
                from .head import load_head

                self._head = load_head(p, self.device)
                self.log(f"[pipeline] learned head loaded from {p}")
            else:
                self.log("[pipeline] no learned head found - using training-free detector")
        return self._head

    # --------------------------------------------------------------------------------
    def features(self, video_path: str, cache: str | Path | None = None) -> dict:
        if cache and Path(cache).exists():
            return load_features(cache)
        with_omni = True
        if self.cfg.skip_omnivore_if_not_cooking:
            from .video_io import keyframes

            kf = keyframes(video_path, size=self.cfg.frame_size, hwaccel=self.cfg.hwaccel)
            if len(kf):
                name, score, _ = self.labeler.predict(self.clip.encode_images(kf))
                with_omni = name == "cooking"
                self.log(f"[pipeline] key-frame pre-check: {name} ({score:.2f}) -> "
                         f"{'running' if with_omni else 'skipping'} Omnivore")
        fx = FeatureExtractor(self.cfg, self.device, omnivore=self.omni if with_omni else None,
                              clip=self.clip)
        feats = fx.extract(video_path, with_omni=with_omni, log=self.log)
        if cache:
            save_features(cache, feats)
        return feats

    def detect_actions(self, feats: dict) -> list[tuple]:
        if self.head is not None and "omni_feat" in feats and len(feats["omni_feat"]):
            from .head import head_detect

            probs = act_mod.window_probs(feats, self.cfg)
            return head_detect(self.head, feats, self.cfg, self.device, omni_probs=probs)
        return act_mod.detect_training_free(feats, self.cfg)

    def from_features(self, feats: dict, video_id: str) -> dict:
        cfg = self.cfg
        duration = float(feats["duration"])
        name, score, _ = self.labeler.predict(feats["clip_emb"])
        if name == "cooking" and "omni_logits" in feats:
            dets = self.detect_actions(feats)
            actions = act_mod.to_json_actions(dets, duration)
            highlights = coverage_highlights(dets, duration, cfg)
        else:
            actions = []
            pos, neg = self.labeler.highlight_text(name)
            grid, imp = importance_curve(feats, pos, neg, cfg)
            highlights = importance_highlights(grid, imp, duration, cfg)
        return {
            "team_id": cfg.team_id,
            "video_id": video_id,
            "duration_sec": round(duration, 2),
            "label": {"name": name, "score": round(score, 4)},
            "actions": actions,
            "highlights": highlights,
        }

    def run(self, video_path: str, out_json: str | Path | None = None, video_id: str | None = None,
            cache: str | Path | None = None) -> dict:
        t = time.time()
        video_id = video_id or Path(video_path).stem
        feats = self.features(video_path, cache)
        result = self.from_features(feats, video_id)
        errors, warnings = validate_obj(result)
        for w in warnings[:5]:
            self.log(f"[validate] warning: {w}")
        for e in errors:  # still write the file: a partially valid answer beats none
            self.log(f"[validate] ERROR: {e}")
        if out_json:
            out_json = Path(out_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_json.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, indent=1))
            tmp.replace(out_json)
        hl = sum(h["segment"][1] - h["segment"][0] for h in result["highlights"])
        self.log(f"[pipeline] {video_id}: label={result['label']['name']} "
                 f"({result['label']['score']:.2f}) actions={len(result['actions'])} "
                 f"highlights={len(result['highlights'])} ({100 * hl / result['duration_sec']:.1f}%) "
                 f"in {time.time() - t:.1f}s")
        return result
