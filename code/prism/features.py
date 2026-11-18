"""Single-pass feature extraction.

One ffmpeg decode at 15 fps feeds three consumers:
  * Omnivore sliding windows (32 frames, 0.5 s stride) -> 1024-d features + 3806 action logits
  * CLIP image embeddings at 1 fps                      -> label + FPV importance
  * frame-difference motion energy at 15 fps            -> FPV importance

Everything is returned as a dict of numpy arrays and can be cached as .npz so heads,
detectors and highlight selectors can be re-run without decoding again.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .video_io import FrameStream, probe

FEATURE_VERSION = 1


class FeatureExtractor:
    def __init__(self, cfg: Config, device: str, omnivore=None, clip=None):
        self.cfg, self.device = cfg, device
        self.omni, self.clip = omnivore, clip
        self.batch = cfg.resolve_batch(device)

    # ------------------------------------------------------------------------------
    def _run_omni(self, batch: list[np.ndarray]):
        try:
            return self._forward(batch)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            self.batch = max(1, len(batch) // 2)  # remember the smaller size for later batches
            h = len(batch) // 2
            f1, l1 = self._run_omni(batch[:h])
            f2, l2 = self._run_omni(batch[h:])
            return np.concatenate([f1, f2]), np.concatenate([l1, l2])

    def _forward(self, batch: list[np.ndarray]):
        x = torch.from_numpy(np.stack(batch)).to(self.device, non_blocking=True)
        use_amp = self.device.startswith("cuda") and self.cfg.fp16
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp) if use_amp else _null():
                feat, logits = self.omni(x)
        return feat.float().cpu().numpy(), logits.float().cpu().numpy()

    def extract(self, video_path: str, with_omni: bool = True, log=print) -> dict:
        cfg = self.cfg
        info = probe(video_path)
        fps, T = cfg.decode_fps, cfg.clip_frames
        stride_f = cfg.window_stride_s * fps
        clip_every = max(1, int(round(fps / cfg.clip_fps)))
        use_omni = with_omni and self.omni is not None

        buf: dict[int, np.ndarray] = {}
        k, next_start = 0, 0
        win_starts: list[int] = []
        o_batch: list[np.ndarray] = []
        feats, logits = [], []
        c_batch, c_idx, c_emb, c_t = [], [], [], []
        motion = []
        prev_small = None
        n = 0
        t0 = time.time()

        def flush_omni():
            if o_batch:
                f, lg = self._run_omni(o_batch)
                feats.append(f.astype(np.float16))
                logits.append(lg.astype(np.float16))
                o_batch.clear()

        def flush_clip():
            if c_batch and self.clip is not None:
                c_emb.append(self.clip.encode_images(np.stack(c_batch)).astype(np.float16))
                c_t.extend(c_idx)
            c_batch.clear()
            c_idx.clear()

        for idx, frame in FrameStream(video_path, fps=fps, size=cfg.frame_size, hwaccel=cfg.hwaccel):
            n = idx + 1
            small = frame[::4, ::4].mean(axis=2, dtype=np.float32)
            motion.append(0.0 if prev_small is None else float(np.abs(small - prev_small).mean()))
            prev_small = small
            if idx % clip_every == 0:
                c_batch.append(frame)
                c_idx.append(idx)
                if len(c_batch) >= cfg.clip_batch:
                    flush_clip()
            if not use_omni:
                continue
            buf[idx] = frame
            while next_start + T - 1 <= idx:
                o_batch.append(np.stack([buf[i] for i in range(next_start, next_start + T)]))
                win_starts.append(next_start)
                k += 1
                next_start = int(round(k * stride_f))
                keep_from = min(next_start, idx - T + 2)  # always keep the last T frames for the tail window
                for i in [i for i in buf if i < keep_from]:
                    del buf[i]
                if len(o_batch) >= self.batch:
                    flush_omni()
        flush_clip()

        if use_omni:
            # tail: make sure the end of the video is covered by one window
            last_needed = n - T
            if not win_starts or (last_needed > win_starts[-1] and n >= T):
                s = max(0, last_needed)
                frames = [buf.get(i) for i in range(s, n)]
                if all(f is not None for f in frames) and frames:
                    while len(frames) < T:
                        frames.append(frames[-1])
                    o_batch.append(np.stack(frames))
                    win_starts.append(s)
            if not win_starts and buf:  # video shorter than one window
                frames = [buf[i] for i in sorted(buf)]
                while len(frames) < T:
                    frames.append(frames[-1])
                o_batch.append(np.stack(frames[:T]))
                win_starts.append(0)
            flush_omni()

        out = {
            "version": np.array(FEATURE_VERSION),
            "duration": np.array(info.duration, dtype=np.float64),
            "src_fps": np.array(info.fps, dtype=np.float64),
            "decode_fps": np.array(fps, dtype=np.float64),
            "n_frames": np.array(n),
            "motion": np.asarray(motion, dtype=np.float16),
            "clip_t": (np.asarray(c_t, dtype=np.float32) + 0.5) / fps,
            "clip_emb": (np.concatenate(c_emb) if c_emb else np.zeros((0, 1), np.float16)),
        }
        if use_omni:
            ws = np.asarray(win_starts, dtype=np.float64)
            out["win_center"] = ((ws + T / 2.0) / fps).astype(np.float32)
            out["win_span"] = np.array(T / fps, dtype=np.float32)
            out["omni_feat"] = np.concatenate(feats) if feats else np.zeros((0, 1024), np.float16)
            out["omni_logits"] = np.concatenate(logits) if logits else np.zeros((0, 3806), np.float16)
        dt = time.time() - t0
        log(f"[features] {Path(video_path).name}: {info.duration:.1f}s video, {n} frames, "
            f"{len(win_starts)} windows, {len(c_t)} clip frames in {dt:.1f}s "
            f"({info.duration / max(dt, 1e-6):.1f}x realtime)")
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def save_features(path: str | Path, feats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **feats)
    tmp.replace(path)


def load_features(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
