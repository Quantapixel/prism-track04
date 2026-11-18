"""ffmpeg-based probing and streaming decode (no OpenCV / decord / PyAV needed).

Frames are decoded once at ``fps`` and resized so the short side is ``size`` and
centre-cropped to ``size x size`` inside ffmpeg, which keeps Python memory flat even
for 30-minute 1080p60 EPIC videos.
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass
class VideoInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int


def _ffbin(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"{name} not found on PATH; install ffmpeg")
    return exe


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        [_ffbin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", "-select_streams", "v:0", path],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    if not info.get("streams"):
        raise RuntimeError(f"no video stream in {path}")
    st = info["streams"][0]
    fmt = info.get("format", {})
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = st.get(key, "0/0")
        try:
            f = Fraction(rate)
            if f > 0:
                fps = float(f)
                break
        except (ZeroDivisionError, ValueError):
            continue
    dur = 0.0
    for src in (fmt.get("duration"), st.get("duration")):
        try:
            if src is not None and float(src) > 0:
                dur = float(src)
                break
        except ValueError:
            pass
    if dur <= 0 and fps > 0 and st.get("nb_frames"):
        dur = int(st["nb_frames"]) / fps
    if dur <= 0:
        raise RuntimeError(f"could not determine duration of {path}")
    return VideoInfo(path, dur, fps, int(st.get("width", 0)), int(st.get("height", 0)))


class FrameStream:
    """Iterate ``(index, HxWx3 uint8)`` frames sampled at a fixed rate.

    Frame ``i`` corresponds to time ``i / fps`` seconds. A background thread drains the
    ffmpeg pipe so decoding overlaps with model inference.
    """

    def __init__(self, path: str, fps: float = 15.0, size: int = 224, hwaccel: str = "",
                 threads: int = 0, prefetch: int = 256):
        self.path, self.fps, self.size = path, fps, size
        vf = (f"fps={fps},scale=w={size}:h={size}:force_original_aspect_ratio=increase:"
              f"flags=bilinear,crop={size}:{size}")
        cmd = [_ffbin("ffmpeg"), "-nostdin", "-v", "error"]
        if hwaccel:
            cmd += ["-hwaccel", hwaccel]
        cmd += ["-threads", str(threads), "-i", path, "-an", "-sn", "-dn", "-vf", vf,
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        self._cmd = cmd
        self._prefetch = prefetch

    def __iter__(self):
        nbytes = self.size * self.size * 3
        proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=nbytes * 4)
        q: queue.Queue = queue.Queue(maxsize=self._prefetch)
        sentinel = object()

        def reader():
            try:
                while True:
                    buf = proc.stdout.read(nbytes)
                    if not buf or len(buf) < nbytes:
                        break
                    q.put(np.frombuffer(buf, dtype=np.uint8).reshape(self.size, self.size, 3))
            finally:
                q.put(sentinel)

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        idx = 0
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield idx, item
                idx += 1
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            th.join(timeout=5)
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            if idx == 0:
                raise RuntimeError(f"ffmpeg produced no frames for {self.path}: {err[-2000:]}")


def keyframes(path: str, size: int = 224, max_frames: int = 48, hwaccel: str = "") -> np.ndarray:
    """Decode only key frames (very fast) and return up to ``max_frames`` evenly spaced.

    Used for a cheap first look at the video (is this a kitchen?) before the full pass.
    """
    vf = (f"scale=w={size}:h={size}:force_original_aspect_ratio=increase:flags=bilinear,"
          f"crop={size}:{size}")
    cmd = [_ffbin("ffmpeg"), "-nostdin", "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-skip_frame", "nokey", "-i", path, "-an", "-sn", "-dn", "-fps_mode", "passthrough",
            "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, check=False)
    if not res.stdout and b"fps_mode" in res.stderr:  # ffmpeg < 5.1
        i = cmd.index("-fps_mode")
        cmd[i:i + 2] = ["-vsync", "0"]
        res = subprocess.run(cmd, capture_output=True, check=False)
    out = res.stdout
    nbytes = size * size * 3
    n = len(out) // nbytes
    if n == 0:
        return np.zeros((0, size, size, 3), np.uint8)
    frames = np.frombuffer(out[: n * nbytes], dtype=np.uint8).reshape(n, size, size, 3)
    if n > max_frames:
        frames = frames[np.linspace(0, n - 1, max_frames).round().astype(int)]
    return frames
