#!/usr/bin/env python3
"""Single-video inference. Usage: python code/infer.py <video.mp4> <out.json> [--config cfg.json]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prism.config import Config  # noqa: E402
from prism.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("out_json")
    ap.add_argument("--config", default=None, help="JSON file overriding prism.config.Config")
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--cache", default=None, help="optional .npz feature cache path")
    ap.add_argument("--video-id", default=None, help="default: video file name without extension")
    a = ap.parse_args()
    cfg = Config.load(a.config, team_id=a.team_id, device=a.device)
    Pipeline(cfg, log=lambda m: print(m, file=sys.stderr)).run(
        a.video, a.out_json, video_id=a.video_id, cache=a.cache)


if __name__ == "__main__":
    main()
