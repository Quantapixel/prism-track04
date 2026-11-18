#!/usr/bin/env python3
"""Download the 49 manifest FPVSum videos from YouTube with yt-dlp.

  pip install yt-dlp        (ffmpeg must be on PATH for merging)
  python code/tools/download_fpvsum.py            # -> data/fpvsum/<video_id>.mp4

The YouTube id comes from FPVSum_videolist.csv (fetch_annotations.sh), not from the file
name: e.g. BikePolo_5yf797Lh4Oc_1 is YouTube 6oOSbkUiJY0.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# <=720p is plenty (frames are resized to 224) and keeps the download small
FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "package" / "manifest.csv"))
    ap.add_argument("--list", default=str(ROOT / "data" / "annotations" / "FPVSum_videolist.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "fpvsum"))
    ap.add_argument("--cookies-from-browser", default=None, help="e.g. firefox, if YouTube asks to sign in")
    a = ap.parse_args()
    exe = shutil.which("yt-dlp") or None
    cmd0 = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    with open(a.list) as fh:
        yt = {r["video_name"]: r["youtube_id"] for r in csv.DictReader(fh)}
    with open(a.manifest) as fh:
        ids = [r["video_id"] for r in csv.DictReader(fh) if r["dataset"] == "fpvsum"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for vid in ids:
        dst = out / f"{vid}.mp4"
        if dst.exists() and dst.stat().st_size > 0:
            print(f"have {vid}")
            continue
        if vid not in yt:
            print(f"!! {vid} not in FPVSum_videolist.csv")
            failed.append(vid)
            continue
        cmd = cmd0 + ["-f", FORMAT, "--merge-output-format", "mp4", "--no-playlist",
                      "-o", str(out / f"{vid}.%(ext)s"), f"https://www.youtube.com/watch?v={yt[vid]}"]
        if a.cookies_from_browser:
            cmd[1:1] = ["--cookies-from-browser", a.cookies_from_browser]
        print("get ", vid)
        if subprocess.run(cmd).returncode != 0 or not dst.exists():
            failed.append(vid)
    print(f"{len(ids) - len(failed)}/{len(ids)} present in {out}")
    if failed:
        print("failed:", ",".join(failed))
    print("next: python code/tools/check_videos.py   (verifies durations against the manifest)")


if __name__ == "__main__":
    main()
