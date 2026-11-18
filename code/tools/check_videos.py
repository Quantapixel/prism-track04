#!/usr/bin/env python3
"""Check every manifest video is present, decodable, and within 0.5 s of the manifest duration.

  python code/tools/check_videos.py [--videos data] [--delete-bad]

A duration off by > 0.5 s makes that video's JSON invalid (scores 0), so re-download it.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "tools"))

from prism.video_io import probe  # noqa: E402
from run_all import index_videos  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "package" / "manifest.csv"))
    ap.add_argument("--videos", nargs="*", default=[str(ROOT / "data")])
    ap.add_argument("--delete-bad", action="store_true", help="delete truncated/broken files")
    a = ap.parse_args()
    idx = index_videos(a.videos)
    with open(a.manifest) as fh:
        rows = list(csv.DictReader(fh))
    missing, bad, ok = [], [], 0
    for r in rows:
        vid, want = r["video_id"], float(r["duration_sec"])
        p = idx.get(vid)
        if p is None:
            missing.append(vid)
            continue
        try:
            got = probe(str(p)).duration
        except Exception as e:  # noqa: BLE001
            got, err = None, str(e)
        if got is None or abs(got - want) > 0.5:
            bad.append(vid)
            print(f"BAD  {vid}: {'unreadable' if got is None else f'{got:.2f}s'} vs manifest {want:.2f}s  ({p})")
            if a.delete_bad and (got is None or got < want - 0.5):
                p.unlink()
                print("     deleted")
        else:
            ok += 1
    print(f"ok={ok} bad={len(bad)} missing={len(missing)} of {len(rows)}")
    if missing:
        print("missing:", ",".join(missing))


if __name__ == "__main__":
    main()
