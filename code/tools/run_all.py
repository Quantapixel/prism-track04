#!/usr/bin/env python3
"""Batch inference over the manifest (models are loaded once; much faster than 187 run.sh calls).

  # everything, caching features so later re-runs (new head / thresholds) take seconds
  python code/tools/run_all.py --videos data/EPIC-KITCHENS data/fpvsum

  # 3 parallel worker processes sharing one GPU (decode overlaps with inference)
  python code/tools/run_all.py --videos data --workers 3

  # rebuild all JSONs from cached features only (no video needed)
  python code/tools/run_all.py --from-cache --overwrite

Produces exactly the same JSON as ./run.sh for each video (same Pipeline code path).
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from prism.config import Config  # noqa: E402
from prism.features import load_features  # noqa: E402
from prism.pipeline import Pipeline  # noqa: E402

EXTS = (".mp4", ".MP4", ".mkv", ".webm", ".mov", ".MOV")


def index_videos(dirs):
    idx = {}
    for d in dirs:
        for p in Path(d).rglob("*"):
            if p.suffix in EXTS and p.is_file() and not p.name.endswith(".part"):
                idx.setdefault(p.stem, p)
    return idx


def spawn_workers(a) -> int:
    """Re-launch this script as N shard processes; longest videos are spread evenly."""
    import subprocess

    args = [x for x in sys.argv[1:]]
    for flag in ("--workers", "--shard"):
        while flag in args:
            i = args.index(flag)
            del args[i:i + 2]
    procs = []
    for i in range(a.workers):
        log = Path(a.out) / f"_worker{i}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "w")
        env = {**__import__("os").environ, "PRISM_WORKERS": str(a.workers)}
        procs.append((subprocess.Popen([sys.executable, __file__, *args, "--shard", f"{i}/{a.workers}"],
                                       stdout=fh, stderr=subprocess.STDOUT, env=env), fh, log))
        print(f"worker {i}: pid {procs[-1][0].pid}, log {log}")
    rc = 0
    for p, fh, log in procs:
        rc |= p.wait()
        fh.close()
        print(f"{log.name}: " + (log.read_text().strip().splitlines() or ["(empty)"])[-1])
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "package" / "manifest.csv"))
    ap.add_argument("--videos", nargs="*", default=[str(ROOT / "data")], help="dirs searched recursively")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--cache", default=str(ROOT / "cache" / "features"), help="'' disables caching")
    ap.add_argument("--config", default=None)
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", choices=["epic_kitchens_100", "fpvsum"], default=None)
    ap.add_argument("--ids", default=None, help="comma-separated subset of video ids")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th video starting at i")
    ap.add_argument("--workers", type=int, default=1,
                    help="spawn this many shard processes in parallel (each loads its own models; "
                         "~3 GB VRAM each with CLIP-L + Omnivore fp16; 2-3 suits a 12 GB card)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--from-cache", action="store_true", help="use cached features only; skip videos without")
    a = ap.parse_args()

    if a.workers > 1:
        sys.exit(spawn_workers(a))
    cfg_path = a.config or (ROOT / "configs" / "submission.json")
    cfg = Config.load(cfg_path if Path(cfg_path).exists() else None, team_id=a.team_id, device=a.device)
    with open(a.manifest) as fh:
        rows = list(csv.DictReader(fh))
    if a.only:
        rows = [r for r in rows if r["dataset"] == a.only]
    if a.ids:
        keep = set(a.ids.split(","))
        rows = [r for r in rows if r["video_id"] in keep]
    i, n = map(int, a.shard.split("/"))
    rows.sort(key=lambda r: -float(r["duration_sec"]))  # longest first: balanced shards, early ETA
    rows = rows[i::n]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vids = {} if a.from_cache else index_videos(a.videos)
    pipe = Pipeline(cfg)
    done = failed = skipped = 0
    t0 = time.time()
    for k, r in enumerate(rows):
        vid = r["video_id"]
        out = out_dir / f"{vid}.json"
        if out.exists() and not a.overwrite:
            skipped += 1
            continue
        cache = Path(a.cache) / f"{vid}.npz" if a.cache else None
        try:
            if a.from_cache:
                if cache is None or not cache.exists():
                    print(f"[{k + 1}/{len(rows)}] {vid}: no cached features, skipped")
                    skipped += 1
                    continue
                import json

                res = pipe.from_features(load_features(cache), vid)
                out.write_text(json.dumps(res, indent=1))
            else:
                if vid not in vids:
                    print(f"[{k + 1}/{len(rows)}] {vid}: video file not found, skipped")
                    skipped += 1
                    continue
                print(f"[{k + 1}/{len(rows)}] {vid}  ({r['duration_sec']} s)")
                res = pipe.run(str(vids[vid]), out, video_id=vid, cache=cache)
            if abs(res["duration_sec"] - float(r["duration_sec"])) > 0.5:
                print(f"   ! duration {res['duration_sec']} vs manifest {r['duration_sec']} "
                      f"(> 0.5 s: this file would fail; re-download the video)")
            done += 1
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"   x {vid} failed:\n{traceback.format_exc()}")
    print(f"done={done} skipped={skipped} failed={failed} in {(time.time() - t0) / 60:.1f} min")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
