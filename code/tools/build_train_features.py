#!/usr/bin/env python3
"""Build Omnivore features for EPIC-100 *train* videos: download -> extract -> delete.

EPIC train is ~75 h / ~1 TB of 1080p video, so each video is downloaded, turned into a
~10-30 MB .npz, and deleted immediately (unless --keep-videos). The next video downloads
in the background while the current one is processed. Resumable: finished videos are
skipped.

  python code/tools/build_train_features.py --hours 25           # ~25 h of train video
  python code/tools/build_train_features.py --hours 75           # the whole train split

Validation (= manifest = hidden test) videos are refused.
"""
import argparse
import csv
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from prism.config import Config  # noqa: E402
from prism.features import FeatureExtractor, save_features  # noqa: E402

ANN = ROOT / "data" / "annotations"


def manifest_ids() -> set:
    with open(ROOT / "package" / "manifest.csv") as fh:
        return {r["video_id"] for r in csv.DictReader(fh)}


def select(hours: float, seed: int) -> list[tuple[str, float]]:
    with open(ANN / "EPIC_100_video_info.csv") as fh:
        dur = {r["video_id"]: float(r["duration"]) for r in csv.DictReader(fh)}
    with open(ANN / "EPIC_100_train.csv") as fh:
        train = sorted({r["video_id"] for r in csv.DictReader(fh)})
    bad = set(train) & manifest_ids()
    assert not bad, f"train/manifest overlap: {bad}"
    by_p = defaultdict(list)
    for v in train:
        by_p[v.split("_")[0]].append(v)
    rng = random.Random(seed)
    for v in by_p.values():
        rng.shuffle(v)
    order, ps = [], sorted(by_p)
    while any(by_p.values()):  # round-robin over participants -> diverse kitchens
        for p in ps:
            if by_p[p]:
                order.append(by_p[p].pop())
    out, tot = [], 0.0
    for v in order:
        if tot >= hours * 3600:
            break
        out.append((v, dur[v]))
        tot += dur[v]
    return out


def download(vid: str, tmp: Path) -> Path | None:
    f = tmp / "EPIC-KITCHENS" / vid.split("_")[0] / "videos" / f"{vid}.MP4"
    if f.exists() and f.stat().st_size > 0:
        return f
    env = {"OUT": str(tmp), "PATH": __import__("os").environ.get("PATH", "")}
    r = subprocess.run(["bash", str(ROOT / "code" / "tools" / "download_epic.sh"), vid], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return f if (r.returncode == 0 and f.exists() and f.stat().st_size > 0) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=str(ROOT / "cache" / "train"))
    ap.add_argument("--tmp", default=str(ROOT / "data" / "train_tmp"), help="download scratch dir")
    ap.add_argument("--keep-videos", action="store_true")
    ap.add_argument("--video-dir", default=None, help="use already-downloaded videos from here (no download)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prefetch", type=int, default=2)
    a = ap.parse_args()

    cfg = Config.load(a.config, device=a.device)
    device = cfg.resolve_device()
    cache, tmp = Path(a.cache), Path(a.tmp)
    cache.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    sel = select(a.hours, a.seed)
    (cache / "selection.txt").write_text("\n".join(v for v, _ in sel) + "\n")
    todo = [(v, d) for v, d in sel if not (cache / f"{v}.npz").exists()]
    print(f"{len(sel)} train videos selected ({sum(d for _, d in sel) / 3600:.1f} h); "
          f"{len(todo)} still to process")
    if not todo:
        return

    from prism.backbones import OmnivoreEpic
    from prism.config import weights_path

    omni = OmnivoreEpic(weights_path(cfg.omni_weights)).eval().to(device)
    fx = FeatureExtractor(cfg, device, omnivore=omni, clip=None)

    q: queue.Queue = queue.Queue(maxsize=max(1, a.prefetch))

    def producer():
        for v, _ in todo:
            if a.video_dir:
                hits = list(Path(a.video_dir).rglob(f"{v}.MP4")) + list(Path(a.video_dir).rglob(f"{v}.mp4"))
                q.put((v, hits[0] if hits else None))
            else:
                q.put((v, download(v, tmp)))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t0, done = time.time(), 0
    while True:
        item = q.get()
        if item is None:
            break
        v, path = item
        if path is None:
            print(f"!! {v}: download failed / not found, skipped")
            continue
        try:
            feats = fx.extract(str(path), with_omni=True)
            save_features(cache / f"{v}.npz", feats)
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"!! {v}: {e}")
        finally:
            if not a.keep_videos and not a.video_dir:
                path.unlink(missing_ok=True)
        el = time.time() - t0
        print(f"   {done}/{len(todo)} done, {el / 60:.1f} min elapsed, "
              f"~{el / max(done, 1) * (len(todo) - done) / 60:.0f} min left")
    if not a.keep_videos and not a.video_dir:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
