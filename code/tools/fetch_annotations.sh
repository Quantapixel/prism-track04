#!/usr/bin/env bash
# Fetch public annotation files into data/annotations/ (kept out of git on purpose:
# the submission must not contain ground truth).
#   EPIC_100_train.csv       - training the head (train split only)
#   EPIC_100_validation.csv  - local scoring ONLY (this is the hidden test set)
#   EPIC_100_video_info.csv  - durations / fps
#   FPVSum_videolist.csv     - YouTube ids for download_fpvsum.py
#   fpvsum_gt_data.mat       - local scoring ONLY
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXT="$ROOT/ext"; OUT="$ROOT/data/annotations"
mkdir -p "$EXT" "$OUT"
[ -d "$EXT/epic-kitchens-100-annotations" ] || git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-100-annotations.git "$EXT/epic-kitchens-100-annotations"
[ -d "$EXT/fpvsum" ] || git clone --depth 1 https://github.com/azuxmioy/fpvsum.git "$EXT/fpvsum"
cp "$EXT/epic-kitchens-100-annotations/"{EPIC_100_train.csv,EPIC_100_validation.csv,EPIC_100_video_info.csv} "$OUT/"
cp "$EXT/fpvsum/crawler/FPVSum_videolist.csv" "$OUT/"
cp "$EXT/fpvsum/annotation/gt_data.mat" "$OUT/fpvsum_gt_data.mat"
ls -la "$OUT"
