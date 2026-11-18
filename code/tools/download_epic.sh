#!/usr/bin/env bash
# Download EPIC-KITCHENS-100 videos with the official downloader (resumable per file).
#   code/tools/download_epic.sh                 # the 138 manifest (validation) videos, ~184 GB
#   code/tools/download_epic.sh P01_01,P01_02   # specific ids
#   OUT=/mnt/big/EPIC code/tools/download_epic.sh   # custom location
# Files land in $OUT/EPIC-KITCHENS/<participant>/videos/<id>.MP4
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXT="$ROOT/ext"; OUT="${OUT:-$ROOT/data}"
mkdir -p "$EXT" "$OUT"
[ -d "$EXT/epic-kitchens-download-scripts" ] || git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git "$EXT/epic-kitchens-download-scripts"
if [ "$#" -ge 1 ]; then
  IDS="$1"
else
  IDS="$(awk -F, 'NR>1 && $3=="epic_kitchens_100" {print $1}' "$ROOT/package/manifest.csv" | paste -sd, -)"
fi
# skip ids that are already fully present
TODO=()
IFS=',' read -ra ALL <<< "$IDS"
for id in "${ALL[@]}"; do
  f="$OUT/EPIC-KITCHENS/${id%%_*}/videos/$id.MP4"
  if [ -s "$f" ]; then echo "have $id"; else TODO+=("$id"); fi
done
if [ "${#TODO[@]}" -eq 0 ]; then echo "all present"; exit 0; fi
echo "downloading ${#TODO[@]} video(s) to $OUT/EPIC-KITCHENS"
cd "$EXT/epic-kitchens-download-scripts"
python3 epic_downloader.py --videos --specific-videos "$(IFS=,; echo "${TODO[*]}")" --output-path "$OUT"
