#!/usr/bin/env bash
# Assemble the upload folder required by TEAM_README section 6:
#   TEAM_ID/{run.sh, code/, weights/, requirements.txt, outputs/, README.md}
#   code/tools/package_submission.sh T07 [dest_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEAM="${1:?usage: package_submission.sh TEAM_ID [dest]}"
DEST="${2:-$ROOT/submission}/$TEAM"
cd "$ROOT"
grep -q "\"team_id\": \"$TEAM\"" configs/submission.json || {
  echo "configs/submission.json team_id is not $TEAM - set it and regenerate outputs first" >&2; exit 1; }
n=$(ls outputs/*.json 2>/dev/null | wc -l)
python3 code/tools/validate.py --dir outputs/ || { echo "outputs/ does not validate ($n files)" >&2; exit 1; }
[ -f weights/omnivore_swinB_epic.torch ] || { echo "weights missing: python code/tools/fetch_weights.py" >&2; exit 1; }
rm -rf "$DEST"; mkdir -p "$DEST"
cp run.sh requirements.txt README.md "$DEST/"
mkdir -p "$DEST/code" "$DEST/configs"
rsync -a --exclude __pycache__ code/ "$DEST/code/"
cp configs/submission.json "$DEST/configs/"
cp -r package "$DEST/package"   # supported_classes.json + manifest for the tools
rsync -a --exclude README.md weights/ "$DEST/weights/"
cp weights/README.md "$DEST/weights/README.md"
rsync -a outputs/ "$DEST/outputs/"
# integrity: no ground truth may ship
if grep -rlE "gt_data|EPIC_100_validation" "$DEST" --include=*.csv --include=*.mat >/dev/null 2>&1; then
  echo "ground-truth file found in $DEST - aborting" >&2; exit 1; fi
du -sh "$DEST"; echo "ready: $DEST"
