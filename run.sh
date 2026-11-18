#!/usr/bin/env bash
# Official entry point:  ./run.sh <video.mp4> <out.json>
# Extra flags are passed through to code/infer.py (e.g. --team-id, --device cpu).
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "usage: ./run.sh <video.mp4> <out.json> [--device cuda|cpu] [--config cfg.json]" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$HERE/.venv/bin/python" ]; then PY="$HERE/.venv/bin/python"; else PY="python3"; fi
fi
CFG=()
if [ -f "$HERE/configs/submission.json" ]; then CFG=(--config "$HERE/configs/submission.json"); fi
exec "$PY" "$HERE/code/infer.py" "$1" "$2" "${CFG[@]}" "${@:3}"
