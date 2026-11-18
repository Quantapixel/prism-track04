#!/usr/bin/env bash
# Run a heavy command without making the desktop unusable:
#   lowest CPU + I/O priority, limited threads, and a hard RAM cap (only this command is
#   killed if it exceeds it - never your browser).
#   code/tools/gentle.sh python code/tools/run_all.py --videos data
#   MEM=6G THREADS=6 code/tools/gentle.sh ./run.sh video.mp4 out.json
set -euo pipefail
MEM="${MEM:-5G}"; THREADS="${THREADS:-4}"
export PRISM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  exec systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 \
       -p CPUQuota="$((THREADS * 100))%" nice -n 19 ionice -c3 "$@"
fi
exec nice -n 19 ionice -c3 "$@"   # no systemd: priority only
