#!/usr/bin/env bash
# Run one command with a hard memory ceiling, and never two at once.
#
#   bash scripts/run_capped.sh bash scripts/chain_ladder.sh
#   LTH_MEM_MAX=6G bash scripts/run_capped.sh python3 -m lth.train ...
#
# - The command runs in its own cgroup with MemoryMax and no swap. If it outgrows the cap,
#   the kernel kills that command only; the desktop is not pushed into swap.
# - A lock stops a second job from starting while one is running.
# - It refuses to start when the machine is already short of memory or swap is in use.
set -euo pipefail

CAP=${LTH_MEM_MAX:-8G}
MIN_FREE_GB=${LTH_MIN_FREE_GB:-10}
LOCK="$(cd "$(dirname "$0")/.." && pwd)/.lth-job.lock"

if [ $# -eq 0 ]; then
  echo "usage: run_capped.sh COMMAND [ARGS...]" >&2
  exit 2
fi

avail_gb=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
swap_used_mb=$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {printf "%d", (t - f) / 1024}' /proc/meminfo)
if [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
  echo "run_capped: only ${avail_gb} GB of RAM available (need ${MIN_FREE_GB}); not starting." >&2
  exit 1
fi
if [ "$swap_used_mb" -gt 256 ]; then
  echo "run_capped: ${swap_used_mb} MB of swap already in use, the machine is under memory pressure; not starting." >&2
  exit 1
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "run_capped: another job holds $LOCK; not starting a second one." >&2
  exit 1
fi
exec systemd-run --user --scope --quiet -p MemoryMax="$CAP" -p MemorySwapMax=0 "$@"
