#!/usr/bin/env bash
# Refresh the hard-MATH E50 status figure and JSON tracker every 60 seconds.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

export MPLCONFIGDIR="$ROOT_DIR/var/cache/matplotlib"
export XDG_CACHE_HOME="$ROOT_DIR/var/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

LOG="$ROOT_DIR/var/artifacts/e50_hard_math_status_monitor.log"
STOP="$ROOT_DIR/var/artifacts/e50_hard_math_status_monitor.stop"

while [[ ! -f "$STOP" ]]; do
  cycle_started="$(date +%s)"
  {
    printf '[%s] refreshing E50 hard-MATH status\n' \
      "$(date '+%F %T %Z')"
    python "$ROOT_DIR/ops/exp_scaling/plot_e50_math_status.py"
  } >>"$LOG" 2>&1
  cycle_elapsed="$(( $(date +%s) - cycle_started ))"
  cycle_delay="$(( 60 - cycle_elapsed ))"
  if (( cycle_delay > 0 )); then
    sleep "$cycle_delay"
  fi
done

printf '[%s] E50 hard-MATH status monitor stopped\n' \
  "$(date '+%F %T %Z')" >>"$LOG"
