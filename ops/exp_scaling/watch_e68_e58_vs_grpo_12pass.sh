#!/usr/bin/env bash
# Refresh the five-domain E68/E69 decision figure once per minute.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
PLOTTER="$ROOT_DIR/ops/exp_scaling/plot_e68_e58_vs_grpo_12pass.py"
if [[ ! -f "$PLOTTER" ]]; then
  echo "E68 live-figure monitor cannot resolve repository root: $ROOT_DIR" >&2
  exit 1
fi

PYTHON_BIN="${OAT_ZERO_E68_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E68_MONITOR_INTERVAL_SECONDS:-60}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e68-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e68-monitor-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$ROOT_DIR"

while true; do
  cycle_started="$SECONDS"
  date -Ins
  "$PYTHON_BIN" "$PLOTTER" || true
  elapsed="$((SECONDS - cycle_started))"
  delay="$((INTERVAL_SECONDS - elapsed))"
  if (( delay < 1 )); then
    delay=1
  fi
  sleep "$delay"
done
