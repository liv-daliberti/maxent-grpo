#!/usr/bin/env bash
# Refresh E61's audit, four curve files, and expanding three-seed graph.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/ops/exp_scaling/audit_e61_e58_vs_grpo_12pass.py" ]]; then
  echo "E61 monitor cannot resolve repository root: $ROOT_DIR" >&2
  exit 1
fi

PYTHON_BIN="${OAT_ZERO_E61_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E61_MONITOR_INTERVAL_SECONDS:-60}"
AUDIT="$ROOT_DIR/var/artifacts/e61_e58_vs_grpo_12pass_audit_latest.json"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e61-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e61-monitor-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e61_e58_vs_grpo_12pass.py || true
  for prefix in \
    gce61_e58_vs_grpo_05b_12ep \
    cde61_e58_vs_grpo_05b_12ep \
    pye61_e58_vs_grpo_05b_12ep \
    mie61_e58_vs_grpo_05b_12ep; do
    "$PYTHON_BIN" ops/exp_scaling/parse_scaling_curve.py \
      --stamp-prefix "$prefix" || true
  done
  "$PYTHON_BIN" ops/exp_scaling/plot_e61_e58_vs_grpo_12pass.py || true

  status="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
      "$AUDIT" 2>/dev/null || true
  )"
  if [[ "$status" == pass ]]; then
    echo "[e61-monitor] terminal clean audit; exiting"
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
