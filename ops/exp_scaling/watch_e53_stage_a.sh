#!/usr/bin/env bash
# Refresh E53 Stage A's audit, dashboard, curve JSON, and 0–50-pass graph.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
PYTHON_BIN="${OAT_ZERO_E53_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E53_MONITOR_INTERVAL_SECONDS:-60}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e53-stage-a-monitor-matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT_DIR"
while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e53_stage_a.py || true
  "$PYTHON_BIN" ops/exp_scaling/refresh_latest_freeform_05b.py \
    --current-canonical-only || true
  "$PYTHON_BIN" ops/exp_scaling/monitor_campaign.py \
    --current-canonical-only --once --no-clear || true
  sleep "$INTERVAL_SECONDS"
done
