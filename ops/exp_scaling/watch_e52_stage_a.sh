#!/usr/bin/env bash
# Refresh the exact 27-run E52 Stage-A audit until the cohort settles.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
PYTHON_BIN="${OAT_ZERO_E52_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E52_MONITOR_INTERVAL_SECONDS:-60}"
AUDIT="$ROOT_DIR/ops/exp_scaling/audit_e52_stage_a.py"
if [[ ! -f "$AUDIT" ]]; then
  echo "E52 Stage-A monitor cannot resolve auditor: $AUDIT" >&2
  exit 1
fi
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" "$AUDIT" || true
  "$PYTHON_BIN" ops/exp_scaling/refresh_latest_freeform_05b.py \
    --current-canonical-only || true
  "$PYTHON_BIN" ops/exp_scaling/monitor_campaign.py \
    --current-canonical-only --once --no-clear || true
  sleep "$INTERVAL_SECONDS"
done
