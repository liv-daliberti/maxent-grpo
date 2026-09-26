#!/usr/bin/env bash
# Refresh E70 Stage A audit, four five-seed curves, and the wide live figure.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/ops/exp_scaling/audit_e70_clean_stage_a_05b.py" ]]; then
  echo "E70 Stage-A monitor cannot resolve repository root: $ROOT_DIR" >&2
  exit 1
fi

PYTHON_BIN="${OAT_ZERO_E70_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E70_MONITOR_INTERVAL_SECONDS:-60}"
AUDIT="$ROOT_DIR/var/artifacts/e70_clean_stage_a_05b_audit_latest.json"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e70-stage-a-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e70-stage-a-monitor-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e70_clean_stage_a_05b.py || true
  for prefix in \
    gce70_clean_stage_a_05b_12pass \
    cde70_clean_stage_a_05b_12pass \
    pye70_clean_stage_a_05b_12pass \
    mie70_clean_stage_a_05b_12pass; do
    "$PYTHON_BIN" ops/exp_scaling/parse_scaling_curve.py \
      --stamp-prefix "$prefix" || true
  done
  "$PYTHON_BIN" ops/exp_scaling/plot_e70_clean_05b_wide_live.py || true
  "$PYTHON_BIN" ops/exp_scaling/audit_e70_full80_campaign.py || true

  status="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
      "$AUDIT" 2>/dev/null || true
  )"
  if [[ "$status" == pass ]]; then
    echo "[e70-stage-a-monitor] terminal clean audit; exiting"
    exit 0
  elif [[ "$status" == fail ]]; then
    echo "[e70-stage-a-monitor] fail-closed audit violation; exiting" >&2
    exit 1
  fi
  sleep "$INTERVAL_SECONDS"
done
