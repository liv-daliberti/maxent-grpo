#!/usr/bin/env bash
# Continue E70 audits, curves, and the historical graph until all 80 cells pass.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
PYTHON_BIN="${OAT_ZERO_E70_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E70_MONITOR_INTERVAL_SECONDS:-60}"
FULL_AUDIT="$ROOT_DIR/var/artifacts/e70_clean_full80_campaign_audit_latest.json"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e70-full80-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e70-full80-monitor-cache"
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
      "$FULL_AUDIT" 2>/dev/null || true
  )"
  if [[ "$status" == pass ]]; then
    if "$PYTHON_BIN" ops/exp_scaling/parse_scaling_curve.py \
        --stamp-prefix pprepair_support_retention_final_v1 \
      && "$PYTHON_BIN" \
        ops/exp_scaling/build_e70_historical_multipage_v7_20260730.py; then
      echo "[e70-full80-monitor] all 80 cells audited terminal; " \
        "complete report refreshed; exiting"
      exit 0
    fi
    echo "[e70-full80-monitor] terminal cells are clean but the complete " \
      "report refresh failed; retrying" >&2
    sleep "$INTERVAL_SECONDS"
    continue
  elif [[ "$status" == fail ]]; then
    echo "[e70-full80-monitor] fail-closed campaign violation; exiting" >&2
    exit 1
  fi
  sleep "$INTERVAL_SECONDS"
done
