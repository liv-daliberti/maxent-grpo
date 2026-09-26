#!/usr/bin/env bash
# Refresh the E52 audit, scheduler dashboard, curve JSON, and 50-pass figure.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/ops/exp_scaling/audit_e52_sentinel.py" ]]; then
  echo "E52 monitor cannot resolve repository root: $ROOT_DIR" >&2
  exit 1
fi
PYTHON_BIN="${OAT_ZERO_E52_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E52_MONITOR_INTERVAL_SECONDS:-60}"
STAGE_A_APPROVAL="$ROOT_DIR/var/artifacts/e52_sentinel_stage_a_approval.json"
STAGE_A_IDENTITY="$ROOT_DIR/var/artifacts/e52_direct_inverse_entropy_canonical_05b_stage_a_v1_identity.json"
STAGE_A_LOCK="$ROOT_DIR/var/artifacts/e52_stage_a_autolaunch.lock"
STAGE_A_LOG="$ROOT_DIR/var/artifacts/logs/e52_stage_a_autolaunch.log"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e52-monitor-matplotlib"
mkdir -p "$MPLCONFIGDIR" "$(dirname "$STAGE_A_LOG")"
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e52_sentinel.py || true
  "$PYTHON_BIN" ops/exp_scaling/refresh_latest_freeform_05b.py \
    --current-canonical-only || true
  "$PYTHON_BIN" ops/exp_scaling/monitor_campaign.py \
    --current-canonical-only --once --no-clear || true
  if [[ -f "$STAGE_A_APPROVAL" && ! -f "$STAGE_A_IDENTITY" ]]; then
    if mkdir "$STAGE_A_LOCK" 2>/dev/null; then
      {
        date
        echo "[e52-monitor] terminal approval detected; replaying Stage A"
        if ops/exp_scaling/launch_e52_stage_a.sh stage_a; then
          echo "[e52-monitor] Stage A submitted and released"
        else
          status="$?"
          echo "[e52-monitor] Stage A launch failed closed with status $status"
          false
        fi
      } >>"$STAGE_A_LOG" 2>&1 || true
    fi
  fi
  sleep "$INTERVAL_SECONDS"
done
