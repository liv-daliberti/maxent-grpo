#!/usr/bin/env bash
# Refresh the current sentinel audit, dashboard, curves, and expanding graph.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/ops/exp_scaling/audit_e57_sentinel.py" ]]; then
  echo "Sentinel monitor cannot resolve repository root: $ROOT_DIR" >&2
  exit 1
fi
PYTHON_BIN="${OAT_ZERO_E55_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
AUDIT_PYTHON_BIN="${OAT_ZERO_E58_AUDIT_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E55_MONITOR_INTERVAL_SECONDS:-60}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e55-monitor-matplotlib"
mkdir -p "$MPLCONFIGDIR" "$ROOT_DIR/var/artifacts/logs"
cd "$ROOT_DIR"

while true; do
  date
  if [[ -f \
    var/artifacts/e58_global_verified_replay_canonical_05b_sentinel_identity.json
  ]]; then
    AUDITOR=ops/exp_scaling/audit_e58_sentinel.py
    SELECTED_AUDIT_PYTHON="$AUDIT_PYTHON_BIN"
  elif [[ -f \
    var/artifacts/e57_verified_first_split_canonical_05b_sentinel_identity.json
  ]]; then
    AUDITOR=ops/exp_scaling/audit_e57_sentinel.py
    SELECTED_AUDIT_PYTHON="$AUDIT_PYTHON_BIN"
  else
    AUDITOR=ops/exp_scaling/audit_e55_sentinel.py
    SELECTED_AUDIT_PYTHON="$AUDIT_PYTHON_BIN"
  fi
  "$SELECTED_AUDIT_PYTHON" "$AUDITOR" || true
  "$PYTHON_BIN" ops/exp_scaling/refresh_latest_freeform_05b.py \
    --current-canonical-only || true
  "$PYTHON_BIN" ops/exp_scaling/monitor_campaign.py \
    --current-canonical-only --once --no-clear || true
  sleep "$INTERVAL_SECONDS"
done
