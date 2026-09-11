#!/usr/bin/env bash
# Refresh E64's audit, standalone figure, and shared five-domain figure.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/ops/exp_scaling/audit_e64_math500_realism_matched.py" ]]; then
  echo "E64 monitor cannot resolve repository root: $ROOT_DIR" >&2
  exit 1
fi

PYTHON_BIN="${OAT_ZERO_E64_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E64_MONITOR_INTERVAL_SECONDS:-60}"
AUDIT="$ROOT_DIR/var/artifacts/e64_math500_realism_matched_audit_latest.json"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e64-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e64-monitor-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e64_math500_realism_matched.py || true
  "$PYTHON_BIN" ops/exp_scaling/plot_e64_math500_realism.py || true
  "$PYTHON_BIN" ops/exp_scaling/plot_e61r1_e58_vs_grpo_12pass.py || true

  status="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
      "$AUDIT" 2>/dev/null || true
  )"
  if [[ "$status" == pass ]]; then
    echo "[e64-monitor] terminal clean audit; exiting"
    exit 0
  fi
  if [[ "$status" == fail ]]; then
    echo "[e64-monitor] fail-closed audit violation; exiting" >&2
    exit 1
  fi
  sleep "$INTERVAL_SECONDS"
done
