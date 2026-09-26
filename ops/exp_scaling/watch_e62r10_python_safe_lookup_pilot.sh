#!/usr/bin/env bash
# Refresh E62-R10's Python pilot audit, curve JSON, and expanding live graph.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/ops/exp_scaling/audit_e62r10_python_safe_lookup_pilot.py" ]]; then
  echo "E62 monitor cannot resolve repository root: $ROOT_DIR" >&2
  exit 1
fi

PYTHON_BIN="${OAT_ZERO_E62_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E62_MONITOR_INTERVAL_SECONDS:-60}"
AUDIT="$ROOT_DIR/var/artifacts/e62r10_python_pilot_audit_latest.json"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e62-python-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e62-python-monitor-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e62r10_python_safe_lookup_pilot.py || true
  "$PYTHON_BIN" ops/exp_scaling/parse_scaling_curve.py \
    --stamp-prefix pye62r10_safe_lookup_05b_1ep_pilot || true
  "$PYTHON_BIN" ops/exp_scaling/plot_e62r10_python_safe_lookup_pilot.py || true

  status="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
      "$AUDIT" 2>/dev/null || true
  )"
  case "$status" in
    pass|fail)
      echo "[e62-monitor] terminal audit status=$status; exiting"
      exit 0
      ;;
  esac
  sleep "$INTERVAL_SECONDS"
done
