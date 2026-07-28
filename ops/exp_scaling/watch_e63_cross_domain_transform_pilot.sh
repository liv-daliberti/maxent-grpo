#!/usr/bin/env bash
# Refresh E63's audit and expanding cross-domain graph.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
PYTHON_BIN="${OAT_ZERO_E63_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
AUDIT="$ROOT_DIR/var/artifacts/e63_cross_domain_transform_pilot_audit_latest.json"
INTERVAL_SECONDS="${OAT_ZERO_E63_MONITOR_INTERVAL_SECONDS:-60}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e63-transform-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e63-transform-monitor-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e63_cross_domain_transform_pilot.py \
    || true
  "$PYTHON_BIN" ops/exp_scaling/plot_e63_cross_domain_transform_pilot.py \
    || true
  status="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
      "$AUDIT" 2>/dev/null || true
  )"
  case "$status" in
    pass|fail)
      echo "[e63-monitor] terminal audit status=$status; exiting"
      exit 0
      ;;
  esac
  sleep "$INTERVAL_SECONDS"
done
