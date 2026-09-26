#!/usr/bin/env bash
# Refresh the E71 audit and both 384/128 five-seed scaling curves.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/ops/exp_scaling/audit_e71_scale384_05b.py" ]]; then
  echo "E71 monitor cannot resolve repository root: $ROOT_DIR" >&2
  exit 1
fi

PYTHON_BIN="${OAT_ZERO_E71_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E71_MONITOR_INTERVAL_SECONDS:-60}"
AUDIT="$ROOT_DIR/var/artifacts/e71_scale384_05b_audit_latest.json"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e71-scale384-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e71-scale384-monitor-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e71_scale384_05b.py || true
  for prefix in \
    gce71_scale384_05b_12pass \
    ppe71_scale384_05b_12pass; do
    "$PYTHON_BIN" ops/exp_scaling/parse_scaling_curve.py \
      --stamp-prefix "$prefix" || true
  done
  # Refresh the manuscript-facing surface. The aggregator prefers an E71 curve
  # over its E70 predecessor as soon as one exists, so this is what carries the
  # supersession through to the paper's results files.
  "$PYTHON_BIN" ops/exp_scaling/aggregate_e70_paper_checkpoints.py || true

  status="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
      "$AUDIT" 2>/dev/null || true
  )"
  if [[ "$status" == pass ]]; then
    echo "[e71-scale384-monitor] terminal clean audit; exiting"
    exit 0
  elif [[ "$status" == fail ]]; then
    echo "[e71-scale384-monitor] fail-closed audit violation; exiting" >&2
    exit 1
  fi
  sleep "$INTERVAL_SECONDS"
done
