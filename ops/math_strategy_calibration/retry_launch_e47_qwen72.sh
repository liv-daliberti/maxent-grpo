#!/usr/bin/env bash
# Retry the idempotent E47 node105 server submission during controller outages.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_RECORD="$ROOT_DIR/var/artifacts/e47_math_strategy_calibration_v1/qwen72_server_job.json"
LAUNCHER="$ROOT_DIR/ops/math_strategy_calibration/launch_e47_qwen72_node105.sh"

for attempt in $(seq 1 240); do
  if [[ -f "$JOB_RECORD" ]]; then
    exit 0
  fi
  if bash "$LAUNCHER"; then
    exit 0
  fi
  sleep 30
done
exit 1
