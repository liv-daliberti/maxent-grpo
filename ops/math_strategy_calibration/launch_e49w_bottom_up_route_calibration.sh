#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULT="$ROOT_DIR/var/artifacts/e49w_bottom_up_05b_route_calibration_v1/result.json"
if [[ -e "$RESULT" ]]; then
  echo "E49W result already exists: $RESULT" >&2
  exit 1
fi

job_id="$(
  sbatch --parsable \
    "$ROOT_DIR/ops/slurm/e49w_bottom_up_route_calibration_node302.slurm"
)"
printf '%s\n' "$job_id" > \
  "$ROOT_DIR/var/artifacts/e49w_bottom_up_route_calibration_job.txt"
echo "submitted E49W bottom-up route calibration job $job_id"
