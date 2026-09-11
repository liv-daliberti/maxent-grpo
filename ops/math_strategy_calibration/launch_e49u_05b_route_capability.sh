#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULT="$ROOT_DIR/var/artifacts/e49u_05b_route_capability_calibration_v1/result.json"
if [[ -e "$RESULT" ]]; then
  echo "Fresh E49U route-capability result required: $RESULT" >&2
  exit 1
fi

job_id="$(sbatch --parsable \
  "$ROOT_DIR/ops/slurm/e49u_05b_route_capability_node302.slurm")"
printf '%s\n' "$job_id" > \
  "$ROOT_DIR/var/artifacts/e49u_05b_route_capability_job.txt"
printf 'submitted E49U 0.5B route-capability job %s\n' "$job_id"
