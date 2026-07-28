#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULT="$ROOT_DIR/var/artifacts/e49t_toy_route_coverage_v1/result.json"
if [[ -e "$RESULT" ]]; then
  echo "Fresh E49T route-coverage result required: $RESULT" >&2
  exit 1
fi

job_id="$(sbatch --parsable \
  --dependency=afterok:30124235:30124236 \
  "$ROOT_DIR/ops/slurm/e49t_toy_route_coverage_node302.slurm")"
printf '%s\n' "$job_id" > \
  "$ROOT_DIR/var/artifacts/e49t_toy_route_coverage_job.txt"
printf 'submitted dependent E49T terminal route-coverage job %s\n' "$job_id"
