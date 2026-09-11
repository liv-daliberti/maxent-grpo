#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULT="$ROOT_DIR/var/artifacts/e49y_observed_route_discovery_v1/result.json"
if [[ -e "$RESULT" ]]; then
  echo "E49Y result already exists: $RESULT" >&2
  exit 1
fi

job_id="$(
  sbatch --parsable \
    "$ROOT_DIR/ops/slurm/e49y_observed_route_discovery_node302.slurm"
)"
printf '%s\n' "$job_id" > \
  "$ROOT_DIR/var/artifacts/e49y_observed_route_discovery_job.txt"
echo "submitted E49Y observed-route discovery job $job_id"
