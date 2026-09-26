#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

OUTPUT_ROOT="$ROOT_DIR/var/artifacts/e50a_consensus_observed_route_calibration_v1"
if [[ -e "$OUTPUT_ROOT/result.json" ]]; then
  echo "refusing to overwrite terminal E50A result: $OUTPUT_ROOT/result.json" >&2
  exit 1
fi

JOB_ID="$(sbatch --parsable \
  "$ROOT_DIR/ops/slurm/e50a_consensus_observed_route_calibration_node302.slurm")"
printf '%s\n' "$JOB_ID" \
  > "$ROOT_DIR/var/artifacts/e50a_consensus_route_calibration_job.txt"
printf 'submitted E50A consensus observed-route calibration job %s\n' "$JOB_ID"
