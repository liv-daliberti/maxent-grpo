#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

result="$ROOT_DIR/var/artifacts/e50c_72b_teacher_route_calibration_v1/result.json"
if [[ ! -f "$result" ]]; then
  echo "E50F waits for terminal E50C: $result" >&2
  exit 1
fi
# E50C/E50F route-identity decisions were quarantined before either became
# terminal after the prospective E50F2/E50F4 audits observed false-new
# errors.  E50F now always runs as the frozen corpus producer for E50G.

output="$ROOT_DIR/var/artifacts/e50f_conditioned_teacher_route_calibration_v1"
if [[ -e "$output/result.json" ]]; then
  echo "fresh E50F result required: $output/result.json" >&2
  exit 1
fi

job_id="$(sbatch --parsable \
  "$ROOT_DIR/ops/slurm/e50f_conditioned_teacher_route_calibration_node302.slurm")"
printf '%s\n' "$job_id" \
  > "$ROOT_DIR/var/artifacts/e50f_conditioned_teacher_route_calibration_job.txt"
printf 'submitted E50F conditioned-teacher route job %s\n' "$job_id"
