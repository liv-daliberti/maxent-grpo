#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

e50k_job_record="$ROOT_DIR/var/artifacts/e50k_fifth_conditioned_teacher_contingency_job.txt"
if [[ ! -f "$e50k_job_record" ]]; then
  echo "E50G requires the submitted E50K predecessor job" >&2
  exit 1
fi
e50k_job="$(tr -d '[:space:]' < "$e50k_job_record")"
if [[ ! "$e50k_job" =~ ^[0-9]+$ ]]; then
  echo "invalid E50K job ID: $e50k_job" >&2
  exit 1
fi

output="$ROOT_DIR/var/artifacts/e50g_safe_signature_teacher_route_calibration_v1"
if [[ -e "$output/result.json" ]]; then
  echo "fresh E50G result required: $output/result.json" >&2
  exit 1
fi

job_id="$(sbatch --parsable \
  --dependency="afterany:$e50k_job" \
  "$ROOT_DIR/ops/slurm/e50g_safe_signature_teacher_route_calibration_node302.slurm")"
printf '%s\n' "$job_id" \
  > "$ROOT_DIR/var/artifacts/e50g_safe_signature_teacher_route_calibration_job.txt"
printf 'submitted E50G safe-signature route job %s after E50K %s\n' \
  "$job_id" "$e50k_job"
