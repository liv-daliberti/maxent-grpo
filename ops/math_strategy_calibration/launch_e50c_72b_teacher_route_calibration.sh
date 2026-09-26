#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

for result in \
  "$ROOT_DIR/var/artifacts/e49aa_calibrated_pairwise_observed_route_discovery_v1/result.json" \
  "$ROOT_DIR/var/artifacts/e49ab_all_observed_persistent_pairwise_route_discovery_v1/result.json" \
  "$ROOT_DIR/var/artifacts/e50a_consensus_observed_route_calibration_v1/result.json" \
  "$ROOT_DIR/var/artifacts/e49ac_confirmed_singleton_observed_route_discovery_v1/result.json"; do
  if [[ ! -f "$result" ]]; then
    echo "E50C waits for terminal predecessor: $result" >&2
    exit 1
  fi
  if jq -e '.pass == true and (.selected_source_indices | length) == 10' \
    "$result" >/dev/null; then
    echo "E50C is inactive because an observed-route source passed: $result" >&2
    exit 1
  fi
done

OUTPUT="$ROOT_DIR/var/artifacts/e50c_72b_teacher_route_calibration_v1"
if [[ -e "$OUTPUT/result.json" ]]; then
  echo "fresh E50C result required: $OUTPUT/result.json" >&2
  exit 1
fi

JOB_ID="$(sbatch --parsable \
  "$ROOT_DIR/ops/slurm/e50c_72b_teacher_route_calibration_node302.slurm")"
printf '%s\n' "$JOB_ID" \
  > "$ROOT_DIR/var/artifacts/e50c_72b_teacher_route_calibration_job.txt"
printf 'submitted E50C 72B-teacher route calibration job %s\n' "$JOB_ID"
