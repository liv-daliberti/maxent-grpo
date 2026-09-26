#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COHORT_DIR="$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1"
if [[ ! -f "$COHORT_DIR/frozen_identity.json" ]] \
  || [[ ! -f "$COHORT_DIR/cohort.jsonl" ]] \
  || [[ ! -f "$COHORT_DIR/private/labels.jsonl" ]]; then
  echo "E49T frozen route-confusion cohort is missing" >&2
  exit 1
fi
if [[ -e "$COHORT_DIR/result.json" ]]; then
  echo "fresh E49T calibration result required" >&2
  exit 1
fi

job_id="$(
  sbatch --parsable ops/slurm/e49t_route_confusion_node302.slurm
)"
printf '%s\n' "$job_id" > "$COHORT_DIR/slurm_job_id.txt"
echo "submitted E49T route-confusion calibration job $job_id"
