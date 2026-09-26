#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

result="$ROOT_DIR/var/artifacts/e50d_matched_math_toy_advancement_v1.json"
if [[ ! -f "$result" ]] || ! jq -e \
  '.schema == "e50_matched_math_toy_advancement_v1"
   and .variant == "e50d"
   and .complete_evidence == true
   and .advance_to_exact_oat_full == true
   and ([.checks[]] | all)' \
  "$result" >/dev/null; then
  echo "E50E requires a passing E50D matched toy" >&2
  exit 1
fi

manifest="$ROOT_DIR/var/data/e50e_exact_oat_math_full/MATERIALIZATION_MANIFEST.json"
if [[ -e "$manifest" ]]; then
  echo "fresh E50E materialization required: $manifest" >&2
  exit 1
fi
for fresh in \
  "$ROOT_DIR/var/artifacts/e50e_qwen72_node302_v1/qwen72_endpoint.json" \
  "$ROOT_DIR/var/artifacts/e50e_route_confusion_continuity_v1/result.json" \
  "$ROOT_DIR/var/artifacts/e50e_declaration_mismatch_continuity_v1/result.json" \
  "$ROOT_DIR/var/artifacts/e50e_qwen72_continuity_v1/certificate.json"; do
  if [[ -e "$fresh" ]]; then
    echo "fresh E50E continuity artifact required: $fresh" >&2
    exit 1
  fi
done
job_id="$(sbatch --parsable \
  "$ROOT_DIR/ops/slurm/e50e_qwen72_continuity_node302.slurm")"
printf '%s\n' "$job_id" \
  > "$ROOT_DIR/var/artifacts/e50e_qwen72_continuity_job.txt"
printf 'submitted E50E calibrated Qwen72 continuity job %s\n' "$job_id"
