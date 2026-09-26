#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

for run in \
  "$ROOT_DIR/var/data/xdr_qwen25_0p5b_instruct_grpo_e49t_natural_menu_math_toy_05b_3ep_v1_grpo_s45" \
  "$ROOT_DIR/var/data/xdr_qwen25_0p5b_instruct_online_canonical_haarnoja_e49t_natural_menu_math_toy_05b_3ep_v1_online_canonical_haarnoja_s45"; do
  if [[ ! -f "$run/TRAINING_COMPLETE.json" ]]; then
    echo "E49T matched arm is not complete: $run" >&2
    exit 1
  fi
done

job_id="$(
  sbatch --parsable \
    "$ROOT_DIR/ops/slurm/e49t_terminal_gate_all.slurm"
)"
printf '%s\n' "$job_id" > \
  "$ROOT_DIR/var/artifacts/e49t_terminal_gate_replacement_job.txt"
echo "submitted E49T general-partition terminal gate job $job_id"
