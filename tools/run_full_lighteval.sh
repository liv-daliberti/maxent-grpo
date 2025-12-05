#!/usr/bin/env bash
# Submit LightEval Slurm jobs for every local checkpoint.
# Defaults mirror the existing GRPO math run; override via env vars.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SLURM_SCRIPT="ops/slurm/evaluate.slurm"
# Set LIGHEVAL_BACKEND=hf to run LightEval without vLLM.
# Accept DATASETS for compatibility with the old pipeline.
TASKS="${TASKS:-${DATASETS:-math_500,aime24,aime25,amc,minerva}}"
MODEL_ROOT="${MODEL_ROOT:-var/data/Qwen2.5-7B-Open-R1-GRPO-math-2k}"
REVISION="${REVISION:-main}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
EXTRA_OPTS="${EXTRA_OPTS:-}"  # space-separated lighteval args to append to lighteval

# Map friendly task names to LightEval task specs.
declare -A TASK_SPECS=(
  ["math_500"]="lighteval|math_500|0|0"
  ["aime24"]="lighteval|aime24|0|0"
  ["aime25"]="lighteval|aime25|0|0"
  ["gpqa"]="lighteval|gpqa:diamond|0|0"
   # Assume LightEval task ids for amc/minerva; adjust if your install differs.
  ["amc"]="lighteval|amc|0|0"
  ["minerva"]="lighteval|minerva|0|0"
)

submit_job() {
  local task_name="$1"
  local task_spec="$2"
  local model="$3"
  echo "[submit] task=${task_name} model=${model}"
  EXTRA_LIGHEVAL_OPTS="$EXTRA_OPTS" sbatch "$SLURM_SCRIPT" "$task_name" "$task_spec" "$model" "$REVISION" "$TENSOR_PARALLEL" "$TRUST_REMOTE_CODE"
}

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "[warn] MODEL_ROOT not found: $MODEL_ROOT" >&2
  exit 1
fi

while IFS= read -r -d '' ckpt; do
  for task in "${TASK_ARRAY[@]}"; do
    spec="${TASK_SPECS[$task]:-}"
    if [[ -z "$spec" ]]; then
      echo "[warn] Unknown task '${task}' (skipping). Known: ${!TASK_SPECS[*]}"
      continue
    fi
    submit_job "$task" "$spec" "$ckpt"
  done
done < <(find "$MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -print0 | sort -zV)
