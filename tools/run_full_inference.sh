#!/usr/bin/env bash
# Helper to submit math inference Slurm jobs for every local checkpoint and HF revision.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SLURM_SCRIPT="ops/slurm/infer_math.slurm"
DATASETS="${DATASETS:-math_500,aime24,aime25,amc,minerva}"
SEEDS="${SEEDS:-0,1,2,3,4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

submit_job() {
  local model="$1"
  local revision="${2:-}"
  local dataset="${3:-$DATASETS}"
  local extra="${4:-}"
  local cmd=(sbatch "$SLURM_SCRIPT" --model "$model" --datasets "$dataset" --seeds "$SEEDS" --num-generations "$NUM_GENERATIONS" --temperature "$TEMPERATURE" --batch-size "$BATCH_SIZE")
  if [[ -n "$revision" ]]; then
    cmd+=(--revision "$revision")
  fi
  if [[ -n "$extra" ]]; then
    cmd+=(--extra-opts "$extra")
  fi
  echo "[submit] ${cmd[*]}"
  "${cmd[@]}"
}

# --- Local checkpoints ------------------------------------------------------
local_roots=(
  "var/data/Qwen2.5-7B-Open-R1-GRPO-math-2k"
)

# Split datasets into an array so each dataset gets its own job.
IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"

for root in "${local_roots[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "[warn] skip missing local root: $root"
    continue
  fi
  while IFS= read -r -d '' ckpt; do
    for dataset in "${DATASET_ARRAY[@]}"; do
      submit_job "$ckpt" "" "$dataset" "$EXTRA_OPTS"
    done
  done < <(find "$root" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -print0 | sort -zV)
done

# --- Hugging Face revisions -------------------------------------------------
declare -A hf_models
#hf_models["od2961/Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-math-v1"]="3500eb3 62b7942 b45b0e3 53baa48 c9e3c93 9ae8546 5fa2604 43cd9cf 1a39473 4577c15 c929c65 140288f e3df397 75ec870 eea25a0"
#hf_models["od2961/Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-BASELINE-math-v1"]="06d9bff 208e3e8 f05074b 2f5351e 8002115 69be8c8 240c5ed 58f2a4b 2687728 d087506 35326ed"

for model in "${!hf_models[@]}"; do
  for rev in ${hf_models[$model]}; do
    submit_job "$model" "$rev"
  done
done
