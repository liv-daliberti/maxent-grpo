#!/bin/bash
# Slurm launcher for multi-benchmark math inference (math_500, AIME24/25, AMC, Minerva).
# Example:
#   sbatch ops/slurm/infer_math.slurm \
#     --model od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-v1 \
#     --datasets math_500,aime24,aime25,amc,minerva \
#     --seeds 0,1,2,3,4 \
#     --num-generations 8 \
#     --temperature 0.6

#SBATCH --job-name=math_eval
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=mltheory
#SBATCH --output=var/logs/%x-%j.out
#SBATCH --error=var/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --account=mltheory
#SBATCH --export=ALL,ENV_MODE=venv,ENV_ACTIVATE=/n/fs/similarity/maxent-grpo/var/openr1/bin/activate

set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-$PWD}"
VAR_DIR="${VAR_DIR:-$ROOT_DIR/var}"
mkdir -p "$VAR_DIR/logs" "$VAR_DIR/cache"

# --- Environment bootstrap (minimal, mirrors train.slurm) ---
ENV_MODE="${ENV_MODE:-venv}"
CONDA_SH="${CONDA_SH:-/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-$VAR_DIR/openr1}"
ENV_ACTIVATE="${ENV_ACTIVATE:-$VAR_DIR/openr1/bin/activate}"
if [[ "$ENV_MODE" == "conda" ]]; then
  if [[ -f "$CONDA_SH" ]]; then source "$CONDA_SH"; fi
  conda activate "$CONDA_ENV" || echo "[warn] conda activate failed"
elif [[ -f "$ENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_ACTIVATE" || echo "[warn] venv activate failed"
fi
export VIRTUAL_ENV="${VIRTUAL_ENV:-$VAR_DIR/openr1}"
export PATH="$VIRTUAL_ENV/bin:${PATH:-}"
export PYTHONNOUSERSITE=1

# Route caches into the repo-local var/ tree
export HF_HOME="$VAR_DIR/cache/hf"
export XDG_CACHE_HOME="$VAR_DIR/cache/xdg"
export XDG_CONFIG_HOME="$VAR_DIR/config"
export HUGGINGFACE_HUB_CACHE="$VAR_DIR/cache/huggingface/hub"
export HF_DATASETS_CACHE="$VAR_DIR/cache/huggingface/datasets"
export TMPDIR="$VAR_DIR/tmp"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
else
  echo "[warn] HF_TOKEN is not set; gated datasets/models may fail." >&2
fi

# --- Argument parsing ---
MODEL="${MODEL:-}"
DATASETS="${DATASETS:-math_500,aime24,aime25,amc,minerva}"
SEEDS="${SEEDS:-0,1,2,3,4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --num-generations) NUM_GENERATIONS="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --extra-opts) EXTRA_OPTS="$2"; shift 2;;
    --help)
      echo "Usage: sbatch ops/slurm/infer_math.slurm [--model HF_ID] [--datasets comma_list] [--seeds comma_list] [--num-generations K] [--temperature T] [--batch-size N] [--extra-opts \"...override...\"]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model (HF id or local checkpoint path) is required." >&2
  exit 1
fi

IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"

echo "[info] Evaluating $MODEL on datasets: ${DATASET_ARRAY[*]}"
echo "[info] Seeds: $SEEDS | num_generations: $NUM_GENERATIONS | temperature: $TEMPERATURE"

for DATASET in "${DATASET_ARRAY[@]}"; do
  echo "[info] Running dataset=$DATASET"
  CMD=(
    maxent-grpo-math-eval
    "inference.dataset=${DATASET}"
    "inference.seeds=[${SEEDS}]"
    "inference.num_generations=${NUM_GENERATIONS}"
    "inference.temperature=${TEMPERATURE}"
    "inference.models=[{model_name_or_path:${MODEL},batch_size:${BATCH_SIZE}}]"
  )
  if [[ -n "$EXTRA_OPTS" ]]; then
    CMD+=(${EXTRA_OPTS})
  fi
  "${CMD[@]}"
done
