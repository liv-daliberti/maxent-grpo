#!/bin/bash
# Single-node sweep runner for baseline vs MaxEnt POC (0.5B) with inline vLLM.
#
# Defaults target 8x A100 with 2 GPUs for vLLM + 6 GPUs for training.
#
# Usage:
#   sbatch ops/slurm/poc_sweep.slurm \
#     --export=ALL,HF_TOKEN=...,WANDB_PROJECT=...,WANDB_ENTITY=...

#SBATCH --job-name=maxent_poc_sweep
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --partition=mltheory
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --account=

set -euo pipefail
set -x

ROOT_DIR="$PWD"
VAR_DIR="${VAR_DIR:-$ROOT_DIR/var}"
mkdir -p "$VAR_DIR/artifacts/logs"

# ---- Environment bootstrap (configure for your cluster) ----
export ENV_MODE="${ENV_MODE:-venv}"
export CONDA_SH="${CONDA_SH:-/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-$VAR_DIR/openr1}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-$VAR_DIR/openr1/bin/activate}"

if [[ "$ENV_MODE" == "conda" ]]; then
  if [[ -f "$CONDA_SH" ]]; then source "$CONDA_SH"; fi
  conda activate "$CONDA_ENV" || echo "[warn] submit-shell: failed to conda activate $CONDA_ENV"
elif [[ -f "$ENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_ACTIVATE" || echo "[warn] submit-shell: failed to source $ENV_ACTIVATE"
else
  echo "[warn] No environment activation performed (ENV_MODE=$ENV_MODE)."
fi

export PYTHONNOUSERSITE=1
export HF_HOME="$VAR_DIR/cache/hf"
export XDG_CACHE_HOME="$VAR_DIR/cache/xdg"
export XDG_CONFIG_HOME="$VAR_DIR/config"
export HUGGINGFACE_HUB_CACHE="$VAR_DIR/cache/huggingface/hub"
export HF_DATASETS_CACHE="$VAR_DIR/cache/huggingface/datasets"
export PIP_CACHE_DIR="$VAR_DIR/cache/pip"
export TORCHINDUCTOR_CACHE_DIR="$VAR_DIR/cache/torchinductor"
export TRITON_CACHE_DIR="$VAR_DIR/cache/triton"
export WANDB_DIR="${WANDB_DIR:-$VAR_DIR/artifacts/wandb}"
export WANDB_CACHE_DIR="$WANDB_DIR"
export WANDB_DISABLED=0
export TMPDIR="$VAR_DIR/tmp"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" \
         "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-maxent-grpo-poc-0-5b}"

# ---- Sweep defaults (override via --export) ----
BASELINE_RECIPE="${BASELINE_RECIPE:-configs/recipes/Qwen2.5-0.5B-Instruct/grpo/config_math_maxent_hparams.yaml}"
MAXENT_RECIPE="${MAXENT_RECIPE:-configs/recipes/Qwen2.5-0.5B-Instruct/maxent-grpo/config_math_maxent_hparams.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/recipes/accelerate_configs/zero3.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-6}"
SEEDS="${SEEDS:-0,1,2}"
TAG="${TAG:-poc-0.5b}"
METRIC="${METRIC:-eval/mean_reward}"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-29525}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_IDS="${VLLM_GPU_IDS:-0,1}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-2,3,4,5,6,7}"

VLLM_LOG="$VAR_DIR/artifacts/logs/vllm_${SLURM_JOB_ID}.out"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT


echo "[info] Starting vLLM on GPUs ${VLLM_GPU_IDS} (model=${VLLM_MODEL})"
CUDA_VISIBLE_DEVICES="$VLLM_GPU_IDS" \
  python -m trl.scripts.vllm_serve \
    --model "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --dtype float16 \
    --gpu_memory_utilization 0.90 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --data_parallel_size 1 \
    --tensor_parallel_size 1 \
    --trust_remote_code \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "[info] Waiting for vLLM health at http://localhost:${VLLM_PORT}/health …"
VLLM_READY=false
for attempt in $(seq 1 180); do
  if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    VLLM_READY=true
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[error] vLLM exited early; see $VLLM_LOG" >&2
    break
  fi
  sleep 2
done
if [[ "$VLLM_READY" != "true" ]]; then
  echo "[error] vLLM failed to become healthy; tailing log:" >&2
  tail -n 200 "$VLLM_LOG" || true
  exit 1
fi
echo "[info] vLLM is healthy."

export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"

SWEEP_ARGS=(
  --use-accelerate
  --accelerate-config "$ACCEL_CONFIG"
  --num-processes "$NUM_PROCESSES"
  --baseline-recipe "$BASELINE_RECIPE"
  --maxent-recipe "$MAXENT_RECIPE"
  --seeds "$SEEDS"
  --tag "$TAG"
  --metric "$METRIC"
  --output-root "$VAR_DIR/data/poc_runs"
  --wandb-dir "$WANDB_DIR"
  --vllm-host "localhost"
  --vllm-port "$VLLM_PORT"
)
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  SWEEP_ARGS+=(--wandb-project "$WANDB_PROJECT")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  SWEEP_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${WANDB_RUN_GROUP:-}" ]]; then
  SWEEP_ARGS+=(--wandb-group "$WANDB_RUN_GROUP")
else
  SWEEP_ARGS+=(--wandb-group "$TAG")
fi

python "$ROOT_DIR/tools/run_poc_sweep.py" "${SWEEP_ARGS[@]}"
