#!/bin/bash
# Slurm launcher for LightEval benchmarks using a local vLLM OpenAI server.
# Usage:
#   sbatch ops/slurm/evaluate.slurm <benchmark_name> <task_list> <model> <revision> <tensor_parallel> <trust_remote_code> [base64_system_prompt]

#SBATCH --job-name=lighteval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --time=24:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err
#SBATCH --export=ALL,ENV_MODE=venv,ENV_ACTIVATE=/n/fs/similarity/maxent-grpo/var/openr1/bin/activate

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

VAR_DIR="${VAR_DIR:-$ROOT_DIR/var}"
mkdir -p "$VAR_DIR/artifacts/logs" "$VAR_DIR/cache" "$VAR_DIR/artifacts/lighteval" "$VAR_DIR/tmp"

# Typo-compatible fallback: keep supporting LIGHEVAL_BACKEND from legacy scripts.
BACKEND="${LIGHTEVAL_BACKEND:-${LIGHEVAL_BACKEND:-vllm}}" # vllm | hf
PORT="${PORT:-29525}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"
WAIT_LOOPS="${WAIT_LOOPS:-300}"   # health retries
WAIT_SLEEP_S="${WAIT_SLEEP_S:-2}" # health retry interval
EXTRA_LIGHTEVAL_OPTS="${EXTRA_LIGHTEVAL_OPTS:-${EXTRA_LIGHEVAL_OPTS:-}}"

ENV_MODE="${ENV_MODE:-venv}"
CONDA_SH="${CONDA_SH:-/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-$VAR_DIR/openr1}"
ENV_ACTIVATE="${ENV_ACTIVATE:-$VAR_DIR/openr1/bin/activate}"

if [[ "$ENV_MODE" == "conda" ]]; then
  if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
  fi
elif [[ -f "$ENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_ACTIVATE"
fi

export VIRTUAL_ENV="${VIRTUAL_ENV:-$VAR_DIR/openr1}"
export PATH="$VIRTUAL_ENV/bin:${PATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="${HF_HOME:-$VAR_DIR/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$VAR_DIR/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$VAR_DIR/config}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$VAR_DIR/cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$VAR_DIR/cache/huggingface/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$VAR_DIR/cache/pip}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$VAR_DIR/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$VAR_DIR/cache/triton}"
export TMPDIR="${TMPDIR:-$VAR_DIR/tmp}"
mkdir -p \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TMPDIR"

if [[ $# -lt 6 ]]; then
  echo "Usage: sbatch ops/slurm/evaluate.slurm <benchmark_name> <task_list> <model> <revision> <tensor_parallel> <trust_remote_code> [base64_system_prompt]" >&2
  exit 1
fi

BENCHMARK="$1"
TASK_LIST_RAW="$2"
MODEL="$3"
REVISION="$4"
TENSOR_PARALLEL="$5"
TRUST_REMOTE_CODE="$6"
SYSTEM_PROMPT_B64="${7:-}"

TASK_LIST="${TASK_LIST_RAW%% *}"
if [[ "$TASK_LIST" != "$TASK_LIST_RAW" ]]; then
  TASK_EXTRA="${TASK_LIST_RAW#"$TASK_LIST"}"
  TASK_EXTRA="${TASK_EXTRA#"${TASK_EXTRA%%[![:space:]]*}"}"
  if [[ -n "$TASK_EXTRA" ]]; then
    if [[ -n "$EXTRA_LIGHTEVAL_OPTS" ]]; then
      EXTRA_LIGHTEVAL_OPTS+=" $TASK_EXTRA"
    else
      EXTRA_LIGHTEVAL_OPTS="$TASK_EXTRA"
    fi
  fi
fi

detect_gpu_count() {
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    echo "${SLURM_GPUS_ON_NODE}"
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local raw="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
    if [[ -n "$raw" ]]; then
      local n=0
      local part
      IFS=',' read -r -a parts <<< "$raw"
      for part in "${parts[@]}"; do
        [[ -z "$part" ]] && continue
        n=$((n + 1))
      done
      if (( n > 0 )); then
        echo "$n"
        return 0
      fi
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local detected
    detected="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$detected" =~ ^[0-9]+$ && "$detected" -gt 0 ]]; then
      echo "$detected"
      return 0
    fi
  fi
  echo 1
}

if [[ "${TENSOR_PARALLEL,,}" == "true" || "$TENSOR_PARALLEL" == "1" ]]; then
  TP_SIZE="$(detect_gpu_count)"
  if [[ ! "$TP_SIZE" =~ ^[0-9]+$ || "$TP_SIZE" -lt 1 ]]; then
    TP_SIZE=1
  fi
else
  TP_SIZE=1
fi

MODEL_BASENAME="$(basename "$MODEL")"
RESULTS_DIR="$VAR_DIR/artifacts/lighteval/${MODEL_BASENAME}"
mkdir -p "$RESULTS_DIR"
SERVER_LOG="$VAR_DIR/artifacts/logs/vllm_${SLURM_JOB_ID:-nojob}.log"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

SYSTEM_PROMPT_FILE=""
if [[ -n "$SYSTEM_PROMPT_B64" ]]; then
  SYSTEM_PROMPT_FILE="$VAR_DIR/tmp/system_prompt_${SLURM_JOB_ID:-nojob}.txt"
  printf '%s' "$SYSTEM_PROMPT_B64" | base64 --decode > "$SYSTEM_PROMPT_FILE"
fi

if [[ "${BACKEND,,}" == "hf" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  export CUDA_VISIBLE_DEVICES
fi

if [[ "${BACKEND,,}" != "hf" ]]; then
  VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --revision "$REVISION"
    --dtype "$VLLM_DTYPE"
    --tensor-parallel-size "$TP_SIZE"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
    --served-model-name "${MODEL_BASENAME}"
  )
  if [[ "${TRUST_REMOTE_CODE,,}" == "true" || "$TRUST_REMOTE_CODE" == "1" ]]; then
    VLLM_CMD+=(--trust-remote-code)
  fi
  "${VLLM_CMD[@]}" >"$SERVER_LOG" 2>&1 &
  VLLM_PID=$!

  echo "[info] Waiting for vLLM health on port $PORT ..."
  VLLM_HEALTHY=0
  for _ in $(seq 1 "$WAIT_LOOPS"); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[error] vLLM exited early; see $SERVER_LOG" >&2
      exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      VLLM_HEALTHY=1
      break
    fi
    sleep "$WAIT_SLEEP_S"
  done
  if [[ "$VLLM_HEALTHY" != "1" ]]; then
    echo "[error] vLLM failed health checks after $((WAIT_LOOPS * WAIT_SLEEP_S))s; see $SERVER_LOG" >&2
    exit 1
  fi
fi

if [[ "${BACKEND,,}" == "hf" ]]; then
  MODEL_ARGS="model_name=${MODEL},revision=${REVISION},batch_size=${BATCH_SIZE},generation_size=${MAX_NEW_TOKENS},trust_remote_code=${TRUST_REMOTE_CODE},model_parallel=false,device=cuda,dtype=${VLLM_DTYPE}"
  LIGHTEVAL_CMD=(lighteval accelerate "$MODEL_ARGS" "$TASK_LIST" --output-dir "$RESULTS_DIR" --save-details)
else
  MODEL_ARGS="model_name=${MODEL_BASENAME},base_url=http://127.0.0.1:${PORT}/v1,provider=openai,api_key=EMPTY,temperature=${TEMPERATURE},top_p=${TOP_P},max_new_tokens=${MAX_NEW_TOKENS}"
  LIGHTEVAL_CMD=(lighteval endpoint litellm "$MODEL_ARGS" "$TASK_LIST" --output-dir "$RESULTS_DIR" --save-details)
fi

if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
  LIGHTEVAL_CMD+=(--system-prompt "$(cat "$SYSTEM_PROMPT_FILE")")
fi
if [[ -n "$EXTRA_LIGHTEVAL_OPTS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_LIGHTEVAL_OPTS)
  for tok in "${EXTRA_ARR[@]}"; do
    if [[ "$tok" == +* ]]; then
      echo "[warn] Skipping unsupported LightEval override token '$tok'." >&2
      continue
    fi
    LIGHTEVAL_CMD+=("$tok")
  done
fi

echo "[info] Running LightEval [backend=${BACKEND}, tp=${TP_SIZE}]: ${LIGHTEVAL_CMD[*]}"
"${LIGHTEVAL_CMD[@]}"

cleanup
trap - EXIT

if compgen -G "$RESULTS_DIR/*.json" >/dev/null; then
  echo "[info] LightEval results available under $RESULTS_DIR"
else
  echo "[warn] LightEval finished but no result JSON was found in $RESULTS_DIR" >&2
fi
