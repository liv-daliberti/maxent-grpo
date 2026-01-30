#!/bin/bash
# Slurm launcher to start one vLLM server and run multiple LightEval tasks sequentially against it.
# Set LIGHEVAL_BACKEND=hf to skip vLLM and run the LightEval HF backend in-process.
# Usage (from repo root):
#   sbatch ops/slurm/evaluate_batch.slurm <tasks_file> <model> <revision> <tensor_parallel> <trust_remote_code> [base64_system_prompt]
# tasks_file: plain text; each non-empty/non-comment line is:
#   <benchmark> <task_list> [extra_lighteval_opts...]
# Example line: math_500 lighteval|math_500|0|0 +inference.models.0.top_p=1.0

#SBATCH --job-name=lighteval-batch
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=mltheory
#SBATCH --time=24:00:00
# Avoid nodes that have previously wedged during vLLM loads; override with sbatch --exclude/--nodelist if desired.
#SBATCH --exclude=node202,node203,node204
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err
#SBATCH --export=ALL,ENV_MODE=venv,ENV_ACTIVATE=/n/fs/similarity/maxent-grpo/var/openr1/bin/activate

set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-$PWD}"
VAR_DIR="${VAR_DIR:-$ROOT_DIR/var}"
BACKEND="${LIGHEVAL_BACKEND:-vllm}" # vllm | hf
mkdir -p "$VAR_DIR/artifacts/logs" "$VAR_DIR/cache" "$VAR_DIR/artifacts/lighteval"

# --- Environment bootstrap (mirrors train/infer scripts) ---
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
# Match training: ensure CUDA toolchain is loaded and pinned.
if command -v module &>/dev/null; then
  module unload cuda || true
  echo "[info] Available CUDA-related modulefiles (if any):"
  module avail cuda 2>&1 | sed -n '1,40p' || true
  if module load cuda/12.4 >/dev/null 2>&1 || module load cudatoolkit/12.4 >/dev/null 2>&1; then
    CUDA_MODULE_VERSION="12.4"
    echo "[info] Loaded CUDA 12.4 via environment modules."
  elif module load cuda/12.6 >/dev/null 2>&1 || module load cudatoolkit/12.6 >/dev/null 2>&1; then
    CUDA_MODULE_VERSION="12.6"
    echo "[info] Loaded CUDA 12.6 via environment modules."
  else
    echo "[warn] No cuda modulefiles could be loaded; defaulting to system installation under /usr/local." >&2
  fi
fi
CUDA_DEFAULT_VERSION="${CUDA_MODULE_VERSION:-12.6}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${CUDA_DEFAULT_VERSION}}"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Route caches into the repo-local var/ tree
export HF_HOME="$VAR_DIR/cache/hf"
export XDG_CACHE_HOME="$VAR_DIR/cache/xdg"
export XDG_CONFIG_HOME="$VAR_DIR/config"
export HUGGINGFACE_HUB_CACHE="$VAR_DIR/cache/huggingface/hub"
export HF_DATASETS_CACHE="$VAR_DIR/cache/huggingface/datasets"
export PIP_CACHE_DIR="$VAR_DIR/cache/pip"
export TORCHINDUCTOR_CACHE_DIR="$VAR_DIR/cache/torchinductor"
export TRITON_CACHE_DIR="$VAR_DIR/cache/triton"
export WANDB_DIR="$VAR_DIR/artifacts/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR"
export TMPDIR="$VAR_DIR/tmp"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
else
  echo "[warn] HF_TOKEN is not set; gated datasets/models may fail." >&2
fi
# Pin deps the same way as training to avoid httpx/hub mismatches and ensure local code is installed.
echo "[info] Pinning huggingface-hub/httpx and installing repo in editable mode"
python -m pip install -q --upgrade 'huggingface-hub[cli,hf_xet]>=0.30.2,<1.0' 'httpx==0.27.2' || true
python -m pip install -q -e "$ROOT_DIR" || echo "[warn] Failed to install editable package from $ROOT_DIR" >&2
# Patch TRL vllm_serve so logprob metadata is available to downstream tooling.
python "$ROOT_DIR/tools/patch_trl_vllm_serve.py"
# Ensure math metrics deps are present for LightEval.
python -m pip install -q 'lighteval[math]' || echo "[warn] Failed to install lighteval[math]; math tasks may error" >&2

# If using HF backend, force a single GPU to avoid accelerate auto model-parallel/device_map surprises.
if [[ "${BACKEND,,}" == "hf" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  export CUDA_VISIBLE_DEVICES
fi

# --- Argument parsing ---
if [[ $# -lt 5 ]]; then
  echo "Usage: sbatch ops/slurm/evaluate_batch.slurm <tasks_file> <model> <revision> <tensor_parallel> <trust_remote_code> [base64_system_prompt]" >&2
  exit 1
fi
TASKS_FILE="$1"        # path to tasks file
MODEL="$2"             # HF id or local path
REVISION="$3"          # git commit / branch / tag or 'main'
TENSOR_PARALLEL="$4"   # true/false string
TRUST_REMOTE_CODE="$5" # true/false string
SYSTEM_PROMPT_B64="${6:-}"

if [[ ! -f "$TASKS_FILE" ]]; then
  echo "[error] tasks_file not found: $TASKS_FILE" >&2
  exit 1
fi

PORT="${PORT:-29525}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
EXTRA_LIGHEVAL_OPTS="${EXTRA_LIGHEVAL_OPTS:-}"
# Optional: stage model to node-local storage to avoid shared FS contention.
STAGE_MODEL="${STAGE_MODEL:-true}"
STAGE_DIR="${STAGE_DIR:-${SLURM_TMPDIR:-/tmp}/vllm_model_${SLURM_JOB_ID}}"
# Optional: skip torch.compile at load time.
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
# Data parallel shards for vLLM server (match training default)
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
# Cap context length to avoid allocating a 131k window from the model config.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8000}"
# Allow overriding GPU memory target and worker multiprocess.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
WORKER_MULTIPROC="${WORKER_MULTIPROC:-true}"

TP_SIZE=1
if [[ "$TENSOR_PARALLEL" == "true" ]]; then
  TP_SIZE=2
fi

SERVER_LOG="$VAR_DIR/artifacts/logs/vllm_${SLURM_JOB_ID}.log"
RESULTS_DIR="$VAR_DIR/artifacts/lighteval/$(basename "$MODEL")"
mkdir -p "$RESULTS_DIR"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Optional system prompt decode
SYSTEM_PROMPT_FILE=""
if [[ -n "$SYSTEM_PROMPT_B64" ]]; then
  SYSTEM_PROMPT_FILE="$VAR_DIR/tmp/system_prompt_${SLURM_JOB_ID}.txt"
  echo "$SYSTEM_PROMPT_B64" | base64 --decode > "$SYSTEM_PROMPT_FILE"
fi

# Optional model staging
if [[ "${STAGE_MODEL,,}" == "true" ]]; then
  echo "[info] Staging model to $STAGE_DIR ..."
  mkdir -p "$STAGE_DIR"
  rsync -a --delete "$MODEL"/ "$STAGE_DIR"/
  MODEL="$STAGE_DIR"
fi

if [[ "${BACKEND,,}" == "hf" ]]; then
  echo "[info] Using LightEval HF backend (no vLLM server)."
else
  # --- Launch vLLM OpenAI server ---
  VLLM_CMD=(
    python -m trl.scripts.vllm_serve
    --model "$MODEL"
    ${REVISION:+--revision "$REVISION"}
    --tensor_parallel_size "$TP_SIZE"
    --data_parallel_size "$DATA_PARALLEL_SIZE"
    --port "$PORT"
    --dtype float16
    --max_model_len "$MAX_MODEL_LEN"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
  )
  # vllm_serve does not accept a CLI flag for multiprocess; use env to control it.
  if [[ "${WORKER_MULTIPROC,,}" == "false" || "$WORKER_MULTIPROC" == "0" ]]; then
    export VLLM_USE_MULTIPROCESS=0
    export VLLM_WORKER_MULTIPROC=0
  fi
  if [[ "${ENFORCE_EAGER,,}" == "true" || "$ENFORCE_EAGER" == "1" ]]; then
    VLLM_CMD+=(--enforce_eager true)
  fi
  if [[ "${TRUST_REMOTE_CODE,,}" == "true" || "$TRUST_REMOTE_CODE" == "1" ]]; then
    VLLM_CMD+=(--trust_remote_code)
  fi
  "${VLLM_CMD[@]}" >"$SERVER_LOG" 2>&1 &
  VLLM_PID=$!

  echo "[info] Waiting for vLLM health on port $PORT ..."
  WAIT_LOOPS=300  # 10 minutes at 2s intervals
  VLLM_HEALTHY=0
  for _ in $(seq 1 "$WAIT_LOOPS"); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[error] vLLM exited early; see $SERVER_LOG" >&2
      exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "[info] vLLM is healthy"
      VLLM_HEALTHY=1
      break
    fi
    sleep 2
  done
  if [[ "$VLLM_HEALTHY" != "1" ]]; then
    echo "[error] vLLM failed health checks after $((WAIT_LOOPS * 2))s; see $SERVER_LOG" >&2
    exit 1
  fi
fi

# --- Run LightEval tasks sequentially ---
RUN_IDX=0
while IFS= read -r LINE || [[ -n "$LINE" ]]; do
  # Skip comments / blank
  [[ -z "$LINE" || "$LINE" =~ ^# ]] && continue
  # Allow arbitrary extra tokens after task_list (per-line extra opts)
  read -r BENCHMARK TASK_LIST EXTRA_LINE <<<"$LINE"
  if [[ -z "${BENCHMARK:-}" || -z "${TASK_LIST:-}" ]]; then
    echo "[warn] Skipping malformed line: $LINE" >&2
    continue
  fi
  RUN_IDX=$((RUN_IDX + 1))

  RESULTS_JSON="$RESULTS_DIR/${BENCHMARK}-tp${TP_SIZE}-job${SLURM_JOB_ID}-run${RUN_IDX}.json"
  if [[ "${BACKEND,,}" == "hf" ]]; then
    MODEL_ARGS="model_name=${MODEL},revision=${REVISION},batch_size=${BATCH_SIZE},generation_size=${MAX_NEW_TOKENS},trust_remote_code=${TRUST_REMOTE_CODE},model_parallel=false,device=cuda"
    LIGHTEVAL_CMD=(lighteval accelerate "$MODEL_ARGS" "$TASK_LIST" --output-dir "$RESULTS_DIR" --save-details)
  else
    MODEL_ARGS="model_name=${MODEL##*/},base_url=http://127.0.0.1:${PORT}/v1,provider=openai,api_key=EMPTY,temperature=${TEMPERATURE},top_p=${TOP_P},max_new_tokens=${MAX_NEW_TOKENS}"
    LIGHTEVAL_CMD=(lighteval endpoint litellm "$MODEL_ARGS" "$TASK_LIST" --output-dir "$RESULTS_DIR" --save-details)
  fi
  if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
    LIGHTEVAL_CMD+=(--system-prompt "$(cat "$SYSTEM_PROMPT_FILE")")
  fi
  if [[ -n "$EXTRA_LIGHEVAL_OPTS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_LIGHEVAL_OPTS)
    for tok in "${EXTRA_ARR[@]}"; do
      # Newer LightEval CLI (0.9.x) does not accept Hydra-style
      # "+inference.*" overrides as extra positional args; skip them
      # here so that legacy task strings from tasks_math_suite.txt
      # do not cause "unexpected extra arguments" failures.
      if [[ "$tok" == +* ]]; then
        echo "[warn] Skipping unsupported LightEval override token '$tok'; please move overrides into a config file instead." >&2
        continue
      fi
      LIGHTEVAL_CMD+=("$tok")
    done
  fi
  if [[ -n "${EXTRA_LINE:-}" ]]; then
    # shellcheck disable=SC2206
    PER_LINE_EXTRA=($EXTRA_LINE)
    for tok in "${PER_LINE_EXTRA[@]}"; do
      if [[ "$tok" == +* ]]; then
        echo "[warn] Skipping unsupported LightEval override token '$tok'; please move overrides into a config file instead." >&2
        continue
      fi
      LIGHTEVAL_CMD+=("$tok")
    done
  fi

  echo "[info] Running LightEval (${RUN_IDX}) [backend=${BACKEND}]: ${LIGHTEVAL_CMD[*]}"
  "${LIGHTEVAL_CMD[@]}"

  if compgen -G "$RESULTS_DIR/*.json" >/dev/null; then
    echo "[info] LightEval run ${RUN_IDX} completed; results under $RESULTS_DIR"
  else
    echo "[warn] LightEval run ${RUN_IDX} finished but no result JSON was found in $RESULTS_DIR" >&2
  fi
done < "$TASKS_FILE"

# --- Teardown ---
if [[ -n "${VLLM_PID:-}" ]]; then
  echo "[info] Stopping vLLM (pid=$VLLM_PID)"
fi
cleanup
trap - EXIT

echo "[info] All LightEval tasks completed."
