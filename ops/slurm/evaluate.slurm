#!/bin/bash
# Slurm launcher for LightEval benchmarks using a local vLLM OpenAI server.
# Set LIGHEVAL_BACKEND=hf to skip vLLM and run the LightEval HF backend in-process.
# Usage (from repo root):
#   sbatch ops/slurm/evaluate.slurm <benchmark_name> <task_list> <model> <revision> <tensor_parallel> <trust_remote_code> [base64_system_prompt]

#SBATCH --job-name=lighteval
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=mltheory
#SBATCH --time=24:00:00
# Avoid nodes that have previously wedged during vLLM loads; override with sbatch --exclude/--nodelist if desired.
# (Kept consistent with evaluate_batch.slurm.)
#SBATCH --exclude=node202,node203,node204
#SBATCH --output=var/logs/%x-%j.out
#SBATCH --error=var/logs/%x-%j.err
#SBATCH --export=ALL,ENV_MODE=venv,ENV_ACTIVATE=/n/fs/similarity/maxent-grpo/var/openr1/bin/activate

set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-$PWD}"
VAR_DIR="${VAR_DIR:-$ROOT_DIR/var}"
BACKEND="${LIGHEVAL_BACKEND:-vllm}" # vllm | hf
mkdir -p "$VAR_DIR/logs" "$VAR_DIR/cache" "$VAR_DIR/artifacts/lighteval"
# Optional: stage model to node-local storage to avoid shared FS contention.
STAGE_MODEL="${STAGE_MODEL:-false}"
STAGE_DIR="${STAGE_DIR:-${SLURM_TMPDIR:-/tmp}/lighteval_model_${SLURM_JOB_ID}}"

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

# Match training/evaluate_batch: ensure CUDA toolchain is loaded and pinned.
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
export WANDB_DIR="$VAR_DIR/cache/wandb"
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

# Pin deps similarly to training/evaluate_batch to avoid hub/httpx mismatches and ensure local code is installed.
echo "[info] Pinning huggingface-hub/httpx and installing repo in editable mode"
python -m pip install -q --upgrade 'huggingface-hub[cli,hf_xet]>=0.30.2,<1.0' 'httpx==0.27.2' || true
python -m pip install -q -e "$ROOT_DIR" || echo "[warn] Failed to install editable package from $ROOT_DIR" >&2
# Ensure math metrics deps are present for LightEval.
python -m pip install -q 'lighteval[math]' || echo "[warn] Failed to install lighteval[math]; math tasks may error" >&2

# If using HF backend, force a single GPU to avoid accelerate auto model-parallel/device_map surprises.
if [[ "${BACKEND,,}" == "hf" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  export CUDA_VISIBLE_DEVICES
fi

# --- Argument parsing ---
if [[ $# -lt 6 ]]; then
  echo "Usage: sbatch ops/slurm/evaluate.slurm <benchmark_name> <task_list> <model> <revision> <tensor_parallel> <trust_remote_code> [base64_system_prompt]" >&2
  exit 1
fi
BENCHMARK="$1"          # e.g., math_500
TASK_LIST_RAW="$2"      # e.g., lighteval|math_500|0|0 [+inference...]
MODEL="$3"              # HF id or local path
REVISION="$4"           # git commit / branch / tag or 'main'
TENSOR_PARALLEL="$5"    # true/false string
TRUST_REMOTE_CODE="$6"  # true/false string
SYSTEM_PROMPT_B64="${7:-}"

PORT="${PORT:-29525}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
EXTRA_LIGHEVAL_OPTS="${EXTRA_LIGHEVAL_OPTS:-}"

# If the task_list was passed with inline +inference overrides (e.g.
#   "lighteval|math_500|0|0 +inference.models.0.top_p=1.0 ...")
# split them so that only the suite|task|few_shot|truncate_few_shots
# part is given as the LightEval task, and the overrides are forwarded
# via EXTRA_LIGHEVAL_OPTS. This keeps compatibility with evaluate_batch.
TASK_LIST="${TASK_LIST_RAW%% *}"
if [[ "$TASK_LIST" != "$TASK_LIST_RAW" ]]; then
  # Everything after the first space is treated as extra LightEval opts.
  TASK_EXTRA="${TASK_LIST_RAW#"$TASK_LIST"}"
  # Trim leading whitespace from TASK_EXTRA
  TASK_EXTRA="${TASK_EXTRA#"${TASK_EXTRA%%[![:space:]]*}"}"
  if [[ -n "$TASK_EXTRA" ]]; then
    if [[ -n "$EXTRA_LIGHEVAL_OPTS" ]]; then
      EXTRA_LIGHEVAL_OPTS+=" $TASK_EXTRA"
    else
      EXTRA_LIGHEVAL_OPTS="$TASK_EXTRA"
    fi
  fi
fi

TP_SIZE=1
if [[ "$TENSOR_PARALLEL" == "true" ]]; then
  TP_SIZE=2
fi

SERVER_LOG="$VAR_DIR/logs/vllm_${SLURM_JOB_ID}.log"
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

# Optional model staging (useful on NFS)
if [[ "${STAGE_MODEL,,}" == "true" ]]; then
  echo "[info] Staging model to $STAGE_DIR ..."
  mkdir -p "$STAGE_DIR"
  rsync -a --delete "$MODEL"/ "$STAGE_DIR"/
  MODEL="$STAGE_DIR"
fi

# --- Launch vLLM OpenAI server ---
VLLM_CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --revision "$REVISION"
  --dtype float16
  --tensor-parallel-size "$TP_SIZE"
  --port "$PORT"
  --max-model-len 8000
  --gpu-memory-utilization 0.9
  --served-model-name "${MODEL##*/}"
)
if [[ "${BACKEND,,}" == "hf" ]]; then
  echo "[info] Using LightEval HF backend (no vLLM server)."
else
  if [[ "${TRUST_REMOTE_CODE,,}" == "true" || "$TRUST_REMOTE_CODE" == "1" ]]; then
    VLLM_CMD+=(--trust-remote-code)
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

# --- Run LightEval ---
RESULTS_JSON="$RESULTS_DIR/${BENCHMARK}-tp${TP_SIZE}-job${SLURM_JOB_ID}.json"

if [[ "${BACKEND,,}" == "hf" ]]; then
  MODEL_ARGS="model_name=${MODEL},revision=${REVISION},batch_size=${BATCH_SIZE},generation_size=${MAX_NEW_TOKENS},trust_remote_code=${TRUST_REMOTE_CODE},model_parallel=false,device=cuda,dtype=float16"
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

echo "[info] Running LightEval [backend=${BACKEND}]: ${LIGHTEVAL_CMD[*]}"
"${LIGHTEVAL_CMD[@]}"

# --- Teardown ---
if [[ -n "${VLLM_PID:-}" ]]; then
  echo "[info] Stopping vLLM (pid=$VLLM_PID)"
fi
cleanup
trap - EXIT

if compgen -G "$RESULTS_DIR/*.json" >/dev/null; then
  echo "[info] LightEval results available under $RESULTS_DIR"
else
  echo "[warn] LightEval finished but no result JSON was found in $RESULTS_DIR" >&2
fi
