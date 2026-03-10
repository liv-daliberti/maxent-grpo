#!/bin/bash
# Single-node dual training launcher:
# - GRPO on GPUs 0-3 (vLLM:0, train:1,2,3)
# - MaxEnt on GPUs 4-7 (vLLM:4, train:5,6,7)
#
# Both stacks run concurrently in one 8-GPU allocation.

#SBATCH --job-name=open_r1_dual_4plus4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --time=48:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err
#SBATCH --export=ALL,ENV_MODE=venv,ENV_ACTIVATE=/n/fs/similarity/maxent-grpo/var/openr1/bin/activate

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

VAR_DIR="${VAR_DIR:-$ROOT_DIR/var}"
mkdir -p "$VAR_DIR/artifacts/logs" "$VAR_DIR/tmp"

CACHE_ROOT="${CACHE_ROOT:-$VAR_DIR/cache}"
mkdir -p \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/hf" \
  "$CACHE_ROOT/tmp" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/torchinductor"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$XDG_CACHE_HOME/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$XDG_CACHE_HOME/huggingface/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENV_MODE="${ENV_MODE:-venv}"
CONDA_SH="${CONDA_SH:-/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-$VAR_DIR/openr1}"
ENV_ACTIVATE="${ENV_ACTIVATE:-$VAR_DIR/openr1/bin/activate}"
VIRTUAL_ENV="${VIRTUAL_ENV:-$VAR_DIR/openr1}"

activate_env() {
  if [[ "$ENV_MODE" == "conda" ]]; then
    if [[ -f "$CONDA_SH" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_SH"
    fi
    conda activate "$CONDA_ENV"
  elif [[ -f "$ENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
  fi
  export VIRTUAL_ENV
  export PATH="$VIRTUAL_ENV/bin:${PATH:-}"
  export PYTHONNOUSERSITE=1
}

to_abs_path() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath "$p"
  elif command -v readlink >/dev/null 2>&1; then
    readlink -f "$p" 2>/dev/null || printf '%s\n' "$p"
  else
    printf '%s\n' "$p"
  fi
}

yaml_scalar() {
  local key="$1"
  local file="$2"
  awk -F': *' -v k="$key" '$1 ~ "^[[:space:]]*"k"[[:space:]]*$" {print $2; exit}' "$file" | tr -d '"' | tr -d "'" || true
}

infer_max_model_len() {
  local cfg="$1"
  local p c
  p="$(yaml_scalar max_prompt_length "$cfg")"
  c="$(yaml_scalar max_completion_length "$cfg")"
  if [[ "$p" =~ ^[0-9]+$ && "$c" =~ ^[0-9]+$ ]]; then
    echo $((p + c))
  elif [[ "$p" =~ ^[0-9]+$ ]]; then
    echo "$p"
  else
    echo 4096
  fi
}

csv_count() {
  local raw="${1//[[:space:]]/}"
  if [[ -z "$raw" ]]; then
    echo 0
    return 0
  fi
  local n=0 part
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    [[ "$part" =~ ^[0-9]+$ ]] || return 1
    n=$((n + 1))
  done
  echo "$n"
}

assert_csv_ids() {
  local name="$1"
  local raw="${2//[[:space:]]/}"
  local max_id="$3"
  local part
  declare -A seen=()
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    [[ "$part" =~ ^[0-9]+$ ]] || { echo "Invalid ${name} id: ${part}" >&2; return 1; }
    (( part >= 0 && part <= max_id )) || { echo "${name} id ${part} out of range [0,${max_id}]" >&2; return 1; }
    [[ -z "${seen[$part]:-}" ]] || { echo "Duplicate ${name} id: ${part}" >&2; return 1; }
    seen[$part]=1
  done
}

contains_id() {
  local csv="${1//[[:space:]]/}"
  local id="$2"
  [[ ",$csv," == *",$id,"* ]]
}

append_flag_value() {
  local args="$1"
  local flag="$2"
  local value="$3"
  if [[ " $args " == *" ${flag} "* || "$args" == *"${flag}="* ]]; then
    printf '%s\n' "$args"
  else
    if [[ -n "$args" ]]; then
      printf '%s %s %s\n' "$args" "$flag" "$value"
    else
      printf '%s %s\n' "$flag" "$value"
    fi
  fi
}

if [[ "$*" == *"--help"* ]]; then
  cat <<'USAGE'
Usage: sbatch ops/slurm/train_dual_4plus4.slurm [options]

Options:
  --model MODEL               Recipe model directory (default: Qwen2.5-0.5B-Instruct)
  --config SUFFIX             Config suffix for both GRPO and MaxEnt (default: math)
  --grpo-config SUFFIX        Optional GRPO config suffix override
  --maxent-config SUFFIX      Optional MaxEnt config suffix override
  --accelerator NAME          Accelerate config name (default: zero3)
  --grpo-args "ARGS"          Extra GRPO args
  --maxent-args "ARGS"        Extra MaxEnt args
  --run-only MODE             both | grpo | maxent (default: both)
USAGE
  exit 0
fi

MODEL="${MODEL:-Qwen2.5-0.5B-Instruct}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
CONFIG_SUFFIX_GRPO="${CONFIG_SUFFIX_GRPO:-$CONFIG_SUFFIX}"
CONFIG_SUFFIX_MAXENT="${CONFIG_SUFFIX_MAXENT:-$CONFIG_SUFFIX}"
ACCELERATOR="${ACCELERATOR:-zero3}"
GRPO_ARGS="${GRPO_ARGS:-}"
MAXENT_ARGS="${MAXENT_ARGS:-}"
RUN_ONLY="$(echo "${RUN_ONLY:-both}" | tr '[:upper:]' '[:lower:]')"

GRPO_VLLM_GPU="${GRPO_VLLM_GPU:-0}"
GRPO_TRAIN_GPUS="${GRPO_TRAIN_GPUS:-1,2,3}"
MAXENT_VLLM_GPU="${MAXENT_VLLM_GPU:-4}"
MAXENT_TRAIN_GPUS="${MAXENT_TRAIN_GPUS:-5,6,7}"

GRPO_VLLM_PORT="${GRPO_VLLM_PORT:-8000}"
MAXENT_VLLM_PORT="${MAXENT_VLLM_PORT:-8001}"
GRPO_GROUP_PORT="${GRPO_GROUP_PORT:-29535}"
MAXENT_GROUP_PORT="${MAXENT_GROUP_PORT:-29536}"
GRPO_MASTER_PORT="${GRPO_MASTER_PORT:-6000}"
MAXENT_MASTER_PORT="${MAXENT_MASTER_PORT:-6001}"
VLLM_INIT_COMMUNICATOR_TIMEOUT="${VLLM_INIT_COMMUNICATOR_TIMEOUT:-300}"

# Prefer the async TRL vLLM communicator patch to avoid startup handshake stalls.
MAXENT_TRL_VLLM_CLIENT_PATCH="${MAXENT_TRL_VLLM_CLIENT_PATCH:-1}"
MAXENT_VLLM_ASYNC_INIT="${MAXENT_VLLM_ASYNC_INIT:-auto}"
MAXENT_VLLM_INIT_RETRIES="${MAXENT_VLLM_INIT_RETRIES:-3}"
MAXENT_VLLM_INIT_RETRY_BACKOFF_S="${MAXENT_VLLM_INIT_RETRY_BACKOFF_S:-2}"
MAXENT_VLLM_INIT_TIMEOUT_S="${MAXENT_VLLM_INIT_TIMEOUT_S:-300}"

VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
# vLLM cold starts (weight load + compile cache warmup) can exceed 8 minutes on first run.
# Default to a longer health window: 900 * 2s = 30 minutes.
VLLM_HEALTH_MAX_ATTEMPTS="${VLLM_HEALTH_MAX_ATTEMPTS:-900}"
VLLM_HEALTH_INTERVAL_S="${VLLM_HEALTH_INTERVAL_S:-2}"
ACC_TEE="${ACC_TEE:-3}"
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-3}"

if ! [[ "$TRAIN_NUM_PROCESSES" =~ ^[0-9]+$ ]] || (( TRAIN_NUM_PROCESSES < 1 )); then
  echo "TRAIN_NUM_PROCESSES must be a positive integer (got: $TRAIN_NUM_PROCESSES)" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --config) CONFIG_SUFFIX="$2"; CONFIG_SUFFIX_GRPO="$2"; CONFIG_SUFFIX_MAXENT="$2"; shift 2 ;;
    --grpo-config) CONFIG_SUFFIX_GRPO="$2"; shift 2 ;;
    --maxent-config) CONFIG_SUFFIX_MAXENT="$2"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --grpo-args) GRPO_ARGS="$2"; shift 2 ;;
    --maxent-args) MAXENT_ARGS="$2"; shift 2 ;;
    --run-only) RUN_ONLY="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "$RUN_ONLY" != "both" && "$RUN_ONLY" != "grpo" && "$RUN_ONLY" != "maxent" ]]; then
  echo "RUN_ONLY must be both|grpo|maxent" >&2
  exit 1
fi

# In single-stack mode, default to a 4-GPU local layout (vLLM:0, train:1,2,3).
# This allows maxent-only/grpo-only jobs to run in a 4-GPU allocation.
if [[ "$RUN_ONLY" == "maxent" ]]; then
  if [[ "$MAXENT_VLLM_GPU" == "4" ]]; then MAXENT_VLLM_GPU="0"; fi
  if [[ "$MAXENT_TRAIN_GPUS" == "5,6,7" ]]; then MAXENT_TRAIN_GPUS="1,2,3"; fi
fi
if [[ "$RUN_ONLY" == "grpo" ]]; then
  if [[ "$GRPO_VLLM_GPU" == "4" ]]; then GRPO_VLLM_GPU="0"; fi
  if [[ "$GRPO_TRAIN_GPUS" == "5,6,7" ]]; then GRPO_TRAIN_GPUS="1,2,3"; fi
fi

GRPO_CFG="configs/recipes/${MODEL}/grpo/config_${CONFIG_SUFFIX_GRPO}.yaml"
MAXENT_CFG="configs/recipes/${MODEL}/maxent-grpo/config_${CONFIG_SUFFIX_MAXENT}.yaml"
ACCEL_FILE="configs/recipes/accelerate_configs/${ACCELERATOR}.yaml"

[[ -f "$GRPO_CFG" ]] || { echo "Missing config: $GRPO_CFG" >&2; exit 1; }
[[ -f "$MAXENT_CFG" ]] || { echo "Missing config: $MAXENT_CFG" >&2; exit 1; }
[[ -f "$ACCEL_FILE" ]] || { echo "Missing accelerate config: $ACCEL_FILE" >&2; exit 1; }

GRPO_CFG_ABS="$(to_abs_path "$GRPO_CFG")"
MAXENT_CFG_ABS="$(to_abs_path "$MAXENT_CFG")"
ACCEL_FILE_ABS="$(to_abs_path "$ACCEL_FILE")"

GPU_COUNT=8
if command -v nvidia-smi >/dev/null 2>&1; then
  detected="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$detected" =~ ^[0-9]+$ ]] && (( detected > 0 )); then
    GPU_COUNT="$detected"
  fi
fi
MIN_REQUIRED_GPUS=8
if [[ "$RUN_ONLY" == "grpo" || "$RUN_ONLY" == "maxent" ]]; then
  MIN_REQUIRED_GPUS=4
fi
if (( GPU_COUNT < MIN_REQUIRED_GPUS )); then
  echo "This launcher expects at least ${MIN_REQUIRED_GPUS} visible GPUs for RUN_ONLY=${RUN_ONLY}; detected ${GPU_COUNT}." >&2
  exit 1
fi
MAX_GPU_ID=$((GPU_COUNT - 1))

if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "grpo" ]]; then
  assert_csv_ids "GRPO_TRAIN_GPUS" "$GRPO_TRAIN_GPUS" "$MAX_GPU_ID"
  [[ "$GRPO_VLLM_GPU" =~ ^[0-9]+$ ]] || { echo "Invalid GRPO_VLLM_GPU=$GRPO_VLLM_GPU" >&2; exit 1; }
  (( GRPO_VLLM_GPU <= MAX_GPU_ID )) || { echo "GRPO_VLLM_GPU out of range" >&2; exit 1; }
  (( $(csv_count "$GRPO_TRAIN_GPUS") == 3 )) || { echo "GRPO_TRAIN_GPUS must contain exactly 3 GPUs" >&2; exit 1; }
  contains_id "$GRPO_TRAIN_GPUS" "$GRPO_VLLM_GPU" && { echo "GRPO vLLM GPU overlaps GRPO train GPUs" >&2; exit 1; }
fi

if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "maxent" ]]; then
  assert_csv_ids "MAXENT_TRAIN_GPUS" "$MAXENT_TRAIN_GPUS" "$MAX_GPU_ID"
  [[ "$MAXENT_VLLM_GPU" =~ ^[0-9]+$ ]] || { echo "Invalid MAXENT_VLLM_GPU=$MAXENT_VLLM_GPU" >&2; exit 1; }
  (( MAXENT_VLLM_GPU <= MAX_GPU_ID )) || { echo "MAXENT_VLLM_GPU out of range" >&2; exit 1; }
  (( $(csv_count "$MAXENT_TRAIN_GPUS") == 3 )) || { echo "MAXENT_TRAIN_GPUS must contain exactly 3 GPUs" >&2; exit 1; }
  contains_id "$MAXENT_TRAIN_GPUS" "$MAXENT_VLLM_GPU" && { echo "MaxEnt vLLM GPU overlaps MaxEnt train GPUs" >&2; exit 1; }
fi

if [[ "$RUN_ONLY" == "both" ]]; then
  contains_id "$GRPO_TRAIN_GPUS" "$MAXENT_VLLM_GPU" && { echo "MaxEnt vLLM GPU overlaps GRPO train GPUs" >&2; exit 1; }
  contains_id "$MAXENT_TRAIN_GPUS" "$GRPO_VLLM_GPU" && { echo "GRPO vLLM GPU overlaps MaxEnt train GPUs" >&2; exit 1; }
  for gid in ${GRPO_TRAIN_GPUS//,/ }; do
    contains_id "$MAXENT_TRAIN_GPUS" "$gid" && { echo "Train GPU overlap between GRPO and MaxEnt: $gid" >&2; exit 1; }
  done
fi

GRPO_MODEL="$(yaml_scalar model_name_or_path "$GRPO_CFG")"
MAXENT_MODEL="$(yaml_scalar model_name_or_path "$MAXENT_CFG")"
GRPO_REV="$(yaml_scalar model_revision "$GRPO_CFG")"
MAXENT_REV="$(yaml_scalar model_revision "$MAXENT_CFG")"
GRPO_MODEL="${GRPO_MODEL:-$MODEL}"
MAXENT_MODEL="${MAXENT_MODEL:-$MODEL}"
GRPO_MAX_LEN="${GRPO_MAX_LEN:-$(infer_max_model_len "$GRPO_CFG")}" 
MAXENT_MAX_LEN="${MAXENT_MAX_LEN:-$(infer_max_model_len "$MAXENT_CFG")}" 

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="${WANDB_RUN_GROUP:-dual_4plus4_${MODEL//\//-}_${STAMP}}"
GRPO_RUN_NAME="${GRPO_RUN_NAME:-${RUN_GROUP}-grpo}"
MAXENT_RUN_NAME="${MAXENT_RUN_NAME:-${RUN_GROUP}-maxent}"

GRPO_ARGS="$(append_flag_value "$GRPO_ARGS" "--run_name" "$GRPO_RUN_NAME")"
GRPO_ARGS="$(append_flag_value "$GRPO_ARGS" "--wandb_run_group" "$RUN_GROUP")"
MAXENT_ARGS="$(append_flag_value "$MAXENT_ARGS" "--run_name" "$MAXENT_RUN_NAME")"
MAXENT_ARGS="$(append_flag_value "$MAXENT_ARGS" "--wandb_run_group" "$RUN_GROUP")"

VLLM_GRPO_LOG="$VAR_DIR/artifacts/logs/vllm-grpo-${SLURM_JOB_ID}.out"
VLLM_MAXENT_LOG="$VAR_DIR/artifacts/logs/vllm-maxent-${SLURM_JOB_ID}.out"
TRAIN_GRPO_LOG="$VAR_DIR/artifacts/logs/train-grpo-${SLURM_JOB_ID}.log"
TRAIN_MAXENT_LOG="$VAR_DIR/artifacts/logs/train-maxent-${SLURM_JOB_ID}.log"
: > "$VLLM_GRPO_LOG"
: > "$VLLM_MAXENT_LOG"
: > "$TRAIN_GRPO_LOG"
: > "$TRAIN_MAXENT_LOG"

activate_env

if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "grpo" ]]; then
  echo "[layout] GRPO:   vLLM GPU=${GRPO_VLLM_GPU}, train GPUs=${GRPO_TRAIN_GPUS}, vLLM port=${GRPO_VLLM_PORT}"
fi
if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "maxent" ]]; then
  echo "[layout] MaxEnt: vLLM GPU=${MAXENT_VLLM_GPU}, train GPUs=${MAXENT_TRAIN_GPUS}, vLLM port=${MAXENT_VLLM_PORT}"
fi
echo "[layout] run_group=${RUN_GROUP}"

GRPO_VLLM_PID=""
MAXENT_VLLM_PID=""
GRPO_TRAIN_PID=""
MAXENT_TRAIN_PID=""
LAST_BG_PID=""

cleanup() {
  for pid in "$GRPO_TRAIN_PID" "$MAXENT_TRAIN_PID" "$GRPO_VLLM_PID" "$MAXENT_VLLM_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

start_vllm() {
  local label="$1"
  local gpu_id="$2"
  local model_id="$3"
  local revision="$4"
  local port="$5"
  local max_len="$6"
  local log_file="$7"
  (
    activate_env
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export VLLM_INIT_COMMUNICATOR_TIMEOUT
    cmd=(
      python -m trl.scripts.vllm_serve
      --model "$model_id"
      --host 127.0.0.1
      --port "$port"
      --tensor_parallel_size 1
      --data_parallel_size 1
      --max_model_len "$max_len"
      --dtype "$VLLM_DTYPE"
      --enforce_eager "$VLLM_ENFORCE_EAGER"
    )
    if [[ -n "$revision" ]]; then
      cmd+=(--revision "$revision")
    fi
    echo "[$label] ${cmd[*]}"
    exec "${cmd[@]}"
  ) >> "$log_file" 2>&1 &
  LAST_BG_PID="$!"
}

wait_vllm_health() {
  local label="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  for attempt in $(seq 1 "$VLLM_HEALTH_MAX_ATTEMPTS"); do
    if curl -fsS "http://127.0.0.1:${port}/health/" >/dev/null 2>&1; then
      echo "[$label] vLLM healthy on port ${port}"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$label] vLLM process exited during startup." >&2
      tail -n 80 "$log_file" >&2 || true
      return 1
    fi
    sleep "$VLLM_HEALTH_INTERVAL_S"
  done
  echo "[$label] vLLM health timeout on port ${port}." >&2
  tail -n 80 "$log_file" >&2 || true
  return 1
}

launch_train() {
  local label="$1"
  local entrypoint="$2"
  local recipe="$3"
  local train_gpus="$4"
  local main_port="$5"
  local group_port="$6"
  local vllm_port="$7"
  local extra_args="$8"
  local log_file="$9"

  (
    activate_env
    export CUDA_VISIBLE_DEVICES="$train_gpus"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
    export MAXENT_VLLM_MODE=server
    export VLLM_MODE=server
    export MAXENT_VLLM_URL="http://127.0.0.1:${vllm_port}/generate"
    export VLLM_URL="$MAXENT_VLLM_URL"
    export VLLM_GROUP_PORT="$group_port"
    export PORT_FOR_COMMUNICATION="$group_port"
    export GRPO_RECIPE="$recipe"
    export GRPO_RECIPE_USED="$recipe"
    export MAXENT_TRL_VLLM_CLIENT_PATCH
    export MAXENT_VLLM_ASYNC_INIT
    export MAXENT_VLLM_INIT_RETRIES
    export MAXENT_VLLM_INIT_RETRY_BACKOFF_S
    export MAXENT_VLLM_INIT_TIMEOUT_S
    export VLLM_INIT_COMMUNICATOR_TIMEOUT

    launch=(
      python -m accelerate.commands.launch
      --config_file "$ACCEL_FILE_ABS"
      --num_machines 1
      --num_processes "$TRAIN_NUM_PROCESSES"
      --main_process_ip 127.0.0.1
      --main_process_port "$main_port"
      --machine_rank 0
      --tee "$ACC_TEE"
      "$entrypoint"
      --use_vllm true
      --vllm_mode server
      --vllm_server_host 127.0.0.1
      --vllm_server_port "$vllm_port"
    )
    cmd=("${launch[@]}")
    if [[ -n "$extra_args" ]]; then
      eval "cmd+=( ${extra_args} )"
    fi

    echo "[$label] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[$label] ${cmd[*]}"
    exec "${cmd[@]}"
  ) >> "$log_file" 2>&1 &
  LAST_BG_PID="$!"
}

if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "grpo" ]]; then
  start_vllm "grpo-vllm" "$GRPO_VLLM_GPU" "$GRPO_MODEL" "$GRPO_REV" "$GRPO_VLLM_PORT" "$GRPO_MAX_LEN" "$VLLM_GRPO_LOG"
  GRPO_VLLM_PID="$LAST_BG_PID"
fi
if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "maxent" ]]; then
  start_vllm "maxent-vllm" "$MAXENT_VLLM_GPU" "$MAXENT_MODEL" "$MAXENT_REV" "$MAXENT_VLLM_PORT" "$MAXENT_MAX_LEN" "$VLLM_MAXENT_LOG"
  MAXENT_VLLM_PID="$LAST_BG_PID"
fi

if [[ -n "$GRPO_VLLM_PID" ]]; then
  wait_vllm_health "grpo-vllm" "$GRPO_VLLM_PORT" "$GRPO_VLLM_PID" "$VLLM_GRPO_LOG"
fi
if [[ -n "$MAXENT_VLLM_PID" ]]; then
  wait_vllm_health "maxent-vllm" "$MAXENT_VLLM_PORT" "$MAXENT_VLLM_PID" "$VLLM_MAXENT_LOG"
fi

if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "grpo" ]]; then
  launch_train "grpo-train" "src/maxent_grpo/grpo.py" "$GRPO_CFG_ABS" "$GRPO_TRAIN_GPUS" "$GRPO_MASTER_PORT" "$GRPO_GROUP_PORT" "$GRPO_VLLM_PORT" "$GRPO_ARGS" "$TRAIN_GRPO_LOG"
  GRPO_TRAIN_PID="$LAST_BG_PID"
fi
if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "maxent" ]]; then
  launch_train "maxent-train" "src/maxent_grpo/grpo.py" "$MAXENT_CFG_ABS" "$MAXENT_TRAIN_GPUS" "$MAXENT_MASTER_PORT" "$MAXENT_GROUP_PORT" "$MAXENT_VLLM_PORT" "$MAXENT_ARGS" "$TRAIN_MAXENT_LOG"
  MAXENT_TRAIN_PID="$LAST_BG_PID"
fi

grpo_status=0
maxent_status=0
if [[ -n "$GRPO_TRAIN_PID" ]]; then
  wait "$GRPO_TRAIN_PID" || grpo_status=$?
fi
if [[ -n "$MAXENT_TRAIN_PID" ]]; then
  wait "$MAXENT_TRAIN_PID" || maxent_status=$?
fi

if [[ -n "$GRPO_VLLM_PID" ]]; then
  kill "$GRPO_VLLM_PID" 2>/dev/null || true
  wait "$GRPO_VLLM_PID" 2>/dev/null || true
fi
if [[ -n "$MAXENT_VLLM_PID" ]]; then
  kill "$MAXENT_VLLM_PID" 2>/dev/null || true
  wait "$MAXENT_VLLM_PID" 2>/dev/null || true
fi

echo "[summary] grpo_status=${grpo_status} maxent_status=${maxent_status}"
echo "[summary] logs:"
echo "  $VLLM_GRPO_LOG"
echo "  $VLLM_MAXENT_LOG"
echo "  $TRAIN_GRPO_LOG"
echo "  $TRAIN_MAXENT_LOG"

if (( grpo_status != 0 || maxent_status != 0 )); then
  exit 1
fi
