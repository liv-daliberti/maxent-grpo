#!/bin/bash
# Single-node Open‑R1 trainer (vLLM colocate)
#
# Example:
#   sbatch --nodes=1 var/repo/ops/slurm/train.slurm \
#     --model Qwen2.5-1.5B-Instruct \
#     --task grpo \
#     --config math \
#     --accelerator zero3 \
#     --dp 8 \
#     --tp 1 \
#     --args "--run_name demo"

#SBATCH --job-name=open_r1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --nodes=1                 # Default: single node
#SBATCH --partition=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:8         # Request A100s explicitly
#SBATCH --account=mltheory      # Adjust for your cluster (e.g., hopper-prod)
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --export=ALL,ENV_MODE=venv,ENV_ACTIVATE=/n/fs/similarity/maxent-grpo/var/openr1/bin/activate


set -euo pipefail
set -x

export ROOT_DIR="$PWD"
export VAR_DIR="${VAR_DIR:-$ROOT_DIR/var}"
mkdir -p "$VAR_DIR/artifacts/logs"

# ---- Environment bootstrap (configure for your cluster) ----
if [[ "${MAXENT_DRY_RUN:-0}" != "1" ]]; then
  # Choose how to activate your environment on all nodes.
  # - For Conda: set ENV_MODE=conda, CONDA_SH to conda.sh, CONDA_ENV to env path or name
  # - For venv:  set ENV_MODE=venv and ENV_ACTIVATE to the venv's activate script path
  # Default to the project venv; override ENV_MODE/ENV_ACTIVATE if you prefer conda.
  export ENV_MODE="${ENV_MODE:-venv}"
  export CONDA_SH="${CONDA_SH:-/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh}"
  export CONDA_ENV="${CONDA_ENV:-$VAR_DIR/openr1}"
  export ENV_ACTIVATE="${ENV_ACTIVATE:-$VAR_DIR/openr1/bin/activate}"

  # Ensure CUDA toolkit matches the PyTorch build (torch 2.7.0 was compiled with CUDA 12.6)
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

  # Activate once in submission shell (best-effort); srun subshells will also activate.
  if [[ "$ENV_MODE" == "conda" ]]; then
    if [[ -f "$CONDA_SH" ]]; then source "$CONDA_SH"; fi
    conda activate "$CONDA_ENV" || echo "[warn] submit-shell: failed to conda activate $CONDA_ENV"
  elif [[ -f "$ENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE" || echo "[warn] submit-shell: failed to source $ENV_ACTIVATE"
  else
    echo "[warn] No environment activation performed (ENV_MODE=$ENV_MODE)."
  fi
  # Ensure downstream shells inherit the venv path even if activation is skipped.
  export VIRTUAL_ENV="${VIRTUAL_ENV:-$VAR_DIR/openr1}"
  export PATH="$VIRTUAL_ENV/bin:${PATH:-}"
  # Sanity check: confirm python and wandb availability on the submit shell (workers will re-run activation).
  echo "[debug] python after activation: $(which python)"
  python - <<'PY' || echo "[warn] wandb import failed in submit shell"
import sys, importlib.util
print("exe", sys.executable)
print("wandb spec", importlib.util.find_spec("wandb"))
try:
    import wandb
    print("wandb version", wandb.__version__, "file", getattr(wandb, "__file__", None))
except Exception as e:
    print("wandb import error:", e)
PY
  # Avoid picking up user-site packages (~/.local) that can conflict (e.g., httpx)
  export PYTHONNOUSERSITE=1
  export HF_TOKEN="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"

  # Route all caches and temp dirs into var/
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
  else
    echo "[warn] HF_TOKEN not set; gated models/datasets may fail to download." >&2
  fi
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
  # Force-enable wandb for training jobs so the Transformers callback sees it.
  export WANDB_DISABLED=0
  export TMPDIR="$VAR_DIR/tmp"
  export VLLM_USAGE_STATS_PATH="$VAR_DIR/cache/vllm/usage_stats.json"
  export VLLM_NO_USAGE_STATS=1
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_HUB_ENABLE_PROGRESS_BARS=1
  # Enable vLLM prompt dedup unless overridden.
  export MAXENT_VLLM_DEDUP="${MAXENT_VLLM_DEDUP:-1}"
  # Debugging aids for distributed/CUDA failures (override to disable).
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
  export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
  mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" \
           "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" \
           "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR" \
           "$(dirname "$VLLM_USAGE_STATS_PATH")"
  echo "Preflight: pin hub + httpx for compatibility"
  # Pin once in the submission shell to avoid concurrent pip on multiple nodes
  python -m pip install -q --upgrade 'huggingface-hub[cli,hf_xet]>=0.30.2,<1.0' 'httpx==0.27.2' || true
  # Ensure the local repo (with recent edits) is installed in editable mode for worker nodes.
  python -m pip install -q -e "$ROOT_DIR" || {
    echo "[warn] Failed to install editable package from $ROOT_DIR" >&2
  }
else
  export ENV_MODE="${ENV_MODE:-venv}"
  export CONDA_ENV="${CONDA_ENV:-$VAR_DIR/openr1}"
  export ENV_ACTIVATE="${ENV_ACTIVATE:-$VAR_DIR/openr1/bin/activate}"
  export VIRTUAL_ENV="${VIRTUAL_ENV:-$VAR_DIR/openr1}"
  export PATH="$VIRTUAL_ENV/bin:${PATH:-}"
  export PYTHONNOUSERSITE=1
fi

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch var/repo/ops/slurm/train.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Recipe model directory under configs/recipes/ (e.g., Qwen2.5-1.5B-Instruct)"
  echo "  --task TASK              Task name: grpo | maxent (alias: maxent-grpo)"
  echo "  --config SUFFIX          Config suffix in configs/recipes/<model>/<task-dir>/config_<SUFFIX>.yaml (e.g., math)"
  echo "  --accelerator NAME       Accelerate config in configs/recipes/accelerate_configs/<NAME>.yaml (e.g., zero3)"
  echo "  --dp N                   vLLM data parallel size on the server node (default: 1)"
  echo "  --tp N                   vLLM tensor parallel size on the server node (default: 1)"
  echo "  --vllm-port PORT         vLLM server port (ignored in colocate; default: 29525)"
  echo "  --vllm-group-port PORT   NCCL group port for weight sync RPCs (ignored in colocate; default: 29535)"
  echo "  --args \"ARGS\"          Extra args to pass to the training entrypoint"
  exit 0
fi

# Defaults (so `sbatch var/repo/ops/slurm/train.slurm` just works)
# - Single node, 8 GPUs; vLLM runs in-process (colocate) with training
# - Qwen 1.5B GRPO math with accelerate zero3
MODEL="${MODEL:-Qwen2.5-1.5B-Instruct}"
TASK="${TASK:-grpo}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
ACCELERATOR="${ACCELERATOR:-zero3}"
DP=${DP:-1}
TP=${TP:-1}
VLLM_PORT=${VLLM_PORT:-29525}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-29535}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8000}
OPTIONAL_ARGS="${OPTIONAL_ARGS:-}"

# Parse CLI
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)        MODEL="$2"; shift 2;;
    --task)         TASK="$2"; shift 2;;
    --config)       CONFIG_SUFFIX="$2"; shift 2;;
    --accelerator)  ACCELERATOR="$2"; shift 2;;
    --dp)           DP="$2"; shift 2;;
    --tp)           TP="$2"; shift 2;;
    --vllm-port)    VLLM_PORT="$2"; shift 2;;
    --vllm-group-port) VLLM_GROUP_PORT="$2"; shift 2;;
    --args)         OPTIONAL_ARGS="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

export VLLM_GROUP_PORT
export PORT_FOR_COMMUNICATION="${PORT_FOR_COMMUNICATION:-$VLLM_GROUP_PORT}"

# Validate
if [[ -z "$MODEL" || -z "$TASK" || -z "$CONFIG_SUFFIX" || -z "$ACCELERATOR" ]]; then
  echo "Error: --model, --task, --config, --accelerator are required" >&2
  exit 1
fi

TASK_NORMALIZED=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')
case "$TASK_NORMALIZED" in
  grpo)
    TASK_DIR="grpo"
    TRAIN_ENTRYPOINT="src/maxent_grpo/grpo.py"
    ;;
  maxent|maxent-grpo)
    TASK_DIR="maxent-grpo"
    TRAIN_ENTRYPOINT="src/maxent_grpo/maxent_grpo.py"
    ;;
  infoseed|info-seed)
    TASK_DIR="infoseed"
    TRAIN_ENTRYPOINT="src/maxent_grpo/infoseed.py"
    ;;
  *)
    echo "Unsupported --task: $TASK (use: grpo | maxent | maxent-grpo | infoseed)" >&2
    exit 1
    ;;
esac

CONFIG_FILE="configs/recipes/${MODEL}/${TASK_DIR}/config_${CONFIG_SUFFIX}.yaml"
ACCEL_FILE="configs/recipes/accelerate_configs/${ACCELERATOR}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config not found: $CONFIG_FILE" >&2
  exit 1
fi
if [[ ! -f "$ACCEL_FILE" ]]; then
  echo "Accelerate config not found: $ACCEL_FILE" >&2
  exit 1
fi
export GRPO_RECIPE_USED="${GRPO_RECIPE_USED:-$CONFIG_FILE}"
# Prefer an absolute recipe path so worker CWD doesn't matter.
GRPO_RECIPE_PATH="$CONFIG_FILE"
if command -v realpath >/dev/null 2>&1; then
  GRPO_RECIPE_PATH="$(realpath "$CONFIG_FILE")"
elif command -v readlink >/dev/null 2>&1; then
  GRPO_RECIPE_PATH="$(readlink -f "$CONFIG_FILE" 2>/dev/null || echo "$CONFIG_FILE")"
fi
# Ensure the training entrypoint loads the recipe via GRPO_RECIPE to avoid
# TRL parser conflicts when reward fields overlap.
export GRPO_RECIPE="${GRPO_RECIPE:-$GRPO_RECIPE_PATH}"
export GRPO_RECIPE_USED="${GRPO_RECIPE_USED:-$GRPO_RECIPE_PATH}"
export ACCELERATE_CONFIG_FILE="$ACCEL_FILE"
if [[ "${MAXENT_DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] model=$MODEL task=$TASK config=$CONFIG_FILE accelerator=$ACCEL_FILE entrypoint=$TRAIN_ENTRYPOINT"
  exit 0
fi
if grep -qi '^[[:space:]]*distributed_type:[[:space:]]*deepspeed' "$ACCEL_FILE"; then
  export ACCELERATE_USE_DEEPSPEED="true"
  # Default to DeepSpeed's native save_checkpoint; allow opting into
  # Accelerate-only state saves per job if needed.
  export MAXENT_PREFER_ACCELERATE_STATE_SAVE="${MAXENT_PREFER_ACCELERATE_STATE_SAVE:-false}"
else
  export ACCELERATE_USE_DEEPSPEED="${ACCELERATE_USE_DEEPSPEED:-false}"
fi
# Optional: skip accelerator.save_state when DeepSpeed is active to avoid heavy shard saves.
export MAXENT_SKIP_DEEPSPEED_STATE_SAVE="${MAXENT_SKIP_DEEPSPEED_STATE_SAVE:-false}"
export MAXENT_ALLOW_DEEPSPEED_STATE_SAVE="${MAXENT_ALLOW_DEEPSPEED_STATE_SAVE:-false}"
# Default to shallow HF metadata on periodic checkpoints to avoid rewriting
# large model.safetensors files each time. Set MAXENT_CHECKPOINT_METADATA_MODE=full
# if you specifically want save_pretrained at every save.
export MAXENT_CHECKPOINT_METADATA_MODE="${MAXENT_CHECKPOINT_METADATA_MODE:-shallow}"
# Optional: base directory to copy static HF metadata from when using shallow
# checkpoints. Leave empty to default to the training output_dir.
export MAXENT_CHECKPOINT_METADATA_SOURCE="${MAXENT_CHECKPOINT_METADATA_SOURCE:-}"

# Best-effort: read num_processes from accelerate config to honor its world size
ACCEL_NUM_PROCS=$(grep -E "^[[:space:]]*num_processes:" "$ACCEL_FILE" | head -n1 | awk '{print $2}' || true)

# Extract model + revision for vLLM from YAML (best-effort)
MODEL_ID=$(grep -E "^[[:space:]]*model_name_or_path:" "$CONFIG_FILE" | awk '{print $2}' || true)
REVISION=$(grep -E "^[[:space:]]*model_revision:" "$CONFIG_FILE" | head -n1 | awk '{print $2}' || true)
VLLM_MODEL="${VLLM_MODEL:-${MODEL_ID:-$MODEL}}"

# Extract grad_acc override (and vLLM overrides) from optional args if present; otherwise from YAML
GRAD_ACC_STEPS=$(grep -E "^[[:space:]]*gradient_accumulation_steps:" "$CONFIG_FILE" | awk '{print $2}' || echo 1)
VLLM_MODE_OVERRIDE=""
USE_VLLM_OVERRIDE=""
VLLM_SYNC_OVERRIDE=""
RUN_NAME_OVERRIDE=""
WANDB_GROUP_OVERRIDE=""
IFS=' ' read -ra ARG_ARR <<< "$OPTIONAL_ARGS"
for idx in "${!ARG_ARR[@]}"; do
  arg="${ARG_ARR[$idx]}"
  case "$arg" in
    --gradient_accumulation_steps=*)
      GRAD_ACC_STEPS="${arg#*=}"
      ;;
    --gradient_accumulation_steps)
      GRAD_ACC_STEPS="${ARG_ARR[$((idx+1))]:-}"
      ;;
    --vllm_mode=*)
      VLLM_MODE_OVERRIDE="${arg#*=}"
      ;;
    --vllm_mode|--vllm-mode)
      VLLM_MODE_OVERRIDE="${ARG_ARR[$((idx+1))]:-}"
      ;;
    --vllm_sync_weights=*)
      VLLM_SYNC_OVERRIDE="${arg#*=}"
      ;;
    --vllm_sync_weights)
      VLLM_SYNC_OVERRIDE="${ARG_ARR[$((idx+1))]:-}"
      ;;
    --run_name=*)
      RUN_NAME_OVERRIDE="${arg#*=}"
      ;;
    --run_name)
      RUN_NAME_OVERRIDE="${ARG_ARR[$((idx+1))]:-}"
      ;;
    --wandb_run_group=*)
      WANDB_GROUP_OVERRIDE="${arg#*=}"
      ;;
    --wandb_run_group)
      WANDB_GROUP_OVERRIDE="${ARG_ARR[$((idx+1))]:-}"
      ;;
    --use_vllm=*)
      USE_VLLM_OVERRIDE="${arg#*=}"
      ;;
    --use_vllm|--use-vllm)
      USE_VLLM_OVERRIDE="${ARG_ARR[$((idx+1))]:-}"
      ;;
  esac
done
echo "gradient_accumulation_steps = $GRAD_ACC_STEPS"

# Detect whether this config wants vLLM (optional args can override)
USE_VLLM=false
if grep -qE '^[[:space:]]*use_vllm:[[:space:]]*true' "$CONFIG_FILE"; then
  USE_VLLM=true
fi
if [[ -n "${USE_VLLM_OVERRIDE}" ]]; then
  case "${USE_VLLM_OVERRIDE,,}" in
    1|true|yes|y) USE_VLLM=true ;;
    0|false|no|n) USE_VLLM=false ;;
    *) echo "Warning: unknown --use_vllm value '${USE_VLLM_OVERRIDE}'; keeping config value." >&2 ;;
  esac
fi
if [[ -n "${RUN_NAME_OVERRIDE}" ]]; then
  export MAXENT_RUN_NAME="${RUN_NAME_OVERRIDE}"
fi
if [[ -n "${WANDB_GROUP_OVERRIDE}" ]]; then
  export WANDB_RUN_GROUP="${WANDB_GROUP_OVERRIDE}"
  export MAXENT_WANDB_RUN_GROUP="${WANDB_GROUP_OVERRIDE}"
fi
if [[ -n "${VLLM_SYNC_OVERRIDE}" ]]; then
  export MAXENT_VLLM_SYNC_WEIGHTS="${VLLM_SYNC_OVERRIDE}"
fi

VLLM_MODE=$(grep -E "^[[:space:]]*vllm_mode:" "$CONFIG_FILE" | head -n1 | awk '{print $2}' | tr -d '"' | tr '[:upper:]' '[:lower:]')
if [[ -n "${VLLM_MODE_OVERRIDE}" ]]; then
  VLLM_MODE="${VLLM_MODE_OVERRIDE}"
fi
VLLM_MODE="$(echo "${VLLM_MODE:-}" | tr -d '"' | tr '[:upper:]' '[:lower:]')"
if [[ "$VLLM_MODE" == "inline" || "$VLLM_MODE" == "local" || "$VLLM_MODE" == "inprocess" || "$VLLM_MODE" == "in-process" ]]; then
  VLLM_MODE="colocate"
fi
if [[ -z "$VLLM_MODE" ]]; then
  VLLM_MODE="colocate"
elif [[ "$VLLM_MODE" != "server" && "$VLLM_MODE" != "colocate" ]]; then
  echo "Warning: unknown vllm_mode '${VLLM_MODE}'; defaulting to colocate." >&2
  VLLM_MODE="colocate"
fi
export VLLM_MODE

if $USE_VLLM && [[ "$VLLM_MODE" != "colocate" ]]; then
  echo "This launcher expects vLLM colocate mode; got '${VLLM_MODE}'." >&2
  exit 1
fi

# SLURM topology
NUM_NODES=${SLURM_NNODES}

# Infer visible GPUs to avoid over-launching when CUDA_VISIBLE_DEVICES is set.
CUDA_VISIBLE_COUNT=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES,,}" != "none" ]]; then
  IFS=',' read -ra _cudas <<< "$CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_COUNT=${#_cudas[@]}
fi

# Respect explicit override, then Slurm hint, then CUDA visibility, else default to 8.
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${CUDA_VISIBLE_COUNT:-8}}}
if [[ -n "$CUDA_VISIBLE_COUNT" ]] && (( CUDA_VISIBLE_COUNT > 0 )) && (( CUDA_VISIBLE_COUNT < GPUS_PER_NODE )); then
  GPUS_PER_NODE=$CUDA_VISIBLE_COUNT
fi
readarray -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_PORT=${MASTER_PORT:-6000}

TRAIN_NODES=("${NODELIST[@]}")

SINGLE_NODE=false
if (( NUM_NODES == 1 )); then SINGLE_NODE=true; fi
if ! $SINGLE_NODE; then
  echo "Colocate mode requires a single node job (nodes=${NUM_NODES})." >&2
  exit 1
fi

MASTER_NODE=${TRAIN_NODES[0]}
MASTER_ADDR=${MASTER_NODE}
ACCELERATE_MAIN_ADDR=${MASTER_ADDR}
if $SINGLE_NODE; then
  ACCELERATE_MAIN_ADDR="${ACCELERATE_MAIN_ADDR_LOCALHOST:-127.0.0.1}"
  export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
fi
TRAIN_NUM_MACHINES=${#TRAIN_NODES[@]}
if (( TRAIN_NUM_MACHINES <= 0 )); then
  echo "No training nodes detected after vLLM allocation." >&2
  exit 1
fi
echo "SLURM nodes: ${NODELIST[*]}"
echo "Train nodes: ${TRAIN_NODES[*]}"

TRAIN_LOCAL_PROCS=${TRAIN_LOCAL_PROCS:-${ACCEL_NUM_PROCS:-$GPUS_PER_NODE}}
if (( TRAIN_LOCAL_PROCS > GPUS_PER_NODE )); then
  echo "Requested TRAIN_LOCAL_PROCS=${TRAIN_LOCAL_PROCS} exceeds visible GPUs (${GPUS_PER_NODE})." >&2
  exit 1
fi
if (( TRAIN_LOCAL_PROCS <= 0 )); then
  echo "Training GPU count per node must be positive (computed $TRAIN_LOCAL_PROCS)." >&2
  exit 1
fi
WORLD_SIZE=$(( TRAIN_LOCAL_PROCS * TRAIN_NUM_MACHINES ))

# Robust NCCL settings
export NCCL_ASYNC_ERROR_HANDLING=1
# vLLM-specific NCCL overrides are applied only when launching vLLM.
# Pre-enable NCCL debug for sync failures; override per job if needed.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

# Build Accelerate launcher and training command
NODELIST_CSV=$(IFS=,; echo "${TRAIN_NODES[*]}")
export CMD="${TRAIN_ENTRYPOINT} ${OPTIONAL_ARGS}"

TRAIN_PROCS=${TRAIN_LOCAL_PROCS}

export ACC_TEE=${ACC_TEE:-3}
if [[ -n "${MAXENT_LOG_LEVEL:-}" ]]; then
  case "${MAXENT_LOG_LEVEL}" in
    trace|debug) MAXENT_LOG_LEVEL=10 ;;
    info) MAXENT_LOG_LEVEL=20 ;;
    warning|warn) MAXENT_LOG_LEVEL=30 ;;
    error) MAXENT_LOG_LEVEL=40 ;;
    critical|fatal) MAXENT_LOG_LEVEL=50 ;;
  esac
fi
export MAXENT_LOG_LEVEL="${MAXENT_LOG_LEVEL:-30}"
export MAXENT_LOG_LEVEL_NUM="${MAXENT_LOG_LEVEL}"
if [[ -n "${MAXENT_LOG_LEVEL_RUN:-}" ]]; then
  case "${MAXENT_LOG_LEVEL_RUN}" in
    trace|debug) MAXENT_LOG_LEVEL_RUN=10 ;;
    info) MAXENT_LOG_LEVEL_RUN=20 ;;
    warning|warn) MAXENT_LOG_LEVEL_RUN=30 ;;
    error) MAXENT_LOG_LEVEL_RUN=40 ;;
    critical|fatal) MAXENT_LOG_LEVEL_RUN=50 ;;
  esac
fi
export MAXENT_LOG_LEVEL_RUN="${MAXENT_LOG_LEVEL_RUN:-20}"
export MAXENT_LOG_LEVEL_RUN_NUM="${MAXENT_LOG_LEVEL_RUN}"
export MAXENT_EVAL_BATCH_TIMEOUT_S="${MAXENT_EVAL_BATCH_TIMEOUT_S:-60}"
export MAXENT_EVAL_ONLY_RANK0="${MAXENT_EVAL_ONLY_RANK0:-0}"
if [[ "${MAXENT_EVAL_ONLY_RANK0:-0}" != "0" ]]; then
  export MAXENT_EVAL_FORCE_LOCAL_VLLM="${MAXENT_EVAL_FORCE_LOCAL_VLLM:-1}"
else
  export MAXENT_EVAL_FORCE_LOCAL_VLLM="${MAXENT_EVAL_FORCE_LOCAL_VLLM:-0}"
fi
export MAXENT_VLLM_LOG_MIRROR_FILE="${MAXENT_VLLM_LOG_MIRROR_FILE:-$VAR_DIR/artifacts/logs/vllm-${SLURM_JOB_ID}.out}"
export MAXENT_VLLM_SYNC_LOG_EVERY="${MAXENT_VLLM_SYNC_LOG_EVERY:-1}"
export MAXENT_VLLM_SYNC_MAX_PARAMS="${MAXENT_VLLM_SYNC_MAX_PARAMS:-200}"
export MAXENT_VLLM_COLOCATE_MAX_MODEL_LEN="${MAXENT_VLLM_COLOCATE_MAX_MODEL_LEN:-2048}"
export MAXENT_VLLM_COLOCATE_GPU_UTIL="${MAXENT_VLLM_COLOCATE_GPU_UTIL:-0.5}"
export MAXENT_VLLM_COLOCATE_TP="${MAXENT_VLLM_COLOCATE_TP:-1}"
export MAXENT_VLLM_COLOCATE_ENFORCE_EAGER="${MAXENT_VLLM_COLOCATE_ENFORCE_EAGER:-1}"
export MAXENT_VLLM_COLOCATE_FORCE_V0="${MAXENT_VLLM_COLOCATE_FORCE_V0:-1}"
export MAXENT_VLLM_COLOCATE_INIT_MODE="${MAXENT_VLLM_COLOCATE_INIT_MODE:-subprocess}"
export MAXENT_VLLM_COLOCATE_INIT_TIMEOUT_S="${MAXENT_VLLM_COLOCATE_INIT_TIMEOUT_S:-120}"
export MAXENT_VLLM_COLOCATE_INIT_STACK_S="${MAXENT_VLLM_COLOCATE_INIT_STACK_S:-30}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
if [[ "${VLLM_USE_V1}" == "1" ]]; then
  export MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND="${MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND:-}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"
else
  export MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND="${MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND:-XFORMERS}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
fi
export LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info $VIRTUAL_ENV/bin/accelerate launch \
  --config_file ${ACCEL_FILE} \
  --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
  --num_machines ${TRAIN_NUM_MACHINES} \
  --num_processes ${TRAIN_PROCS} \
  --main_process_ip ${ACCELERATE_MAIN_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${SLURM_PROCID} \
  --tee ${ACC_TEE}"

export CMD

# vLLM host/port should be supplied via recipe config (vllm_url), not CLI flags.

if $SINGLE_NODE; then
  SINGLE_NODE_GRES=${GPUS_PER_NODE}
  SRUN_ARGS=(
    --wait=60
    --kill-on-bad-exit=1
    --nodes=1
    --ntasks=1
    --gres=gpu:${SINGLE_NODE_GRES}
    --nodelist="${MASTER_ADDR}"
  )
else
  SRUN_ARGS=(
    --wait=60
    --kill-on-bad-exit=1
    --nodes=${NUM_NODES}
    --ntasks=${NUM_NODES}
    --nodelist="${NODELIST_CSV}"
  )
fi

echo "Launching training across nodes…"

# Predefine trainer log path in the outer shell (avoid set -u issues)
export TRAINING_LOG="${TRAINING_LOG:-$VAR_DIR/artifacts/logs/train_${SLURM_JOB_ID}.log}"
mkdir -p "$(dirname "$TRAINING_LOG")"
: > "$TRAINING_LOG"

srun "${SRUN_ARGS[@]}" /bin/bash -lc "set -euo pipefail; set -x; \
  if [[ '$ENV_MODE' == 'conda' ]]; then if [[ -f '$CONDA_SH' ]]; then source '$CONDA_SH'; fi; conda activate '$CONDA_ENV'; \
  elif [[ -f '$ENV_ACTIVATE' ]]; then source '$ENV_ACTIVATE'; fi; \
  export VLLM_HOST_IP=\"${VLLM_HOST_IP:-127.0.0.1}\"; \
  # Ensure vLLM colocate stays single-process and avoids implicit DP groups.
  export VLLM_ENABLE_V1_MULTIPROCESSING=\"${VLLM_ENABLE_V1_MULTIPROCESSING:-0}\"; \
  export VLLM_DP_SIZE=\"${VLLM_DP_SIZE:-1}\"; \
  export VLLM_DP_RANK=\"${VLLM_DP_RANK:-0}\"; \
  export VLLM_DP_RANK_LOCAL=\"${VLLM_DP_RANK_LOCAL:-0}\"; \
  export VLLM_DP_MASTER_IP=\"${VLLM_DP_MASTER_IP:-127.0.0.1}\"; \
  export VLLM_DP_MASTER_PORT=\"${VLLM_DP_MASTER_PORT:-0}\"; \
  MAXENT_VLLM_COLOCATE_DEVICE=\"\${MAXENT_VLLM_COLOCATE_DEVICE:-}\"; \
  _CUDA_DEVS=(); \
  if [[ -z \"\${MAXENT_VLLM_COLOCATE_DEVICE}\" ]]; then \
    if [[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ]]; then \
      IFS=',' read -r -a _CUDA_DEVS <<< \"\${CUDA_VISIBLE_DEVICES}\"; \
      if [[ \${#_CUDA_DEVS[@]} -gt ${TRAIN_PROCS} ]]; then \
        MAXENT_VLLM_COLOCATE_DEVICE=\"cuda:\${_CUDA_DEVS[-1]}\"; \
      fi; \
    elif [[ ${GPUS_PER_NODE} -gt ${TRAIN_PROCS} ]]; then \
      MAXENT_VLLM_COLOCATE_DEVICE=\"cuda:$((GPUS_PER_NODE-1))\"; \
    fi; \
  fi; \
  export MAXENT_LOG_LEVEL=\"${MAXENT_LOG_LEVEL_RUN_NUM}\"; \
  export MAXENT_EVAL_BATCH_TIMEOUT_S=\"${MAXENT_EVAL_BATCH_TIMEOUT_S}\"; \
  export MAXENT_EVAL_ONLY_RANK0=\"${MAXENT_EVAL_ONLY_RANK0}\"; \
  export MAXENT_EVAL_FORCE_LOCAL_VLLM=\"${MAXENT_EVAL_FORCE_LOCAL_VLLM}\"; \
  export MAXENT_VLLM_LOG_MIRROR_FILE=\"${MAXENT_VLLM_LOG_MIRROR_FILE}\"; \
  export MAXENT_VLLM_SYNC_LOG_EVERY=\"${MAXENT_VLLM_SYNC_LOG_EVERY}\"; \
  export MAXENT_VLLM_SYNC_MAX_PARAMS=\"${MAXENT_VLLM_SYNC_MAX_PARAMS}\"; \
  export MAXENT_VLLM_COLOCATE_MAX_MODEL_LEN=\"${MAXENT_VLLM_COLOCATE_MAX_MODEL_LEN}\"; \
  export MAXENT_VLLM_COLOCATE_GPU_UTIL=\"${MAXENT_VLLM_COLOCATE_GPU_UTIL}\"; \
  export MAXENT_VLLM_COLOCATE_TP=\"${MAXENT_VLLM_COLOCATE_TP}\"; \
  export MAXENT_VLLM_COLOCATE_ENFORCE_EAGER=\"${MAXENT_VLLM_COLOCATE_ENFORCE_EAGER}\"; \
  export MAXENT_VLLM_COLOCATE_FORCE_V0=\"${MAXENT_VLLM_COLOCATE_FORCE_V0}\"; \
  export MAXENT_VLLM_COLOCATE_INIT_MODE=\"${MAXENT_VLLM_COLOCATE_INIT_MODE}\"; \
  export MAXENT_VLLM_COLOCATE_INIT_STACK_S=\"${MAXENT_VLLM_COLOCATE_INIT_STACK_S}\"; \
  export MAXENT_VLLM_COLOCATE_DEVICE=\"\${MAXENT_VLLM_COLOCATE_DEVICE}\"; \
  export MAXENT_VLLM_COLOCATE_INIT_TIMEOUT_S=\"${MAXENT_VLLM_COLOCATE_INIT_TIMEOUT_S}\"; \
  export MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND=\"${MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND}\"; \
  export VLLM_USE_V1=\"${VLLM_USE_V1}\"; \
  export VLLM_ATTENTION_BACKEND=\"${VLLM_ATTENTION_BACKEND}\"; \
  export GRPO_RECIPE=\"${GRPO_RECIPE}\"; \
  export GRPO_RECIPE_USED=\"${GRPO_RECIPE_USED}\"; \
  export PYTHONPATH=\"${ROOT_DIR}/src:\${PYTHONPATH:-}\"; \
  export PYTHONNOUSERSITE=1; \
  echo 'Trainer node uses python:'; which python || true; python --version || true; \
  export NCCL_DEBUG=\"\${NCCL_DEBUG:-INFO}\"; \
  export NCCL_DEBUG_SUBSYS=\"\${NCCL_DEBUG_SUBSYS:-INIT,NET}\"; \
  export NCCL_DEBUG_FILE=\"${VAR_DIR}/artifacts/logs/nccl-train-${SLURM_JOB_ID}-%h-%p.log\"; \
  echo \"NCCL env: NCCL_DEBUG=\${NCCL_DEBUG:-} NCCL_DEBUG_SUBSYS=\${NCCL_DEBUG_SUBSYS:-} NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME:-} NCCL_P2P_DISABLE=\${NCCL_P2P_DISABLE:-} NCCL_IB_DISABLE=\${NCCL_IB_DISABLE:-} NCCL_DEBUG_FILE=\${NCCL_DEBUG_FILE:-}\"; \
  echo 'Trainer node has accelerate:'; "$VIRTUAL_ENV/bin/accelerate" --version || which accelerate || true; python - <<'PY'
import sys
try:
    import accelerate
    print(getattr(accelerate, '__version__', 'unknown'))
except Exception as e:
    print(f'could not import accelerate: {e}', file=sys.stderr)
PY
  true; \
  # Route caches to PWD on the node
  export ROOT_DIR=\"\$(pwd)\"; \
  export HF_HOME=\"$HF_HOME\"; export XDG_CACHE_HOME=\"$XDG_CACHE_HOME\"; \
  export HUGGINGFACE_HUB_CACHE=\"$HUGGINGFACE_HUB_CACHE\"; \
  export HF_DATASETS_CACHE=\"$HF_DATASETS_CACHE\"; \
  export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\"; export TORCHINDUCTOR_CACHE_DIR=\"$TORCHINDUCTOR_CACHE_DIR\"; \
  export TRITON_CACHE_DIR=\"$TRITON_CACHE_DIR\"; export WANDB_DIR=\"$WANDB_DIR\"; export TMPDIR=\"$TMPDIR\"; \
  mkdir -p \"$HF_HOME\" \"$XDG_CACHE_HOME\" \"$HUGGINGFACE_HUB_CACHE\" \"$HF_DATASETS_CACHE\" \"$PIP_CACHE_DIR\" \"$TORCHINDUCTOR_CACHE_DIR\" \"$TRITON_CACHE_DIR\" \"$WANDB_DIR\" \"$TMPDIR\"; \
  export HF_HUB_ENABLE_HF_TRANSFER=1; export HF_HUB_ENABLE_PROGRESS_BARS=1; export PYTHONUNBUFFERED=1; export MAXENT_FAULTHANDLER=1; \
  export TRAINING_LOG=\"$VAR_DIR/artifacts/logs/train_${SLURM_JOB_ID}.log\"; \
  mkdir -p \"$VAR_DIR/artifacts/logs\"; : > \"$TRAINING_LOG\"; \
  echo \"Writing training logs to \$TRAINING_LOG (ACC_TEE=$ACC_TEE)\"; \
  echo 'Launch command:'; echo \"$LAUNCHER $CMD\"; echo 'About to start accelerate launch'; date; \
  echo \"Effective MAXENT_LOG_LEVEL=\${MAXENT_LOG_LEVEL}\"; \
  echo \"Effective GRPO_RECIPE=\${GRPO_RECIPE}\"; \
  echo 'Spawning accelerate launcher'; \
  set +e; \
  $LAUNCHER $CMD 2>&1 | tee -a \"\$TRAINING_LOG\"; \
  TRAIN_STATUS=\${PIPESTATUS[0]}; \
  set -e; \
  echo 'Accelerate launch finished'; date; \
  echo \"Training finished with exit code: \$TRAIN_STATUS\" | tee -a \"\$TRAINING_LOG\"; \
  exit \$TRAIN_STATUS;"

SRUN_STATUS=$?
echo "srun finished with exit code: ${SRUN_STATUS}"
if [[ ${SRUN_STATUS} -eq 0 ]]; then
  echo "Done."
else
  echo "Done (with errors)."
fi
