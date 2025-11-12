#!/bin/bash
# Multi-node Open‑R1 trainer with optional vLLM server on a dedicated node
#
# Example:
#   sbatch --nodes=2 slurm/train.slurm \
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
#SBATCH --gres=gpu:8              # Default: 8 GPUs on the node
#SBATCH --partition=mltheory      # Adjust for your cluster (e.g., hopper-prod)
#SBATCH --exclude=node915,node916,node917 # Avoid flaky nodes; adjust as needed
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=128:00:00
#SBATCH --account=mltheory      # Adjust for your cluster (e.g., hopper-prod)


set -euo pipefail
set -x

mkdir -p logs

# ---- Environment bootstrap (configure for your cluster) ----
# Choose how to activate your environment on all nodes.
# - For Conda: set ENV_MODE=conda, CONDA_SH to conda.sh, CONDA_ENV to env path or name
# - For venv:  set ENV_MODE=venv and ENV_ACTIVATE to the venv's activate script path
export ENV_MODE="${ENV_MODE:-conda}"
export CONDA_SH="${CONDA_SH:-/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-$PWD/openr1}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-$PWD/openr1/bin/activate}"

# Ensure CUDA toolkit matches the PyTorch build (torch 2.7.0 was compiled with CUDA 12.6)
if command -v module &>/dev/null; then
  module unload cuda || true
  module load cuda/12.6 || {
    echo "[warn] Failed to load cuda/12.6 module; ensure CUDA_HOME already points to a 12.6 toolkit." >&2
  }
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
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
# Avoid picking up user-site packages (~/.local) that can conflict (e.g., httpx)
export PYTHONNOUSERSITE=1

# Route all caches and temp dirs into the current working directory
export ROOT_DIR="$PWD"
export HF_TOKEN="<ENTER THIS>"
export HF_HOME="$ROOT_DIR/.hf_cache"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export XDG_CONFIG_HOME="$ROOT_DIR/.config"
export HUGGINGFACE_HUB_CACHE="$ROOT_DIR/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$ROOT_DIR/.cache/huggingface/datasets"
export PIP_CACHE_DIR="$ROOT_DIR/.pip_cache"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/.torchinductor"
export TRITON_CACHE_DIR="$ROOT_DIR/.triton"
export WANDB_DIR="$ROOT_DIR/.wandb"
export TMPDIR="$ROOT_DIR/.tmp"
export VLLM_USAGE_STATS_PATH="$ROOT_DIR/.cache/vllm/usage_stats.json"
export VLLM_NO_USAGE_STATS=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ENABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" \
         "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR" \
         "$(dirname "$VLLM_USAGE_STATS_PATH")"
echo "Preflight: pin hub + httpx for compatibility"
# Pin once in the submission shell to avoid concurrent pip on multiple nodes
python -m pip install -q --upgrade 'huggingface-hub[cli,hf_xet]>=0.30.2,<1.0' 'httpx==0.27.2' || true

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/train.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Recipe model directory under recipes/ (e.g., Qwen2.5-1.5B-Instruct)"
  echo "  --task TASK              Task name: grpo | maxent"
  echo "  --config SUFFIX          Config suffix in recipes/<model>/<task>/config_<SUFFIX>.yaml (e.g., math)"
  echo "  --accelerator NAME       Accelerate config in recipes/accelerate_configs/<NAME>.yaml (e.g., zero3)"
  echo "  --dp N                   vLLM data parallel size on the server node (default: 1)"
  echo "  --tp N                   vLLM tensor parallel size on the server node (default: 1)"
  echo "  --vllm-port PORT         vLLM server port (default: 8000)"
  echo "  --vllm-group-port PORT   NCCL group port for weight sync RPCs (default: 29535)"
  echo "  --args \"ARGS\"          Extra args to pass to the training entrypoint"
  exit 0
fi

# Defaults (so `sbatch slurm/train.slurm` just works)
# - Single node, 8 GPUs; vLLM on 1 GPU and GRPO training on remaining 7 GPUs
# - Qwen 1.5B GRPO math with accelerate zero3
MODEL="${MODEL:-Qwen2.5-1.5B-Instruct}"
TASK="${TASK:-grpo}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
ACCELERATOR="${ACCELERATOR:-zero3}"
DP=${DP:-1}
TP=${TP:-1}
VLLM_PORT=${VLLM_PORT:-29525}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-29535}
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

CONFIG_FILE="recipes/${MODEL}/${TASK}/config_${CONFIG_SUFFIX}.yaml"
ACCEL_FILE="recipes/accelerate_configs/${ACCELERATOR}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config not found: $CONFIG_FILE" >&2
  exit 1
fi
if [[ ! -f "$ACCEL_FILE" ]]; then
  echo "Accelerate config not found: $ACCEL_FILE" >&2
  exit 1
fi

# Resolve training entrypoint for this repo
case "$TASK" in
  grpo)   TRAIN_ENTRYPOINT="src/grpo.py";;
  maxent) TRAIN_ENTRYPOINT="src/maxent-grpo.py";;
  *) echo "Unsupported --task: $TASK (use: grpo | maxent)" >&2; exit 1;;
esac

# Extract model + revision for vLLM from YAML (best-effort)
MODEL_ID=$(grep -E "^[[:space:]]*model_name_or_path:" "$CONFIG_FILE" | awk '{print $2}' || true)
REVISION=$(grep -E "^[[:space:]]*model_revision:" "$CONFIG_FILE" | head -n1 | awk '{print $2}' || true)
VLLM_MODEL="${VLLM_MODEL:-${MODEL_ID:-$MODEL}}"

# Extract grad_acc override from optional args if present; otherwise from YAML
GRAD_ACC_STEPS=$(grep -E "^[[:space:]]*gradient_accumulation_steps:" "$CONFIG_FILE" | awk '{print $2}' || echo 1)
IFS=' ' read -ra ARG_ARR <<< "$OPTIONAL_ARGS"
for arg in "${ARG_ARR[@]:-}"; do
  if [[ "$arg" == --gradient_accumulation_steps=* ]]; then
    GRAD_ACC_STEPS="${arg#*=}"
    break
  fi
done
echo "gradient_accumulation_steps = $GRAD_ACC_STEPS"

# Detect whether this config wants vLLM server mode
USE_VLLM=false
if grep -qE '^[[:space:]]*use_vllm:[[:space:]]*true' "$CONFIG_FILE"; then
  USE_VLLM=true
fi

VLLM_MODE=$(grep -E "^[[:space:]]*vllm_mode:" "$CONFIG_FILE" | head -n1 | awk '{print $2}' | tr -d '"' | tr '[:upper:]' '[:lower:]')
if [[ -z "$VLLM_MODE" ]]; then
  VLLM_MODE="server"
fi
export VLLM_MODE

USE_VLLM_SERVER=false
if $USE_VLLM && [[ "$VLLM_MODE" == "server" ]]; then
  USE_VLLM_SERVER=true
fi
export USE_VLLM_SERVER

# SLURM topology
NUM_NODES=${SLURM_NNODES}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORLD_SIZE=$(( NUM_NODES * GPUS_PER_NODE ))
readarray -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=${MASTER_PORT:-6000}

TRAIN_NODES=("${NODELIST[@]}")
VLLM_NODE=""

SINGLE_NODE=false
if (( NUM_NODES == 1 )); then SINGLE_NODE=true; fi
SINGLE_NODE_VLLM=false
INLINE_VLLM=false

if $USE_VLLM_SERVER; then
  if ! $SINGLE_NODE; then
    # Reserve last node for vLLM server
    VLLM_NODE=${NODELIST[-1]}
    TRAIN_NODES=("${NODELIST[@]:0:$((NUM_NODES-1))}")
    NUM_NODES=$(( NUM_NODES - 1 ))
    WORLD_SIZE=$(( WORLD_SIZE - GPUS_PER_NODE ))
  else
    # Single-node: dedicate 1 GPU on the lone node to the vLLM server
    VLLM_NODE=${NODELIST[0]}
    SINGLE_NODE_VLLM=true
    INLINE_VLLM=true
  fi
elif $USE_VLLM && [[ "$VLLM_MODE" == "colocate" ]]; then
  if ! $SINGLE_NODE; then
    echo "vLLM colocate mode requires a single node job (nodes=${NUM_NODES})." >&2
    exit 1
  fi
  VLLM_NODE=${NODELIST[0]}
  SINGLE_NODE_VLLM=true
  INLINE_VLLM=true
fi
export INLINE_VLLM

echo "SLURM nodes: ${NODELIST[*]}"
echo "Train nodes: ${TRAIN_NODES[*]}"
if $USE_VLLM_SERVER; then
  echo "vLLM node:   $VLLM_NODE (dp=$DP, tp=$TP, port=$VLLM_PORT, group_port=$VLLM_GROUP_PORT)"
fi

if $SINGLE_NODE_VLLM && (( GPUS_PER_NODE < 2 )); then
  echo "Inline vLLM requires at least 2 GPUs on the node; detected $GPUS_PER_NODE" >&2
  exit 1
fi

VLLM_GROUP_PORT_FLAG=""
if $USE_VLLM; then
  if python -m trl.scripts.vllm_serve --help 2>&1 | grep -q -- "--group-port"; then
    VLLM_GROUP_PORT_FLAG="--group-port ${VLLM_GROUP_PORT}"
  else
    echo "[info] trl.scripts.vllm_serve does not advertise --group-port; relying on env overrides only."
  fi
fi
export VLLM_GROUP_PORT_FLAG

# Robust NCCL settings
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO

# Start vLLM server if requested (multi-node or dedicated node scenarios)
if $USE_VLLM_SERVER && ! $INLINE_VLLM; then
  # Ensure DP does not exceed GPUs on the vLLM head node
  if (( DP > GPUS_PER_NODE )); then
    echo "[warn] --dp ($DP) > GPUs on head ($GPUS_PER_NODE); clamping to $GPUS_PER_NODE"
    DP=$GPUS_PER_NODE
  fi
  # Prefer model from YAML; fallback to --model
  VLLM_MODEL="${MODEL_ID:-$MODEL}"
  echo "Launching vLLM server: $VLLM_MODEL (rev=${REVISION:-none})"
  if $SINGLE_NODE; then VLLM_GRES=1; else VLLM_GRES=$GPUS_PER_NODE; fi
  srun --nodes=1 --ntasks=1 --gres=gpu:${VLLM_GRES} --nodelist="$VLLM_NODE" \
    bash -lc "set -euo pipefail; \
    if [[ '$ENV_MODE' == 'conda' ]]; then if [[ -f '$CONDA_SH' ]]; then source '$CONDA_SH'; fi; conda activate '$CONDA_ENV'; \
    elif [[ -f '$ENV_ACTIVATE' ]]; then source '$ENV_ACTIVATE'; fi; \
    export PYTHONNOUSERSITE=1; \
    echo 'vLLM node uses python:'; which python || true; python --version || true; \
    # Route caches to PWD on the node
    export ROOT_DIR=\"\$(pwd)\"; \
    export HF_HOME=\"$HF_HOME\"; export XDG_CACHE_HOME=\"$XDG_CACHE_HOME\"; export XDG_CONFIG_HOME=\"$XDG_CONFIG_HOME\"; \
    export HUGGINGFACE_HUB_CACHE=\"$HUGGINGFACE_HUB_CACHE\"; \
    export HF_DATASETS_CACHE=\"$HF_DATASETS_CACHE\"; \
    export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\"; export TORCHINDUCTOR_CACHE_DIR=\"$TORCHINDUCTOR_CACHE_DIR\"; \
    export TRITON_CACHE_DIR=\"$TRITON_CACHE_DIR\"; export WANDB_DIR=\"$WANDB_DIR\"; export TMPDIR=\"$TMPDIR\"; \
    export VLLM_USAGE_STATS_PATH=\"$VLLM_USAGE_STATS_PATH\"; export VLLM_NO_USAGE_STATS=1; \
    mkdir -p \"$HF_HOME\" \"$XDG_CACHE_HOME\" \"$HUGGINGFACE_HUB_CACHE\" \"$HF_DATASETS_CACHE\" \"$PIP_CACHE_DIR\" \"$TORCHINDUCTOR_CACHE_DIR\" \"$TRITON_CACHE_DIR\" \"$WANDB_DIR\" \"$TMPDIR\"; \
    mkdir -p \"$(dirname "$VLLM_USAGE_STATS_PATH")\"; \
    export NCCL_DEBUG=WARN; \
    PORT_FOR_COMMUNICATION=$VLLM_GROUP_PORT \
    VLLM_GROUP_PORT=$VLLM_GROUP_PORT \
    python -m trl.scripts.vllm_serve \
      --model "$VLLM_MODEL" \
      ${REVISION:+--revision $REVISION} \
      --tensor_parallel_size $TP \
      --data_parallel_size $DP \
      --port $VLLM_PORT ${VLLM_GROUP_PORT_FLAG:+$VLLM_GROUP_PORT_FLAG} \
      --dtype float16 \
      --gpu_memory_utilization 0.90 \
      --enforce_eager false" \
    > logs/vllm-${SLURM_JOB_ID}.out 2>&1 &

  # Wait for health
  echo "Waiting for vLLM health at http://$VLLM_NODE:$VLLM_PORT/health …"
  for i in $(seq 1 180); do
    if curl -sf "http://$VLLM_NODE:$VLLM_PORT/health" >/dev/null 2>&1; then
      echo "vLLM is healthy."
      break
    fi
    sleep 2
  done
fi

# Build Accelerate launcher and training command
NODELIST_CSV=$(IFS=,; echo "${TRAIN_NODES[*]}")
export CMD="${TRAIN_ENTRYPOINT} --config ${CONFIG_FILE} ${OPTIONAL_ARGS}"

# For single-node with vLLM server, use remaining GPUs for training
if $SINGLE_NODE && $USE_VLLM; then
  TRAIN_PROCS=$(( GPUS_PER_NODE - 1 ))
  TRAIN_NUM_MACHINES=1
  # Train step gets remaining GPUs; Slurm will allocate around the vLLM step
else
  TRAIN_PROCS=${WORLD_SIZE}
  TRAIN_NUM_MACHINES=${NUM_NODES}
fi

export ACC_TEE=${ACC_TEE:-3}
export TRAIN_SEED=${TRAIN_SEED:-42}
export LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
  --config_file ${ACCEL_FILE} \
  --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
  --num_machines ${TRAIN_NUM_MACHINES} \
  --num_processes ${TRAIN_PROCS} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${SLURM_PROCID} \
  --tee ${ACC_TEE}"

CMD+=" --seed ${TRAIN_SEED}"
export CMD

# If vLLM server is used, forward host/port to trainer unless already set in OPTIONAL_ARGS
if $USE_VLLM_SERVER; then
  if [[ "$OPTIONAL_ARGS" != *"--vllm_server_base_url="* && "$OPTIONAL_ARGS" != *"--vllm_server_host="* ]]; then
    if $INLINE_VLLM; then
      VLLM_CLIENT_HOST="localhost"
    else
      VLLM_CLIENT_HOST="${VLLM_NODE}"
    fi
    export CMD+=" --vllm_server_host ${VLLM_CLIENT_HOST} --vllm_server_port ${VLLM_PORT}"
  fi
fi

if $SINGLE_NODE; then
  if $SINGLE_NODE_VLLM; then
    SINGLE_NODE_GRES=${GPUS_PER_NODE}
  else
    SINGLE_NODE_GRES=${GPUS_PER_NODE}
  fi
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
export TRAINING_LOG="${TRAINING_LOG:-$PWD/logs/train_${SLURM_JOB_ID}.log}"
mkdir -p "$(dirname "$TRAINING_LOG")"
: > "$TRAINING_LOG"
export INLINE_VLLM_FALLBACK_GPUS="$(seq -s, 0 $((GPUS_PER_NODE - 1)))"

srun "${SRUN_ARGS[@]}" bash -lc "set -euo pipefail; set -x; \
  if [[ '$ENV_MODE' == 'conda' ]]; then if [[ -f '$CONDA_SH' ]]; then source '$CONDA_SH'; fi; conda activate '$CONDA_ENV'; \
  elif [[ -f '$ENV_ACTIVATE' ]]; then source '$ENV_ACTIVATE'; fi; \
  export PYTHONNOUSERSITE=1; \
  echo 'Trainer node uses python:'; which python || true; python --version || true; \
  echo 'Trainer node has accelerate:'; which accelerate || true; python - <<'PY'
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
  if [ \"${INLINE_VLLM}\" = \"true\" ]; then \
    ORIGINAL_CUDA_VISIBLE_DEVICES=\"\${CUDA_VISIBLE_DEVICES:-}\"; \
    if [ -z \"\$ORIGINAL_CUDA_VISIBLE_DEVICES\" ]; then \
      ORIGINAL_CUDA_VISIBLE_DEVICES=\"${INLINE_VLLM_FALLBACK_GPUS}\"; \
    fi; \
    IFS=, read -ra ALL_GPU_IDS <<< \"\$ORIGINAL_CUDA_VISIBLE_DEVICES\"; \
    if [ \"\${#ALL_GPU_IDS[@]}\" -lt 2 ]; then \
      echo \"Inline vLLM requires at least 2 GPUs, saw: \$ORIGINAL_CUDA_VISIBLE_DEVICES\" >&2; exit 1; \
    fi; \
    INLINE_VLLM_GPU_ID=\"\${ALL_GPU_IDS[0]}\"; \
    TRAIN_GPU_IDS=(\"\${ALL_GPU_IDS[@]:1}\"); \
    INLINE_VLLM_TRAIN_GPUS=\"\${TRAIN_GPU_IDS[*]}\"; \
    INLINE_VLLM_TRAIN_GPUS=\"\${INLINE_VLLM_TRAIN_GPUS// /,}\"; \
    export INLINE_VLLM_GPU_ID INLINE_VLLM_TRAIN_GPUS; \
    echo \"Starting inline vLLM server on GPU \$INLINE_VLLM_GPU_ID…\"; \
    echo \"Trainer GPUs: \$INLINE_VLLM_TRAIN_GPUS\"; \
    VLLM_LOG=\"$ROOT_DIR/logs/vllm-${SLURM_JOB_ID}.out\"; \
    mkdir -p \"\$(dirname \"\$VLLM_LOG\")\"; \
    : > \"\$VLLM_LOG\"; \
    tail -n +1 -F \"\$VLLM_LOG\" & \
    VLLM_TAIL_PID=\$!; \
    PORT_FOR_COMMUNICATION=$VLLM_GROUP_PORT \
    VLLM_GROUP_PORT=$VLLM_GROUP_PORT \
    CUDA_VISIBLE_DEVICES=\$INLINE_VLLM_GPU_ID \
    python -m trl.scripts.vllm_serve \
      --model "$VLLM_MODEL" ${REVISION:+--revision $REVISION} \
      --tensor_parallel_size $TP --data_parallel_size $DP \
      --port $VLLM_PORT ${VLLM_GROUP_PORT_FLAG:+$VLLM_GROUP_PORT_FLAG} \
      --dtype float16 --gpu_memory_utilization 0.90 \
      --enforce_eager false > \"\$VLLM_LOG\" 2>&1 & \
    VLLM_PID=\$!; \
    echo \"Waiting for inline vLLM health at http://localhost:$VLLM_PORT/health …\"; \
    INLINE_VLLM_READY=false; \
    INLINE_VLLM_FAILURE_REASON=\"timeout\"; \
    for attempt in \$(seq 1 180); do \
      if curl -sf http://localhost:$VLLM_PORT/health >/dev/null 2>&1; then \
        INLINE_VLLM_READY=true; \
        break; \
      fi; \
      if ! kill -0 \$VLLM_PID 2>/dev/null; then \
        INLINE_VLLM_FAILURE_REASON=\"crashed\"; \
        break; \
      fi; \
      sleep 2; \
    done; \
    if [ \"\$INLINE_VLLM_READY\" != \"true\" ]; then \
      echo \"Inline vLLM failed to become healthy (reason: \$INLINE_VLLM_FAILURE_REASON).\" >&2; \
      tail -n 200 \"\$VLLM_LOG\" || true; \
      kill \$VLLM_PID || true; \
      wait \$VLLM_PID || true; \
      if [ -n \"\${VLLM_TAIL_PID:-}\" ]; then kill \$VLLM_TAIL_PID || true; fi; \
      exit 1; \
    fi; \
    echo 'Inline vLLM is healthy.'; \
  fi; \
  export HF_HUB_ENABLE_HF_TRANSFER=1; export HF_HUB_ENABLE_PROGRESS_BARS=1; export PYTHONUNBUFFERED=1; \
  export TRAINING_LOG=\"$ROOT_DIR/logs/train_${SLURM_JOB_ID}.log\"; \
  mkdir -p \"$ROOT_DIR/logs\"; : > \"$TRAINING_LOG\"; \
  echo \"Writing training logs to \$TRAINING_LOG (ACC_TEE=$ACC_TEE)\"; \
  echo 'Launch command:'; echo \"$LAUNCHER $CMD\"; echo 'About to start accelerate launch'; date; \
  echo 'Restricting training GPUs when inline vLLM is enabled'; \
  if [ "${INLINE_VLLM}" = "true" ]; then export CUDA_VISIBLE_DEVICES=\"\$INLINE_VLLM_TRAIN_GPUS\"; fi; \
  echo 'Spawning accelerate launcher'; \
  set +e; \
  $LAUNCHER $CMD 2>&1 | tee -a \"\$TRAINING_LOG\"; \
  TRAIN_STATUS=\${PIPESTATUS[0]}; \
  set -e; \
  echo 'Accelerate launch finished'; date; \
  echo \"Training finished with exit code: \$TRAIN_STATUS\" | tee -a \"\$TRAINING_LOG\"; \
  if [ "${INLINE_VLLM}" = "true" ]; then \
    echo 'Stopping inline vLLM…'; \
    kill \$VLLM_PID || true; \
    wait \$VLLM_PID || true; \
    if [ -n \"\${VLLM_TAIL_PID:-}\" ]; then \
      kill \$VLLM_TAIL_PID || true; \
      wait \$VLLM_TAIL_PID 2>/dev/null || true; \
    fi; \
  fi; \
  exit \$TRAIN_STATUS;"

SRUN_STATUS=$?
echo "srun finished with exit code: ${SRUN_STATUS}"
if [[ ${SRUN_STATUS} -eq 0 ]]; then
  echo "Done."
else
  echo "Done (with errors)."
fi
