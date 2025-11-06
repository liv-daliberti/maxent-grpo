#!/bin/bash
##
# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

#
# Unified training launcher (vLLM + GRPO via Accelerate/DeepSpeed)
# This consolidates the logic from training-math-grpo.sh so that README
# can simply instruct: ./training.sh
#

set -euo pipefail
set -E

# Simple timestamped logging helpers
ts() { date +%Y-%m-%dT%H:%M:%S; }
log() { echo "[$(ts)] $*"; }
trap 'echo "[$(ts)] ERR at line $LINENO: $BASH_COMMAND" >&2' ERR
if [ "${DEBUG:-0}" = "1" ]; then set -x; fi

# ----------------------------
# SLURM (optional) headers
# ----------------------------
#SBATCH --job-name=OpenR1_GRPO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:7
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=00:59:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --account=mltheory
#SBATCH --exclusive

# ----------------------------
# MODULES & PYTHON ENVIRONMENT
# ----------------------------
#module load cudatoolkit/12.4
export PATH="$HOME/.local/bin:$PATH"
# Avoid conda alias conflicts injected by cluster environment
unset CONDA_ENVS_PATH  # prefer CONDA_ENVS_DIRS
unset PYTHONHOME PYTHONPATH  # ensure stdlib is resolved from the env
export PYTHONNOUSERSITE=1

# (setup moved below after activating conda and defining ROOT_DIR)

# ----------------------------
# GPU parameterization
# ----------------------------
# Allow overriding the GPU split via env:
#  - GPUS_PER_JOB: total GPUs requested in the allocation (default 7)
#  - VLLM_GPUS:    GPUs for vLLM server (default 1)
#  - TRAIN_GPUS:   GPUs for training (default GPUS_PER_JOB - VLLM_GPUS)
export GPUS_PER_JOB=${GPUS_PER_JOB:-7}
export VLLM_GPUS=${VLLM_GPUS:-1}
if [ "$VLLM_GPUS" -lt 1 ]; then VLLM_GPUS=1; fi
MAX_TRAIN=$(( GPUS_PER_JOB - VLLM_GPUS ))
if [ "$MAX_TRAIN" -lt 1 ]; then MAX_TRAIN=1; fi
export TRAIN_GPUS=${TRAIN_GPUS:-$MAX_TRAIN}
# Accelerate processes = training GPUs
NUM_TOTAL=$GPUS_PER_JOB
NUM_TRAINING=$TRAIN_GPUS

# ----------------------------
# WandB cache and artifact dirs on /n/fs
# ----------------------------
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export VLLM_USAGE_STATS_PATH=/n/fs/similarity/vllm/usage_stats.json
export TMPDIR=/n/fs/similarity/wandb-offload/tmp
export HF_HUB_REQUEST_TIMEOUT=60

#mkdir -p /n/fs/similarity/vllm
#mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

# Optional: Set WANDB_CONFIG_DIR if needed (e.g. wandb/settings)
export WANDB_CONFIG_DIR=/n/fs/similarity/wandb-offload/config
#mkdir -p /n/fs/similarity/wandb-offload/{tmp,artifacts,cache,config}
#mkdir -p logs .cache .hf_cache .tmp .torchinductor .triton

# (local cache paths defined later using ROOT_DIR)

# ─── Load modules and conda ─────────────────────────────────────────────
log "Sourcing conda profile script"
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh

# ─── Localized Conda + Pip Setup ───────────────────────────────────────
export ROOT_DIR="$PWD"
log "ROOT_DIR=$ROOT_DIR"
# Brief node + GPU summary at start to mirror prior script behavior
log "Node(s): ${SLURM_NODELIST:-unknown}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; fi
export ENV_NAME="openr1"
export ENV_DIR="$ROOT_DIR/$ENV_NAME"

export CONDA_PKGS_DIRS="$ROOT_DIR/.conda_pkgs"
export CONDA_ENVS_DIRS="$ROOT_DIR/.conda_envs"
export CONDA_CACHEDIR="$ROOT_DIR/.conda_cache"
export PYTHONUSERBASE="$ROOT_DIR/.local"
export CONDARC="$ROOT_DIR/.condarc"
export PIP_CACHE_DIR="$ROOT_DIR/.pip_cache"
export PIP_CONFIG_FILE="$ROOT_DIR/.pip/pip.conf"
export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$CONDA_CACHEDIR" "$PIP_CACHE_DIR" "$ROOT_DIR/.pip"

# Create local env if missing (kept under repo)
if [ ! -x "$ENV_DIR/bin/python" ]; then
  log "Creating local conda env at $ENV_DIR from environment.yml…"
  conda env create -p "$ENV_DIR" -f "$ROOT_DIR/environment.yml"
fi

# ─── Activate Environment ───────────────────────────────────────────────
log "Activating env: $ENV_DIR"
conda activate "$ENV_DIR"
# If stdlib looks corrupted (encodings missing), recreate the env once
if [ ! -d "$ENV_DIR/lib/python3.11/encodings" ]; then
  log "Python stdlib appears incomplete (missing encodings). Recreating env…"
  conda env remove -p "$ENV_DIR" -y || true
  conda env create -p "$ENV_DIR" -f "$ROOT_DIR/environment.yml"
  conda activate "$ENV_DIR"
fi

# Ensure PyTorch stack matches vLLM/xformers requirements; install only if mismatched.
TORCH_VER=2.7.0
VISION_VER=0.22.0
AUDIO_VER=2.7.0
if ! python - <<PY
import sys
ok=False
try:
  import torch, torchvision, torchaudio
  ok = torch.__version__.startswith('${TORCH_VER}') and \
       torchvision.__version__.startswith('${VISION_VER}') and \
       torchaudio.__version__.startswith('${AUDIO_VER}')
except Exception:
  ok=False
print('PY TORCH_MATCH =', ok, ' torch=', globals().get('torch', type('x',(),{})()).__dict__.get('__version__','n/a'))
sys.exit(0 if ok else 1)
PY
then
  log "Upgrading PyTorch stack to ${TORCH_VER}/${VISION_VER}/${AUDIO_VER}"
  python -m pip uninstall -y torch torchvision torchaudio triton torchtriton pytorch-triton || true
  conda remove -y -p "$ENV_DIR" pytorch pytorch-cuda torchtriton triton || true
  conda clean -a -y
  STREAM_CANDIDATES=""
  if [ -n "${TORCH_CUDA_STREAM:-}" ]; then STREAM_CANDIDATES+=" ${TORCH_CUDA_STREAM}"; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -n1 || true)"
    case "$CUDA_VER" in
      12.6) STREAM_CANDIDATES+=" cu126" ;;
      12.4) STREAM_CANDIDATES+=" cu124" ;;
      12.1) STREAM_CANDIDATES+=" cu121" ;;
    esac
  fi
  STREAM_CANDIDATES+=" cu126 cu124 cu121"
  INSTALL_OK=0
  for S in $STREAM_CANDIDATES; do
    [ "${LAST_TRIED:-}" = "$S" ] && continue
    LAST_TRIED="$S"
    log "Trying PyTorch CUDA wheels stream: $S (torch==$TORCH_VER, torchvision==$VISION_VER, torchaudio==$AUDIO_VER)"
    set +e
    python -m pip install --index-url https://download.pytorch.org/whl/${S} \
      "torch==${TORCH_VER}" "torchvision==${VISION_VER}" "torchaudio==${AUDIO_VER}"
    STATUS=$?
    set -e
    if [ $STATUS -eq 0 ]; then INSTALL_OK=1; break; fi
  done
  if [ $INSTALL_OK -ne 1 ]; then
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] ERROR: Failed to install CUDA PyTorch ${TORCH_VER}. Tried:${STREAM_CANDIDATES}" >&2
    exit 2
  fi
else
  log "PyTorch stack already matches ${TORCH_VER}/${VISION_VER}/${AUDIO_VER}; skipping install"
fi


log "Conda env active at: $(which python)"
python --version

# Ensure pip uses this env and local cache; keep installs local to repo
log "Upgrading pip"
python -m pip install --upgrade pip
# Ensure huggingface-hub is in the compatible range for transformers (<1.0)
log "Ensuring huggingface-hub pinned to <1.0"
python -m pip install 'huggingface-hub[cli,hf_xet]>=0.30.2,<1.0'
# yq for YAML edit convenience
log "Installing yq"
python -m pip install --upgrade 'yq>=3.4,<4'
# TRL CLI imports rich.markdown which depends on markdown-it-py; ensure present
log "Installing rich + markdown-it-py for TRL CLI"
python -m pip install 'markdown-it-py>=3,<4' 'rich>=13,<14'

# Ensure Torch has CUDA support; auto-repair with pip CUDA wheels if missing.
# Use `if ! ...; then` form so set -e doesn't abort the script on non-zero.
# Removed redundant CUDA availability check; torch CUDA wheels are installed above.

# ─── Environment Identifiers ────────────────────────────────────────────
export RUN_NAME="Qwen1.5B-GRPO-Finetune"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Anchor log directory to this repository to avoid picking up cluster-wide LOG_DIR
export LOG_DIR="$ROOT_DIR/logs"
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml"
export CONFIG_FILE="recipes/accelerate_configs/zero3.yaml"
export SERVER_LOG="$LOG_DIR/liv_vllm_${RUN_NAME}_${TIMESTAMP}.log"
export TRAINING_LOG="$LOG_DIR/liv_train_${RUN_NAME}_${TIMESTAMP}.log"

# Ensure log directory exists for file redirections and Slurm output path
mkdir -p "$LOG_DIR"
log "Logs directory: $LOG_DIR"

# Update accelerate config so that num_processes = num_training_gpus
log "Patching accelerate config: num_processes -> $NUM_TRAINING"
cp "${CONFIG_FILE}" "${CONFIG_FILE}.bak"
python -m yq -y --in-place ".num_processes = $NUM_TRAINING" "$CONFIG_FILE"
log "Set accelerate num_processes to $NUM_TRAINING (total GPUs: $NUM_TOTAL)"

# ─── Local cache + logging paths ────────────────────────────────────────
# ensure old HF_TOKEN does not take precedence
unset HF_TOKEN
export HF_HOME="$ROOT_DIR/.hf_cache"
export HF_DATASETS_CACHE="$ROOT_DIR/.cache/huggingface/datasets"
# Prefer HF_HOME; avoid deprecated TRANSFORMERS_CACHE to reduce warnings
unset TRANSFORMERS_CACHE
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export TMPDIR="$ROOT_DIR/.tmp"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/.torchinductor"
export TRITON_CACHE_DIR="$ROOT_DIR/.triton"
export TORCH_LOAD_WEIGHTS_ONLY=0

# ─── Optional: disable vLLM usage stats ────────────────────────────────
export VLLM_API_KEY="dummy"
export VLLM_ATTENTION_BACKEND="xformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Log summary ────────────────────────────────────────────────────────
log "Setup complete. Ready to run GRPO."
log "Env:        $ENV_DIR"
log "Config:     $CONFIG"
log "Log Files:  $SERVER_LOG, $TRAINING_LOG"

# (WANDB /n/fs offload paths already set above; avoid re-defining here)

# -----------------------------------
# 1) Launch vLLM server on GPU 0
# -----------------------------------
export VLLM_ATTENTION_BACKEND=xformers
export TORCH_FORCE_FULL_STATE_DICT=1
export FLASH_ATTENTION_FORCE_DISABLED=1
export TRANSFORMERS_NO_FLASH_ATTN=1
export WANDB_DATA_DIR=/n/fs/similarity/open-r1/wandb

# -----------------------------------
# Single srun step (7 GPUs): vLLM on GPU0 + training on GPU1-6
# -----------------------------------

log "Launching vLLM and training inline (no srun)"

# 1) vLLM on GPU-0 (inline)
export CUDA_VISIBLE_DEVICES=0
log "Launching vLLM on GPU $CUDA_VISIBLE_DEVICES (server log: $SERVER_LOG)"
stdbuf -oL -eL trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  >> "$SERVER_LOG" 2>&1 &

VLLM_PID=$!
log "vLLM PID: $VLLM_PID"

# Health-check loop
attempt=0
until curl -sf http://localhost:8000/health > /dev/null; do
  attempt=$((attempt+1))
  log "Waiting for vLLM… (attempt $attempt)"
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    log "vLLM process exited early. Tailing server log:"
    tail -n 200 "$SERVER_LOG" || true
    exit 3
  fi
  if [ $attempt -ge 90 ]; then
    log "vLLM health check timed out. Tailing server log:"
    tail -n 200 "$SERVER_LOG" || true
    exit 3
  fi
  sleep 2
done
log "vLLM is healthy"


# 2) Training on GPU 1–6 (inline)
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
log "Launching training on GPUs $CUDA_VISIBLE_DEVICES (training log: $TRAINING_LOG)"
accelerate launch \
  --main_process_port 29525 \
  --config_file "$CONFIG_FILE" \
  src/grpo.py \
  --config "$CONFIG" \
  --use_vllm \
  --run_name "${RUN_NAME}-${TIMESTAMP}" \
  --ignore_data_skip \
  --overwrite_output_dir false \
  --resume_from_checkpoint /n/fs/similarity/open-r1/data/Qwen2.5-1.5B-Open-R1-GRPO-math-v1/checkpoint-850 \
  --seed 42 \
  > "$TRAINING_LOG" 2>&1

# Wait for vLLM to exit after training finishes
wait $VLLM_PID || true
log "vLLM exited"
