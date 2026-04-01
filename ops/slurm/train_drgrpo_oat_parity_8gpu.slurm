#!/bin/bash
# Single-node 8xA100 Dr.GRPO parity launcher on the local TRL/vLLM stack.

#SBATCH --job-name=drgrpo_oat_parity_1p5b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --time=48:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

mkdir -p var/artifacts/logs var/tmp var/cache

source "$ROOT_DIR/ops/repo_env.sh"

export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT_DIR/var/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/var/cache/torchinductor}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PRIMARY_ENV_DIR="${PRIMARY_ENV_DIR:-/n/fs/similarity/.conda/envs/openr1}"
if [[ ! -x "$PRIMARY_ENV_DIR/bin/python" ]]; then
  echo "Missing primary runtime python at $PRIMARY_ENV_DIR/bin/python" >&2
  exit 1
fi
export PATH="$PRIMARY_ENV_DIR/bin:${PATH}"
export VIRTUAL_ENV="$PRIMARY_ENV_DIR"
export PYTHONNOUSERSITE=1
OVERLAY_SITE_PACKAGES="${OVERLAY_SITE_PACKAGES:-$ROOT_DIR/var/e2e-venv/lib/python3.11/site-packages:$ROOT_DIR/var/seed_paper_eval/paper_venv/lib/python3.11/site-packages}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${OVERLAY_SITE_PACKAGES}:${PYTHONPATH}"
else
  export PYTHONPATH="${OVERLAY_SITE_PACKAGES}"
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-var/data/drgrpo_1p5b_oat_parity_trl_r1_${RUN_STAMP}}"
RUN_NAME="${RUN_NAME:-drgrpo-1p5b-oat-parity-trl-r1-${RUN_STAMP}}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OUTPUT_DIR
export RUN_NAME

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[slurm] output_dir=${OUTPUT_DIR}"
echo "[slurm] run_name=${RUN_NAME}"
echo "[slurm] python=$(command -v python)"

exec "$ROOT_DIR/ops/run_drgrpo_1p5b_oat_parity_single_node.sh" "$@"
