#!/bin/bash
# Run a short single-GPU tiny-model Dr.GRPO/Dr.X probe on node302.
#
# Override with, for example:
#   sbatch --export=ALL,OAT_ZERO_TINY_VARIANT=grpo ops/slurm/train_tiny_probe_node302.slurm
#   sbatch --export=ALL,OAT_ZERO_TINY_MODEL=smollm2-135m,OAT_ZERO_TINY_VARIANT=current_drx ops/slurm/train_tiny_probe_node302.slurm

#SBATCH --job-name=tiny_drx_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=24:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
README_PYTHON="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"
HOST_TAG="${SLURMD_NODENAME:-$(hostname -s)}"

source "${ROOT_DIR}/ops/repo_env.sh"

export OAT_ZERO_PYTHON="${OAT_ZERO_PYTHON:-${README_PYTHON}}"
export OAT_ZERO_SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:-${ROOT_DIR}/src}"
export OAT_ZERO_TINY_TOPOLOGY="${OAT_ZERO_TINY_TOPOLOGY:-single_gpu}"
export OAT_ZERO_N_GPU="${OAT_ZERO_N_GPU:-1}"
export OAT_ZERO_ENABLE_FLASH_ATTN="${OAT_ZERO_ENABLE_FLASH_ATTN:-0}"
export OAT_ZERO_USE_WB="${OAT_ZERO_USE_WB:-0}"
export OAT_ZERO_LOCAL_ROOT="${OAT_ZERO_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-oat-zero-${HOST_TAG}-tiny-probe}"
export OAT_ZERO_SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-8000}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] oat_zero_python=${OAT_ZERO_PYTHON}"
echo "[slurm] oat_zero_tiny_model=${OAT_ZERO_TINY_MODEL:-qwen2.5-0.5b-instruct}"
echo "[slurm] oat_zero_tiny_variant=${OAT_ZERO_TINY_VARIANT:-current_drx}"
echo "[slurm] oat_zero_tiny_topology=${OAT_ZERO_TINY_TOPOLOGY}"
echo "[slurm] oat_zero_n_gpu=${OAT_ZERO_N_GPU}"
echo "[slurm] oat_zero_use_wb=${OAT_ZERO_USE_WB}"
echo "[slurm] oat_zero_local_root=${OAT_ZERO_LOCAL_ROOT}"
echo "[slurm] hf_home=${HF_HOME:-unset}"
echo "[slurm] wandb_dir=${WANDB_DIR:-unset}"

exec "${ROOT_DIR}/ops/run_oat_zero_tiny_probe.sh"
