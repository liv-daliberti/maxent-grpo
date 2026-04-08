#!/bin/bash
# Run the upstream understand-r1-zero Qwen2.5-Math-1.5B README recipe on node302
# with the README training stack and on-node flash-attn bootstrap.

#SBATCH --job-name=u_r1z_q15b_rf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=48:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_DIR="${ROOT_DIR}/understand-r1-zero"
README_PYTHON="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"
HOST_TAG="${SLURMD_NODENAME:-$(hostname -s)}"

source "${ROOT_DIR}/ops/repo_env.sh"

export OAT_ZERO_UPSTREAM_ROOT="${REPO_DIR}"
export OAT_ZERO_PYTHON="${OAT_ZERO_PYTHON:-${README_PYTHON}}"
export OAT_ZERO_ENABLE_FLASH_ATTN=1
export OAT_ZERO_FLASH_ATTN_VERSION="${OAT_ZERO_FLASH_ATTN_VERSION:-2.7.4.post1}"
export OAT_ZERO_PROMPT_TEMPLATE="${OAT_ZERO_PROMPT_TEMPLATE:-r1}"
export OAT_ZERO_N_GPU="${OAT_ZERO_N_GPU:-8}"
export OAT_ZERO_PROMPT_DATA="${REPO_DIR}/datasets/train/math_12k"
export OAT_ZERO_EVAL_DATA="${REPO_DIR}/datasets/evaluation_suite"
export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-${OAT_ZERO_EVAL_STEPS}}"
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"
export OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"
export OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"
export OAT_ZERO_WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
export OAT_ZERO_WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-qwen2.5-Math-1.5b-r1-zero-readmeflash-node302}"
export OAT_ZERO_LOCAL_ROOT="${OAT_ZERO_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-oat-zero-${HOST_TAG}-readmeflash}"
export OAT_ZERO_SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-20000}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] repo_dir=${REPO_DIR}"
echo "[slurm] oat_zero_upstream_root=${OAT_ZERO_UPSTREAM_ROOT}"
echo "[slurm] oat_zero_python=${OAT_ZERO_PYTHON}"
echo "[slurm] oat_zero_enable_flash_attn=${OAT_ZERO_ENABLE_FLASH_ATTN}"
echo "[slurm] oat_zero_flash_attn_version=${OAT_ZERO_FLASH_ATTN_VERSION}"
echo "[slurm] oat_zero_prompt_template=${OAT_ZERO_PROMPT_TEMPLATE}"
echo "[slurm] oat_zero_eval_steps=${OAT_ZERO_EVAL_STEPS}"
echo "[slurm] oat_zero_save_steps=${OAT_ZERO_SAVE_STEPS}"
echo "[slurm] oat_zero_save_from=${OAT_ZERO_SAVE_FROM}"
echo "[slurm] oat_zero_save_initial_model=${OAT_ZERO_SAVE_INITIAL_MODEL}"
echo "[slurm] oat_zero_max_save_num=${OAT_ZERO_MAX_SAVE_NUM}"
echo "[slurm] oat_zero_max_save_mem=${OAT_ZERO_MAX_SAVE_MEM}"
echo "[slurm] oat_zero_wb_project=${OAT_ZERO_WB_PROJECT}"
echo "[slurm] oat_zero_wb_run_name=${OAT_ZERO_WB_RUN_NAME}"
echo "[slurm] oat_zero_local_root=${OAT_ZERO_LOCAL_ROOT}"
echo "[slurm] oat_zero_shm_size_mb=${OAT_ZERO_SHM_SIZE_MB}"
echo "[slurm] hf_home=${HF_HOME:-unset}"
echo "[slurm] wandb_dir=${WANDB_DIR:-unset}"

exec "${ROOT_DIR}/ops/run_oat_zero_exact_1p5b_upstream.sh"
