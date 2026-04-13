#!/bin/bash
# One-hour A6000 validation run for the baseline 1.5B OAT Dr.GRPO recipe.
# This keeps the base objective and effective train batch size, but uses the
# upstream-safe learner microbatch for 48 GB A6000 GPUs.

#SBATCH --job-name=t_r1z_q15b_a6
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=all
#SBATCH --exclude=node105
#SBATCH --time=00:55:00
#SBATCH --output=var/artifacts/logs/%x-%A_%a.out
#SBATCH --error=var/artifacts/logs/%x-%A_%a.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
README_PYTHON="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"
HOST_TAG="${SLURMD_NODENAME:-$(hostname -s)}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

SEEDS_CSV="${OAT_ZERO_SEEDS_CSV:-42}"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#SEEDS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${ARRAY_INDEX} is out of range for seeds: ${SEEDS_CSV}" >&2
  exit 1
fi

SEED="${SEEDS[$ARRAY_INDEX]}"
RUN_BASE="${OAT_ZERO_RUN_BASENAME:-qwen2.5-Math-1.5b-r1-zero-readmeflash-a6000-1h-test}"
RUN_STAMP_DEFAULT="$(date +%Y%m%d_%H%M%S)_a6000test_seed${SEED}"

source "${ROOT_DIR}/ops/repo_env.sh"

export RUN_STAMP="${RUN_STAMP:-${RUN_STAMP_DEFAULT}}"
export OAT_ZERO_SOURCE_ROOT="${ROOT_DIR}/src"
export OAT_ZERO_PYTHON="${OAT_ZERO_PYTHON:-${README_PYTHON}}"
export OAT_ZERO_ENABLE_FLASH_ATTN=1
export OAT_ZERO_FLASH_ATTN_VERSION="${OAT_ZERO_FLASH_ATTN_VERSION:-2.7.4.post1}"
export OAT_ZERO_PROMPT_TEMPLATE="${OAT_ZERO_PROMPT_TEMPLATE:-r1}"
export OAT_ZERO_OBJECTIVE="${OAT_ZERO_OBJECTIVE:-grpo}"
export OAT_ZERO_N_GPU="${OAT_ZERO_N_GPU:-8}"
export OAT_ZERO_SEED="${OAT_ZERO_SEED:-${SEED}}"
export OAT_ZERO_RND_SEED="${OAT_ZERO_RND_SEED:-0}"
export OAT_ZERO_PROMPT_DATA="${ROOT_DIR}/datasets/train/math_12k"
export OAT_ZERO_EVAL_DATA="${ROOT_DIR}/datasets/evaluation_suite"
export OAT_ZERO_NUM_PROMPT_EPOCH="${OAT_ZERO_NUM_PROMPT_EPOCH:-1}"
export OAT_ZERO_TRAIN_BATCH_SIZE="${OAT_ZERO_TRAIN_BATCH_SIZE:-128}"
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-1}"
export OAT_ZERO_ROLLOUT_BATCH_SIZE="${OAT_ZERO_ROLLOUT_BATCH_SIZE:-128}"
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-16}"
export OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE="${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE:-128}"
export OAT_ZERO_VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.35}"
export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-${OAT_ZERO_EVAL_STEPS}}"
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"
export OAT_ZERO_SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-0}"
export OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-0}"
export OAT_ZERO_ALLOW_NO_SAVE="${OAT_ZERO_ALLOW_NO_SAVE:-1}"
export OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS="${OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS:-0}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"
export OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"
export OAT_ZERO_WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
export OAT_ZERO_WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-${RUN_BASE}-seed${SEED}-${RUN_STAMP}}"
export OAT_ZERO_LOCAL_ROOT="${OAT_ZERO_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-oat-zero-${HOST_TAG}-a6000-1h-seed${SEED}-${SLURM_JOB_ID:-manual}}"
export OAT_ZERO_SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-20000}"
if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] || [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:"* ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
fi
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] array_task_id=${ARRAY_INDEX}"
echo "[slurm] seed=${OAT_ZERO_SEED}"
echo "[slurm] run_stamp=${RUN_STAMP}"
echo "[slurm] source_root=${OAT_ZERO_SOURCE_ROOT}"
echo "[slurm] oat_zero_python=${OAT_ZERO_PYTHON}"
echo "[slurm] oat_zero_objective=${OAT_ZERO_OBJECTIVE}"
echo "[slurm] oat_zero_num_prompt_epoch=${OAT_ZERO_NUM_PROMPT_EPOCH}"
echo "[slurm] oat_zero_train_batch_size=${OAT_ZERO_TRAIN_BATCH_SIZE}"
echo "[slurm] oat_zero_train_batch_size_per_device=${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE}"
echo "[slurm] oat_zero_rollout_batch_size=${OAT_ZERO_ROLLOUT_BATCH_SIZE}"
echo "[slurm] oat_zero_rollout_batch_size_per_device=${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE}"
echo "[slurm] oat_zero_n_gpu=${OAT_ZERO_N_GPU}"
echo "[slurm] oat_zero_vllm_gpu_ratio=${OAT_ZERO_VLLM_GPU_RATIO}"
echo "[slurm] oat_zero_eval_steps=${OAT_ZERO_EVAL_STEPS}"
echo "[slurm] oat_zero_save_steps=${OAT_ZERO_SAVE_STEPS}"
echo "[slurm] oat_zero_save_ckpt=${OAT_ZERO_SAVE_CKPT}"
echo "[slurm] oat_zero_save_initial_model=${OAT_ZERO_SAVE_INITIAL_MODEL}"
echo "[slurm] oat_zero_wb_run_name=${OAT_ZERO_WB_RUN_NAME}"
echo "[slurm] oat_zero_local_root=${OAT_ZERO_LOCAL_ROOT}"
echo "[slurm] pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-unset}"
echo "[slurm] hf_home=${HF_HOME:-unset}"
echo "[slurm] wandb_dir=${WANDB_DIR:-unset}"

exec "${ROOT_DIR}/ops/run_oat_zero_exact_1p5b_upstream.sh"
