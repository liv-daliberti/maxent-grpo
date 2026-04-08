#!/bin/bash
# Run the upstream understand-r1-zero Qwen2.5-Math-7B R1-template Dr.GRPO
# recipe on node105 using the canonical README-flash OAT runtime.

#SBATCH --job-name=u_r1z_qm7b_r1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node105
#SBATCH --time=48:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_DIR="${ROOT_DIR}/understand-r1-zero"
README_PYTHON="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"
HOST_TAG="${SLURMD_NODENAME:-$(hostname -s)}"

source "${ROOT_DIR}/ops/repo_env.sh"

export PYTHONPATH="${ROOT_DIR}/tools/runtime_patches${PYTHONPATH:+:${PYTHONPATH}}"

export OAT_ZERO_UPSTREAM_ROOT="${REPO_DIR}"
export OAT_ZERO_PYTHON="${OAT_ZERO_PYTHON:-${README_PYTHON}}"
export OAT_ZERO_ENABLE_FLASH_ATTN=1
export OAT_ZERO_FLASH_ATTN_VERSION="${OAT_ZERO_FLASH_ATTN_VERSION:-2.7.4.post1}"
export OAT_ZERO_N_GPU="${OAT_ZERO_N_GPU:-8}"
export OAT_ZERO_COLLOCATE="${OAT_ZERO_COLLOCATE:-0}"
export OAT_ZERO_VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.75}"
export OAT_ZERO_VLLM_SLEEP="${OAT_ZERO_VLLM_SLEEP:-1}"
export OAT_ZERO_CUDA_HOME="${OAT_ZERO_CUDA_HOME:-/usr/local/cuda-12.4}"
export OAT_ZERO_ZERO_STAGE="${OAT_ZERO_ZERO_STAGE:-3}"
export OAT_ZERO_ADAM_OFFLOAD="${OAT_ZERO_ADAM_OFFLOAD:-1}"
export OAT_ZERO_MAX_NORM="${OAT_ZERO_MAX_NORM:-0}"
export OAT_ZERO_ACTIVATION_OFFLOADING="${OAT_ZERO_ACTIVATION_OFFLOADING:-1}"
export OAT_ZERO_DISABLE_TRACE_CACHE="${OAT_ZERO_DISABLE_TRACE_CACHE:-1}"
export OAT_ZERO_GRAD_ACCUM_DTYPE="${OAT_ZERO_GRAD_ACCUM_DTYPE:-bf16}"
export OAT_ZERO_TRAIN_BATCH_SIZE="${OAT_ZERO_TRAIN_BATCH_SIZE:-32}"
export OAT_ZERO_ROLLOUT_BATCH_SIZE="${OAT_ZERO_ROLLOUT_BATCH_SIZE:-32}"
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-4}"
export OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE="${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE:-32}"
export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
export OAT_ZERO_USE_WB="${OAT_ZERO_USE_WB:-1}"
export OAT_ZERO_WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
export OAT_ZERO_WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-qwen2.5-Math-7b-drgrpo-r1template-readmeflash-node105}"
export OAT_ZERO_LOCAL_ROOT="${OAT_ZERO_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-oat-zero-${HOST_TAG}-readmeflash-7b-r1}"
export OAT_ZERO_SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-20000}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
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
echo "[slurm] oat_zero_collocate=${OAT_ZERO_COLLOCATE}"
echo "[slurm] oat_zero_vllm_gpu_ratio=${OAT_ZERO_VLLM_GPU_RATIO}"
echo "[slurm] oat_zero_vllm_sleep=${OAT_ZERO_VLLM_SLEEP}"
echo "[slurm] oat_zero_cuda_home=${OAT_ZERO_CUDA_HOME}"
echo "[slurm] oat_zero_zero_stage=${OAT_ZERO_ZERO_STAGE}"
echo "[slurm] oat_zero_adam_offload=${OAT_ZERO_ADAM_OFFLOAD}"
echo "[slurm] oat_zero_max_norm=${OAT_ZERO_MAX_NORM}"
echo "[slurm] oat_zero_activation_offloading=${OAT_ZERO_ACTIVATION_OFFLOADING}"
echo "[slurm] oat_zero_disable_trace_cache=${OAT_ZERO_DISABLE_TRACE_CACHE}"
echo "[slurm] oat_zero_grad_accum_dtype=${OAT_ZERO_GRAD_ACCUM_DTYPE}"
echo "[slurm] oat_zero_train_batch_size=${OAT_ZERO_TRAIN_BATCH_SIZE}"
echo "[slurm] oat_zero_rollout_batch_size=${OAT_ZERO_ROLLOUT_BATCH_SIZE}"
echo "[slurm] oat_zero_rollout_batch_size_per_device=${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE}"
echo "[slurm] oat_zero_pi_buffer_maxlen_per_device=${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE}"
echo "[slurm] oat_zero_eval_steps=${OAT_ZERO_EVAL_STEPS}"
echo "[slurm] oat_zero_use_wb=${OAT_ZERO_USE_WB}"
echo "[slurm] oat_zero_wb_project=${OAT_ZERO_WB_PROJECT}"
echo "[slurm] oat_zero_wb_run_name=${OAT_ZERO_WB_RUN_NAME}"
echo "[slurm] oat_zero_local_root=${OAT_ZERO_LOCAL_ROOT}"
echo "[slurm] oat_zero_shm_size_mb=${OAT_ZERO_SHM_SIZE_MB}"
echo "[slurm] pythonunbuffered=${PYTHONUNBUFFERED}"
echo "[slurm] pythonioencoding=${PYTHONIOENCODING}"
echo "[slurm] pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[slurm] hf_home=${HF_HOME:-unset}"
echo "[slurm] wandb_dir=${WANDB_DIR:-unset}"
echo "[slurm] pythonpath=${PYTHONPATH}"

exec bash "${ROOT_DIR}/ops/run_oat_zero_math_7b_r1_upstream.sh"
