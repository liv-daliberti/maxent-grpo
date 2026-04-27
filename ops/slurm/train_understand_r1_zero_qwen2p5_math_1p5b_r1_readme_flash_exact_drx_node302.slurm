#!/bin/bash
# Run the exact listwise / DrX 1.5B OAT recipe on node302 with the README
# training stack, on-node flash-attn bootstrap, and row-sharded batch-1 learner
# microbatches (one owned sample row per rank, exact grouped target rebuild).

#SBATCH --job-name=u_r1z_q15b_drx
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
README_PYTHON="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"
HOST_TAG="${SLURMD_NODENAME:-$(hostname -s)}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

source "${ROOT_DIR}/ops/repo_env.sh"

export OAT_ZERO_SOURCE_ROOT="${ROOT_DIR}/src"
export OAT_ZERO_PYTHON="${OAT_ZERO_PYTHON:-${README_PYTHON}}"
export OAT_ZERO_ENABLE_FLASH_ATTN=1
export OAT_ZERO_FLASH_ATTN_VERSION="${OAT_ZERO_FLASH_ATTN_VERSION:-2.7.4.post1}"
export OAT_ZERO_PROMPT_TEMPLATE="${OAT_ZERO_PROMPT_TEMPLATE:-r1}"
export OAT_ZERO_OBJECTIVE="${OAT_ZERO_OBJECTIVE:-maxent_listwise}"
export OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY:-1}"
export OAT_ZERO_N_GPU="${OAT_ZERO_N_GPU:-8}"
export OAT_ZERO_PROMPT_DATA="${ROOT_DIR}/datasets/train/math_12k"
export OAT_ZERO_EVAL_DATA="${ROOT_DIR}/datasets/evaluation_suite"
export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-${OAT_ZERO_EVAL_STEPS}}"
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"
export OAT_ZERO_SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-1}"
export OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS="${OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS:-1}"
export OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"
export OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"
export OAT_ZERO_WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
export OAT_ZERO_WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-qwen2.5-Math-1.5b-r1-zero-readmeflash-drx-outputmaxent-lam010-node302-${RUN_STAMP}}"
export OAT_ZERO_LOCAL_ROOT="${OAT_ZERO_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-oat-zero-${HOST_TAG}-readmeflash-exact-drx}"
export OAT_ZERO_SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-20000}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Keep the exact DrX job on the batch-1 row-sharded path while reusing the
# safer node302 memory split and smaller ZeRO-2 communication buckets.
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-1}"
export OAT_ZERO_VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.30}"
export OAT_ZERO_DEEPSPEED_ALLGATHER_BUCKET_SIZE="${OAT_ZERO_DEEPSPEED_ALLGATHER_BUCKET_SIZE:-200000000}"
export OAT_ZERO_DEEPSPEED_REDUCE_BUCKET_SIZE="${OAT_ZERO_DEEPSPEED_REDUCE_BUCKET_SIZE:-200000000}"
export OAT_ZERO_DEEPSPEED_OVERLAP_COMM="${OAT_ZERO_DEEPSPEED_OVERLAP_COMM:-0}"
export OAT_ZERO_DEEPSPEED_CONTIGUOUS_GRADIENTS="${OAT_ZERO_DEEPSPEED_CONTIGUOUS_GRADIENTS:-0}"
export OAT_ZERO_DEEPSPEED_USE_MULTI_RANK_BUCKET_ALLREDUCE="${OAT_ZERO_DEEPSPEED_USE_MULTI_RANK_BUCKET_ALLREDUCE:-0}"

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] source_root=${OAT_ZERO_SOURCE_ROOT}"
echo "[slurm] oat_zero_python=${OAT_ZERO_PYTHON}"
echo "[slurm] oat_zero_enable_flash_attn=${OAT_ZERO_ENABLE_FLASH_ATTN}"
echo "[slurm] oat_zero_flash_attn_version=${OAT_ZERO_FLASH_ATTN_VERSION}"
echo "[slurm] oat_zero_prompt_template=${OAT_ZERO_PROMPT_TEMPLATE}"
echo "[slurm] oat_zero_objective=${OAT_ZERO_OBJECTIVE}"
echo "[slurm] oat_zero_maxent_drgrpo_token_primary=${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY}"
echo "[slurm] oat_zero_eval_steps=${OAT_ZERO_EVAL_STEPS}"
echo "[slurm] oat_zero_save_steps=${OAT_ZERO_SAVE_STEPS}"
echo "[slurm] oat_zero_save_from=${OAT_ZERO_SAVE_FROM}"
echo "[slurm] oat_zero_save_ckpt=${OAT_ZERO_SAVE_CKPT}"
echo "[slurm] oat_zero_require_full_eval_checkpoints=${OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS}"
echo "[slurm] oat_zero_save_initial_model=${OAT_ZERO_SAVE_INITIAL_MODEL}"
echo "[slurm] oat_zero_max_save_num=${OAT_ZERO_MAX_SAVE_NUM}"
echo "[slurm] oat_zero_max_save_mem=${OAT_ZERO_MAX_SAVE_MEM}"
echo "[slurm] oat_zero_wb_project=${OAT_ZERO_WB_PROJECT}"
echo "[slurm] oat_zero_wb_run_name=${OAT_ZERO_WB_RUN_NAME}"
echo "[slurm] oat_zero_wandb_run_id=${OAT_ZERO_WANDB_RUN_ID:-unset}"
echo "[slurm] oat_zero_maxent_semantic_cluster_method=${OAT_ZERO_MAXENT_SEMANTIC_CLUSTER_METHOD:-unset}"
echo "[slurm] oat_zero_maxent_semantic_embedding_similarity_threshold=${OAT_ZERO_MAXENT_SEMANTIC_EMBEDDING_SIMILARITY_THRESHOLD:-unset}"
echo "[slurm] oat_zero_maxent_semantic_spectral_max_clusters=${OAT_ZERO_MAXENT_SEMANTIC_SPECTRAL_MAX_CLUSTERS:-unset}"
echo "[slurm] oat_zero_maxent_semantic_spectral_eigengap_min=${OAT_ZERO_MAXENT_SEMANTIC_SPECTRAL_EIGENGAP_MIN:-unset}"
echo "[slurm] oat_zero_maxent_drgrpo_token_advantage_source=${OAT_ZERO_MAXENT_DRGRPO_TOKEN_ADVANTAGE_SOURCE:-unset}"
echo "[slurm] oat_zero_maxent_drgrpo_token_length_normalizer=${OAT_ZERO_MAXENT_DRGRPO_TOKEN_LENGTH_NORMALIZER:-unset}"
echo "[slurm] oat_zero_local_root=${OAT_ZERO_LOCAL_ROOT}"
echo "[slurm] oat_zero_shm_size_mb=${OAT_ZERO_SHM_SIZE_MB}"
echo "[slurm] oat_zero_train_batch_size_per_device=${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE}"
echo "[slurm] oat_zero_vllm_gpu_ratio=${OAT_ZERO_VLLM_GPU_RATIO}"
echo "[slurm] oat_zero_ds_allgather_bucket_size=${OAT_ZERO_DEEPSPEED_ALLGATHER_BUCKET_SIZE}"
echo "[slurm] oat_zero_ds_reduce_bucket_size=${OAT_ZERO_DEEPSPEED_REDUCE_BUCKET_SIZE}"
echo "[slurm] oat_zero_ds_overlap_comm=${OAT_ZERO_DEEPSPEED_OVERLAP_COMM}"
echo "[slurm] oat_zero_ds_contiguous_gradients=${OAT_ZERO_DEEPSPEED_CONTIGUOUS_GRADIENTS}"
echo "[slurm] oat_zero_ds_use_multi_rank_bucket_allreduce=${OAT_ZERO_DEEPSPEED_USE_MULTI_RANK_BUCKET_ALLREDUCE}"
echo "[slurm] hf_home=${HF_HOME:-unset}"
echo "[slurm] wandb_dir=${WANDB_DIR:-unset}"

exec "${ROOT_DIR}/ops/run_oat_zero_exact_drx_1p5b_upstream.sh"
