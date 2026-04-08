#!/bin/bash
# Launch baseline DR.GRPO and DR.GRPO-Explorer concurrently on node302 using
# the canonical README-flash OAT environment, split across GPUs 0-3 and 4-7.

#SBATCH --job-name=u_r1z_q15b_pair4
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

PAIR_STAMP="${PAIR_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PAIR_ROOT="${PAIR_ROOT:-$ROOT_DIR/var/data/oat_zero_pair_4plus4_${PAIR_STAMP}}"
PAIR_WANDB_ROOT="${PAIR_WANDB_ROOT:-$ROOT_DIR/var/wandb/pair_4plus4_${PAIR_STAMP}}"
PAIR_LOG_ROOT="${PAIR_LOG_ROOT:-$ROOT_DIR/var/artifacts/logs}"

BASE_SAVE_PATH="${BASE_SAVE_PATH:-$PAIR_ROOT/drgrpo}"
EXPLORER_SAVE_PATH="${EXPLORER_SAVE_PATH:-$PAIR_ROOT/drgrpo_explorer}"
BASE_RUN_NAME="${BASE_RUN_NAME:-qwen2.5-Math-1.5b-r1-zero-readmeflash-4gpu-baseline-${PAIR_STAMP}}"
EXPLORER_RUN_NAME="${EXPLORER_RUN_NAME:-qwen2.5-Math-1.5b-r1-zero-readmeflash-4gpu-explorer-${PAIR_STAMP}}"

BASE_LOCAL_ROOT="${BASE_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-oat-zero-${HOST_TAG}-pair4-baseline}"
EXPLORER_LOCAL_ROOT="${EXPLORER_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-oat-zero-${HOST_TAG}-pair4-explorer}"

BASE_STDOUT="${PAIR_LOG_ROOT}/oat_zero_pair4_baseline-${SLURM_JOB_ID}.out"
BASE_STDERR="${PAIR_LOG_ROOT}/oat_zero_pair4_baseline-${SLURM_JOB_ID}.err"
EXPLORER_STDOUT="${PAIR_LOG_ROOT}/oat_zero_pair4_explorer-${SLURM_JOB_ID}.out"
EXPLORER_STDERR="${PAIR_LOG_ROOT}/oat_zero_pair4_explorer-${SLURM_JOB_ID}.err"

mkdir -p \
  "$PAIR_ROOT" \
  "$PAIR_WANDB_ROOT/baseline" \
  "$PAIR_WANDB_ROOT/explorer" \
  "$PAIR_LOG_ROOT"

export OAT_ZERO_UPSTREAM_ROOT="${REPO_DIR}"
export OAT_ZERO_PYTHON="${OAT_ZERO_PYTHON:-${README_PYTHON}}"
export OAT_ZERO_ENABLE_FLASH_ATTN=1
export OAT_ZERO_FLASH_ATTN_VERSION="${OAT_ZERO_FLASH_ATTN_VERSION:-2.7.4.post1}"
export OAT_ZERO_PROMPT_TEMPLATE="${OAT_ZERO_PROMPT_TEMPLATE:-r1}"
export OAT_ZERO_PROMPT_DATA="${REPO_DIR}/datasets/train/math_12k"
export OAT_ZERO_EVAL_DATA="${REPO_DIR}/datasets/evaluation_suite"
export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-${OAT_ZERO_EVAL_STEPS}}"
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"
export OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"
export OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"
export OAT_ZERO_WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
export OAT_ZERO_SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-20000}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Scale the known-good 8-GPU geometry down to 4 GPUs per run while keeping the
# same per-device batch structure.
export PAIR_N_GPU="${PAIR_N_GPU:-4}"
export PAIR_TRAIN_BATCH_SIZE="${PAIR_TRAIN_BATCH_SIZE:-64}"
export PAIR_ROLLOUT_BATCH_SIZE="${PAIR_ROLLOUT_BATCH_SIZE:-64}"
export PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE="${PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE:-16}"
export PAIR_PI_BUFFER_MAXLEN_PER_DEVICE="${PAIR_PI_BUFFER_MAXLEN_PER_DEVICE:-64}"

# Explorer-specific defaults follow the latest OAT-side choices we have been
# iterating on: beta fixed at zero, learnable tau, PPO-like token clip mode,
# and a gentle sharp -> loose -> sharper entropy schedule.
export PAIR_EXPLORER_BETA="${PAIR_EXPLORER_BETA:-0}"
export PAIR_EXPLORER_TAU="${PAIR_EXPLORER_TAU:-0.5}"
export PAIR_EXPLORER_TAU_LEARNABLE="${PAIR_EXPLORER_TAU_LEARNABLE:-1}"
export PAIR_EXPLORER_TAU_LR="${PAIR_EXPLORER_TAU_LR:-0.02}"
export PAIR_EXPLORER_TAU_MIN="${PAIR_EXPLORER_TAU_MIN:-0}"
export PAIR_EXPLORER_TAU_MAX="${PAIR_EXPLORER_TAU_MAX:-0}"
export PAIR_EXPLORER_TAU_WARMUP_STEPS="${PAIR_EXPLORER_TAU_WARMUP_STEPS:-0}"
export PAIR_EXPLORER_CLIP_MODE="${PAIR_EXPLORER_CLIP_MODE:-token}"

terminate_children() {
  local pid
  for pid in "${BASE_PID:-}" "${EXPLORER_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap terminate_children INT TERM

launch_baseline() {
  (
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export OAT_ZERO_OBJECTIVE="grpo"
    export OAT_ZERO_N_GPU="${PAIR_N_GPU}"
    export OAT_ZERO_TRAIN_BATCH_SIZE="${PAIR_TRAIN_BATCH_SIZE}"
    export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-1}"
    export OAT_ZERO_ROLLOUT_BATCH_SIZE="${PAIR_ROLLOUT_BATCH_SIZE}"
    export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE="${PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE}"
    export OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE="${PAIR_PI_BUFFER_MAXLEN_PER_DEVICE}"
    export OAT_ZERO_WB_RUN_NAME="${BASE_RUN_NAME}"
    export OAT_ZERO_LOCAL_ROOT="${BASE_LOCAL_ROOT}"
    export WANDB_DIR="${PAIR_WANDB_ROOT}/baseline"
    export RUN_STAMP="${PAIR_STAMP}_baseline"
    export SAVE_PATH="${BASE_SAVE_PATH}"
    exec "${ROOT_DIR}/ops/run_oat_zero_exact_1p5b_upstream.sh"
  ) >"${BASE_STDOUT}" 2>"${BASE_STDERR}" &
  BASE_PID=$!
}

launch_explorer() {
  (
    export CUDA_VISIBLE_DEVICES="4,5,6,7"
    export OAT_ZERO_OBJECTIVE="maxent_listwise"
    export OAT_ZERO_N_GPU="${PAIR_N_GPU}"
    export OAT_ZERO_TRAIN_BATCH_SIZE="${PAIR_TRAIN_BATCH_SIZE}"
    export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE_EXPLORER:-8}"
    export OAT_ZERO_ROLLOUT_BATCH_SIZE="${PAIR_ROLLOUT_BATCH_SIZE}"
    export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE="${PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE}"
    export OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE="${PAIR_PI_BUFFER_MAXLEN_PER_DEVICE}"
    export OAT_ZERO_BETA="${PAIR_EXPLORER_BETA}"
    export OAT_ZERO_MAXENT_TAU="${PAIR_EXPLORER_TAU}"
    export OAT_ZERO_MAXENT_TAU_LEARNABLE="${PAIR_EXPLORER_TAU_LEARNABLE}"
    export OAT_ZERO_MAXENT_TAU_LR="${PAIR_EXPLORER_TAU_LR}"
    export OAT_ZERO_MAXENT_TAU_MIN="${PAIR_EXPLORER_TAU_MIN}"
    export OAT_ZERO_MAXENT_TAU_MAX="${PAIR_EXPLORER_TAU_MAX}"
    export OAT_ZERO_MAXENT_TAU_WARMUP_STEPS="${PAIR_EXPLORER_TAU_WARMUP_STEPS}"
    export OAT_ZERO_MAXENT_BETA_CONTROLLER_ENABLED=0
    export OAT_ZERO_MAXENT_CLIP_MODE="${PAIR_EXPLORER_CLIP_MODE}"
    export OAT_ZERO_WB_RUN_NAME="${EXPLORER_RUN_NAME}"
    export OAT_ZERO_LOCAL_ROOT="${EXPLORER_LOCAL_ROOT}"
    export WANDB_DIR="${PAIR_WANDB_ROOT}/explorer"
    export RUN_STAMP="${PAIR_STAMP}_explorer"
    export SAVE_PATH="${EXPLORER_SAVE_PATH}"
    exec "${ROOT_DIR}/ops/run_oat_zero_explorer_1p5b_upstream.sh"
  ) >"${EXPLORER_STDOUT}" 2>"${EXPLORER_STDERR}" &
  EXPLORER_PID=$!
}

echo "[pair] host=$(hostname)"
echo "[pair] job_id=${SLURM_JOB_ID:-}"
echo "[pair] repo_dir=${REPO_DIR}"
echo "[pair] pair_root=${PAIR_ROOT}"
echo "[pair] pair_wandb_root=${PAIR_WANDB_ROOT}"
echo "[pair] baseline_save_path=${BASE_SAVE_PATH}"
echo "[pair] explorer_save_path=${EXPLORER_SAVE_PATH}"
echo "[pair] baseline_run_name=${BASE_RUN_NAME}"
echo "[pair] explorer_run_name=${EXPLORER_RUN_NAME}"
echo "[pair] baseline_logs=${BASE_STDOUT} ${BASE_STDERR}"
echo "[pair] explorer_logs=${EXPLORER_STDOUT} ${EXPLORER_STDERR}"
echo "[pair] pair_n_gpu=${PAIR_N_GPU}"
echo "[pair] pair_train_batch_size=${PAIR_TRAIN_BATCH_SIZE}"
echo "[pair] pair_rollout_batch_size=${PAIR_ROLLOUT_BATCH_SIZE}"
echo "[pair] pair_rollout_batch_size_per_device=${PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE}"
echo "[pair] pair_pi_buffer_maxlen_per_device=${PAIR_PI_BUFFER_MAXLEN_PER_DEVICE}"
echo "[pair] explorer_beta=${PAIR_EXPLORER_BETA}"
echo "[pair] explorer_tau=${PAIR_EXPLORER_TAU}"
echo "[pair] explorer_tau_learnable=${PAIR_EXPLORER_TAU_LEARNABLE}"
echo "[pair] explorer_tau_lr=${PAIR_EXPLORER_TAU_LR}"
echo "[pair] explorer_clip_mode=${PAIR_EXPLORER_CLIP_MODE}"

launch_baseline
launch_explorer

echo "[pair] baseline_pid=${BASE_PID}"
echo "[pair] explorer_pid=${EXPLORER_PID}"

baseline_status=0
explorer_status=0

wait "${BASE_PID}" || baseline_status=$?
wait "${EXPLORER_PID}" || explorer_status=$?

echo "[pair] baseline_status=${baseline_status}"
echo "[pair] explorer_status=${explorer_status}"

if [[ "${baseline_status}" -ne 0 || "${explorer_status}" -ne 0 ]]; then
  exit 1
fi
