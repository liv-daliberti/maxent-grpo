#!/bin/bash
# Launch pure Dr.GRPO and pure Dr.GRPO-Explorer concurrently on node302.
# - baseline Dr.GRPO uses GPUs 0,1,2,3
# - explorer uses GPUs 4,5,6,7

#SBATCH --job-name=drgrpo_pure_pair_4plus4
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
cd "$ROOT_DIR"

mkdir -p var/artifacts/logs var/tmp var/cache

PAIR_STAMP="${PAIR_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PAIR_ROOT="${PAIR_ROOT:-$ROOT_DIR/var/data/pure_pair_${PAIR_STAMP}}"
GRPO_SAVE_PATH="${GRPO_SAVE_PATH:-$PAIR_ROOT/drgrpo}"
EXPLORER_SAVE_PATH="${EXPLORER_SAVE_PATH:-$PAIR_ROOT/drgrpo_explorer}"
GRPO_WB_RUN_NAME="${GRPO_WB_RUN_NAME:-qwen2.5-Math-1.5b-drgrpo-r1template-pair-${PAIR_STAMP}}"
EXPLORER_WB_RUN_NAME="${EXPLORER_WB_RUN_NAME:-qwen2.5-Math-1.5b-drgrpo-explorer-r1template-pair-${PAIR_STAMP}}"

PAIR_TMP_ROOT="${PAIR_TMP_ROOT:-$ROOT_DIR/var/tmp/pure_pair_${SLURM_JOB_ID:-manual}}"
PAIR_CACHE_ROOT="${PAIR_CACHE_ROOT:-$ROOT_DIR/var/cache/pure_pair_${SLURM_JOB_ID:-manual}}"
mkdir -p \
  "$PAIR_ROOT" \
  "$PAIR_TMP_ROOT/grpo" \
  "$PAIR_TMP_ROOT/explorer" \
  "$PAIR_CACHE_ROOT/triton_grpo" \
  "$PAIR_CACHE_ROOT/triton_explorer" \
  "$PAIR_CACHE_ROOT/torchinductor_grpo" \
  "$PAIR_CACHE_ROOT/torchinductor_explorer" \
  "$PAIR_CACHE_ROOT/extensions_grpo" \
  "$PAIR_CACHE_ROOT/extensions_explorer"

GRPO_STDOUT="$ROOT_DIR/var/artifacts/logs/drgrpo_pure_pair_grpo-${SLURM_JOB_ID}.out"
GRPO_STDERR="$ROOT_DIR/var/artifacts/logs/drgrpo_pure_pair_grpo-${SLURM_JOB_ID}.err"
EXPLORER_STDOUT="$ROOT_DIR/var/artifacts/logs/drgrpo_pure_pair_explorer-${SLURM_JOB_ID}.out"
EXPLORER_STDERR="$ROOT_DIR/var/artifacts/logs/drgrpo_pure_pair_explorer-${SLURM_JOB_ID}.err"

terminate_children() {
  local pid
  for pid in "${GRPO_PID:-}" "${EXPLORER_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap terminate_children INT TERM

PAIR_PURE_VERIFIER_VERSION="${PAIR_PURE_VERIFIER_VERSION:-fast}"
PAIR_N_SAMPLE="${PAIR_N_SAMPLE:-8}"
PAIR_ROLLOUT_BATCH_SIZE="${PAIR_ROLLOUT_BATCH_SIZE:-128}"
PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE="${PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE:-32}"
PAIR_PI_BUFFER_MAXLEN_PER_DEVICE="${PAIR_PI_BUFFER_MAXLEN_PER_DEVICE:-256}"
PAIR_SAVE_STEPS="${PAIR_SAVE_STEPS:--1}"
PAIR_SAVE_CKPT="${PAIR_SAVE_CKPT:-0}"
PAIR_BASELINE_BETA="${PAIR_BASELINE_BETA:-0.0}"
PAIR_EXPLORER_BETA="${PAIR_EXPLORER_BETA:-0.08}"
PAIR_EXPLORER_LISTWISE_TAU="${PAIR_EXPLORER_LISTWISE_TAU:-0.25}"
PAIR_PURE_PRECISION="${PAIR_PURE_PRECISION:-fp16}"
PAIR_PURE_ENABLE_FLASH_ATTN="${PAIR_PURE_ENABLE_FLASH_ATTN:-1}"
PAIR_PURE_BOOTSTRAP_FLASH_ATTN="${PAIR_PURE_BOOTSTRAP_FLASH_ATTN:-1}"
PAIR_PURE_FLASH_ATTN_VERSION="${PAIR_PURE_FLASH_ATTN_VERSION:-2.7.0.post2}"
PAIR_PURE_CORRECT_REWARD="${PAIR_PURE_CORRECT_REWARD:-1.0}"
PAIR_PURE_INCORRECT_REWARD="${PAIR_PURE_INCORRECT_REWARD:-0.0}"

launch_baseline() {
  (
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export N_GPU=4
    export RUN_STAMP="${PAIR_STAMP}_grpo"
    export SAVE_PATH="$GRPO_SAVE_PATH"
    export WB_RUN_NAME="$GRPO_WB_RUN_NAME"
    export PURE_DRGRPO_PREBUILD_FUSED_ADAM=0
    export PURE_DRGRPO_PREWARM_MODEL_CACHE=0
    export PURE_DRGRPO_TMPDIR="$PAIR_TMP_ROOT/grpo"
    export TRITON_CACHE_DIR="$PAIR_CACHE_ROOT/triton_grpo"
    export TORCHINDUCTOR_CACHE_DIR="$PAIR_CACHE_ROOT/torchinductor_grpo"
    export TORCH_EXTENSIONS_DIR="$PAIR_CACHE_ROOT/extensions_grpo"
    export PURE_VERIFIER_VERSION="$PAIR_PURE_VERIFIER_VERSION"
    export N_SAMPLE="$PAIR_N_SAMPLE"
    export ROLLOUT_BATCH_SIZE="$PAIR_ROLLOUT_BATCH_SIZE"
    export ROLLOUT_BATCH_SIZE_PER_DEVICE="$PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE"
    export PI_BUFFER_MAXLEN_PER_DEVICE="$PAIR_PI_BUFFER_MAXLEN_PER_DEVICE"
    export PURE_BETA="$PAIR_BASELINE_BETA"
    export SAVE_STEPS="$PAIR_SAVE_STEPS"
    export SAVE_CKPT="$PAIR_SAVE_CKPT"
    export PURE_DRGRPO_PRECISION="$PAIR_PURE_PRECISION"
    export PURE_DRGRPO_ENABLE_FLASH_ATTN="$PAIR_PURE_ENABLE_FLASH_ATTN"
    export PURE_DRGRPO_BOOTSTRAP_FLASH_ATTN="$PAIR_PURE_BOOTSTRAP_FLASH_ATTN"
    export PURE_DRGRPO_FLASH_ATTN_VERSION="$PAIR_PURE_FLASH_ATTN_VERSION"
    export PURE_CORRECT_REWARD="$PAIR_PURE_CORRECT_REWARD"
    export PURE_INCORRECT_REWARD="$PAIR_PURE_INCORRECT_REWARD"
    exec "$ROOT_DIR/archive/trl/ops/run_drgrpo_1p5b_pure_single_node.sh"
  ) >"$GRPO_STDOUT" 2>"$GRPO_STDERR" &
  GRPO_PID=$!
}

launch_explorer() {
  (
    export CUDA_VISIBLE_DEVICES="4,5,6,7"
    export N_GPU=4
    export RUN_STAMP="${PAIR_STAMP}_explorer"
    export SAVE_PATH="$EXPLORER_SAVE_PATH"
    export WB_RUN_NAME="$EXPLORER_WB_RUN_NAME"
    export PURE_DRGRPO_PREBUILD_FUSED_ADAM=0
    export PURE_DRGRPO_PREWARM_MODEL_CACHE=0
    export PURE_DRGRPO_TMPDIR="$PAIR_TMP_ROOT/explorer"
    export TRITON_CACHE_DIR="$PAIR_CACHE_ROOT/triton_explorer"
    export TORCHINDUCTOR_CACHE_DIR="$PAIR_CACHE_ROOT/torchinductor_explorer"
    export TORCH_EXTENSIONS_DIR="$PAIR_CACHE_ROOT/extensions_explorer"
    export PURE_VERIFIER_VERSION="$PAIR_PURE_VERIFIER_VERSION"
    export N_SAMPLE="$PAIR_N_SAMPLE"
    export ROLLOUT_BATCH_SIZE="$PAIR_ROLLOUT_BATCH_SIZE"
    export ROLLOUT_BATCH_SIZE_PER_DEVICE="$PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE"
    export PI_BUFFER_MAXLEN_PER_DEVICE="$PAIR_PI_BUFFER_MAXLEN_PER_DEVICE"
    export PURE_BETA="$PAIR_EXPLORER_BETA"
    export PURE_LISTWISE_TAU="$PAIR_EXPLORER_LISTWISE_TAU"
    export SAVE_STEPS="$PAIR_SAVE_STEPS"
    export SAVE_CKPT="$PAIR_SAVE_CKPT"
    export PURE_DRGRPO_PRECISION="$PAIR_PURE_PRECISION"
    export PURE_DRGRPO_ENABLE_FLASH_ATTN="$PAIR_PURE_ENABLE_FLASH_ATTN"
    export PURE_DRGRPO_BOOTSTRAP_FLASH_ATTN="$PAIR_PURE_BOOTSTRAP_FLASH_ATTN"
    export PURE_DRGRPO_FLASH_ATTN_VERSION="$PAIR_PURE_FLASH_ATTN_VERSION"
    export PURE_CORRECT_REWARD="$PAIR_PURE_CORRECT_REWARD"
    export PURE_INCORRECT_REWARD="$PAIR_PURE_INCORRECT_REWARD"
    exec "$ROOT_DIR/archive/trl/ops/run_drgrpo_explorer_1p5b_pure_single_node.sh"
  ) >"$EXPLORER_STDOUT" 2>"$EXPLORER_STDERR" &
  EXPLORER_PID=$!
}

echo "[pair] host=$(hostname)"
echo "[pair] job_id=${SLURM_JOB_ID:-}"
echo "[pair] pair_root=${PAIR_ROOT}"
echo "[pair] baseline_save_path=${GRPO_SAVE_PATH}"
echo "[pair] explorer_save_path=${EXPLORER_SAVE_PATH}"
echo "[pair] baseline_wb_run_name=${GRPO_WB_RUN_NAME}"
echo "[pair] explorer_wb_run_name=${EXPLORER_WB_RUN_NAME}"
echo "[pair] baseline_logs=${GRPO_STDOUT} ${GRPO_STDERR}"
echo "[pair] explorer_logs=${EXPLORER_STDOUT} ${EXPLORER_STDERR}"
echo "[pair] pure_verifier_version=${PAIR_PURE_VERIFIER_VERSION}"
echo "[pair] num_samples=${PAIR_N_SAMPLE}"
echo "[pair] rollout_batch_size=${PAIR_ROLLOUT_BATCH_SIZE}"
echo "[pair] rollout_batch_size_per_device=${PAIR_ROLLOUT_BATCH_SIZE_PER_DEVICE}"
echo "[pair] pi_buffer_maxlen_per_device=${PAIR_PI_BUFFER_MAXLEN_PER_DEVICE}"
echo "[pair] baseline_beta=${PAIR_BASELINE_BETA}"
echo "[pair] explorer_beta=${PAIR_EXPLORER_BETA}"
echo "[pair] explorer_listwise_tau=${PAIR_EXPLORER_LISTWISE_TAU}"
echo "[pair] save_steps=${PAIR_SAVE_STEPS}"
echo "[pair] save_ckpt=${PAIR_SAVE_CKPT}"
echo "[pair] pure_precision=${PAIR_PURE_PRECISION}"
echo "[pair] pure_enable_flash_attn=${PAIR_PURE_ENABLE_FLASH_ATTN}"
echo "[pair] pure_bootstrap_flash_attn=${PAIR_PURE_BOOTSTRAP_FLASH_ATTN}"
echo "[pair] pure_flash_attn_version=${PAIR_PURE_FLASH_ATTN_VERSION}"
echo "[pair] pure_correct_reward=${PAIR_PURE_CORRECT_REWARD}"
echo "[pair] pure_incorrect_reward=${PAIR_PURE_INCORRECT_REWARD}"

launch_baseline
launch_explorer

echo "[pair] baseline_pid=${GRPO_PID}"
echo "[pair] explorer_pid=${EXPLORER_PID}"

grpo_status=0
explorer_status=0

wait "$GRPO_PID" || grpo_status=$?
wait "$EXPLORER_PID" || explorer_status=$?

echo "[pair] baseline_status=${grpo_status}"
echo "[pair] explorer_status=${explorer_status}"

if (( grpo_status != 0 || explorer_status != 0 )); then
  exit 1
fi
