#!/usr/bin/env bash
# Submit the interim/general-compute pair as two 4-GPU single-stack jobs:
# - entropy MaxEnt
# - SEED-GRPO
#
# Defaults are pinned to the interim/general-compute profile so these jobs do
# not land on the shared mltheory quartet A100 nodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_WRAPPER="${DUAL_WRAPPER:-$SCRIPT_DIR/run_dual_4plus4_single_node.sh}"
if [[ ! -f "$DUAL_WRAPPER" ]]; then
  echo "Missing wrapper: $DUAL_WRAPPER" >&2
  exit 1
fi

CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_fair}"
MODEL="${MODEL:-}"
RECIPE_PROFILE="${RECIPE_PROFILE:-experiment}"
MAXENT_JOB_NAME="${MAXENT_JOB_NAME:-cs_interim_maxent}"
SEED_JOB_NAME="${SEED_JOB_NAME:-cs_interim_seed_drgrpo}"
MAXENT_ARGS="${MAXENT_ARGS:-}"
SEED_ARGS="${SEED_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
RESOURCE_PROFILE="${RESOURCE_PROFILE:-interim_a6000}"
SINGLE_STACK_PARTITION="${SINGLE_STACK_PARTITION:-lowprio}"
SINGLE_STACK_ACCOUNT="${SINGLE_STACK_ACCOUNT:-allcs}"
SINGLE_STACK_GRES="${SINGLE_STACK_GRES:-gpu:a6000:4}"
SINGLE_STACK_CPUS_PER_TASK="${SINGLE_STACK_CPUS_PER_TASK:-32}"
SINGLE_STACK_MEM="${SINGLE_STACK_MEM:-128G}"
SINGLE_STACK_NUM_PROCESSES="${SINGLE_STACK_NUM_PROCESSES:-3}"
SINGLE_STACK_VLLM_GPU="${SINGLE_STACK_VLLM_GPU:-0}"
SINGLE_STACK_TRAIN_GPUS="${SINGLE_STACK_TRAIN_GPUS:-1,2,3}"
MAXENT_VLLM_PORT="${MAXENT_VLLM_PORT:-8002}"
MAXENT_GROUP_PORT="${MAXENT_GROUP_PORT:-29537}"
MAXENT_MASTER_PORT="${MAXENT_MASTER_PORT:-6002}"
SEED_VLLM_PORT="${SEED_VLLM_PORT:-8003}"
SEED_GROUP_PORT="${SEED_GROUP_PORT:-29538}"
SEED_MASTER_PORT="${SEED_MASTER_PORT:-6003}"

if [[ -z "$MODEL" ]]; then
  case "$CONFIG_SUFFIX" in
    math|math_fair|math_stable) MODEL="Qwen2.5-1.5B-Instruct" ;;
    code_mbpp) MODEL="Qwen2.5-0.5B-Instruct" ;;
    *) MODEL="Qwen2.5-1.5B-Instruct" ;;
  esac
fi

RUN_GROUP="${WANDB_RUN_GROUP:-interim_pair_${MODEL//\//-}_${CONFIG_SUFFIX}_$(date +%Y%m%d_%H%M%S)}"

echo "[interim] run_group=${RUN_GROUP}"
echo "[interim] maxent job: ${MAXENT_JOB_NAME}"
echo "[interim] seed job:   ${SEED_JOB_NAME}"

submit_interim_job() {
  local stack="$1"
  local job_name="$2"
  local variant_args="$3"
  local stack_vllm_port="$MAXENT_VLLM_PORT"
  local stack_group_port="$MAXENT_GROUP_PORT"
  local stack_master_port="$MAXENT_MASTER_PORT"
  if [[ "$stack" == "seed" ]]; then
    stack_vllm_port="$SEED_VLLM_PORT"
    stack_group_port="$SEED_GROUP_PORT"
    stack_master_port="$SEED_MASTER_PORT"
  fi
  local output=""
  output="$(
    env \
      WANDB_RUN_GROUP="$RUN_GROUP" \
      MODEL="$MODEL" \
      CONFIG_SUFFIX="$CONFIG_SUFFIX" \
      RECIPE_PROFILE="$RECIPE_PROFILE" \
      RESOURCE_PROFILE="$RESOURCE_PROFILE" \
      SBATCH_PARTITION="$SINGLE_STACK_PARTITION" \
      SBATCH_ACCOUNT="$SINGLE_STACK_ACCOUNT" \
      SBATCH_GRES="$SINGLE_STACK_GRES" \
      SBATCH_CPUS_PER_TASK="$SINGLE_STACK_CPUS_PER_TASK" \
      SBATCH_MEM="$SINGLE_STACK_MEM" \
      TRAIN_NUM_PROCESSES="$SINGLE_STACK_NUM_PROCESSES" \
      MAXENT_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
      MAXENT_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
      MAXENT_VLLM_PORT="$stack_vllm_port" \
      MAXENT_GROUP_PORT="$stack_group_port" \
      MAXENT_MASTER_PORT="$stack_master_port" \
      JOB_NAME="$job_name" \
      RUN_ONLY="$stack" \
      MAXENT_ARGS="$variant_args" \
      DRY_RUN="$DRY_RUN" \
      "$DUAL_WRAPPER"
  )"
  printf '%s\n' "$output"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'dryrun-%s\n' "$stack"
    return 0
  fi
  local job_id
  job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse submitted batch job id for ${stack}" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

maxent_output_and_id="$(submit_interim_job "maxent" "$MAXENT_JOB_NAME" "$MAXENT_ARGS")"
printf '%s\n' "$maxent_output_and_id"

seed_output_and_id="$(submit_interim_job "seed" "$SEED_JOB_NAME" "$SEED_ARGS")"
printf '%s\n' "$seed_output_and_id"
