#!/usr/bin/env bash
# Submit the three-way experiment presets as single-stack jobs:
# - listwise MaxEnt
# - entropy MaxEnt
# - GRPO (after the first two finish)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_WRAPPER="${DUAL_WRAPPER:-$SCRIPT_DIR/run_dual_4plus4_single_node.sh}"
if [[ ! -f "$DUAL_WRAPPER" ]]; then
  echo "Missing wrapper: $DUAL_WRAPPER" >&2
  exit 1
fi

CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
MODEL="${MODEL:-}"
RECIPE_PROFILE="${RECIPE_PROFILE:-experiment}"
GRPO_JOB_NAME="${GRPO_JOB_NAME:-open_r1_triplet_grpo}"
MAXENT_JOB_NAME="${MAXENT_JOB_NAME:-open_r1_triplet_maxent}"
LISTWISE_JOB_NAME="${LISTWISE_JOB_NAME:-open_r1_triplet_listwise}"
GRPO_ARGS="${GRPO_ARGS:-}"
MAXENT_ARGS="${MAXENT_ARGS:-}"
LISTWISE_ARGS="${LISTWISE_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
SINGLE_STACK_GRES="${SINGLE_STACK_GRES:-gpu:a100:4}"
SINGLE_STACK_NUM_PROCESSES="${SINGLE_STACK_NUM_PROCESSES:-3}"
SINGLE_STACK_VLLM_GPU="${SINGLE_STACK_VLLM_GPU:-0}"
SINGLE_STACK_TRAIN_GPUS="${SINGLE_STACK_TRAIN_GPUS:-1,2,3}"
LISTWISE_VLLM_PORT="${LISTWISE_VLLM_PORT:-8001}"
LISTWISE_GROUP_PORT="${LISTWISE_GROUP_PORT:-29536}"
LISTWISE_MASTER_PORT="${LISTWISE_MASTER_PORT:-6001}"
ENTROPY_VLLM_PORT="${ENTROPY_VLLM_PORT:-8002}"
ENTROPY_GROUP_PORT="${ENTROPY_GROUP_PORT:-29537}"
ENTROPY_MASTER_PORT="${ENTROPY_MASTER_PORT:-6002}"
GRPO_VLLM_PORT="${GRPO_VLLM_PORT:-8000}"
GRPO_GROUP_PORT="${GRPO_GROUP_PORT:-29535}"
GRPO_MASTER_PORT="${GRPO_MASTER_PORT:-6000}"

if [[ -z "$MODEL" ]]; then
  case "$CONFIG_SUFFIX" in
    math|math_fair|math_stable) MODEL="Qwen2.5-1.5B-Instruct" ;;
    code_mbpp) MODEL="Qwen2.5-0.5B-Instruct" ;;
    *) MODEL="Qwen2.5-1.5B-Instruct" ;;
  esac
fi

RUN_GROUP="${WANDB_RUN_GROUP:-triplet_${MODEL//\//-}_${CONFIG_SUFFIX}_$(date +%Y%m%d_%H%M%S)}"

echo "[triplet] run_group=${RUN_GROUP}"
echo "[triplet] listwise job: ${LISTWISE_JOB_NAME} (listwise maxent)"
echo "[triplet] maxent job:   ${MAXENT_JOB_NAME} (entropy maxent)"
echo "[triplet] grpo job:     ${GRPO_JOB_NAME}"

submit_triplet_job() {
  local stack="$1"
  local job_name="$2"
  local variant_args="$3"
  local dependency="${4:-}"
  local output=""
  if [[ "$stack" == "grpo" ]]; then
    output="$(
      env \
        WANDB_RUN_GROUP="$RUN_GROUP" \
        MODEL="$MODEL" \
        CONFIG_SUFFIX="$CONFIG_SUFFIX" \
        RECIPE_PROFILE="$RECIPE_PROFILE" \
        SBATCH_GRES="$SINGLE_STACK_GRES" \
        TRAIN_NUM_PROCESSES="$SINGLE_STACK_NUM_PROCESSES" \
        GRPO_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
        GRPO_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
        GRPO_VLLM_PORT="$GRPO_VLLM_PORT" \
        GRPO_GROUP_PORT="$GRPO_GROUP_PORT" \
        GRPO_MASTER_PORT="$GRPO_MASTER_PORT" \
        JOB_NAME="$job_name" \
        RUN_ONLY="$stack" \
        GRPO_ARGS="$variant_args" \
        SBATCH_DEPENDENCY="$dependency" \
        DRY_RUN="$DRY_RUN" \
        "$DUAL_WRAPPER"
    )"
  else
    local stack_vllm_port="$ENTROPY_VLLM_PORT"
    local stack_group_port="$ENTROPY_GROUP_PORT"
    local stack_master_port="$ENTROPY_MASTER_PORT"
    if [[ "$stack" == "listwise" ]]; then
      stack_vllm_port="$LISTWISE_VLLM_PORT"
      stack_group_port="$LISTWISE_GROUP_PORT"
      stack_master_port="$LISTWISE_MASTER_PORT"
    fi
    output="$(
      env \
        WANDB_RUN_GROUP="$RUN_GROUP" \
        MODEL="$MODEL" \
        CONFIG_SUFFIX="$CONFIG_SUFFIX" \
        RECIPE_PROFILE="$RECIPE_PROFILE" \
        SBATCH_GRES="$SINGLE_STACK_GRES" \
        TRAIN_NUM_PROCESSES="$SINGLE_STACK_NUM_PROCESSES" \
        MAXENT_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
        MAXENT_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
        MAXENT_VLLM_PORT="$stack_vllm_port" \
        MAXENT_GROUP_PORT="$stack_group_port" \
        MAXENT_MASTER_PORT="$stack_master_port" \
        JOB_NAME="$job_name" \
        RUN_ONLY="$stack" \
        MAXENT_ARGS="$variant_args" \
        SBATCH_DEPENDENCY="$dependency" \
        DRY_RUN="$DRY_RUN" \
        "$DUAL_WRAPPER"
    )"
  fi
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

listwise_output_and_id="$(submit_triplet_job "listwise" "$LISTWISE_JOB_NAME" "$LISTWISE_ARGS")"
printf '%s\n' "$listwise_output_and_id"
LISTWISE_JOB_ID="$(printf '%s\n' "$listwise_output_and_id" | tail -n 1)"

maxent_output_and_id="$(submit_triplet_job "maxent" "$MAXENT_JOB_NAME" "$MAXENT_ARGS")"
printf '%s\n' "$maxent_output_and_id"
MAXENT_JOB_ID="$(printf '%s\n' "$maxent_output_and_id" | tail -n 1)"

GRPO_DEPENDENCY="afterany:${LISTWISE_JOB_ID}:${MAXENT_JOB_ID}"
grpo_output_and_id="$(submit_triplet_job "grpo" "$GRPO_JOB_NAME" "$GRPO_ARGS" "$GRPO_DEPENDENCY")"
printf '%s\n' "$grpo_output_and_id"
