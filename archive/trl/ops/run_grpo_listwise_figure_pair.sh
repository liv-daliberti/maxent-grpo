#!/usr/bin/env bash
# Submit a GRPO + listwise pair for distribution-figure generation.
#
# Defaults:
# - single-stack jobs on the lowprio A6000 path
# - shared math_fair 1.5B backbone
# - step-0 paper eval disabled so jobs reach training quickly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_WRAPPER="${DUAL_WRAPPER:-$SCRIPT_DIR/run_dual_4plus4_single_node.sh}"
if [[ ! -f "$DUAL_WRAPPER" ]]; then
  echo "Missing wrapper: $DUAL_WRAPPER" >&2
  exit 1
fi

CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_fair}"
MODEL="${MODEL:-Qwen2.5-1.5B-Instruct}"
RECIPE_PROFILE="${RECIPE_PROFILE:-experiment}"
RESOURCE_PROFILE="${RESOURCE_PROFILE:-interim_a6000}"
SBATCH_PARTITION="${SBATCH_PARTITION:-lowprio}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-allcs}"
SBATCH_GRES="${SBATCH_GRES:-gpu:a6000:4}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-32}"
SBATCH_MEM="${SBATCH_MEM:-128G}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-3}"
SINGLE_STACK_VLLM_GPU="${SINGLE_STACK_VLLM_GPU:-0}"
SINGLE_STACK_TRAIN_GPUS="${SINGLE_STACK_TRAIN_GPUS:-1,2,3}"
GRPO_JOB_NAME="${GRPO_JOB_NAME:-cs_figure_grpo}"
LISTWISE_JOB_NAME="${LISTWISE_JOB_NAME:-cs_figure_listwise}"
GRPO_ARGS="${GRPO_ARGS:-}"
LISTWISE_ARGS="${LISTWISE_ARGS:-}"
STEP0_PAPER_EVAL_ENABLED="${STEP0_PAPER_EVAL_ENABLED:-0}"
STEP0_PAPER_EVAL_PASS_AT_8_ENABLED="${STEP0_PAPER_EVAL_PASS_AT_8_ENABLED:-0}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-0}"

RUN_GROUP="${WANDB_RUN_GROUP:-figure_${MODEL//\//-}_${CONFIG_SUFFIX}_$(date +%Y%m%d_%H%M%S)}"

submit_pair_job() {
  local stack="$1"
  local job_name="$2"
  local variant_args="$3"
  local output=""
  if [[ "$stack" == "grpo" ]]; then
    output="$(
      env \
        WANDB_RUN_GROUP="$RUN_GROUP" \
        MODEL="$MODEL" \
        CONFIG_SUFFIX="$CONFIG_SUFFIX" \
        RECIPE_PROFILE="$RECIPE_PROFILE" \
        RESOURCE_PROFILE="$RESOURCE_PROFILE" \
        SBATCH_PARTITION="$SBATCH_PARTITION" \
        SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
        SBATCH_GRES="$SBATCH_GRES" \
        SBATCH_CPUS_PER_TASK="$SBATCH_CPUS_PER_TASK" \
        SBATCH_MEM="$SBATCH_MEM" \
        SBATCH_TIME="$SBATCH_TIME" \
        TRAIN_NUM_PROCESSES="$TRAIN_NUM_PROCESSES" \
        GRPO_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
        GRPO_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
        MAXENT_STEP0_PAPER_EVAL_ENABLED="$STEP0_PAPER_EVAL_ENABLED" \
        MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_ENABLED="$STEP0_PAPER_EVAL_PASS_AT_8_ENABLED" \
        JOB_NAME="$job_name" \
        RUN_ONLY="$stack" \
        MAXENT_ARGS="" \
        GRPO_ARGS="$variant_args" \
        SBATCH_DEPENDENCY="$SBATCH_DEPENDENCY" \
        DRY_RUN="$DRY_RUN" \
        "$DUAL_WRAPPER"
    )"
  else
    output="$(
      env \
        WANDB_RUN_GROUP="$RUN_GROUP" \
        MODEL="$MODEL" \
        CONFIG_SUFFIX="$CONFIG_SUFFIX" \
        RECIPE_PROFILE="$RECIPE_PROFILE" \
        RESOURCE_PROFILE="$RESOURCE_PROFILE" \
        SBATCH_PARTITION="$SBATCH_PARTITION" \
        SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
        SBATCH_GRES="$SBATCH_GRES" \
        SBATCH_CPUS_PER_TASK="$SBATCH_CPUS_PER_TASK" \
        SBATCH_MEM="$SBATCH_MEM" \
        SBATCH_TIME="$SBATCH_TIME" \
        TRAIN_NUM_PROCESSES="$TRAIN_NUM_PROCESSES" \
        MAXENT_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
        MAXENT_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
        MAXENT_STEP0_PAPER_EVAL_ENABLED="$STEP0_PAPER_EVAL_ENABLED" \
        MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_ENABLED="$STEP0_PAPER_EVAL_PASS_AT_8_ENABLED" \
        JOB_NAME="$job_name" \
        RUN_ONLY="$stack" \
        GRPO_ARGS="" \
        MAXENT_ARGS="$variant_args" \
        SBATCH_DEPENDENCY="$SBATCH_DEPENDENCY" \
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

echo "[figure-pair] run_group=${RUN_GROUP}"
echo "[figure-pair] grpo_job=${GRPO_JOB_NAME}"
echo "[figure-pair] listwise_job=${LISTWISE_JOB_NAME}"

grpo_output_and_id="$(submit_pair_job "grpo" "$GRPO_JOB_NAME" "$GRPO_ARGS")"
printf '%s\n' "$grpo_output_and_id"
GRPO_JOB_ID="$(printf '%s\n' "$grpo_output_and_id" | tail -n 1)"

listwise_output_and_id="$(submit_pair_job "listwise" "$LISTWISE_JOB_NAME" "$LISTWISE_ARGS")"
printf '%s\n' "$listwise_output_and_id"
LISTWISE_JOB_ID="$(printf '%s\n' "$listwise_output_and_id" | tail -n 1)"

echo "[figure-pair] grpo_job_id=${GRPO_JOB_ID}"
echo "[figure-pair] listwise_job_id=${LISTWISE_JOB_ID}"
