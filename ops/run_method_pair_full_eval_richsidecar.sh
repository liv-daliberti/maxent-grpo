#!/usr/bin/env bash
# Submit a two-method rich-sidecar pair with periodic live official SEED eval
# and final full-suite eval jobs. This wrapper is method-agnostic across the
# supported single-stack modes in train_dual_4plus4.slurm: grpo|maxent|listwise|seed.
#
# Unlike the specialized GRPO-vs-Listwise wrapper, this script does not submit
# the pairwise mass-distribution plots, because those diagnostics reconstruct
# objective-specific update weights and are not valid for arbitrary method pairs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DUAL_WRAPPER="${DUAL_WRAPPER:-$SCRIPT_DIR/run_dual_4plus4_single_node.sh}"
EVAL_SLURM_SCRIPT="${EVAL_SLURM_SCRIPT:-$SCRIPT_DIR/slurm/eval_saved_model_full_seed.slurm}"

CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_fair}"
MODEL="${MODEL:-Qwen2.5-1.5B-Instruct}"
RESOURCE_PROFILE="${RESOURCE_PROFILE:-interim_a6000}"
SBATCH_PARTITION="${SBATCH_PARTITION:-cs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-allcs}"
SBATCH_GRES="${SBATCH_GRES:-gpu:a6000:4}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-32}"
SBATCH_MEM="${SBATCH_MEM:-128G}"
SBATCH_TIME="${SBATCH_TIME:-7-00:00:00}"
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-3}"
SINGLE_STACK_VLLM_GPU="${SINGLE_STACK_VLLM_GPU:-0}"
SINGLE_STACK_TRAIN_GPUS="${SINGLE_STACK_TRAIN_GPUS:-1,2,3}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:--1}"
TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-20}"
TRAIN_EVAL_STEPS="${TRAIN_EVAL_STEPS:-25}"
TRAIN_SAVE_STEPS="${TRAIN_SAVE_STEPS:-25}"
TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-1000}"
TRAIN_EVAL_ON_START="${TRAIN_EVAL_ON_START:-true}"
TRAIN_LIVE_EVAL_ENABLED="${TRAIN_LIVE_EVAL_ENABLED:-true}"
TRAIN_LIVE_PASS_AT_8_ENABLED="${TRAIN_LIVE_PASS_AT_8_ENABLED:-true}"
TRAIN_LIVE_EVAL_FAIL_ON_ERROR="${TRAIN_LIVE_EVAL_FAIL_ON_ERROR:-false}"
TRAIN_LIVE_EVAL_TEMPLATE="${TRAIN_LIVE_EVAL_TEMPLATE:-no}"
TASKS="${TASKS:-aime,amc,math,minerva,olympiad_bench}"
EVAL_TEMPLATE="${EVAL_TEMPLATE:-no}"
PASS_AT_8_SAMPLES="${PASS_AT_8_SAMPLES:-8}"
EVAL_SBATCH_GRES="${EVAL_SBATCH_GRES:-gpu:a6000:1}"
EVAL_SBATCH_CPUS_PER_TASK="${EVAL_SBATCH_CPUS_PER_TASK:-16}"
EVAL_SBATCH_MEM="${EVAL_SBATCH_MEM:-96G}"
EVAL_SBATCH_TIME="${EVAL_SBATCH_TIME:-24:00:00}"
DRY_RUN="${DRY_RUN:-0}"

STACK_A="${STACK_A:-seed}"
STACK_B="${STACK_B:-maxent}"
LABEL_A="${LABEL_A:-SEED-Dr.GRPO}"
LABEL_B="${LABEL_B:-Token MaxEnt}"
STACK_A_EXTRA_ARGS="${STACK_A_EXTRA_ARGS:-}"
STACK_B_EXTRA_ARGS="${STACK_B_EXTRA_ARGS:-}"

RUN_GROUP="${WANDB_RUN_GROUP:-full_eval_richsidecar_pair_${MODEL//\//-}_${CONFIG_SUFFIX}_${STACK_A}_vs_${STACK_B}_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-var/artifacts/full_eval_pairs/${RUN_GROUP}}"
DATA_ROOT="${DATA_ROOT:-var/data/full_eval_pairs/${RUN_GROUP}}"

mkdir -p "$PIPELINE_ROOT" "$DATA_ROOT"

STACK_A_RUN_NAME="${STACK_A_RUN_NAME:-${RUN_GROUP}-${STACK_A}}"
STACK_B_RUN_NAME="${STACK_B_RUN_NAME:-${RUN_GROUP}-${STACK_B}}"
STACK_A_OUTPUT_DIR="${STACK_A_OUTPUT_DIR:-$DATA_ROOT/${STACK_A}}"
STACK_B_OUTPUT_DIR="${STACK_B_OUTPUT_DIR:-$DATA_ROOT/${STACK_B}}"
STACK_A_RESULTS_DIR="${STACK_A_RESULTS_DIR:-$PIPELINE_ROOT/${STACK_A}_seed_eval}"
STACK_B_RESULTS_DIR="${STACK_B_RESULTS_DIR:-$PIPELINE_ROOT/${STACK_B}_seed_eval}"

STACK_A_JOB_NAME="${STACK_A_JOB_NAME:-cs_full_eval_${STACK_A}_rich_long}"
STACK_B_JOB_NAME="${STACK_B_JOB_NAME:-cs_full_eval_${STACK_B}_rich_long}"
STACK_A_EVAL_JOB_NAME="${STACK_A_EVAL_JOB_NAME:-cs_full_eval_${STACK_A}_eval_rich_long}"
STACK_B_EVAL_JOB_NAME="${STACK_B_EVAL_JOB_NAME:-cs_full_eval_${STACK_B}_eval_rich_long}"

COMMON_TRAIN_ARGS=(
  "--max_steps ${TRAIN_MAX_STEPS}"
  "--num_train_epochs ${TRAIN_NUM_EPOCHS}"
  "--save_strategy steps"
  "--save_steps ${TRAIN_SAVE_STEPS}"
  "--save_total_limit ${TRAIN_SAVE_TOTAL_LIMIT}"
  "--final_model_save_enabled true"
  "--seed_paper_eval_enabled ${TRAIN_LIVE_EVAL_ENABLED}"
  "--seed_paper_eval_pass_at_8_enabled ${TRAIN_LIVE_PASS_AT_8_ENABLED}"
  "--seed_paper_eval_fail_on_error ${TRAIN_LIVE_EVAL_FAIL_ON_ERROR}"
  "--seed_paper_eval_template ${TRAIN_LIVE_EVAL_TEMPLATE}"
  "--eval_on_start ${TRAIN_EVAL_ON_START}"
  "--eval_steps ${TRAIN_EVAL_STEPS}"
  "--logging_steps 1"
  "--logging_first_step true"
  "--log_completions false"
  "--rich_log_completions true"
  "--rich_log_completions_to_wandb false"
  "--rich_log_completions_synchronize_ranks true"
  "--rich_log_completions_key rich_completions"
)

STACK_A_ARGS="${STACK_A_ARGS:-${COMMON_TRAIN_ARGS[*]} --output_dir ${STACK_A_OUTPUT_DIR} --run_name ${STACK_A_RUN_NAME} ${STACK_A_EXTRA_ARGS}}"
STACK_B_ARGS="${STACK_B_ARGS:-${COMMON_TRAIN_ARGS[*]} --output_dir ${STACK_B_OUTPUT_DIR} --run_name ${STACK_B_RUN_NAME} ${STACK_B_EXTRA_ARGS}}"

submit_train_job() {
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
        RESOURCE_PROFILE="$RESOURCE_PROFILE" \
        SBATCH_PARTITION="$SBATCH_PARTITION" \
        SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
        SBATCH_GRES="$SBATCH_GRES" \
        SBATCH_CPUS_PER_TASK="$SBATCH_CPUS_PER_TASK" \
        SBATCH_MEM="$SBATCH_MEM" \
        SBATCH_TIME="$SBATCH_TIME" \
        TRAIN_NUM_PROCESSES="$TRAIN_NUM_PROCESSES" \
        MAXENT_STEP0_PAPER_EVAL_TEMPLATE="$TRAIN_LIVE_EVAL_TEMPLATE" \
        MAXENT_STEP0_PAPER_EVAL_ENABLED=0 \
        MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_ENABLED=0 \
        GRPO_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
        GRPO_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
        JOB_NAME="$job_name" \
        RUN_ONLY="$stack" \
        MAXENT_ARGS="" \
        GRPO_ARGS="$variant_args" \
        DRY_RUN="$DRY_RUN" \
        "$DUAL_WRAPPER"
    )"
  else
    output="$(
      env \
        WANDB_RUN_GROUP="$RUN_GROUP" \
        MODEL="$MODEL" \
        CONFIG_SUFFIX="$CONFIG_SUFFIX" \
        RESOURCE_PROFILE="$RESOURCE_PROFILE" \
        SBATCH_PARTITION="$SBATCH_PARTITION" \
        SBATCH_ACCOUNT="$SBATCH_ACCOUNT" \
        SBATCH_GRES="$SBATCH_GRES" \
        SBATCH_CPUS_PER_TASK="$SBATCH_CPUS_PER_TASK" \
        SBATCH_MEM="$SBATCH_MEM" \
        SBATCH_TIME="$SBATCH_TIME" \
        TRAIN_NUM_PROCESSES="$TRAIN_NUM_PROCESSES" \
        MAXENT_STEP0_PAPER_EVAL_TEMPLATE="$TRAIN_LIVE_EVAL_TEMPLATE" \
        MAXENT_STEP0_PAPER_EVAL_ENABLED=0 \
        MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_ENABLED=0 \
        MAXENT_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
        MAXENT_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
        JOB_NAME="$job_name" \
        RUN_ONLY="$stack" \
        GRPO_ARGS="" \
        MAXENT_ARGS="$variant_args" \
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
    echo "Failed to parse training job id for ${stack}" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

submit_eval_job() {
  local job_name="$1"
  local dependency_job_id="$2"
  local model_path="$3"
  local results_dir="$4"
  local port="$5"
  local output
  output="$(
    sbatch \
      --job-name "$job_name" \
      --nodes 1 \
      --partition "$SBATCH_PARTITION" \
      --account "$SBATCH_ACCOUNT" \
      --gres "$EVAL_SBATCH_GRES" \
      --cpus-per-task "$EVAL_SBATCH_CPUS_PER_TASK" \
      --mem "$EVAL_SBATCH_MEM" \
      --time "$EVAL_SBATCH_TIME" \
      --dependency "afterok:${dependency_job_id}" \
      --export "ALL,MODEL_PATH=${model_path},RESULTS_DIR=${results_dir},TASKS=${TASKS},TEMPLATE=${EVAL_TEMPLATE},PASS_AT_8_SAMPLES=${PASS_AT_8_SAMPLES},VLLM_PORT=${port}" \
      "$EVAL_SLURM_SCRIPT"
  )"
  printf '%s\n' "$output"
  local job_id
  job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse eval job id for ${job_name}" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

echo "[pair-rich] run_group=${RUN_GROUP}"
echo "[pair-rich] stack_a=${STACK_A} label_a=${LABEL_A}"
echo "[pair-rich] stack_b=${STACK_B} label_b=${LABEL_B}"
echo "[pair-rich] train_max_steps=${TRAIN_MAX_STEPS} train_num_epochs=${TRAIN_NUM_EPOCHS}"
echo "[pair-rich] live_eval_enabled=${TRAIN_LIVE_EVAL_ENABLED} live_pass_at_8=${TRAIN_LIVE_PASS_AT_8_ENABLED} eval_steps=${TRAIN_EVAL_STEPS}"
echo "[pair-rich] tasks=${TASKS} template=${TRAIN_LIVE_EVAL_TEMPLATE}"

stack_a_output_and_id="$(submit_train_job "$STACK_A" "$STACK_A_JOB_NAME" "$STACK_A_ARGS")"
printf '%s\n' "$stack_a_output_and_id"
STACK_A_JOB_ID="$(printf '%s\n' "$stack_a_output_and_id" | tail -n 1)"

stack_b_output_and_id="$(submit_train_job "$STACK_B" "$STACK_B_JOB_NAME" "$STACK_B_ARGS")"
printf '%s\n' "$stack_b_output_and_id"
STACK_B_JOB_ID="$(printf '%s\n' "$stack_b_output_and_id" | tail -n 1)"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

mkdir -p "$STACK_A_RESULTS_DIR" "$STACK_B_RESULTS_DIR"

stack_a_eval_output_and_id="$(submit_eval_job "$STACK_A_EVAL_JOB_NAME" "$STACK_A_JOB_ID" "$STACK_A_OUTPUT_DIR" "$STACK_A_RESULTS_DIR" "8127")"
printf '%s\n' "$stack_a_eval_output_and_id"
STACK_A_EVAL_JOB_ID="$(printf '%s\n' "$stack_a_eval_output_and_id" | tail -n 1)"

stack_b_eval_output_and_id="$(submit_eval_job "$STACK_B_EVAL_JOB_NAME" "$STACK_B_JOB_ID" "$STACK_B_OUTPUT_DIR" "$STACK_B_RESULTS_DIR" "8128")"
printf '%s\n' "$stack_b_eval_output_and_id"
STACK_B_EVAL_JOB_ID="$(printf '%s\n' "$stack_b_eval_output_and_id" | tail -n 1)"

MANIFEST_PATH="$PIPELINE_ROOT/manifest.tsv"
cat >"$MANIFEST_PATH" <<EOF
role	job_id	path
train_${STACK_A}	${STACK_A_JOB_ID}	${STACK_A_OUTPUT_DIR}
train_${STACK_B}	${STACK_B_JOB_ID}	${STACK_B_OUTPUT_DIR}
eval_${STACK_A}	${STACK_A_EVAL_JOB_ID}	${STACK_A_RESULTS_DIR}
eval_${STACK_B}	${STACK_B_EVAL_JOB_ID}	${STACK_B_RESULTS_DIR}
EOF

NOTE_PATH="$PIPELINE_ROOT/README.txt"
cat >"$NOTE_PATH" <<EOF
Run group: ${RUN_GROUP}
Stack A: ${STACK_A} (${LABEL_A})
Stack B: ${STACK_B} (${LABEL_B})

This launcher mirrors the rich-sidecar training/eval pipeline but intentionally
does not submit the GRPO-vs-Listwise diagnostic plots. Those plots reconstruct
objective-specific rollout masses and are not valid for arbitrary method pairs.
EOF

echo "[pair-rich] manifest=${MANIFEST_PATH}"
echo "[pair-rich] train_${STACK_A}=${STACK_A_JOB_ID}"
echo "[pair-rich] train_${STACK_B}=${STACK_B_JOB_ID}"
echo "[pair-rich] eval_${STACK_A}=${STACK_A_EVAL_JOB_ID}"
echo "[pair-rich] eval_${STACK_B}=${STACK_B_EVAL_JOB_ID}"
