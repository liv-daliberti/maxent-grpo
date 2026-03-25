#!/usr/bin/env bash
# Submit a long GRPO + listwise pair with rich sidecars, periodic live paper eval,
# final full pass@8 eval, and rich-sidecar comparison plots.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DUAL_WRAPPER="${DUAL_WRAPPER:-$SCRIPT_DIR/run_dual_4plus4_single_node.sh}"
EVAL_SLURM_SCRIPT="${EVAL_SLURM_SCRIPT:-$SCRIPT_DIR/slurm/eval_saved_model_full_seed.slurm}"

CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_fair}"
MODEL="${MODEL:-Qwen2.5-1.5B-Instruct}"
RESOURCE_PROFILE="${RESOURCE_PROFILE:-quartet_a100}"
SBATCH_PARTITION="${SBATCH_PARTITION:-mltheory}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mltheory}"
SBATCH_GRES="${SBATCH_GRES:-gpu:a100:4}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-48}"
SBATCH_MEM="${SBATCH_MEM:-240G}"
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
TRAIN_LIVE_PASS_AT_8_ENABLED="${TRAIN_LIVE_PASS_AT_8_ENABLED:-false}"
TRAIN_LIVE_EVAL_FAIL_ON_ERROR="${TRAIN_LIVE_EVAL_FAIL_ON_ERROR:-false}"
TASKS="${TASKS:-aime,amc,math,minerva,olympiad_bench}"
PASS_AT_8_SAMPLES="${PASS_AT_8_SAMPLES:-8}"
LISTWISE_TAU="${LISTWISE_TAU:-0.35}"
LISTWISE_BETA="${LISTWISE_BETA:-0.12}"
LISTWISE_Q_TEMPERATURE="${LISTWISE_Q_TEMPERATURE:-2.0}"
EVAL_SBATCH_GRES="${EVAL_SBATCH_GRES:-gpu:a100:1}"
EVAL_SBATCH_CPUS_PER_TASK="${EVAL_SBATCH_CPUS_PER_TASK:-16}"
EVAL_SBATCH_MEM="${EVAL_SBATCH_MEM:-96G}"
EVAL_SBATCH_TIME="${EVAL_SBATCH_TIME:-24:00:00}"
RICH_PLOT_SBATCH_CPUS_PER_TASK="${RICH_PLOT_SBATCH_CPUS_PER_TASK:-8}"
RICH_PLOT_SBATCH_MEM="${RICH_PLOT_SBATCH_MEM:-32G}"
RICH_PLOT_SBATCH_TIME="${RICH_PLOT_SBATCH_TIME:-02:00:00}"
DRY_RUN="${DRY_RUN:-0}"

RUN_GROUP="${WANDB_RUN_GROUP:-full_eval_richsidecar_${MODEL//\//-}_${CONFIG_SUFFIX}_mltheory_bestlistwise_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-var/artifacts/full_eval_pairs/${RUN_GROUP}}"
DATA_ROOT="${DATA_ROOT:-var/data/full_eval_pairs/${RUN_GROUP}}"

mkdir -p "$PIPELINE_ROOT" "$DATA_ROOT"

GRPO_RUN_NAME="${GRPO_RUN_NAME:-${RUN_GROUP}-grpo}"
LISTWISE_RUN_NAME="${LISTWISE_RUN_NAME:-${RUN_GROUP}-listwise}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-$DATA_ROOT/grpo}"
LISTWISE_OUTPUT_DIR="${LISTWISE_OUTPUT_DIR:-$DATA_ROOT/listwise}"
GRPO_RESULTS_DIR="${GRPO_RESULTS_DIR:-$PIPELINE_ROOT/grpo_seed_eval}"
LISTWISE_RESULTS_DIR="${LISTWISE_RESULTS_DIR:-$PIPELINE_ROOT/listwise_seed_eval}"
PLOT_OUTPUT_SVG="${PLOT_OUTPUT_SVG:-$PIPELINE_ROOT/listwise_vs_grpo_full_eval_pass8.svg}"
PLOT_OUTPUT_PNG="${PLOT_OUTPUT_PNG:-$PIPELINE_ROOT/listwise_vs_grpo_full_eval_pass8.png}"
PLOT_SUMMARY_JSON="${PLOT_SUMMARY_JSON:-$PIPELINE_ROOT/listwise_vs_grpo_full_eval_pass8.summary.json}"
RICH_PLOT_SENTINEL="${RICH_PLOT_SENTINEL:-$PIPELINE_ROOT/listwise_vs_grpo_accuracy_dynamics.png}"

GRPO_JOB_NAME="${GRPO_JOB_NAME:-cs_full_eval_grpo_rich_long}"
LISTWISE_JOB_NAME="${LISTWISE_JOB_NAME:-cs_full_eval_listwise_rich_long}"
GRPO_EVAL_JOB_NAME="${GRPO_EVAL_JOB_NAME:-cs_full_eval_grpo_eval_rich_long}"
LISTWISE_EVAL_JOB_NAME="${LISTWISE_EVAL_JOB_NAME:-cs_full_eval_listwise_eval_rich_long}"
PLOT_JOB_NAME="${PLOT_JOB_NAME:-cs_full_eval_plot_rich_long}"
RICH_PLOT_JOB_NAME="${RICH_PLOT_JOB_NAME:-cs_full_eval_rich_plot_long}"

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

GRPO_ARGS="${GRPO_ARGS:-${COMMON_TRAIN_ARGS[*]} --output_dir ${GRPO_OUTPUT_DIR} --run_name ${GRPO_RUN_NAME}}"
LISTWISE_ARGS="${LISTWISE_ARGS:-${COMMON_TRAIN_ARGS[*]} --output_dir ${LISTWISE_OUTPUT_DIR} --run_name ${LISTWISE_RUN_NAME} --maxent_tau ${LISTWISE_TAU} --beta ${LISTWISE_BETA} --maxent_q_temperature ${LISTWISE_Q_TEMPERATURE}}"

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
        GRPO_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
        GRPO_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
        MAXENT_STEP0_PAPER_EVAL_ENABLED=0 \
        MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_ENABLED=0 \
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
        MAXENT_VLLM_GPU="$SINGLE_STACK_VLLM_GPU" \
        MAXENT_TRAIN_GPUS="$SINGLE_STACK_TRAIN_GPUS" \
        MAXENT_STEP0_PAPER_EVAL_ENABLED=0 \
        MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_ENABLED=0 \
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
      --export "ALL,MODEL_PATH=${model_path},RESULTS_DIR=${results_dir},TASKS=${TASKS},PASS_AT_8_SAMPLES=${PASS_AT_8_SAMPLES},VLLM_PORT=${port}" \
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

submit_rich_plot_job() {
  local dependency_jobs="$1"
  local rich_plot_cmd
  rich_plot_cmd=$(
    cat <<EOF
cd '$ROOT_DIR' && source ops/repo_env.sh && \
python tools/plot_listwise_vs_grpo_distribution.py \
  --grpo-run '$GRPO_RUN_NAME' \
  --listwise-run '$LISTWISE_RUN_NAME' \
  --q-temperature '$LISTWISE_Q_TEMPERATURE' \
  --output '$PIPELINE_ROOT/listwise_vs_grpo_within_prompt_mass.svg' \
  --summary-json '$PIPELINE_ROOT/listwise_vs_grpo_within_prompt_mass.summary.json' && \
convert '$PIPELINE_ROOT/listwise_vs_grpo_within_prompt_mass.svg' '$PIPELINE_ROOT/listwise_vs_grpo_within_prompt_mass.png' && \
python tools/plot_listwise_vs_grpo_over_time.py \
  --grpo-run '$GRPO_RUN_NAME' \
  --listwise-run '$LISTWISE_RUN_NAME' \
  --q-temperature '$LISTWISE_Q_TEMPERATURE' \
  --output '$PIPELINE_ROOT/listwise_vs_grpo_mass_over_time.svg' \
  --summary-json '$PIPELINE_ROOT/listwise_vs_grpo_mass_over_time.summary.json' && \
convert '$PIPELINE_ROOT/listwise_vs_grpo_mass_over_time.svg' '$PIPELINE_ROOT/listwise_vs_grpo_mass_over_time.png' && \
python tools/plot_listwise_vs_grpo_prompt_dynamics.py \
  --grpo-run '$GRPO_RUN_NAME' \
  --listwise-run '$LISTWISE_RUN_NAME' \
  --q-temperature '$LISTWISE_Q_TEMPERATURE' \
  --output '$PIPELINE_ROOT/listwise_vs_grpo_prompt_dynamics.svg' \
  --summary-json '$PIPELINE_ROOT/listwise_vs_grpo_prompt_dynamics.summary.json' && \
convert '$PIPELINE_ROOT/listwise_vs_grpo_prompt_dynamics.svg' '$PIPELINE_ROOT/listwise_vs_grpo_prompt_dynamics.png' && \
python tools/plot_listwise_vs_grpo_prompt_level.py \
  --grpo-run '$GRPO_RUN_NAME' \
  --listwise-run '$LISTWISE_RUN_NAME' \
  --q-temperature '$LISTWISE_Q_TEMPERATURE' \
  --output '$PIPELINE_ROOT/listwise_vs_grpo_prompt_level.svg' \
  --summary-json '$PIPELINE_ROOT/listwise_vs_grpo_prompt_level.summary.json' && \
convert '$PIPELINE_ROOT/listwise_vs_grpo_prompt_level.svg' '$PIPELINE_ROOT/listwise_vs_grpo_prompt_level.png' && \
python tools/plot_listwise_vs_grpo_accuracy_dynamics.py \
  --grpo-run '$GRPO_RUN_NAME' \
  --listwise-run '$LISTWISE_RUN_NAME' \
  --q-temperature '$LISTWISE_Q_TEMPERATURE' \
  --output '$PIPELINE_ROOT/listwise_vs_grpo_accuracy_dynamics.png' \
  --summary-json '$PIPELINE_ROOT/listwise_vs_grpo_accuracy_dynamics.summary.json'
EOF
  )
  local output
  output="$(
    sbatch \
      --job-name "$RICH_PLOT_JOB_NAME" \
      --nodes 1 \
      --partition "$SBATCH_PARTITION" \
      --account "$SBATCH_ACCOUNT" \
      --cpus-per-task "$RICH_PLOT_SBATCH_CPUS_PER_TASK" \
      --mem "$RICH_PLOT_SBATCH_MEM" \
      --time "$RICH_PLOT_SBATCH_TIME" \
      --dependency "afterok:${dependency_jobs}" \
      --wrap "$rich_plot_cmd"
  )"
  printf '%s\n' "$output"
  local job_id
  job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse rich plot job id" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

echo "[full-eval-rich] run_group=${RUN_GROUP}"
echo "[full-eval-rich] listwise_tau=${LISTWISE_TAU} beta=${LISTWISE_BETA} q_temperature=${LISTWISE_Q_TEMPERATURE}"
echo "[full-eval-rich] train_max_steps=${TRAIN_MAX_STEPS} train_num_epochs=${TRAIN_NUM_EPOCHS}"
echo "[full-eval-rich] live_eval_enabled=${TRAIN_LIVE_EVAL_ENABLED} live_pass_at_8=${TRAIN_LIVE_PASS_AT_8_ENABLED} eval_steps=${TRAIN_EVAL_STEPS}"

grpo_output_and_id="$(submit_train_job "grpo" "$GRPO_JOB_NAME" "$GRPO_ARGS")"
printf '%s\n' "$grpo_output_and_id"
GRPO_JOB_ID="$(printf '%s\n' "$grpo_output_and_id" | tail -n 1)"

listwise_output_and_id="$(submit_train_job "listwise" "$LISTWISE_JOB_NAME" "$LISTWISE_ARGS")"
printf '%s\n' "$listwise_output_and_id"
LISTWISE_JOB_ID="$(printf '%s\n' "$listwise_output_and_id" | tail -n 1)"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

mkdir -p "$GRPO_RESULTS_DIR" "$LISTWISE_RESULTS_DIR"

grpo_eval_output_and_id="$(submit_eval_job "$GRPO_EVAL_JOB_NAME" "$GRPO_JOB_ID" "$GRPO_OUTPUT_DIR" "$GRPO_RESULTS_DIR" "8127")"
printf '%s\n' "$grpo_eval_output_and_id"
GRPO_EVAL_JOB_ID="$(printf '%s\n' "$grpo_eval_output_and_id" | tail -n 1)"

listwise_eval_output_and_id="$(submit_eval_job "$LISTWISE_EVAL_JOB_NAME" "$LISTWISE_JOB_ID" "$LISTWISE_OUTPUT_DIR" "$LISTWISE_RESULTS_DIR" "8128")"
printf '%s\n' "$listwise_eval_output_and_id"
LISTWISE_EVAL_JOB_ID="$(printf '%s\n' "$listwise_eval_output_and_id" | tail -n 1)"

PLOT_CMD=$(
  cat <<EOF
cd '$ROOT_DIR' && source ops/repo_env.sh && \
python tools/plot_listwise_vs_grpo_eval_pass8.py \
  --grpo-pass8-json '$GRPO_RESULTS_DIR/seed_paper_eval_outputs_pass_at_8_n${PASS_AT_8_SAMPLES}.json' \
  --listwise-pass8-json '$LISTWISE_RESULTS_DIR/seed_paper_eval_outputs_pass_at_8_n${PASS_AT_8_SAMPLES}.json' \
  --listwise-tau '${LISTWISE_TAU}' \
  --listwise-beta '${LISTWISE_BETA}' \
  --listwise-q-temperature '${LISTWISE_Q_TEMPERATURE}' \
  --output '$PLOT_OUTPUT_SVG' \
  --summary-json '$PLOT_SUMMARY_JSON' && \
convert '$PLOT_OUTPUT_SVG' '$PLOT_OUTPUT_PNG'
EOF
)

PLOT_OUTPUT="$(
  sbatch \
    --job-name "$PLOT_JOB_NAME" \
    --nodes 1 \
    --partition "$SBATCH_PARTITION" \
    --account "$SBATCH_ACCOUNT" \
    --cpus-per-task 4 \
    --mem 16G \
    --time 02:00:00 \
    --dependency "afterok:${GRPO_EVAL_JOB_ID}:${LISTWISE_EVAL_JOB_ID}" \
    --wrap "$PLOT_CMD"
)"
printf '%s\n' "$PLOT_OUTPUT"
PLOT_JOB_ID="$(printf '%s\n' "$PLOT_OUTPUT" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"

rich_plot_output_and_id="$(submit_rich_plot_job "${GRPO_JOB_ID}:${LISTWISE_JOB_ID}")"
printf '%s\n' "$rich_plot_output_and_id"
RICH_PLOT_JOB_ID="$(printf '%s\n' "$rich_plot_output_and_id" | tail -n 1)"

MANIFEST_PATH="$PIPELINE_ROOT/manifest.tsv"
cat >"$MANIFEST_PATH" <<EOF
role	job_id	path
train_grpo	${GRPO_JOB_ID}	${GRPO_OUTPUT_DIR}
train_listwise	${LISTWISE_JOB_ID}	${LISTWISE_OUTPUT_DIR}
eval_grpo	${GRPO_EVAL_JOB_ID}	${GRPO_RESULTS_DIR}
eval_listwise	${LISTWISE_EVAL_JOB_ID}	${LISTWISE_RESULTS_DIR}
plot_eval_pass8	${PLOT_JOB_ID}	${PLOT_OUTPUT_PNG}
plot_rich_sidecar	${RICH_PLOT_JOB_ID}	${RICH_PLOT_SENTINEL}
EOF

echo "[full-eval-rich] manifest=${MANIFEST_PATH}"
echo "[full-eval-rich] train_grpo=${GRPO_JOB_ID}"
echo "[full-eval-rich] train_listwise=${LISTWISE_JOB_ID}"
echo "[full-eval-rich] eval_grpo=${GRPO_EVAL_JOB_ID}"
echo "[full-eval-rich] eval_listwise=${LISTWISE_EVAL_JOB_ID}"
echo "[full-eval-rich] plot_eval_pass8=${PLOT_JOB_ID}"
echo "[full-eval-rich] plot_rich_sidecar=${RICH_PLOT_JOB_ID}"
