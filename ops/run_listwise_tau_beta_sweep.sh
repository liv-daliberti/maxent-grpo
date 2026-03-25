#!/usr/bin/env bash
# Submit a short listwise tau/beta sweep on the 1.5B math_fair backbone.
#
# Defaults:
# - resource profile: interim/general-compute A6000s
# - objective: listwise only
# - grid: tau in {0.35, 0.50, 0.70}, beta in {0.04, 0.08, 0.12}
# - short horizon: 50 steps
# - reduced official eval for tuning: aime,amc,math

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
TAU_VALUES="${TAU_VALUES:-0.35,0.50,0.70}"
BETA_VALUES="${BETA_VALUES:-0.04,0.08,0.12}"
SWEEP_MAX_STEPS="${SWEEP_MAX_STEPS:-50}"
SWEEP_EVAL_STEPS="${SWEEP_EVAL_STEPS:-25}"
SWEEP_TASKS="${SWEEP_TASKS:-aime,amc,math}"
SWEEP_MAX_TEST="${SWEEP_MAX_TEST:-}"
SWEEP_SEEDS="${SWEEP_SEEDS:-${SWEEP_SEED:-42}}"
MAXENT_ARGS_BASE="${MAXENT_ARGS_BASE:-}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-cs_listwise_sweep}"
DRY_RUN="${DRY_RUN:-0}"

RUN_GROUP="${WANDB_RUN_GROUP:-listwise_tau_beta_sweep_${MODEL//\//-}_${CONFIG_SUFFIX}_$(date +%Y%m%d_%H%M%S)}"
VAR_DIR="${VAR_DIR:-$SCRIPT_DIR/../var}"
VAR_DIR="$(cd "$VAR_DIR" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VAR_DIR/data/listwise_tau_beta_sweep/${RUN_GROUP}}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$VAR_DIR/artifacts/sweeps/listwise_tau_beta/${RUN_GROUP}}"
MANIFEST_PATH="$MANIFEST_ROOT/manifest.tsv"

mkdir -p "$OUTPUT_ROOT" "$MANIFEST_ROOT"

slugify_float() {
  local raw="$1"
  raw="${raw//-/m}"
  raw="${raw//./p}"
  printf '%s\n' "$raw"
}

parse_csv_into_array() {
  local raw="${1//[[:space:]]/}"
  local -n out_ref="$2"
  if [[ -z "$raw" ]]; then
    echo "CSV value list must not be empty" >&2
    exit 1
  fi
  IFS=',' read -r -a out_ref <<< "$raw"
}

build_variant_args() {
  local tau="$1"
  local beta="$2"
  local seed="$3"
  local run_name="$4"
  local output_dir="$5"
  local args="$MAXENT_ARGS_BASE"
  args+=" --maxent_tau $tau"
  args+=" --beta $beta"
  args+=" --max_steps $SWEEP_MAX_STEPS"
  args+=" --num_train_epochs 1"
  args+=" --eval_steps $SWEEP_EVAL_STEPS"
  args+=" --save_strategy no"
  args+=" --final_model_save_enabled false"
  args+=" --seed_paper_eval_tasks $SWEEP_TASKS"
  args+=" --seed $seed"
  args+=" --output_dir $output_dir"
  args+=" --run_name $run_name"
  if [[ -n "$SWEEP_MAX_TEST" ]]; then
    args+=" --seed_paper_eval_max_test $SWEEP_MAX_TEST"
  fi
  printf '%s\n' "$args"
}

submit_sweep_job() {
  local tau="$1"
  local beta="$2"
  local seed="$3"
  local tau_slug beta_slug seed_slug run_name job_name output_dir variant_args output job_id
  tau_slug="$(slugify_float "$tau")"
  beta_slug="$(slugify_float "$beta")"
  seed_slug="$(slugify_float "$seed")"
  run_name="${RUN_GROUP}-seed${seed_slug}-tau${tau_slug}-beta${beta_slug}"
  job_name="${JOB_NAME_PREFIX}-s${seed_slug}-t${tau_slug}-b${beta_slug}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  variant_args="$(build_variant_args "$tau" "$beta" "$seed" "$run_name" "$output_dir")"
  output="$(
    env \
      WANDB_RUN_GROUP="$RUN_GROUP" \
      MODEL="$MODEL" \
      CONFIG_SUFFIX="$CONFIG_SUFFIX" \
      RECIPE_PROFILE="$RECIPE_PROFILE" \
      MAXENT_STEP0_PAPER_EVAL_TASKS="$SWEEP_TASKS" \
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
      JOB_NAME="$job_name" \
      RUN_ONLY="listwise" \
      MAXENT_ARGS="$variant_args" \
      DRY_RUN="$DRY_RUN" \
      "$DUAL_WRAPPER"
  )"
  printf '%s\n' "$output"
  if [[ "$DRY_RUN" == "1" ]]; then
    job_id="dryrun-${seed_slug}-${tau_slug}-${beta_slug}"
  else
    job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
    if [[ -z "$job_id" ]]; then
      echo "Failed to parse submitted batch job id for seed=${seed} tau=${tau} beta=${beta}" >&2
      exit 1
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tau" "$beta" "$seed" "$job_id" "$job_name" "$run_name" "$output_dir" >> "$MANIFEST_PATH"
}

tau_values=()
beta_values=()
seed_values=()
parse_csv_into_array "$TAU_VALUES" tau_values
parse_csv_into_array "$BETA_VALUES" beta_values
parse_csv_into_array "$SWEEP_SEEDS" seed_values

printf 'tau\tbeta\tseed\tjob_id\tjob_name\trun_name\toutput_dir\n' > "$MANIFEST_PATH"

echo "[listwise-sweep] run_group=${RUN_GROUP}"
echo "[listwise-sweep] manifest=${MANIFEST_PATH}"
echo "[listwise-sweep] output_root=${OUTPUT_ROOT}"
echo "[listwise-sweep] tau_values=${TAU_VALUES}"
echo "[listwise-sweep] beta_values=${BETA_VALUES}"
echo "[listwise-sweep] seeds=${SWEEP_SEEDS}"
echo "[listwise-sweep] max_steps=${SWEEP_MAX_STEPS} eval_steps=${SWEEP_EVAL_STEPS} tasks=${SWEEP_TASKS}"

for seed in "${seed_values[@]}"; do
  for tau in "${tau_values[@]}"; do
    for beta in "${beta_values[@]}"; do
      submit_sweep_job "$tau" "$beta" "$seed"
    done
  done
done

echo "[listwise-sweep] manifest written to ${MANIFEST_PATH}"
echo "[listwise-sweep] report command:"
echo "  python $SCRIPT_DIR/../tools/listwise_sweep_report.py --manifest $MANIFEST_PATH"
