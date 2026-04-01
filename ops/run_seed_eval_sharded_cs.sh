#!/usr/bin/env bash
# Submit SEED paper eval as CS shards, with optional prompt-range subshards per task.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SLURM_SCRIPT="${EVAL_SLURM_SCRIPT:-$ROOT_DIR/ops/slurm/eval_saved_model_full_seed.slurm}"
MERGE_SCRIPT="${MERGE_SCRIPT:-$ROOT_DIR/tools/merge_seed_eval_shards.py}"
TASK_MERGE_SCRIPT="${TASK_MERGE_SCRIPT:-$ROOT_DIR/tools/merge_seed_eval_task_subshards.py}"

RUN_NAME="${RUN_NAME:-oat_parity_1p5b_cs_sharded_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/var/artifacts/seed_paper_eval/manual/${RUN_NAME}}"
CURRENT_LINK="${CURRENT_LINK:-$ROOT_DIR/var/artifacts/seed_paper_eval/manual/current_oat_parity_1p5b_cs_sharded}"
MANIFEST_PATH="${MANIFEST_PATH:-$RESULT_ROOT/manifest.tsv}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/var/artifacts/logs}"

SBATCH_PARTITION="${SBATCH_PARTITION:-cs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-allcs}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-8}"
SBATCH_MEM="${SBATCH_MEM:-48G}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

MERGE_PARTITION="${MERGE_PARTITION:-$SBATCH_PARTITION}"
MERGE_ACCOUNT="${MERGE_ACCOUNT:-$SBATCH_ACCOUNT}"
MERGE_CPUS_PER_TASK="${MERGE_CPUS_PER_TASK:-2}"
MERGE_MEM="${MERGE_MEM:-8G}"
MERGE_TIME="${MERGE_TIME:-01:00:00}"
MERGE_EXTRA_ARGS="${MERGE_EXTRA_ARGS:-}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$ROOT_DIR/var/data/drgrpo_1p5b_oat_parity_trl_r1_20260331_144832}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}"
STEPS="${STEPS:-0,50,100}"
TEMPLATES="${TEMPLATES:-no,qwen_math,r1}"
TASKS="${TASKS:-aime,amc,math,minerva,olympiad_bench}"
TASK_SUBSHARDS="${TASK_SUBSHARDS:-}"
SKIP_EXISTING_TASKS="${SKIP_EXISTING_TASKS:-false}"
PASS_AT_8_SAMPLES="${PASS_AT_8_SAMPLES:-8}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"

mkdir -p "$RESULT_ROOT" "$LOG_DIR"
ln -sfn "$RESULT_ROOT" "$CURRENT_LINK"

SBATCH_EXTRA_ARR=()
MERGE_EXTRA_ARR=()
if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  SBATCH_EXTRA_ARR=($SBATCH_EXTRA_ARGS)
fi
if [[ -n "$MERGE_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  MERGE_EXTRA_ARR=($MERGE_EXTRA_ARGS)
fi

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -r -a out_ref <<< "$raw"
}

is_truthy() {
  local raw="${1:-}"
  case "${raw,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

declare -A TASK_SUBSHARD_MAP=()

parse_task_subshards() {
  local raw="${1:-}"
  TASK_SUBSHARD_MAP=()
  [[ -n "$raw" ]] || return 0
  local pairs=()
  split_csv "$raw" pairs
  local pair task count
  for pair in "${pairs[@]}"; do
    pair="${pair//[[:space:]]/}"
    [[ -n "$pair" ]] || continue
    if [[ "$pair" != *=* ]]; then
      echo "Invalid TASK_SUBSHARDS entry: $pair" >&2
      exit 1
    fi
    task="${pair%%=*}"
    count="${pair#*=}"
    if ! [[ "$count" =~ ^[0-9]+$ ]] || (( count < 1 )); then
      echo "Invalid shard count for $task: $count" >&2
      exit 1
    fi
    TASK_SUBSHARD_MAP["$task"]="$count"
  done
}

task_subshard_count_for() {
  local task="$1"
  if [[ -n "${TASK_SUBSHARD_MAP[$task]:-}" ]]; then
    printf '%s\n' "${TASK_SUBSHARD_MAP[$task]}"
  else
    printf '1\n'
  fi
}

task_total_prompts() {
  case "$1" in
    aime) printf '30\n' ;;
    amc) printf '83\n' ;;
    math) printf '500\n' ;;
    minerva) printf '272\n' ;;
    olympiad_bench) printf '675\n' ;;
    *)
      echo "Unknown official SEED task: $1" >&2
      exit 1
      ;;
  esac
}

task_shard_bounds() {
  local total="$1"
  local shard_idx="$2"
  local shard_count="$3"
  local start=$(( total * shard_idx / shard_count ))
  local end=$(( total * (shard_idx + 1) / shard_count ))
  printf '%s %s\n' "$start" "$end"
}

task_root_summary_exists() {
  local task_dir="$1"
  compgen -G "${task_dir}/*.summary.json" >/dev/null
}

first_task_root_summary() {
  local task_dir="$1"
  compgen -G "${task_dir}/*.summary.json" | head -n 1
}

model_path_for_step() {
  local step="$1"
  if [[ "$step" == "0" ]]; then
    printf '%s\n' "$BASE_MODEL_PATH"
    return 0
  fi
  local checkpoint_path="$TRAIN_OUTPUT_DIR/checkpoint-${step}"
  if [[ ! -d "$checkpoint_path" ]]; then
    echo "Missing checkpoint for step ${step}: $checkpoint_path" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint_path"
}

submit_task_job() {
  local job_name="$1"
  local model_path="$2"
  local results_dir="$3"
  local task_name="$4"
  local template_name="$5"
  local prompt_start="${6:-}"
  local prompt_end="${7:-}"
  local export_vars
  export_vars="ALL,ROOT_DIR=${ROOT_DIR},MODEL_PATH=${model_path},RESULTS_DIR=${results_dir},TASKS=${task_name},TEMPLATE=${template_name},PASS_AT_8_SAMPLES=${PASS_AT_8_SAMPLES},VLLM_DTYPE=${VLLM_DTYPE},VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN},VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE},PROMPT_START=${prompt_start},PROMPT_END=${prompt_end}"
  local output
  output="$(
    sbatch \
      --job-name "$job_name" \
      --nodes 1 \
      --partition "$SBATCH_PARTITION" \
      --account "$SBATCH_ACCOUNT" \
      --gres "$SBATCH_GRES" \
      --cpus-per-task "$SBATCH_CPUS_PER_TASK" \
      --mem "$SBATCH_MEM" \
      --time "$SBATCH_TIME" \
      --output "$LOG_DIR/%x-%j.out" \
      --error "$LOG_DIR/%x-%j.err" \
      "${SBATCH_EXTRA_ARR[@]}" \
      --export "$export_vars" \
      "$EVAL_SLURM_SCRIPT"
  )"
  printf '%s\n' "$output"
  local job_id
  job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse job id for ${job_name}" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

submit_task_merge_job() {
  local job_name="$1"
  local task_dir="$2"
  local dependency_csv="$3"
  local -a cmd=(
    sbatch
    --job-name "$job_name"
    --nodes 1
    --partition "$MERGE_PARTITION"
    --account "$MERGE_ACCOUNT"
    --cpus-per-task "$MERGE_CPUS_PER_TASK"
    --mem "$MERGE_MEM"
    --time "$MERGE_TIME"
    --output "$LOG_DIR/%x-%j.out"
    --error "$LOG_DIR/%x-%j.err"
  )
  if [[ -n "$dependency_csv" ]]; then
    cmd+=(--dependency "afterok:${dependency_csv}")
  fi
  cmd+=("${MERGE_EXTRA_ARR[@]}")
  cmd+=(
    --wrap
    "cd '$ROOT_DIR' && source ops/repo_env.sh && python '$TASK_MERGE_SCRIPT' --task-dir '$task_dir'"
  )
  local output
  output="$("${cmd[@]}")"
  printf '%s\n' "$output"
  local job_id
  job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse task merge job id for ${job_name}" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

submit_merge_job() {
  local job_name="$1"
  local parent_dir="$2"
  local dependency_csv="${3:-}"
  local -a cmd=(
    sbatch
    --job-name "$job_name"
    --nodes 1
    --partition "$MERGE_PARTITION"
    --account "$MERGE_ACCOUNT"
    --cpus-per-task "$MERGE_CPUS_PER_TASK"
    --mem "$MERGE_MEM"
    --time "$MERGE_TIME"
    --output "$LOG_DIR/%x-%j.out"
    --error "$LOG_DIR/%x-%j.err"
  )
  if [[ -n "$dependency_csv" ]]; then
    cmd+=(--dependency "afterok:${dependency_csv}")
  fi
  cmd+=("${MERGE_EXTRA_ARR[@]}")
  cmd+=(
    --wrap
    "cd '$ROOT_DIR' && source ops/repo_env.sh && python '$MERGE_SCRIPT' --parent-dir '$parent_dir' --tasks '$TASKS'"
  )
  local output
  output="$("${cmd[@]}")"
  printf '%s\n' "$output"
  local job_id
  job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse merge job id for ${job_name}" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

step_values=()
template_values=()
task_values=()
split_csv "$STEPS" step_values
split_csv "$TEMPLATES" template_values
split_csv "$TASKS" task_values
parse_task_subshards "$TASK_SUBSHARDS"

{
  printf 'kind\tstep\ttemplate\ttask\tjob_id\tpath\n'
} >"$MANIFEST_PATH"

for raw_step in "${step_values[@]}"; do
  step="${raw_step//[[:space:]]/}"
  [[ -n "$step" ]] || continue
  model_path="$(model_path_for_step "$step")"
  for raw_template in "${template_values[@]}"; do
    template="${raw_template//[[:space:]]/}"
    [[ -n "$template" ]] || continue
    combo_root="$RESULT_ROOT/step${step}/${template}"
    mkdir -p "$combo_root"
    combo_dependency_ids=()
    for raw_task in "${task_values[@]}"; do
      task="${raw_task//[[:space:]]/}"
      [[ -n "$task" ]] || continue
      task_dir="$combo_root/$task"
      mkdir -p "$task_dir"
      if is_truthy "$SKIP_EXISTING_TASKS" && task_root_summary_exists "$task_dir"; then
        existing_summary="$(first_task_root_summary "$task_dir")"
        printf 'existing\t%s\t%s\t%s\t-\t%s\n' \
          "$step" "$template" "$task" "$existing_summary" >>"$MANIFEST_PATH"
        echo "[seed-eval-sharded] skipping existing task summary: step=${step} template=${template} task=${task}" >&2
        continue
      fi

      shard_count="$(task_subshard_count_for "$task")"
      if (( shard_count <= 1 )); then
        job_name="cs_s${step}_${template}_${task}"
        output_and_id="$(submit_task_job "$job_name" "$model_path" "$task_dir" "$task" "$template")"
        printf '%s\n' "$output_and_id"
        job_id="$(printf '%s\n' "$output_and_id" | tail -n 1)"
        combo_dependency_ids+=("$job_id")
        printf 'task\t%s\t%s\t%s\t%s\t%s\n' \
          "$step" "$template" "$task" "$job_id" "$task_dir" >>"$MANIFEST_PATH"
        continue
      fi

      total_prompts="$(task_total_prompts "$task")"
      task_dependency_ids=()
      for (( shard_idx=0; shard_idx<shard_count; shard_idx++ )); do
        read -r prompt_start prompt_end < <(task_shard_bounds "$total_prompts" "$shard_idx" "$shard_count")
        shard_label="$(printf 'shard_%02d_of_%02d' "$((shard_idx + 1))" "$shard_count")"
        shard_dir="$task_dir/$shard_label"
        mkdir -p "$shard_dir"
        job_name="cs_s${step}_${template}_${task}_p$((shard_idx + 1))of${shard_count}"
        output_and_id="$(submit_task_job "$job_name" "$model_path" "$shard_dir" "$task" "$template" "$prompt_start" "$prompt_end")"
        printf '%s\n' "$output_and_id"
        job_id="$(printf '%s\n' "$output_and_id" | tail -n 1)"
        task_dependency_ids+=("$job_id")
        printf 'subshard\t%s\t%s\t%s\t%s\t%s\n' \
          "$step" "$template" "$task" "$job_id" "$shard_dir" >>"$MANIFEST_PATH"
      done
      task_dependency_csv="$(IFS=:; printf '%s' "${task_dependency_ids[*]}")"
      task_merge_job_name="cs_merge_task_s${step}_${template}_${task}"
      task_merge_output_and_id="$(submit_task_merge_job "$task_merge_job_name" "$task_dir" "$task_dependency_csv")"
      printf '%s\n' "$task_merge_output_and_id"
      task_merge_job_id="$(printf '%s\n' "$task_merge_output_and_id" | tail -n 1)"
      combo_dependency_ids+=("$task_merge_job_id")
      printf 'task_merge\t%s\t%s\t%s\t%s\t%s\n' \
        "$step" "$template" "$task" "$task_merge_job_id" "$task_dir/seed_paper_eval_sharded.summary.json" >>"$MANIFEST_PATH"
    done

    dependency_csv=""
    if (( ${#combo_dependency_ids[@]} > 0 )); then
      dependency_csv="$(IFS=:; printf '%s' "${combo_dependency_ids[*]}")"
    fi
    merge_job_name="cs_merge_s${step}_${template}"
    merge_output_and_id="$(submit_merge_job "$merge_job_name" "$combo_root" "$dependency_csv")"
    printf '%s\n' "$merge_output_and_id"
    merge_job_id="$(printf '%s\n' "$merge_output_and_id" | tail -n 1)"
    printf 'merge\t%s\t%s\t-\t%s\t%s\n' \
      "$step" "$template" "$merge_job_id" "$combo_root/seed_paper_eval_sharded.summary.json" >>"$MANIFEST_PATH"
  done
done

echo "[seed-eval-sharded] result_root=$RESULT_ROOT"
echo "[seed-eval-sharded] current_link=$CURRENT_LINK"
echo "[seed-eval-sharded] manifest=$MANIFEST_PATH"
