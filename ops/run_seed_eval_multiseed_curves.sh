#!/usr/bin/env bash
# Launch multi-seed sharded SEED eval sweeps and aggregate CI curves.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHARDED_SCRIPT="${SHARDED_SCRIPT:-$ROOT_DIR/ops/run_seed_eval_sharded_cs.sh}"
AGG_SCRIPT="${AGG_SCRIPT:-$ROOT_DIR/tools/render_seed_eval_multiseed_curves.py}"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/var/data/oat_zero_exact_1p5b_20260404_101832/qwen2.5-Math-1.5b-r1-zero-readmeflash-node302_0404T10:19:27}"
STEP_SOURCE_DIR="${STEP_SOURCE_DIR:-$RUN_ROOT}"
MODEL_ROOT="${MODEL_ROOT:-$RUN_ROOT}"
MODEL_STEP_FORMATS="${MODEL_STEP_FORMATS:-saved_models/step_%05d,checkpoint-%d,checkpoints/step_%05d}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}"
PREFER_STEP0_MODEL_ARTIFACT="${PREFER_STEP0_MODEL_ARTIFACT:-0}"

INFERENCE_SEEDS="${INFERENCE_SEEDS:-0,1,2,3,4}"
TEMPLATES="${TEMPLATES:-no,qwen,r1}"
TASKS="${TASKS:-aime,amc,math,minerva,olympiad_bench}"
TASK_SUBSHARDS="${TASK_SUBSHARDS:-math=8,minerva=4,olympiad_bench=10}"
STEPS="${STEPS:-}"
SKIP_EXISTING_TASKS="${SKIP_EXISTING_TASKS:-false}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

PASS_AT_8_SAMPLES="${PASS_AT_8_SAMPLES:-8}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"

SBATCH_PARTITION="${SBATCH_PARTITION:-all}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-allcs}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-4}"
SBATCH_MEM="${SBATCH_MEM:-24G}"
SBATCH_TIME="${SBATCH_TIME:-00:55:00}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

MERGE_PARTITION="${MERGE_PARTITION:-$SBATCH_PARTITION}"
MERGE_ACCOUNT="${MERGE_ACCOUNT:-$SBATCH_ACCOUNT}"
MERGE_CPUS_PER_TASK="${MERGE_CPUS_PER_TASK:-1}"
MERGE_MEM="${MERGE_MEM:-4G}"
MERGE_TIME="${MERGE_TIME:-00:20:00}"
MERGE_EXTRA_ARGS="${MERGE_EXTRA_ARGS:-}"

AGG_PARTITION="${AGG_PARTITION:-$MERGE_PARTITION}"
AGG_ACCOUNT="${AGG_ACCOUNT:-$MERGE_ACCOUNT}"
AGG_CPUS_PER_TASK="${AGG_CPUS_PER_TASK:-2}"
AGG_MEM="${AGG_MEM:-8G}"
AGG_TIME="${AGG_TIME:-00:20:00}"
AGG_EXTRA_ARGS="${AGG_EXTRA_ARGS:-}"
AGG_PYTHON="${AGG_PYTHON:-python}"

RUN_BASENAME="$(basename "$RUN_ROOT")"
RUN_NAME="${RUN_NAME:-${RUN_BASENAME}_seed_curves_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/var/artifacts/seed_paper_eval/multiseed/${RUN_NAME}}"
CURRENT_LINK="${CURRENT_LINK:-$ROOT_DIR/var/artifacts/seed_paper_eval/multiseed/current_${RUN_BASENAME}}"
TOP_MANIFEST="${TOP_MANIFEST:-$RESULTS_ROOT/multiseed_manifest.tsv}"
AGG_SUMMARY_JSON="${AGG_SUMMARY_JSON:-$RESULTS_ROOT/multiseed_curves.summary.json}"
AGG_ROWS_TSV="${AGG_ROWS_TSV:-$RESULTS_ROOT/multiseed_curves.tsv}"
AGG_SVG="${AGG_SVG:-$RESULTS_ROOT/multiseed_curves.svg}"

mkdir -p "$RESULTS_ROOT"
mkdir -p "$(dirname "$CURRENT_LINK")"
ln -sfn "$RESULTS_ROOT" "$CURRENT_LINK"

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

normalize_templates_csv() {
  local raw="$1"
  local entries=()
  local normalized=()
  local value
  split_csv "$raw" entries
  for value in "${entries[@]}"; do
    value="${value//[[:space:]]/}"
    [[ -n "$value" ]] || continue
    case "${value,,}" in
      qwen|qwen_math) normalized+=("qwen_math") ;;
      no|none) normalized+=("no") ;;
      r1) normalized+=("r1") ;;
      *) normalized+=("$value") ;;
    esac
  done
  if (( ${#normalized[@]} == 0 )); then
    echo "No valid templates were provided." >&2
    exit 1
  fi
  local IFS=','
  printf '%s\n' "${normalized[*]}"
}

discover_steps() {
  python - "$STEP_SOURCE_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()

sources: list[tuple[str, list[int]]] = []
saved_dir = root / "saved_models"
if saved_dir.is_dir():
    saved_steps = sorted(
        int(path.name.split("_", 1)[1])
        for path in saved_dir.glob("step_*")
        if path.is_dir() and "_" in path.name and path.name.split("_", 1)[1].isdigit()
    )
    if saved_steps:
        sources.append(("saved_models", saved_steps))

checkpoint_steps = sorted(
    int(path.name.split("-", 1)[1])
    for path in root.glob("checkpoint-*")
    if path.is_dir() and "-" in path.name and path.name.split("-", 1)[1].isdigit()
)
if checkpoint_steps:
    sources.append(("checkpoint_dirs", checkpoint_steps))

nested_checkpoint_root = root / "checkpoints"
if nested_checkpoint_root.is_dir():
    nested_steps = sorted(
        int(path.name.split("_", 1)[1])
        for path in nested_checkpoint_root.glob("step_*")
        if path.is_dir() and "_" in path.name and path.name.split("_", 1)[1].isdigit()
    )
    if nested_steps:
        sources.append(("checkpoints_step_dirs", nested_steps))

eval_dir = root / "eval_results"
if eval_dir.is_dir():
    eval_steps = sorted(
        {
            int(path.stem.split("_", 1)[0])
            for path in eval_dir.glob("*.json")
            if "_" in path.stem and path.stem.split("_", 1)[0].isdigit()
        }
    )
    if eval_steps:
        sources.append(("eval_results", eval_steps))

if not sources:
    raise SystemExit(
        f"Unable to discover checkpoint steps under {root}. "
        "Expected saved_models/, checkpoint-*/ dirs, checkpoints/step_*/ dirs, or eval_results/*.json."
    )

source_name, steps = sources[0]
print(source_name)
print(",".join(str(step) for step in steps))
PY
}

validate_model_artifacts() {
  local step_source="$1"
  local steps_csv="$2"
  python - "$MODEL_ROOT" "$MODEL_STEP_FORMATS" "$steps_csv" "$PREFER_STEP0_MODEL_ARTIFACT" "$step_source" <<'PY'
from pathlib import Path
import sys

model_root = Path(sys.argv[1]).resolve()
formats = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
steps = [int(item) for item in sys.argv[3].split(",") if item.strip()]
prefer_step0 = sys.argv[4].strip().lower() in {"1", "true", "yes", "on"}
step_source = sys.argv[5]

missing: list[int] = []
for step in steps:
    if step == 0 and not prefer_step0:
        continue
    found = False
    for fmt in formats:
        candidate = model_root / (fmt % step)
        if candidate.is_dir():
            found = True
            break
    if not found:
        missing.append(step)

if missing:
    preview = ",".join(str(step) for step in missing[:10])
    total = len(steps)
    raise SystemExit(
        "Missing model artifacts for "
        f"{len(missing)}/{total} discovered steps under {model_root}. "
        f"Checked MODEL_STEP_FORMATS={','.join(formats)}. "
        f"First missing steps: {preview}. "
        f"Step source was {step_source}. "
        "If this run only contains eval_results/*.json, point MODEL_ROOT at the exported checkpoints."
    )
PY
}

collect_merge_job_ids() {
  python - "$1" <<'PY'
from pathlib import Path
import csv
import sys

manifest_path = Path(sys.argv[1])
if not manifest_path.exists():
    raise SystemExit(f"Manifest not found: {manifest_path}")

job_ids: list[str] = []
with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        if row.get("kind") != "merge":
            continue
        job_id = (row.get("job_id") or "").strip()
        if job_id and job_id != "-":
            job_ids.append(job_id)
print(":".join(job_ids))
PY
}

submit_agg_job() {
  local dependency_csv="${1:-}"
  local -a cmd=(
    sbatch
    --job-name "seed_eval_curve_agg"
    --nodes 1
    --partition "$AGG_PARTITION"
    --account "$AGG_ACCOUNT"
    --cpus-per-task "$AGG_CPUS_PER_TASK"
    --mem "$AGG_MEM"
    --time "$AGG_TIME"
    --output "$RESULTS_ROOT/%x-%j.out"
    --error "$RESULTS_ROOT/%x-%j.err"
  )
  if [[ -n "$dependency_csv" ]]; then
    cmd+=(--dependency "afterok:${dependency_csv}")
  fi
  if [[ -n "$AGG_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local -a agg_extra_arr=($AGG_EXTRA_ARGS)
    cmd+=("${agg_extra_arr[@]}")
  fi
  cmd+=(
    --wrap
    "cd '$ROOT_DIR' && source ops/repo_env.sh && '$AGG_PYTHON' '$AGG_SCRIPT' --results-root '$RESULTS_ROOT' --summary-json '$AGG_SUMMARY_JSON' --rows-tsv '$AGG_ROWS_TSV' --output-svg '$AGG_SVG'"
  )
  local output
  output="$("${cmd[@]}")"
  printf '%s\n' "$output"
  local job_id
  job_id="$(printf '%s\n' "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse aggregation job id" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

NORMALIZED_TEMPLATES="$(normalize_templates_csv "$TEMPLATES")"
if [[ -z "$STEPS" ]]; then
  mapfile -t discovered_info < <(discover_steps)
  STEP_SOURCE="${discovered_info[0]}"
  STEPS="${discovered_info[1]}"
else
  STEP_SOURCE="manual_override"
fi
validate_model_artifacts "$STEP_SOURCE" "$STEPS"

seed_values=()
template_values=()
step_values=()
split_csv "$INFERENCE_SEEDS" seed_values
split_csv "$NORMALIZED_TEMPLATES" template_values
split_csv "$STEPS" step_values

echo "[multiseed] run_root=$RUN_ROOT"
echo "[multiseed] model_root=$MODEL_ROOT"
echo "[multiseed] step_source=$STEP_SOURCE"
echo "[multiseed] steps=${#step_values[@]} templates=${#template_values[@]} seeds=${#seed_values[@]}"
echo "[multiseed] results_root=$RESULTS_ROOT"
echo "[multiseed] sbatch_time=$SBATCH_TIME merge_time=$MERGE_TIME agg_time=$AGG_TIME"

if is_truthy "$VALIDATE_ONLY"; then
  echo "[multiseed] validation complete; no jobs submitted because VALIDATE_ONLY=$VALIDATE_ONLY"
  exit 0
fi

{
  printf 'kind\tsampling_seed\tjob_id\tpath\tmanifest\n'
} >"$TOP_MANIFEST"

aggregate_dependencies=()
for raw_seed in "${seed_values[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  [[ -n "$seed" ]] || continue
  seed_result_root="$RESULTS_ROOT/seed_$(printf '%03d' "$seed")"
  seed_manifest="$seed_result_root/manifest.tsv"
  seed_current_link="$RESULTS_ROOT/current_seed_$(printf '%03d' "$seed")"

  (
    export RUN_NAME="${RUN_NAME}_seed${seed}"
    export RESULT_ROOT="$seed_result_root"
    export CURRENT_LINK="$seed_current_link"
    export MANIFEST_PATH="$seed_manifest"
    export TRAIN_OUTPUT_DIR="$RUN_ROOT"
    export MODEL_ROOT="$MODEL_ROOT"
    export MODEL_STEP_FORMATS="$MODEL_STEP_FORMATS"
    export BASE_MODEL_PATH="$BASE_MODEL_PATH"
    export PREFER_STEP0_MODEL_ARTIFACT="$PREFER_STEP0_MODEL_ARTIFACT"
    export STEPS="$STEPS"
    export TEMPLATES="$NORMALIZED_TEMPLATES"
    export TASKS="$TASKS"
    export TASK_SUBSHARDS="$TASK_SUBSHARDS"
    export SKIP_EXISTING_TASKS="$SKIP_EXISTING_TASKS"
    export PASS_AT_8_SAMPLES="$PASS_AT_8_SAMPLES"
    export VLLM_DTYPE="$VLLM_DTYPE"
    export VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN"
    export VLLM_BATCH_SIZE="$VLLM_BATCH_SIZE"
    export SAMPLING_SEED="$seed"
    export SBATCH_PARTITION="$SBATCH_PARTITION"
    export SBATCH_ACCOUNT="$SBATCH_ACCOUNT"
    export SBATCH_GRES="$SBATCH_GRES"
    export SBATCH_CPUS_PER_TASK="$SBATCH_CPUS_PER_TASK"
    export SBATCH_MEM="$SBATCH_MEM"
    export SBATCH_TIME="$SBATCH_TIME"
    export SBATCH_EXTRA_ARGS="$SBATCH_EXTRA_ARGS"
    export MERGE_PARTITION="$MERGE_PARTITION"
    export MERGE_ACCOUNT="$MERGE_ACCOUNT"
    export MERGE_CPUS_PER_TASK="$MERGE_CPUS_PER_TASK"
    export MERGE_MEM="$MERGE_MEM"
    export MERGE_TIME="$MERGE_TIME"
    export MERGE_EXTRA_ARGS="$MERGE_EXTRA_ARGS"
    bash "$SHARDED_SCRIPT"
  )

  merge_job_ids="$(collect_merge_job_ids "$seed_manifest")"
  if [[ -n "$merge_job_ids" ]]; then
    IFS=':' read -r -a seed_dep_values <<< "$merge_job_ids"
    aggregate_dependencies+=("${seed_dep_values[@]}")
  fi
  printf 'seed_merge\t%s\t%s\t%s\t%s\n' \
    "$seed" "${merge_job_ids:--}" "$seed_result_root" "$seed_manifest" >>"$TOP_MANIFEST"
done

agg_dependency_csv=""
if (( ${#aggregate_dependencies[@]} > 0 )); then
  agg_dependency_csv="$(IFS=:; printf '%s' "${aggregate_dependencies[*]}")"
fi

agg_output_and_id="$(submit_agg_job "$agg_dependency_csv")"
printf '%s\n' "$agg_output_and_id"
agg_job_id="$(printf '%s\n' "$agg_output_and_id" | tail -n 1)"
printf 'aggregate\t-\t%s\t%s\t%s\n' \
  "$agg_job_id" "$AGG_SVG" "$TOP_MANIFEST" >>"$TOP_MANIFEST"

echo "[multiseed] top_manifest=$TOP_MANIFEST"
echo "[multiseed] figure=$AGG_SVG"
echo "[multiseed] aggregate_job_id=$agg_job_id"
