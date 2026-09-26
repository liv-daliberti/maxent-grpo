#!/usr/bin/env bash
# Four fixed-seed K=8 terminal evaluations for the clean 0.5B free-form
# treatment/control cohorts. Every generated completion is retained.
set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_ROOT="${OAT_ZERO_EVAL_CODE_ROOT:-$ROOT_DIR}"
TASK="${OAT_ZERO_COMPARATIVE_TASK:?set countdown or graph_coloring}"
RUN_DATA_ROOT="${OAT_ZERO_RUN_DATA_ROOT:-$ROOT_DIR/var/data}"
OUTPUT_ROOT="${OAT_ZERO_EVAL_OUTPUT_ROOT:-$ROOT_DIR/var/artifacts/freeform_05b_repeated_eval_v1}"
PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
EVAL_SEEDS_CSV="${OAT_ZERO_EVAL_SEEDS:-1001,1002,1003,1004}"
SAMPLE_COUNT="${OAT_ZERO_EVAL_SAMPLE_COUNT:-8}"

case "$TASK" in
  countdown)
    CONTROL_PREFIX="cde22_freeform_conditional_dual_05b_v2"
    TREATMENT_PREFIX="cde27_freeform_conditional_dual_05b_v1"
    DIAGNOSTIC_STAMP="cde30_freeform_05b_fixed_k8x4_v1"
    DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_countdown_easy3_probe}"
    ;;
  graph_coloring)
    CONTROL_PREFIX="gce22_freeform_conditional_dual_05b_v2"
    TREATMENT_PREFIX="gce27_freeform_conditional_dual_05b_v1"
    DIAGNOSTIC_STAMP="gce30_freeform_05b_fixed_k8x4_v1"
    DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_answer_mode_probe}"
    ;;
  *)
    echo "Unknown task $TASK; use countdown or graph_coloring." >&2
    exit 1
    ;;
esac

latest_checkpoint() {
  local prefix="$1" arm="$2" seed="$3"
  local best="" best_step=-1 run_dir step_dir step_text step
  shopt -s nullglob
  for run_dir in \
    "$RUN_DATA_ROOT"/xdr_*_"${prefix}_${arm}_s${seed}" \
    "$RUN_DATA_ROOT"/oat_zero_tiny_*_"${prefix}_${arm}_s${seed}"; do
    for step_dir in "$run_dir"/saved_models/step_* "$run_dir"/debug_*/saved_models/step_*; do
      [[ -f "$step_dir/config.json" ]] || continue
      step_text="${step_dir##*step_}"
      step="$((10#$step_text))"
      if (( step > best_step )); then
        best="$step_dir"
        best_step="$step"
      fi
    done
  done
  shopt -u nullglob
  [[ -n "$best" ]] || {
    echo "No final checkpoint for ${prefix}_${arm}_s${seed}" >&2
    return 1
  }
  printf '%s\n' "$best"
}

checkpoint_specs=()
for seed in 43 44 45; do
  control_checkpoint="$(latest_checkpoint "$CONTROL_PREFIX" grpo "$seed")"
  treatment_checkpoint="$(latest_checkpoint "$TREATMENT_PREFIX" maxent_dual "$seed")"
  checkpoint_specs+=(--checkpoint "grpo_s${seed}=${control_checkpoint}")
  checkpoint_specs+=(--checkpoint "maxent_dual_s${seed}=${treatment_checkpoint}")
  echo "[fixed-eval] grpo_s${seed} -> ${control_checkpoint}"
  echo "[fixed-eval] maxent_dual_s${seed} -> ${treatment_checkpoint}"
done

mkdir -p "$OUTPUT_ROOT"
IFS=',' read -r -a eval_seeds <<< "$EVAL_SEEDS_CSV"
for eval_seed in "${eval_seeds[@]}"; do
  "$PYTHON_BIN" "$CODE_ROOT/ops/eval_exact_answer_mode_coverage.py" \
    --stamp-prefix "${DIAGNOSTIC_STAMP}_e${eval_seed}" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --splits multi_answer \
    --sample-count "$SAMPLE_COUNT" \
    --temperature 1 \
    --top-p 1 \
    --seed "$eval_seed" \
    --include-text \
    "${checkpoint_specs[@]}"
done

"$PYTHON_BIN" "$CODE_ROOT/ops/eval_exact_answer_mode_coverage.py" \
  --stamp-prefix "${DIAGNOSTIC_STAMP}_greedy" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --splits multi_answer \
  --sample-count 1 \
  --temperature 0 \
  --seed 0 \
  --include-text \
  "${checkpoint_specs[@]}"

"$PYTHON_BIN" "$CODE_ROOT/ops/analyze_countdown_comparative.py" \
  --output-root "$OUTPUT_ROOT" \
  --stamp-prefix "$DIAGNOSTIC_STAMP" \
  --baseline grpo \
  --primary-arm maxent_dual \
  --out "$OUTPUT_ROOT/${DIAGNOSTIC_STAMP}_regression.json"

echo "[fixed-eval] complete: $OUTPUT_ROOT/$DIAGNOSTIC_STAMP"
