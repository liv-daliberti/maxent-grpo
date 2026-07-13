#!/usr/bin/env bash
# Evaluate every trained comparative arm with the paper protocol: for each
# evaluation seed, draw K sampled completions per prompt (default K=8,
# temperature 1.0) plus one greedy pass, using explicit checkpoint specs
# discovered from the submit manifest naming. Run on a GPU node, e.g.:
#   srun --gres=gpu:1 --nodelist=node917 -- \
#     RUN_STAMP_PREFIX=<stamp> ops/run_countdown_comparative_eval.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP_PREFIX="${RUN_STAMP_PREFIX:?set RUN_STAMP_PREFIX to the submit stamp}"
TASK="${OAT_ZERO_COMPARATIVE_TASK:-countdown}"
case "$TASK" in
  countdown)
    DEFAULT_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
    ;;
  graph_coloring)
    DEFAULT_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
    ;;
  *)
    echo "Unknown OAT_ZERO_COMPARATIVE_TASK=${TASK}; use countdown or graph_coloring." >&2
    exit 1
    ;;
esac
DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$DEFAULT_DATA_ROOT}"
RUN_DATA_ROOT="${OAT_ZERO_RUN_DATA_ROOT:-$ROOT_DIR/var/data}"
OUTPUT_ROOT="${OAT_ZERO_EVAL_OUTPUT_ROOT:-$ROOT_DIR/var/artifacts}"
EVAL_SEEDS_CSV="${OAT_ZERO_EVAL_SEEDS:-1001,1002,1003}"
SAMPLE_COUNT="${OAT_ZERO_EVAL_SAMPLE_COUNT:-8}"
TEMPERATURE="${OAT_ZERO_EVAL_TEMPERATURE:-1.0}"
PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
MANIFEST="$ROOT_DIR/var/artifacts/${STAMP_PREFIX}_comparative_jobs.tsv"

# Discover final checkpoints for every arm/seed run of this stamp prefix. The
# submit manifest is authoritative (a loose glob on the prefix would also
# match runs from stamps that merely share it, e.g. "cd" matching "cd_v2").
declare -a run_stamps=()
if [[ -f "$MANIFEST" ]]; then
  while IFS=$'\t' read -r _arm _seed _job run_stamp; do
    [[ "$run_stamp" == "run_stamp" || -z "$run_stamp" ]] && continue
    run_stamps+=("$run_stamp")
  done < "$MANIFEST"
  echo "[comparative-eval] using manifest ${MANIFEST} (${#run_stamps[@]} runs)"
else
  echo "[comparative-eval] WARNING: no manifest at ${MANIFEST}; falling back to glob" >&2
fi

_run_dirs_for_stamp() {
  if [[ ${#run_stamps[@]} -gt 0 ]]; then
    local stamp
    for stamp in "${run_stamps[@]}"; do
      compgen -G "$RUN_DATA_ROOT/oat_zero_tiny_*_${stamp}" || true
    done
  else
    compgen -G "$RUN_DATA_ROOT/oat_zero_tiny_*_${STAMP_PREFIX}_*" || true
  fi
}

checkpoint_specs=()
shopt -s nullglob
while IFS= read -r run_dir; do
  [[ -d "$run_dir" ]] || continue
  # alias = the part of the run stamp after the stamp prefix: <arm>_s<seed>
  alias="${run_dir##*"${STAMP_PREFIX}"_}"
  final_step=""
  final_step_num=-1
  for step_dir in "$run_dir"/saved_models/step_* "$run_dir"/debug_*/saved_models/step_*; do
    [[ -f "$step_dir/config.json" ]] || continue
    step_num="${step_dir##*step_}"
    step_num="$((10#$step_num))"
    if (( step_num > final_step_num )); then
      final_step="$step_dir"
      final_step_num="$step_num"
    fi
  done
  if [[ -n "$final_step" ]]; then
    checkpoint_specs+=("--checkpoint" "${alias}=${final_step}")
    echo "[comparative-eval] ${alias} -> ${final_step}"
  else
    echo "[comparative-eval] WARNING: no vLLM-loadable checkpoint under ${run_dir}" >&2
  fi
done < <(_run_dirs_for_stamp)
shopt -u nullglob

if [[ ${#checkpoint_specs[@]} -eq 0 ]]; then
  echo "No checkpoints found for stamp prefix ${STAMP_PREFIX} under ${RUN_DATA_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a eval_seeds <<< "$EVAL_SEEDS_CSV"

# Each coverage-eval invocation loads its checkpoints into vLLM sequentially
# in one process; chunking bounds peak memory when the comparative has many
# arms (per-alias outputs accumulate under the same stamp-prefix directory).
CHUNK="${OAT_ZERO_EVAL_CHECKPOINT_CHUNK:-8}"
run_coverage_eval() {
  local stamp="$1" sample_count="$2" temperature="$3" sampling_seed="$4"
  local i
  for ((i = 0; i < ${#checkpoint_specs[@]}; i += 2 * CHUNK)); do
    "$PYTHON_BIN" "$ROOT_DIR/ops/eval_exact_answer_mode_coverage.py" \
      --stamp-prefix "$stamp" \
      --data-root "$DATA_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --sample-count "$sample_count" \
      --temperature "$temperature" \
      --seed "$sampling_seed" \
      "${checkpoint_specs[@]:i:2*CHUNK}"
  done
}

# Sampled passes: one coverage-eval sweep per evaluation seed.
for eseed in "${eval_seeds[@]}"; do
  run_coverage_eval "${STAMP_PREFIX}_e${eseed}" "$SAMPLE_COUNT" "$TEMPERATURE" "$eseed"
done

# Greedy pass@1: temperature 0, single sample, seed-independent.
run_coverage_eval "${STAMP_PREFIX}_greedy" 1 0.0 0

echo "[comparative-eval] done; analyze with:"
echo "  $PYTHON_BIN ops/analyze_countdown_comparative.py --output-root $OUTPUT_ROOT --stamp-prefix $STAMP_PREFIX"
