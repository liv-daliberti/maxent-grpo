#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

OUTPUT="$ROOT_DIR/var/data/e50d_teacher_route_math_toy"
result="$ROOT_DIR/var/artifacts/e50g_safe_signature_teacher_route_calibration_v1/result.json"
if [[ ! -f "$result" ]] || ! jq -e \
  '.schema == "e50g_safe_signature_teacher_route_calibration_v1"
   and .pass == true
   and (.selected_source_indices | length) == 10
   and .bidirectionally_executable_count >= 10
   and .wrong_route_success_count == 0
   and .cross_rendering_false_new_count == 0
   and ([.checks[]] | all)' \
  "$result" >/dev/null; then
  echo "E50D requires the passing E50G safe-signature result" >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "fresh E50D output required: $OUTPUT" >&2
  exit 1
fi

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/materialize_e50d_teacher_route_math_toy.py" \
  --output-root "$OUTPUT"
