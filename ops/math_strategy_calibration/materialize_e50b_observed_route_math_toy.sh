#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

OUTPUT="$ROOT_DIR/var/data/e50b_observed_route_math_toy"
if [[ -e "$OUTPUT" ]]; then
  echo "fresh E50B output required: $OUTPUT" >&2
  exit 1
fi

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/materialize_e50b_observed_route_math_toy.py" \
  --output-root "$OUTPUT"
