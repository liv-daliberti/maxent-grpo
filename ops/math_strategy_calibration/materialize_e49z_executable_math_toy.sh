#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULT="$ROOT_DIR/var/artifacts/e49y_compiled_bottom_up_route_calibration_v1/result.json"
OUTPUT="$ROOT_DIR/var/data/e49z_executable_route_math_toy"
if [[ ! -f "$RESULT" ]]; then
  echo "E49Y result is missing: $RESULT" >&2
  exit 1
fi
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - "$RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    result.get("schema")
    != "e49y_compiled_bottom_up_route_calibration_v1"
    or result.get("pass") is not True
    or len(result.get("selected_problem_ids") or []) != 10
    or result.get("bidirectionally_executable_count", 0) < 10
):
    raise SystemExit("E49Y has not passed its frozen route gate")
PY
if [[ -e "$OUTPUT" ]]; then
  echo "Fresh E49Z output required: $OUTPUT" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/materialize_e49z_executable_math_toy.py" \
  --output-root "$OUTPUT"
