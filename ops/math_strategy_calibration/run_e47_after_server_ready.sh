#!/usr/bin/env bash
# Resume-safe E47 calibration once the node105 Qwen72 health record appears.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$ROOT_DIR/var/artifacts/e47_math_strategy_calibration_v1"
ENDPOINT="$OUTPUT/qwen72_endpoint.json"
PIPELINE="$ROOT_DIR/ops/math_strategy_calibration/e47_calibration.py"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

for _ in $(seq 1 360); do
  if [[ -f "$ENDPOINT" ]]; then
    break
  fi
  sleep 10
done
if [[ ! -f "$ENDPOINT" ]]; then
  echo "E47 Qwen72 endpoint did not become ready within one hour" >&2
  exit 1
fi

# Each successful per-problem/pass response is atomic. A structural/API
# failure can therefore resume without repeating accepted judge outcomes.
for attempt in 1 2 3; do
  if "$PYTHON_BIN" "$PIPELINE" judge \
    --workers 4 \
    --endpoint "$ENDPOINT"; then
    break
  fi
  if [[ "$attempt" == 3 ]]; then
    exit 1
  fi
  sleep 30
done

"$PYTHON_BIN" "$PIPELINE" analyze
"$PYTHON_BIN" "$PIPELINE" audit-packet
