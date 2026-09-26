#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
SCRIPT="$ROOT_DIR/ops/math_strategy_calibration/e47m_semantic_regressions.py"
PROTOCOL="$ROOT_DIR/paper/preregistration/e47p_clean_distinct_route_regression.md"
ENDPOINT="$ROOT_DIR/var/artifacts/e49_math_strategy_qwen72_v1/qwen72_endpoint.json"
OUTPUT_ROOT="$ROOT_DIR/var/artifacts/e47p_clean_distinct_route_regression_v1"
LOG="$OUTPUT_ROOT.log"

cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "=== E47P frozen clean distinct-route regression ==="
sha256sum "$PROTOCOL" "$SCRIPT" \
  "$ROOT_DIR/src/oat_drgrpo/math_strategy_canonicalizer.py"
"$PYTHON_BIN" "$SCRIPT" \
  --endpoint "$ENDPOINT" \
  --output "$OUTPUT_ROOT/semantic_regressions.json"
