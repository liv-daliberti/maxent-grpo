#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PIPELINE="$ROOT_DIR/ops/math_strategy_calibration/e47b_bounded_calibration.py"
REGRESSIONS="$ROOT_DIR/ops/math_strategy_calibration/e47s_reduced_veto_regressions.py"
PROTOCOL="$ROOT_DIR/paper/preregistration/e47w_pairwise_math_strategy_calibration.md"
ENDPOINT="$ROOT_DIR/var/artifacts/e49_math_strategy_qwen72_v1/qwen72_endpoint.json"
OUTPUT="$ROOT_DIR/var/artifacts/e47w_pairwise_math_strategy_calibration_v1"
LOG="$ROOT_DIR/var/artifacts/e47w_pairwise_math_strategy_calibration_v1.log"

cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export E47_BOUNDED_PROTOCOL_ID=E47W-CAL
export E47_BOUNDED_SCHEMA_PREFIX=e47w_pairwise_veto

if [[ ! -x "$PYTHON_BIN" ]] \
  || [[ ! -f "$ENDPOINT" ]] \
  || ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E47W frozen prerequisite is missing" >&2
  exit 2
fi

mkdir -p "$OUTPUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== E47W frozen pairwise-veto calibration ==="
sha256sum "$PROTOCOL" "$PIPELINE" "$REGRESSIONS" \
  "$ROOT_DIR/src/oat_drgrpo/math_strategy_canonicalizer.py"
"$PYTHON_BIN" "$REGRESSIONS" \
  --endpoint "$ENDPOINT" \
  --output "$OUTPUT/semantic_regressions.json"
"$PYTHON_BIN" "$PIPELINE" run \
  --output "$OUTPUT" \
  --endpoint "$ENDPOINT" \
  --workers 4 \
  --timeout 900
"$PYTHON_BIN" "$PIPELINE" analyze --output "$OUTPUT"
"$PYTHON_BIN" "$PIPELINE" audit-packet --output "$OUTPUT"
echo "=== E47W automatic calibration complete; blinded audit ready ==="
