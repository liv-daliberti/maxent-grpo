#!/usr/bin/env bash
# Freeze and submit E49K's eval-only distinct-route buffer.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export E49H_EXPERIMENT=e49k
export E49H_EXPECTED_TRAIN=0
export E49H_EXPECTED_EVAL=10
export E49H_STATUS_MARKER=E49K
export E49H_PROTOCOL="$ROOT_DIR/paper/preregistration/e49k_curated_eval_buffer_20260724.md"
export E49H_AMENDMENT="$E49H_PROTOCOL"
export E49H_CONTRACTS_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49k_curated_distinct_routes_eval_buffer.json"
export E49H_OUTPUT_OVERRIDE="$ROOT_DIR/var/artifacts/e49k_curated_eval_buffer_v1"
exec bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e49h_curated_routes.sh"
