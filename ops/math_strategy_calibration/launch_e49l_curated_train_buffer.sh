#!/usr/bin/env bash
# Freeze and submit E49L's train-only distinct-route buffer.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export E49H_EXPERIMENT=e49l
export E49H_EXPECTED_TRAIN=5
export E49H_EXPECTED_EVAL=0
export E49H_STATUS_MARKER=E49L
export E49H_PROTOCOL="$ROOT_DIR/paper/preregistration/e49l_curated_train_buffer_20260724.md"
export E49H_AMENDMENT="$E49H_PROTOCOL"
export E49H_CONTRACTS_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49l_curated_distinct_routes_train_buffer.json"
export E49H_OUTPUT_OVERRIDE="$ROOT_DIR/var/artifacts/e49l_curated_train_buffer_v1"
exec bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e49h_curated_routes.sh"
