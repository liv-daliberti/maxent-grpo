#!/usr/bin/env bash
# Freeze and submit E49N's eval recovery cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export E49H_EXPERIMENT=e49n
export E49H_EXPECTED_TRAIN=0
export E49H_EXPECTED_EVAL=6
export E49H_STATUS_MARKER=E49N
export E49H_PROTOCOL="$ROOT_DIR/paper/preregistration/e49n_curated_eval_recovery_20260724.md"
export E49H_AMENDMENT="$E49H_PROTOCOL"
export E49H_CONTRACTS_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49n_curated_distinct_routes_eval_recovery.json"
export E49H_OUTPUT_OVERRIDE="$ROOT_DIR/var/artifacts/e49n_curated_eval_recovery_v1"
exec bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e49h_curated_routes.sh"
