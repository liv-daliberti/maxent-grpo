#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

: "${E50E_ENDPOINT_RECORD:?missing E50E_ENDPOINT_RECORD}"
: "${E50E_ROUTE_CALIBRATION:?missing E50E_ROUTE_CALIBRATION}"
: "${E50E_DECLARATION_CALIBRATION:?missing E50E_DECLARATION_CALIBRATION}"
: "${E50E_CONTINUITY_CERTIFICATE:?missing E50E_CONTINUITY_CERTIFICATE}"

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/materialize_e50e_exact_oat_math.py" \
  --output-root "$ROOT_DIR/var/data/e50e_exact_oat_math_full" \
  --evidence-root "$ROOT_DIR/var/artifacts/e50e_exact_oat_math_full_v1" \
  --endpoint-record "$E50E_ENDPOINT_RECORD" \
  --workers 4 \
  --timeout 1200

bash "$ROOT_DIR/ops/exp_scaling/launch_e50e_exact_oat_math_05b.sh" run
