#!/usr/bin/env bash
set -euo pipefail

: "${E50E_ROUTE_VARIANT:?missing E50E_ROUTE_VARIANT}"

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"
: "${E50E_ENDPOINT_RECORD:?missing E50E_ENDPOINT_RECORD}"
: "${E50E_ROUTE_CALIBRATION:?missing E50E_ROUTE_CALIBRATION}"
: "${E50E_DECLARATION_CALIBRATION:?missing E50E_DECLARATION_CALIBRATION}"
: "${E50E_CONTINUITY_CERTIFICATE:?missing E50E_CONTINUITY_CERTIFICATE}"
prefix=e50e_exact_oat_math_05b_3ep_v1
result="$ROOT_DIR/var/artifacts/e50e_exact_oat_math_result_v1.json"
curve="$ROOT_DIR/var/artifacts/${prefix}_scaling_curve.json"
mechanism="$ROOT_DIR/var/artifacts/e50e_e46_mechanism_comparison_v1.json"

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/exp_scaling/analyze_e50e_exact_oat_math.py" \
  --route-variant "$E50E_ROUTE_VARIANT" \
  --base-route-probe \
    "$ROOT_DIR/var/artifacts/${E50E_ROUTE_VARIANT}_baseline_route_probe_v1/result.json" \
  --control-route-probe \
    "$ROOT_DIR/var/artifacts/${E50E_ROUTE_VARIANT}_full_control_terminal_route_probe_v1/result.json" \
  --treatment-route-probe \
    "$ROOT_DIR/var/artifacts/${E50E_ROUTE_VARIANT}_full_treatment_terminal_route_probe_v1/result.json" \
  --continuity-certificate "$E50E_CONTINUITY_CERTIFICATE" \
  --out "$result"
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/exp_scaling/parse_scaling_curve.py" \
  --stamp-prefix "$prefix" \
  --out "$curve" \
  --prompt-pool-size 384 \
  --num-samples 16 \
  --max-training-passes 3 \
  --eval-splits math
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/exp_scaling/summarize_e49t_e46_mechanism.py" \
  --advancement "$result" \
  --math-curve "$curve" \
  --expected-advancement-schema e50e_exact_oat_math_result_v1 \
  --output-schema e50e_e46_mechanism_comparison_v1 \
  --route-calibration "$E50E_ROUTE_CALIBRATION" \
  --declaration-calibration "$E50E_DECLARATION_CALIBRATION" \
  --out "$mechanism"
