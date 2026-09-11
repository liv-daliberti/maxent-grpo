#!/usr/bin/env bash
set -euo pipefail

: "${E50_ANALYSIS_VARIANT:?missing E50_ANALYSIS_VARIANT}"

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"
case "$E50_ANALYSIS_VARIANT" in
  e50b) prefix=e50b_observed_route_math_toy_05b_3ep_v1 ;;
  e50d) prefix=e50d_teacher_route_math_toy_05b_3ep_v1 ;;
  *) echo "invalid E50 analysis variant: $E50_ANALYSIS_VARIANT" >&2; exit 1 ;;
esac
out="$ROOT_DIR/var/artifacts/${E50_ANALYSIS_VARIANT}_matched_math_toy_advancement_v1.json"
curve="$ROOT_DIR/var/artifacts/${prefix}_scaling_curve.json"
mechanism="$ROOT_DIR/var/artifacts/${E50_ANALYSIS_VARIANT}_e46_mechanism_comparison_v1.json"

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/exp_scaling/analyze_e50_matched_math.py" \
  --variant "$E50_ANALYSIS_VARIANT" \
  --stamp-prefix "$prefix" \
  --base-route-probe \
    "$ROOT_DIR/var/artifacts/${E50_ANALYSIS_VARIANT}_baseline_route_probe_v1/result.json" \
  --control-route-probe \
    "$ROOT_DIR/var/artifacts/${E50_ANALYSIS_VARIANT}_control_terminal_route_probe_v1/result.json" \
  --treatment-route-probe \
    "$ROOT_DIR/var/artifacts/${E50_ANALYSIS_VARIANT}_treatment_terminal_route_probe_v1/result.json" \
  --out "$out"

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/exp_scaling/parse_scaling_curve.py" \
  --stamp-prefix "$prefix" \
  --out "$curve" \
  --prompt-pool-size 50 \
  --num-samples 16 \
  --max-training-passes 3 \
  --eval-splits math
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/exp_scaling/summarize_e49t_e46_mechanism.py" \
  --advancement "$out" \
  --math-curve "$curve" \
  --expected-advancement-schema e50_matched_math_toy_advancement_v1 \
  --output-schema e50_e46_mechanism_comparison_v1 \
  --out "$mechanism"

if jq -e \
  '.complete_evidence == true
   and .advance_to_exact_oat_full == true
   and ([.checks[]] | all)' \
  "$out" >/dev/null \
  && jq -e '.qualitative_mechanism_match == true' \
    "$mechanism" >/dev/null; then
  bash \
    "$ROOT_DIR/ops/math_strategy_calibration/launch_e50e_exact_oat_materialization.sh"
else
  echo "$E50_ANALYSIS_VARIANT did not pass; exact full run remains gated"
fi
