#!/usr/bin/env bash
# Refresh E65 curves, fail-closed audit, and the shared five-domain figure.
set -u

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$PWD}"
PYTHON_BIN="${OAT_ZERO_E65_MONITOR_PYTHON:-/usr/local/anaconda3/2024.02/bin/python}"
CHECKPOINT_PYTHON="${OAT_ZERO_E68_CHECKPOINT_AUDIT_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
INTERVAL_SECONDS="${OAT_ZERO_E65_MONITOR_INTERVAL_SECONDS:-60}"
AUDIT="$ROOT_DIR/var/artifacts/e65_five_domain_confirmation_audit_latest.json"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/e65-monitor-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/e65-monitor-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$ROOT_DIR"

while true; do
  date
  "$PYTHON_BIN" ops/exp_scaling/audit_e61r1_e58_vs_grpo_12pass.py || true
  "$PYTHON_BIN" ops/exp_scaling/audit_e64_math500_realism_matched.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e64_math500_verifier_sensitivity.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e65r1_objective_mismatch.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e67_preoptimizer_invalidation.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e68_separated_support_actuator_ablation.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e68_preintervention_equivalence.py || true
  "$CHECKPOINT_PYTHON" \
    ops/exp_scaling/audit_e68_checkpoint_separation.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/analyze_e68_paired_prompt_uncertainty_v2.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e66_same_plumbing_actuator_ablation.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e65_eval_cadence.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e65_fixed_checkpoint_coverage.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e65_five_domain_confirmation.py || true
  for prefix in \
    gce61r1_e58_vs_grpo_05b_12ep \
    cde61r1_e58_vs_grpo_05b_12ep \
    pye61r1_e58_vs_grpo_05b_12ep \
    mie61r1_e58_vs_grpo_05b_12ep \
    gce68_separated_support_actuator_05b_12ep \
    cde68_separated_support_actuator_05b_12ep \
    pye68_separated_support_actuator_05b_12ep \
    mie68_separated_support_actuator_05b_12ep \
    gce66_same_plumbing_control_05b_12ep \
    cde66_same_plumbing_control_05b_12ep \
    pye66_same_plumbing_control_05b_12ep \
    mie66_same_plumbing_control_05b_12ep; do
    "$PYTHON_BIN" ops/exp_scaling/parse_scaling_curve.py \
      --stamp-prefix "$prefix" || true
  done
  "$PYTHON_BIN" \
    ops/exp_scaling/summarize_e65_five_domain_confirmation.py || true
  "$PYTHON_BIN" \
    ops/exp_scaling/audit_e65_legitimate_result_readiness.py || true
  "$PYTHON_BIN" ops/exp_scaling/plot_e64_math500_realism.py || true
  "$PYTHON_BIN" ops/exp_scaling/plot_e61r1_e58_vs_grpo_12pass.py || true
  "$PYTHON_BIN" ops/exp_scaling/plot_e65_all_epoch_diagnostic.py || true

  status="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
      "$AUDIT" 2>/dev/null || true
  )"
  if [[ "$status" == pass ]]; then
    echo "[e65-monitor] five-domain campaign has a terminal clean audit; exiting"
    exit 0
  fi
  if [[ "$status" == fail ]]; then
    echo "[e65-monitor] five-domain campaign has a fail-closed violation; exiting" >&2
    exit 1
  fi
  sleep "$INTERVAL_SECONDS"
done
