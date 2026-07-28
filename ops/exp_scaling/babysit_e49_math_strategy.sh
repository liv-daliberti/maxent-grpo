#!/usr/bin/env bash
# Advance E49B from the frozen toy gate to full MATH only after a clean pass.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TOY_PREFIX=e49b_math_strategy_toy_05b_v1
FULL_PREFIX=e49b_math_strategy_full_05b_v1
TOY_GRPO="$ROOT_DIR/var/data/xdr_qwen25_0p5b_instruct_grpo_${TOY_PREFIX}_grpo_s45"
TOY_MAXENT="$ROOT_DIR/var/data/xdr_qwen25_0p5b_instruct_online_canonical_haarnoja_${TOY_PREFIX}_online_canonical_haarnoja_s45"
FULL_GRPO="$ROOT_DIR/var/data/xdr_qwen25_0p5b_instruct_grpo_${FULL_PREFIX}_grpo_s45"
FULL_MAXENT="$ROOT_DIR/var/data/xdr_qwen25_0p5b_instruct_online_canonical_haarnoja_${FULL_PREFIX}_online_canonical_haarnoja_s45"
FULL_MANIFEST="$ROOT_DIR/var/artifacts/${FULL_PREFIX}_comparative_jobs.tsv"
CONTINUITY_DIR="$ROOT_DIR/var/artifacts/e49_math_strategy_qwen72_continuity_v1"
CONTINUITY_RECORD="$CONTINUITY_DIR/qwen72_endpoint.json"
CONTINUITY_JOB_RECORD="$CONTINUITY_DIR/server_job.json"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

wait_for_pair() {
  local stage="$1"
  local left="$2"
  local right="$3"
  while [[ ! -f "$left/TRAINING_COMPLETE.json" \
    || ! -f "$right/TRAINING_COMPLETE.json" ]]; do
    log "waiting for matched ${stage} completion"
    sleep 60
  done
  log "matched ${stage} completion markers observed"
}

wait_for_pair toy "$TOY_GRPO" "$TOY_MAXENT"
if ! python "$ROOT_DIR/ops/exp_scaling/analyze_e49_math_strategy.py" \
  --stage toy; then
  log "toy gate failed; preserving evidence for a prospective iteration"
  exit 2
fi

if [[ ! -f "$FULL_MANIFEST" ]]; then
  if [[ ! -f "$CONTINUITY_JOB_RECORD" ]]; then
    log "toy gate passed; submitting a fresh identical full-stage judge"
    bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e49_qwen72_continuity_node105.sh"
  fi
  for _ in $(seq 1 180); do
    if [[ -f "$CONTINUITY_RECORD" ]]; then
      break
    fi
    log "waiting for the full-stage judge to become ready"
    sleep 10
  done
  if [[ ! -f "$CONTINUITY_RECORD" ]]; then
    log "full-stage judge did not become ready within 30 minutes"
    exit 2
  fi
  log "fresh judge is ready; launching exact OAT MATH-500 stage"
  E49B_QWEN72_ENDPOINT_RECORD="$CONTINUITY_RECORD" \
    bash "$ROOT_DIR/ops/exp_scaling/launch_e49_validator_bound_math_strategy_05b.sh" \
      full
else
  log "full manifest already exists; refusing to submit a duplicate cohort"
fi

if tmux has-session -t e49b_math_monitor 2>/dev/null; then
  tmux kill-session -t e49b_math_monitor
fi
tmux new-session -d -s e49b_math_monitor \
  "bash '$ROOT_DIR/ops/exp_scaling/monitor_e49_math_strategy.sh' full"
log "full 60-second dashboard monitor is live"

wait_for_pair full "$FULL_GRPO" "$FULL_MAXENT"
if python "$ROOT_DIR/ops/exp_scaling/analyze_e49_math_strategy.py" \
  --stage full; then
  log "full E49B gate passed"
else
  log "full E49B gate failed; preserving evidence for a prospective iteration"
  exit 2
fi
