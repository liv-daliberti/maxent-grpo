#!/usr/bin/env bash
# E9b Stage A: frozen-policy entropy calibration and fixed-alpha dose smokes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${1:-stage-a}"

common_env() {
  export OAT_ZERO_ONLY_ARMS=maxent
  export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
  export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
  export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
  export OAT_ZERO_INCLUDE_SEED_ARM=0
  export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
  export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
  export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
  export OAT_ZERO_TRAIN_SEEDS=9005
  export OAT_ZERO_NUM_SAMPLES=16
  export OAT_ZERO_MAX_QUERIES=100000000
  export OAT_ZERO_NUM_PROMPT_EPOCH=1
  export OAT_ZERO_EVAL_PROMPT_INTERVAL=32
  export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
  export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0
  export OAT_ZERO_TRAIN_NODELIST=node023,node024
  export OAT_ZERO_TRAIN_GRES=gpu:rtx_3090:1
  export OAT_ZERO_TRAIN_PARTITION=lowprio
  export OAT_ZERO_TRAIN_ACCOUNT=allcs
  export OAT_ZERO_TRAIN_MEMORY=32G
  export OAT_ZERO_TRAIN_TIME_LIMIT=02:00:00
  export OAT_ZERO_TRAIN_BATCH_SIZE=16
  export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
  export OAT_ZERO_NUM_PPO_EPOCHS=1
  export OAT_ZERO_SAVE_CKPT=0
  export OAT_ZERO_MAX_SAVE_NUM=1
}

submit_fixed() {
  local stamp="$1" alpha="$2" learning_rate="$3" max_train="$4"
  echo "[e9b] stamp=${stamp} alpha=${alpha} lr=${learning_rate} max_train=${max_train}"
  common_env
  RUN_STAMP_PREFIX="$stamp" \
  OAT_ZERO_MAXENT_ALPHA="$alpha" \
  OAT_ZERO_LEARNING_RATE="$learning_rate" \
  OAT_ZERO_MAX_TRAIN="$max_train" \
    bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
}

submit_adaptive() {
  local target_entropy
  target_entropy="$(python3 "$SCRIPT_DIR/check_e9b_maxent_actuator_smoke.py" \
    stage-a --run-data-root "$ROOT_DIR/var/data" --target-only)"
  echo "[e9b] adaptive target_entropy=${target_entropy}"
  common_env
  export OAT_ZERO_ONLY_ARMS=maxent_control,maxent_dual
  export OAT_ZERO_INCLUDE_MAXENT_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
  RUN_STAMP_PREFIX=gce9b_maxent_adaptive_v1 \
  OAT_ZERO_MAXENT_ALPHA=0.05 \
  OAT_ZERO_MAXENT_CONTROL_RATIO=0.8 \
  OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY="$target_entropy" \
  OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=64 \
  OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=0.5 \
  OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=0.9 \
  OAT_ZERO_MAXENT_CONTROL_GAIN=1.0 \
  OAT_ZERO_MAXENT_DUAL_RATIO=0.8 \
  OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$target_entropy" \
  OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64 \
  OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.005 \
  OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.5 \
  OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.03 \
  OAT_ZERO_LEARNING_RATE=0.0000002 \
  OAT_ZERO_MAX_TRAIN=128 \
    bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
}

case "$TARGET" in
  calibration)
    submit_fixed gce9b_maxent_calibration_v1 0.05 0.0 64
    ;;
  dose)
    submit_fixed gce9b_maxent_dose_a0p05_v1 0.05 0.0000002 128
    submit_fixed gce9b_maxent_dose_a0p10_v1 0.10 0.0000002 128
    submit_fixed gce9b_maxent_dose_a0p20_v1 0.20 0.0000002 128
    submit_fixed gce9b_maxent_dose_a0p50_v1 0.50 0.0000002 128
    ;;
  stage-a)
    bash "$0" calibration
    bash "$0" dose
    ;;
  stage-b)
    submit_adaptive
    ;;
  *)
    echo "Use calibration, dose, stage-a, or stage-b." >&2
    exit 2
    ;;
esac
