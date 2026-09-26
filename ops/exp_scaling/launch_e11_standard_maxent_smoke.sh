#!/usr/bin/env bash
# Submit E11's literal standard-alpha gate or gated three-arm comparison.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${1:-literal}"
RAW_TARGET=9.9680345401152

common_env() {
  export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
  export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
  export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
  export OAT_ZERO_INCLUDE_SEED_ARM=0
  export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
  export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
  export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
  export OAT_ZERO_TRAIN_SEEDS=9005
  export OAT_ZERO_NUM_SAMPLES=16
  export OAT_ZERO_LEARNING_RATE=0.0000002
  export OAT_ZERO_MAX_QUERIES=100000000
  export OAT_ZERO_NUM_PROMPT_EPOCH=1
  export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
  export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0
  export OAT_ZERO_TRAIN_NODELIST=node023,node024
  export OAT_ZERO_TRAIN_GRES=gpu:rtx_3090:1
  export OAT_ZERO_TRAIN_PARTITION=lowprio
  export OAT_ZERO_TRAIN_ACCOUNT=allcs
  export OAT_ZERO_TRAIN_MEMORY=32G
  export OAT_ZERO_TRAIN_BATCH_SIZE=16
  export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
  export OAT_ZERO_NUM_PPO_EPOCHS=1
  export OAT_ZERO_SAVE_CKPT=0
  export OAT_ZERO_MAX_SAVE_NUM=1
}

launch_literal() {
  echo "[e11] literal standard-MaxEnt alpha=0.05 gate"
  common_env
  export OAT_ZERO_ONLY_ARMS=maxent
  export OAT_ZERO_INCLUDE_MAXENT_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
  RUN_STAMP_PREFIX=gce11_standard_maxent_literal_a0p05_v1 \
  OAT_ZERO_MAXENT_ALPHA=0.05 \
  OAT_ZERO_MAX_TRAIN=32 \
  OAT_ZERO_EVAL_PROMPT_INTERVAL=16 \
  OAT_ZERO_TRAIN_TIME_LIMIT=00:20:00 \
    bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
}

launch_comparative() {
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 "$SCRIPT_DIR/check_e11_standard_maxent_smoke.py" \
      --run-data-root "$ROOT_DIR/var/data" --stage literal
  echo "[e11] standard-MaxEnt three-arm smoke"
  common_env
  export OAT_ZERO_ONLY_ARMS=maxent,maxent_control,maxent_dual
  export OAT_ZERO_INCLUDE_MAXENT_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
  RUN_STAMP_PREFIX=gce11_standard_maxent_smoke_v1 \
  OAT_ZERO_MAXENT_ALPHA=0.05 \
  OAT_ZERO_MAXENT_CONTROL_RATIO=0.8 \
  OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY="$RAW_TARGET" \
  OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=64 \
  OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=0.5 \
  OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=0.9 \
  OAT_ZERO_MAXENT_CONTROL_GAIN=1.0 \
  OAT_ZERO_MAXENT_DUAL_RATIO=0.8 \
  OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$RAW_TARGET" \
  OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64 \
  OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.005 \
  OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.5 \
  OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.03 \
  OAT_ZERO_MAX_TRAIN=128 \
  OAT_ZERO_EVAL_PROMPT_INTERVAL=32 \
  OAT_ZERO_TRAIN_TIME_LIMIT=00:45:00 \
    bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
}

case "$TARGET" in
  literal)
    launch_literal
    ;;
  comparative)
    launch_comparative
    ;;
  *)
    echo "Use literal or comparative." >&2
    exit 2
    ;;
esac
