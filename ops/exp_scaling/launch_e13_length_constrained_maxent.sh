#!/usr/bin/env bash
# Submit E13's two projected expected-length-dual engineering arms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

common_env() {
  export OAT_ZERO_ONLY_ARMS=maxent_length_dual
  export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
  export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
  export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
  export OAT_ZERO_INCLUDE_SEED_ARM=0
  export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=1
  export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
  export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
  export OAT_ZERO_TRAIN_SEEDS=9005
  export OAT_ZERO_NUM_SAMPLES=16
  export OAT_ZERO_LEARNING_RATE=0.0000002
  export OAT_ZERO_MAX_TRAIN=128
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
  export OAT_ZERO_TRAIN_TIME_LIMIT=00:45:00
  export OAT_ZERO_TRAIN_BATCH_SIZE=16
  export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
  export OAT_ZERO_NUM_PPO_EPOCHS=1
  export OAT_ZERO_SAVE_CKPT=0
  export OAT_ZERO_MAX_SAVE_NUM=1
  export OAT_ZERO_MAXENT_ALPHA=0.002
  export OAT_ZERO_MAXENT_LENGTH_TARGET=16
  export OAT_ZERO_MAXENT_LENGTH_LAMBDA_INIT=0
  export OAT_ZERO_MAXENT_LENGTH_LAMBDA_MAX=0.02
  export OAT_ZERO_MAXENT_LENGTH_EMA_DECAY=0.9
}

launch_arm() {
  local label="$1" dual_lr="$2"
  local stamp="gce13_length_constrained_maxent_${label}_v1"
  echo "[e13] label=${label} dual_lr=${dual_lr} stamp=${stamp}"
  common_env
  RUN_STAMP_PREFIX="$stamp" \
  OAT_ZERO_MAXENT_LENGTH_DUAL_LR="$dual_lr" \
    bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
}

launch_arm eta5em5 0.00005
launch_arm eta2em4 0.00020
