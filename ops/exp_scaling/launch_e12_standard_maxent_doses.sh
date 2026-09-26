#!/usr/bin/env bash
# Submit E12's four fixed standard-MaxEnt coefficient calibration arms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
}

launch_dose() {
  local label="$1" alpha="$2"
  # v1/a0p0005 was reserved by a scheduler-inaccessible dry launch.  Use one
  # fresh immutable namespace for all four actual calibration jobs.
  local stamp="gce12_standard_maxent_dose_${label}_v2"
  echo "[e12] label=${label} alpha=${alpha} stamp=${stamp}"
  common_env
  RUN_STAMP_PREFIX="$stamp" \
  OAT_ZERO_MAXENT_ALPHA="$alpha" \
    bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
}

launch_dose a0p0005 0.0005
launch_dose a0p0010 0.0010
launch_dose a0p0015 0.0015
launch_dose a0p0020 0.0020
