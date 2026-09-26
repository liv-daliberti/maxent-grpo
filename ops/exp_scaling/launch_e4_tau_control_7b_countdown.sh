#!/usr/bin/env bash
# Full 7B Countdown matrix on non-preemptible CS A6000s: Dr.GRPO, fixed xDr,
# and entropy-feedback xDr at seeds 43--45. Uses the easy3 pool shared by the
# landed 0.5B/3B Countdown scaling rows and a five-pass analytical budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-cde4_taucontrol_7b_full_2xa6000_cs_v1}"
export OAT_ZERO_COMPARATIVE_TASK=countdown
export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"

# Full 384-prompt pool, capped at five complete passes.
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_NUM_PROMPT_EPOCH=5
export OAT_ZERO_EVAL_PROMPT_INTERVAL=96 # quarter of the 384-prompt pool
export OAT_ZERO_SAVE_STEPS=1152
export OAT_ZERO_SAVE_FROM=1152

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node205,node206,node207}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a6000:2}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_TRAIN_PARTITION:-cs}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_TRAIN_ACCOUNT:-allcs}"

export OAT_ZERO_N_GPU="${OAT_ZERO_N_GPU:-2}"
export OAT_ZERO_NUM_GPUS_PER_ACTOR="${OAT_ZERO_NUM_GPUS_PER_ACTOR:-2}"
export OAT_ZERO_ROLLOUT_BATCH_SIZE="${OAT_ZERO_ROLLOUT_BATCH_SIZE:-2}"
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-1}"
export OAT_ZERO_TRAIN_BATCH_SIZE="${OAT_ZERO_TRAIN_BATCH_SIZE:-32}"
# Same-stamp recoveries can lower the microbatch while preserving the global
# batch and analytical objective. This is useful after a long-sequence OOM.
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-16}"

exec "$SCRIPT_DIR/launch_e4_tau_control.sh" 7b
