#!/usr/bin/env bash
# Full analytical 7B E4 campaign: seeds 43--45 for Dr.GRPO, fixed xDr, and
# entropy-feedback xDr. Each job uses one tensor-parallel actor and a ZeRO-2
# learner spanning two A6000s. Requires OAT_ZERO_7B_ANALYTICAL_APPROVED=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-gce4_taucontrol_7b_full_2xa6000_v1}"
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node103,node104,node208}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a6000:2}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_TRAIN_PARTITION:-lowprio}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_TRAIN_ACCOUNT:-allcs}"

export OAT_ZERO_N_GPU=2
export OAT_ZERO_NUM_GPUS_PER_ACTOR=2
export OAT_ZERO_ROLLOUT_BATCH_SIZE=2
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_TRAIN_BATCH_SIZE=32
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=16

exec "$SCRIPT_DIR/launch_e4_tau_control.sh" 7b
