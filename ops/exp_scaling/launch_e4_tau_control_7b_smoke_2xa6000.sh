#!/usr/bin/env bash
# Operational 7B smoke: one tensor-parallel actor spanning two A6000s.
# The dedicated stamp keeps this retry separate from the failed single-GPU
# attempt. As with every 7B smoke, seed 9001 is excluded from outcome analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-gce4_taucontrol_7b_smoke_2xa6000_v6}"
# The dedicated CS partition can leave long smoke jobs outside its backfill
# window. These idle A6000 nodes are available through `lowprio` under the
# same allcs account; Slurm may place each arm independently. Low-priority
# jobs are requeued if an owning partition needs the node.
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node103,node104,node208}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a6000:2}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_TRAIN_PARTITION:-lowprio}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_TRAIN_ACCOUNT:-allcs}"

# With collocation enabled, both the actor and learner see GPUs 0 and 1. The
# actor uses tensor parallelism across the pair instead of placing a complete
# vLLM copy on each 48 GB card.
export OAT_ZERO_N_GPU=2
export OAT_ZERO_NUM_GPUS_PER_ACTOR=2

# OAT requires the global rollout batch to be divisible by both the actor
# count and the tensor-parallel width. Keep one rollout per learner device,
# but make the global smoke batch span the two-device actor.
export OAT_ZERO_ROLLOUT_BATCH_SIZE=2
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1

# Preserve the original global train batch of 32 while sharding it over the
# two learner ranks. DeepSpeed requires:
#   global batch = micro batch per GPU * data-parallel world size * grad accum.
export OAT_ZERO_TRAIN_BATCH_SIZE=32
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=16

exec "$SCRIPT_DIR/launch_e4_tau_control.sh" 7b-smoke
