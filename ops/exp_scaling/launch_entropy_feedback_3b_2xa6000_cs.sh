#!/usr/bin/env bash
# Run only the missing 3B entropy-feedback arms on non-preemptible CS A6000s.
# Landed Dr.GRPO and fixed-xDr controls are reused; no controls are duplicated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"

launch_one() {
  local target="$1"
  local stamp launcher recovery_save_steps
  case "$target" in
    countdown)
      stamp=cde4_taucontrol_3b_feedback_2xa6000_cs_v1
      launcher=launch_e1_3b_cd_easy3.sh
      recovery_save_steps=384
      ;;
    graph)
      stamp=gce4_taucontrol_3b_feedback_2xa6000_cs_v1
      launcher=launch_e1_3b.sh
      recovery_save_steps=1024
      ;;
    *)
      echo "Unknown target: $target (use all, countdown, or graph)." >&2
      return 2
      ;;
  esac

  RUN_STAMP_PREFIX="$stamp" \
  OAT_ZERO_ONLY_ARMS=xdr_tau_control \
  OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=1 \
  OAT_ZERO_SAVE_CKPT=1 \
  OAT_ZERO_SAVE_STEPS="$recovery_save_steps" \
  OAT_ZERO_SAVE_FROM="$recovery_save_steps" \
  OAT_ZERO_MAX_SAVE_NUM=1 \
  OAT_ZERO_TRAIN_NODELIST=node205,node206,node207 \
  OAT_ZERO_TRAIN_GRES=gpu:a6000:2 \
  OAT_ZERO_TRAIN_PARTITION=cs \
  OAT_ZERO_TRAIN_ACCOUNT=allcs \
  OAT_ZERO_TRAIN_MEMORY=96G \
  OAT_ZERO_N_GPU=2 \
  OAT_ZERO_NUM_GPUS_PER_ACTOR=2 \
  OAT_ZERO_ROLLOUT_BATCH_SIZE=2 \
  OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1 \
  OAT_ZERO_TRAIN_BATCH_SIZE=32 \
  OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=16 \
    bash "$SCRIPT_DIR/$launcher"
}

if [[ "$TARGET" == "all" ]]; then
  launch_one countdown
  launch_one graph
else
  launch_one "$TARGET"
fi
