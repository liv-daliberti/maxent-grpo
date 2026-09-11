#!/usr/bin/env bash
# Add only the prospective Haarnoja-style entropy-dual xDr arm to the existing
# 2 environments x 3 scales x 3 seeds compute-divergence grid. Existing
# Dr.GRPO, fixed-xDr, and proportional-feedback controls are reused.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${1:-all}"

launch_e1_extension() {
  local target="$1"
  local stamp="$2"
  local launcher="$3"
  local memory="$4"
  local nodelist="$5"
  local gres="$6"
  local partition="$7"
  local account="$8"
  local n_gpu="$9"
  local adam_offload="${10:-0}"
  local activation_offloading="${11:-0}"

  echo "[haarnoja-extension] target=${target} stamp=${stamp}"
  RUN_STAMP_PREFIX="$stamp" \
  OAT_ZERO_ONLY_ARMS=xdr_sac_dual \
  OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=1 \
  OAT_ZERO_XDR_SAC_DUAL_BASE=0.05 \
  OAT_ZERO_XDR_SAC_DUAL_RATIO=0.8 \
  OAT_ZERO_XDR_SAC_DUAL_WARMUP_STEPS=64 \
  OAT_ZERO_XDR_SAC_DUAL_MIN_TAU=0.005 \
  OAT_ZERO_XDR_SAC_DUAL_MAX_TAU=0.5 \
  OAT_ZERO_XDR_SAC_DUAL_ALPHA_LR=0.003 \
  OAT_ZERO_SAVE_CKPT=0 \
  OAT_ZERO_MAX_SAVE_NUM=1 \
  OAT_ZERO_TRAIN_MEMORY="$memory" \
  OAT_ZERO_TRAIN_NODELIST="$nodelist" \
  OAT_ZERO_TRAIN_GRES="$gres" \
  OAT_ZERO_TRAIN_PARTITION="$partition" \
  OAT_ZERO_TRAIN_ACCOUNT="$account" \
  OAT_ZERO_N_GPU="$n_gpu" \
  OAT_ZERO_NUM_GPUS_PER_ACTOR="$n_gpu" \
  OAT_ZERO_ADAM_OFFLOAD="$adam_offload" \
  OAT_ZERO_ACTIVATION_OFFLOADING="$activation_offloading" \
  OAT_ZERO_ROLLOUT_BATCH_SIZE="$n_gpu" \
  OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1 \
  OAT_ZERO_TRAIN_BATCH_SIZE="${OAT_ZERO_HAARNOJA_TRAIN_BATCH_SIZE:-32}" \
  OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_HAARNOJA_TRAIN_BATCH_SIZE_PER_DEVICE:-16}" \
    bash "$SCRIPT_DIR/$launcher"
}

launch_7b_extension() {
  local target="$1"
  local stamp task data_root max_train eval_prompt_interval
  case "$target" in
    countdown-7b)
      stamp=cde5_haarnoja_7b_v1
      task=countdown
      data_root="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
      max_train=384
      eval_prompt_interval=96
      ;;
    graph-7b)
      stamp=gce5_haarnoja_7b_v1
      task=graph_coloring
      data_root="$ROOT_DIR/var/data/exact_gc_large_probe"
      max_train=1024
      eval_prompt_interval=256
      ;;
    *)
      echo "Unknown 7B target: $target" >&2
      return 2
      ;;
  esac

  echo "[haarnoja-extension] target=${target} stamp=${stamp}"
  RUN_STAMP_PREFIX="$stamp" \
  OAT_ZERO_ONLY_ARMS=xdr_sac_dual \
  OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=1 \
  OAT_ZERO_XDR_SAC_DUAL_BASE=0.05 \
  OAT_ZERO_XDR_SAC_DUAL_RATIO=0.8 \
  OAT_ZERO_XDR_SAC_DUAL_WARMUP_STEPS=64 \
  OAT_ZERO_XDR_SAC_DUAL_MIN_TAU=0.005 \
  OAT_ZERO_XDR_SAC_DUAL_MAX_TAU=0.5 \
  OAT_ZERO_XDR_SAC_DUAL_ALPHA_LR=0.003 \
  OAT_ZERO_7B_ANALYTICAL_APPROVED=1 \
  OAT_ZERO_COMPARATIVE_TASK="$task" \
  OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3 \
  OAT_ZERO_COMPARATIVE_DATA_ROOT="$data_root" \
  OAT_ZERO_MAX_TRAIN="$max_train" \
  OAT_ZERO_NUM_PROMPT_EPOCH=5 \
  OAT_ZERO_EVAL_PROMPT_INTERVAL="$eval_prompt_interval" \
  OAT_ZERO_SAVE_CKPT=0 \
  OAT_ZERO_MAX_SAVE_NUM=1 \
  OAT_ZERO_TRAIN_NODELIST=node302 \
  OAT_ZERO_TRAIN_GRES=gpu:a100:2 \
  OAT_ZERO_TRAIN_PARTITION=mltheory \
  OAT_ZERO_TRAIN_ACCOUNT=mltheory \
  OAT_ZERO_TRAIN_MEMORY=192G \
  OAT_ZERO_N_GPU=2 \
  OAT_ZERO_NUM_GPUS_PER_ACTOR=2 \
  OAT_ZERO_ROLLOUT_BATCH_SIZE=2 \
  OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1 \
  OAT_ZERO_TRAIN_BATCH_SIZE=32 \
  OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=16 \
  OAT_ZERO_ADAM_OFFLOAD=1 \
  OAT_ZERO_ACTIVATION_OFFLOADING=1 \
    bash "$SCRIPT_DIR/launch_e4_tau_control.sh" 7b
}

launch_one() {
  case "$1" in
    countdown-05b)
      OAT_ZERO_HAARNOJA_TRAIN_BATCH_SIZE=16 \
      OAT_ZERO_HAARNOJA_TRAIN_BATCH_SIZE_PER_DEVICE=16 \
        launch_e1_extension "$1" cde5_haarnoja_05b_v1 \
          launch_e1_05b_cd_pilot.sh 32G node202,node203,node204 \
          gpu:a5000:1 cs allcs 1 0 0
      ;;
    graph-05b)
      OAT_ZERO_HAARNOJA_TRAIN_BATCH_SIZE=16 \
      OAT_ZERO_HAARNOJA_TRAIN_BATCH_SIZE_PER_DEVICE=16 \
        launch_e1_extension "$1" gce5_haarnoja_05b_v1 \
          launch_e1_05b_pilot.sh 32G node202,node203,node204 \
          gpu:a5000:1 cs allcs 1 0 0
      ;;
    countdown-3b)
      launch_e1_extension "$1" cde5_haarnoja_3b_2xa5000_v1 \
        launch_e1_3b_cd_easy3.sh 96G node105 \
        gpu:a5000:2 mltheory mltheory 2 1 1
      ;;
    graph-3b)
      launch_e1_extension "$1" gce5_haarnoja_3b_2xa5000_v1 \
        launch_e1_3b.sh 96G node105 \
        gpu:a5000:2 mltheory mltheory 2 1 1
      ;;
    countdown-7b|graph-7b)
      launch_7b_extension "$1"
      ;;
    *)
      echo "Unknown target: $1" >&2
      echo "Use all, countdown-{05b,3b,7b}, or graph-{05b,3b,7b}." >&2
      return 2
      ;;
  esac
}

if [[ "$TARGET" == all ]]; then
  for target in \
    countdown-05b graph-05b \
    countdown-3b graph-3b \
    countdown-7b graph-7b
  do
    launch_one "$target"
  done
else
  launch_one "$TARGET"
fi
