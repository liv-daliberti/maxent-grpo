#!/usr/bin/env bash
# Submit E11's three standard-objective direct-MaxEnt methods across the maintained
# 2 environments x 3 scales x 3 seeds grid. Retired estimators are never
# resumed here, and the analytical grid fails closed on E11's smoke gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${1:-all}"
MAXENT_ARMS=maxent,maxent_control,maxent_dual

common_maxent_env() {
  export OAT_ZERO_ONLY_ARMS="$MAXENT_ARMS"
  export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
  export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
  export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
  export OAT_ZERO_INCLUDE_SEED_ARM=0
  export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
  export OAT_ZERO_MAXENT_ALPHA=0.05
  export OAT_ZERO_MAXENT_CONTROL_RATIO=0.8
  export OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=9.9680345401152
  export OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=64
  export OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=0.5
  export OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=0.9
  export OAT_ZERO_MAXENT_CONTROL_GAIN=1.0
  export OAT_ZERO_MAXENT_DUAL_RATIO=0.8
  export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=9.9680345401152
  export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64
  export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.005
  export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.5
  export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.03
  # Full-vocabulary categorical entropy carries a larger backward graph than
  # the sampled E8 estimator. Keep the effective train batch unchanged while
  # accumulating four-row microbatches on every model scale.
  export OAT_ZERO_MAXENT_TRAIN_BATCH_SIZE_PER_DEVICE=4
  export OAT_ZERO_NUM_PPO_EPOCHS=1
  export OAT_ZERO_SAVE_CKPT=0
  export OAT_ZERO_MAX_SAVE_NUM=1
}

require_e11_smoke() {
  if [[ "${OAT_ZERO_COMPARATIVE_CONFIG_ONLY:-0}" == "1" ]]; then
    echo "[on-policy-maxent] configuration-only mode: smoke gate not evaluated"
    return 0
  fi
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 "$SCRIPT_DIR/check_e11_standard_maxent_smoke.py" \
      --run-data-root "$ROOT_DIR/var/data"
}

launch_standard() {
  local target="$1" stamp="$2" launcher="$3" memory="$4"
  local nodelist="$5" gres="$6" partition="$7" account="$8"
  local n_gpu="$9" adam_offload="${10:-0}" activation_offloading="${11:-0}"

  echo "[on-policy-maxent] target=${target} stamp=${stamp}"
  common_maxent_env
  RUN_STAMP_PREFIX="$stamp" \
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
  OAT_ZERO_TRAIN_BATCH_SIZE="${OAT_ZERO_MAXENT_TRAIN_BATCH_SIZE:-32}" \
  OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="$OAT_ZERO_MAXENT_TRAIN_BATCH_SIZE_PER_DEVICE" \
    bash "$SCRIPT_DIR/$launcher"
}

launch_7b() {
  local target="$1" stamp task data_root max_train eval_prompt_interval
  case "$target" in
    countdown-7b)
      stamp=cde11_standard_maxent_7b_v1
      task=countdown
      data_root="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
      max_train=384
      eval_prompt_interval=96
      ;;
    graph-7b)
      stamp=gce11_standard_maxent_7b_v1
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

  echo "[on-policy-maxent] target=${target} stamp=${stamp}"
  common_maxent_env
  RUN_STAMP_PREFIX="$stamp" \
  OAT_ZERO_7B_ANALYTICAL_APPROVED=1 \
  OAT_ZERO_COMPARATIVE_TASK="$task" \
  OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-7b-instruct \
  OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3 \
  OAT_ZERO_COMPARATIVE_DATA_ROOT="$data_root" \
  OAT_ZERO_TRAIN_SEEDS=43,44,45 \
  OAT_ZERO_NUM_SAMPLES=32 \
  OAT_ZERO_LEARNING_RATE=0.0000002 \
  OAT_ZERO_MAX_TRAIN="$max_train" \
  OAT_ZERO_MAX_QUERIES=100000000 \
  OAT_ZERO_NUM_PROMPT_EPOCH=5 \
  OAT_ZERO_EVAL_PROMPT_INTERVAL="$eval_prompt_interval" \
  OAT_ZERO_EVAL_MODE_COVERAGE_K=8 \
  OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0 \
  OAT_ZERO_TRAIN_NODELIST=node302 \
  OAT_ZERO_TRAIN_GRES=gpu:a100:2 \
  OAT_ZERO_TRAIN_PARTITION=mltheory \
  OAT_ZERO_TRAIN_ACCOUNT=mltheory \
  OAT_ZERO_TRAIN_MEMORY=192G \
  OAT_ZERO_TRAIN_TIME_LIMIT=168:00:00 \
  OAT_ZERO_N_GPU=2 \
  OAT_ZERO_NUM_GPUS_PER_ACTOR=2 \
  OAT_ZERO_ROLLOUT_BATCH_SIZE=2 \
  OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1 \
  OAT_ZERO_TRAIN_BATCH_SIZE=32 \
  OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="$OAT_ZERO_MAXENT_TRAIN_BATCH_SIZE_PER_DEVICE" \
  OAT_ZERO_ADAM_OFFLOAD=1 \
  OAT_ZERO_ACTIVATION_OFFLOADING=1 \
    bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
}

launch_smoke() {
  bash "$SCRIPT_DIR/launch_e11_standard_maxent_smoke.sh" literal
}

launch_one() {
  case "$1" in
    countdown-05b)
      OAT_ZERO_MAXENT_TRAIN_BATCH_SIZE=16 \
      OAT_ZERO_MAXENT_TRAIN_BATCH_SIZE_PER_DEVICE=4 \
        launch_standard "$1" cde11_standard_maxent_05b_v1 \
          launch_e1_05b_cd_pilot.sh 32G node202,node203,node204 \
          gpu:a5000:1 cs allcs 1 0 0
      ;;
    graph-05b)
      OAT_ZERO_MAXENT_TRAIN_BATCH_SIZE=16 \
      OAT_ZERO_MAXENT_TRAIN_BATCH_SIZE_PER_DEVICE=4 \
        launch_standard "$1" gce11_standard_maxent_05b_v1 \
          launch_e1_05b_pilot.sh 32G node202,node203,node204 \
          gpu:a5000:1 cs allcs 1 0 0
      ;;
    countdown-3b)
      launch_standard "$1" cde11_standard_maxent_3b_2xa6000_v1 \
        launch_e1_3b_cd_easy3.sh 96G node205,node206,node207 \
        gpu:a6000:2 cs allcs 2 0 0
      ;;
    graph-3b)
      launch_standard "$1" gce11_standard_maxent_3b_2xa6000_v1 \
        launch_e1_3b.sh 96G node205,node206,node207 \
        gpu:a6000:2 cs allcs 2 0 0
      ;;
    countdown-7b|graph-7b)
      launch_7b "$1"
      ;;
    *)
      echo "Unknown target: $1" >&2
      return 2
      ;;
  esac
}

case "$TARGET" in
  smoke)
    launch_smoke
    ;;
  all)
    require_e11_smoke
    for target in \
      countdown-05b graph-05b \
      countdown-3b graph-3b \
      countdown-7b graph-7b
    do
      launch_one "$target"
    done
    ;;
  *)
    require_e11_smoke
    launch_one "$TARGET"
    ;;
esac
