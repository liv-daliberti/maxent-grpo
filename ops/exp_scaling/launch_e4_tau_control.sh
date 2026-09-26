#!/usr/bin/env bash
# Single scale-aware entry point for the staged E4 entropy-feedback experiment.
# This script submits jobs; use the 7B smoke target before approving a full 7B run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCALE="${1:-}"
campaign_kind=analytical
save_ckpt=1

case "$SCALE" in
  1p5b)
    model=qwen2.5-1.5b-instruct
    default_stamp=gce4_taucontrol_1p5b
    train_seeds=43,44,45
    max_train=1024
    prompt_epochs=5
    eval_prompt_interval=256
    save_steps=1024
    save_from=1024
    max_save_num=20
    time_limit=48:00:00
    train_memory=96G
    adam_offload=0
    activation_offloading=0
    ;;
  3b)
    model=qwen2.5-3b-instruct
    default_stamp=gce4_taucontrol_3b
    train_seeds=43,44,45
    max_train=1024
    prompt_epochs=5
    eval_prompt_interval=256
    save_steps=1024
    save_from=1024
    max_save_num=20
    time_limit=72:00:00
    train_memory=96G
    adam_offload=0
    activation_offloading=0
    ;;
  7b-smoke)
    campaign_kind=operational-smoke
    model=qwen2.5-7b-instruct
    default_stamp=gce4_taucontrol_7b_smoke
    train_seeds=9001
    # The operational smoke obeys the same five-pass ceiling as analytical
    # runs. The full 1,024-row pool is loaded on every pass.
    max_train=1024
    prompt_epochs=5
    eval_prompt_interval=64
    save_steps=160
    save_from=160
    max_save_num=2
    time_limit=12:00:00
    train_memory=192G
    adam_offload=1
    activation_offloading=1
    ;;
  7b)
    if [[ "${OAT_ZERO_7B_ANALYTICAL_APPROVED:-0}" != "1" ]]; then
      echo "Full 7B E4 submits nine high-compute analytical jobs." >&2
      echo "Rerun with OAT_ZERO_7B_ANALYTICAL_APPROVED=1 to confirm." >&2
      exit 2
    fi
    model=qwen2.5-7b-instruct
    default_stamp=gce4_taucontrol_7b
    train_seeds=43,44,45
    max_train=1024
    prompt_epochs=5
    eval_prompt_interval=256
    save_steps=1024
    save_from=1024
    # Inline metrics and checkpoint evaluations carry the complete analytical
    # curve. Persist only the latest model snapshot, not ZeRO optimizer state:
    # one 7B ZeRO-2 checkpoint is roughly 120 GB, so nine analytical arms would
    # exhaust the shared filesystem without adding evidence to the paper.
    save_ckpt=0
    max_save_num=1
    time_limit=168:00:00
    train_memory=192G
    adam_offload=1
    activation_offloading=1
    ;;
  *)
    echo "Usage: $0 {1p5b|3b|7b-smoke|7b}" >&2
    exit 2
    ;;
esac

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-$default_stamp}"

# Core E4 has three arms: Dr.GRPO, fixed xDr tau=.05, and proportional
# entropy-feedback xDr. A filtered extension may opt into the prospective
# Haarnoja-style dual arm without resubmitting these controls.
export OAT_ZERO_COMPARATIVE_TASK="${OAT_ZERO_COMPARATIVE_TASK:-graph_coloring}"
export OAT_ZERO_COMPARATIVE_MODEL="$model"
export OAT_ZERO_COMPARATIVE_DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_gc_large_probe}"
export OAT_ZERO_XDR_TAUS=0.05
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=1
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM="${OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM:-0}"
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_TRAIN_SEEDS="${OAT_ZERO_TRAIN_SEEDS:-$train_seeds}"
export OAT_ZERO_NUM_SAMPLES=32
export OAT_ZERO_LEARNING_RATE=0.0000002

# Identical controller at every scale; its entropy target is calibrated within
# each run, so raw entropy magnitudes need not match across model sizes.
export OAT_ZERO_XDR_TAU_CONTROL_BASE=0.05
export OAT_ZERO_XDR_TAU_CONTROL_RATIO=0.8
export OAT_ZERO_XDR_TAU_CONTROL_WARMUP_STEPS=64
export OAT_ZERO_XDR_TAU_CONTROL_MIN=0.005
export OAT_ZERO_XDR_TAU_CONTROL_EMA_DECAY=0.9
export OAT_ZERO_XDR_TAU_CONTROL_GAIN=20.0

export OAT_ZERO_MAX_TRAIN="${OAT_ZERO_MAX_TRAIN:-$max_train}"
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH="${OAT_ZERO_NUM_PROMPT_EPOCH:-$prompt_epochs}"
export OAT_ZERO_EVAL_PROMPT_INTERVAL="${OAT_ZERO_EVAL_PROMPT_INTERVAL:-$eval_prompt_interval}"
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0

export OAT_ZERO_SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-$save_ckpt}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-$save_steps}"
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-$save_from}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-$max_save_num}"

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a100:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_TRAIN_PARTITION:-}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_TRAIN_ACCOUNT:-}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_TRAIN_TIME_LIMIT:-$time_limit}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_TRAIN_MEMORY:-$train_memory}"
export OAT_ZERO_ADAM_OFFLOAD="${OAT_ZERO_ADAM_OFFLOAD:-$adam_offload}"
export OAT_ZERO_ACTIVATION_OFFLOADING="${OAT_ZERO_ACTIVATION_OFFLOADING:-$activation_offloading}"

echo "[e4-${SCALE}] mode=${campaign_kind} model=${model} stamp=${RUN_STAMP_PREFIX} seeds=${OAT_ZERO_TRAIN_SEEDS}"
echo "[e4-${SCALE}] memory=${OAT_ZERO_TRAIN_MEMORY} adam_offload=${OAT_ZERO_ADAM_OFFLOAD} activation_offload=${OAT_ZERO_ACTIVATION_OFFLOADING}"
echo "[e4-${SCALE}] budget max_train=${OAT_ZERO_MAX_TRAIN} max_queries=${OAT_ZERO_MAX_QUERIES} prompt_epochs=${OAT_ZERO_NUM_PROMPT_EPOCH} eval_prompt_interval=${OAT_ZERO_EVAL_PROMPT_INTERVAL} save_steps=${OAT_ZERO_SAVE_STEPS} save_ckpt=${OAT_ZERO_SAVE_CKPT} max_save_num=${OAT_ZERO_MAX_SAVE_NUM}"
echo "[e4-${SCALE}] placement=${OAT_ZERO_TRAIN_PARTITION:-template-default}/${OAT_ZERO_TRAIN_ACCOUNT:-template-default} nodelist=${OAT_ZERO_TRAIN_NODELIST} gres=${OAT_ZERO_TRAIN_GRES}"
echo "[e4-${SCALE}] gpu_layout total=${OAT_ZERO_N_GPU:-1} per_actor=${OAT_ZERO_NUM_GPUS_PER_ACTOR:-1}"
if [[ "${OAT_ZERO_E4_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "[e4-${SCALE}] configuration only; no jobs submitted"
  exit 0
fi
exec "$ROOT_DIR/ops/submit_countdown_comparative.sh"
