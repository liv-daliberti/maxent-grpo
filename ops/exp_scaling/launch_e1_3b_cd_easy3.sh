#!/usr/bin/env bash
# E1 compute-scaling divergence -- 3B COUNTDOWN (easy3) domain-generality arm.
#
# Countdown-4 is floor-bound at 3B (baseline pass@8 .073), so the 3B dynamics
# test moves to the easy3 pool (3 numbers, 384/128), where a 3B model has real
# reward signal. Expect pass@8 near ceiling; the informative dynamics are
# coverage@8 / distinct@8 (mode collapse vs. retention). Protocol matches the
# 3B graph-coloring E1 runs: Dr.GRPO vs xDr tau=0.05, G=32, 3 seeds, 5 passes.
#
# Pool = 384 prompts; the standardized ceiling is five complete traversals.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-cde1_3b}"

export OAT_ZERO_COMPARATIVE_TASK=countdown
export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct
export OAT_ZERO_XDR_TAUS=0.05
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_NUM_SAMPLES=32               # G=32 at 3B (protocol match)
export OAT_ZERO_LEARNING_RATE=0.0000002

export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=5

export OAT_ZERO_EVAL_PROMPT_INTERVAL=96 # quarter of the 384-prompt pool
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0

export OAT_ZERO_SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-1}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-1152}" # every 3 passes
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-1152}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-8}"

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a100:1}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_TRAIN_TIME_LIMIT:-48:00:00}"

echo "[e1-3b-cd-easy3] stamp=${RUN_STAMP_PREFIX} passes=5 nodes=${OAT_ZERO_TRAIN_NODELIST} gres=${OAT_ZERO_TRAIN_GRES}"
exec "$ROOT_DIR/ops/submit_countdown_comparative.sh"
