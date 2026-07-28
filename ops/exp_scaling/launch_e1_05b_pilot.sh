#!/usr/bin/env bash
# E1 compute-scaling divergence -- 0.5B PILOT (hypothesis-generating).
#
# Purpose: locate the collapse timescale and validate the inline coverage-curve
# machinery before committing the long 3B runs. Runs OFF the contended A100:
# node105 has 10x idle a5000 (24GB), plenty for 0.5B.
#
# Two arms only -- Dr.GRPO (tau=inf) vs xDr.GRPO (tau=0.05) -- graph coloring,
# the domain with traction. Five-epoch horizon with a dense inline
# coverage eval every quarter prompt epoch so we watch the divergence form. See
# paper/preregistration/exploration_compute_scaling.md (E1).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-gce1_05b}"

# --- what to run ---
export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"   # 192-prompt 0.5B pool
export OAT_ZERO_XDR_TAUS=0.05
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_NUM_SAMPLES=16               # G=16 at 0.5B
export OAT_ZERO_LEARNING_RATE=0.0000002

# --- standardized horizon: the full 192-prompt pool for five passes. MAX_TRAIN
#     caps rows loaded per pass; NUM_PROMPT_EPOCH controls repeated traversals. ---
export OAT_ZERO_MAX_TRAIN=192
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=5

# --- inline curve: coverage@8 + greedy every quarter prompt epoch ---
export OAT_ZERO_EVAL_PROMPT_INTERVAL=48 # quarter of the 192-prompt pool
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0

# --- keep periodic checkpoints as a bulletproof fallback curve source ---
export OAT_ZERO_SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-1}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-288}"
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-288}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-24}"

# --- off-A100 placement ---
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node105}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a5000:1}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_TRAIN_TIME_LIMIT:-12:00:00}"

echo "[e1-05b-pilot] stamp=${RUN_STAMP_PREFIX} epochs=${OAT_ZERO_NUM_PROMPT_EPOCH} node=node105 a5000"
exec "$ROOT_DIR/ops/submit_countdown_comparative.sh"
