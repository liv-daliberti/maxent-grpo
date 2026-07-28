#!/usr/bin/env bash
# E1 compute-scaling divergence -- 3B CONFIRMATORY (the money result).
#
# Dr.GRPO (tau=inf) vs xDr.GRPO (tau=0.05), Qwen2.5-3B-Instruct, graph coloring
# (exact_gc_large_probe, 1024/256+256). Five-epoch horizon with a dense
# inline coverage eval so the arm x log(compute) divergence forms across the run;
# epoch-boundary checkpoints are retained as a bulletproof fallback curve source
# AND as the Phase-A init for the E2 two-stage experiment.
#
# Runs on node302 (8x A100, mltheory, 7-day walltime). Jobs will PEND behind the
# user's agora array holding the GPUs and start as GPUs free. Early-stoppable:
# scancel once the curve plateaus.  Prereg: exploration_compute_scaling.md (E1).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-gce1_3b}"

# --- what to run ---
export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$ROOT_DIR/var/data/exact_gc_large_probe"   # 1024-prompt 3B pool
export OAT_ZERO_XDR_TAUS=0.05
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_TRAIN_SEEDS="${OAT_ZERO_TRAIN_SEEDS:-43,44,45}"
export OAT_ZERO_NUM_SAMPLES=32               # G=32 at 3B
export OAT_ZERO_LEARNING_RATE=0.0000002

# --- standardized horizon: the full 1,024-prompt pool for five passes. ---
export OAT_ZERO_MAX_TRAIN=1024
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=5

# --- dense inline curve: coverage@8 + greedy every quarter prompt epoch ---
export OAT_ZERO_EVAL_PROMPT_INTERVAL=256 # quarter of the 1,024-prompt pool
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0

# --- epoch-boundary checkpoints: fallback curve + E2 Phase-A init ---
export OAT_ZERO_SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-1}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-1024}" # every epoch
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-1024}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-20}"

# --- placement (A100 by default; overridable for the two-A6000 CS path) ---
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a100:1}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_TRAIN_TIME_LIMIT:-72:00:00}"

echo "[e1-3b] stamp=${RUN_STAMP_PREFIX} epochs=${OAT_ZERO_NUM_PROMPT_EPOCH} nodes=${OAT_ZERO_TRAIN_NODELIST} gres=${OAT_ZERO_TRAIN_GRES}"
exec "$ROOT_DIR/ops/submit_countdown_comparative.sh"
