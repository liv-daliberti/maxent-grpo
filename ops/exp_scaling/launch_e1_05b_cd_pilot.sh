#!/usr/bin/env bash
# E1 compute-scaling divergence -- 0.5B COUNTDOWN (easy3) domain-generality arm.
#
# The graph-coloring rows of fig:compute-divergence carry the headline; this
# adds the second domain at the scale where Countdown has traction (0.5B on
# easy3: baseline pass@8 ~.40 in paper Sec. 6). At 3B, Countdown-4 sits at its
# reward floor (baseline pass@8 .073; Sec. 7 + the mixed-pool null), so
# collapse dynamics have nothing to show there -- 0.5B is the informative test.
#
# Same protocol as launch_e1_05b_pilot.sh: Dr.GRPO vs xDr tau=0.05, 3 seeds,
# five passes over the 384-prompt pool, with inline coverage evaluation.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-cde1_05b}"

export OAT_ZERO_COMPARATIVE_TASK=countdown
export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_XDR_TAUS=0.05
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002

# MAX_TRAIN caps unique rows loaded per pass; NUM_PROMPT_EPOCH sets the number
# of complete traversals. Use the full pool for exactly five passes.
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=5

export OAT_ZERO_EVAL_PROMPT_INTERVAL=96 # quarter of the 384-prompt pool
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0

export OAT_ZERO_SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-1}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-1152}" # every 3 passes
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-1152}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-12}"

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node105}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a5000:1}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_TRAIN_TIME_LIMIT:-12:00:00}"

echo "[e1-05b-cd] stamp=${RUN_STAMP_PREFIX} passes=5 node=node105 a5000"
exec "$ROOT_DIR/ops/submit_countdown_comparative.sh"
