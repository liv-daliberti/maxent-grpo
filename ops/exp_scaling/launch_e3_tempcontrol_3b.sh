#!/usr/bin/env bash
# E3 raised-rollout-temperature control (rival-explanation rebuttal).
#
# A Dr.GRPO arm with hotter rollouts (T=1.2) under an UNCHANGED update, matched
# budget/pool/seeds to E1's Dr.GRPO. Tests whether xDr's coverage gain is just
# "sample hotter." Predicted: no -- hotter sampling raises surface variation but
# does not reproduce xDr's coverage@8 gain at flat accuracy, because it perturbs
# the sampling distribution, not the credit assignment. Compared post-hoc against
# gce1_3b grpo (T=1.0) and xdr0.05. Prereg: exploration_compute_scaling.md (E3).
#
# STAGED: launch after E1 shows the divergence (control is gated on E1 in the
# prereg), or now if queuing everything. grpo-only arm at T=1.2.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-gce3_3b}"

export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$ROOT_DIR/var/data/exact_gc_large_probe"
export OAT_ZERO_XDR_TAUS=""                    # grpo-only: no xdr arms
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_NUM_SAMPLES=32
export OAT_ZERO_LEARNING_RATE=0.0000002

export OAT_ZERO_TEMPERATURE=1.2                # the control knob: hotter rollouts

# Match E1's five-pass horizon and full 1,024-prompt pool.
export OAT_ZERO_MAX_TRAIN=1024
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PROMPT_EPOCH=5
export OAT_ZERO_EVAL_PROMPT_INTERVAL=256 # quarter of the 1,024-prompt pool
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_SAVE_STEPS=1024
export OAT_ZERO_SAVE_FROM=1024
export OAT_ZERO_MAX_SAVE_NUM=20

export OAT_ZERO_TRAIN_NODELIST=node302
export OAT_ZERO_TRAIN_GRES=gpu:a100:1
export OAT_ZERO_TRAIN_TIME_LIMIT=72:00:00

echo "[e3-tempcontrol] stamp=${RUN_STAMP_PREFIX} rollout_T=1.2 grpo-only epochs=${OAT_ZERO_NUM_PROMPT_EPOCH}"
exec "$ROOT_DIR/ops/submit_countdown_comparative.sh"
