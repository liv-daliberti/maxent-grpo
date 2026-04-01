#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/drgrpo_explorer_1p5b_pure_oat_r1_${RUN_STAMP}}"
export WB_PROJECT="${WB_PROJECT:-oat-zero}"
export WB_RUN_NAME="${WB_RUN_NAME:-qwen2.5-Math-1.5b-drgrpo-explorer-r1template}"

# Dr.GRPO-Explorer: listwise MaxEnt objective on top of the same OAT/Dr.GRPO
# rollout backend, with the "fair" math recipe defaults from the TRL comparison.
export PURE_OBJECTIVE="${PURE_OBJECTIVE:-maxent_listwise}"
export PURE_BETA="${PURE_BETA:-0.08}"
export PURE_LISTWISE_TAU="${PURE_LISTWISE_TAU:-0.5}"
export PURE_LISTWISE_Q_TEMPERATURE="${PURE_LISTWISE_Q_TEMPERATURE:-2.0}"
export PURE_LISTWISE_Q_EPSILON="${PURE_LISTWISE_Q_EPSILON:-1e-6}"
export PURE_MAXENT_LOGPROB_CHUNK_SIZE="${PURE_MAXENT_LOGPROB_CHUNK_SIZE:-2}"

# Listwise loss needs whole prompt groups per microbatch. Setting the local train
# microbatch to num_samples preserves the same 16-prompt global optimizer batch as
# the pure Dr.GRPO run while keeping each prompt-group intact.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export TRAIN_BATCH_SIZE_PER_DEVICE="${TRAIN_BATCH_SIZE_PER_DEVICE:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"

exec "$ROOT_DIR/ops/run_drgrpo_1p5b_pure_single_node.sh" "$@"
