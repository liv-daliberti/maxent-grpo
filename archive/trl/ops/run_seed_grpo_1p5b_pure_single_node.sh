#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/seed_grpo_1p5b_pure_oat_r1_${RUN_STAMP}}"
export WB_PROJECT="${WB_PROJECT:-oat-zero}"
export WB_RUN_NAME="${WB_RUN_NAME:-qwen2.5-Math-1.5b-seed-grpo-r1template}"

export PURE_OBJECTIVE="${PURE_OBJECTIVE:-grpo}"
export PURE_BETA="${PURE_BETA:-0.0}"
export PURE_SEED_GRPO_ENABLED="${PURE_SEED_GRPO_ENABLED:-1}"
export PURE_SEED_GRPO_ALPHA="${PURE_SEED_GRPO_ALPHA:-0.0417}"
export PURE_SEED_GRPO_ALPHA_NORMALIZE_BY_MAX_ENTROPY="${PURE_SEED_GRPO_ALPHA_NORMALIZE_BY_MAX_ENTROPY:-1}"
export PURE_SEED_GRPO_LENGTH_NORMALIZE_LOGPROBS="${PURE_SEED_GRPO_LENGTH_NORMALIZE_LOGPROBS:-1}"

exec "$ROOT_DIR/archive/trl/ops/run_drgrpo_1p5b_pure_single_node.sh" "$@"
