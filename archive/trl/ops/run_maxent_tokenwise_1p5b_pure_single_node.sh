#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/maxent_tokenwise_1p5b_pure_oat_r1_${RUN_STAMP}}"
export WB_PROJECT="${WB_PROJECT:-oat-zero}"
export WB_RUN_NAME="${WB_RUN_NAME:-qwen2.5-Math-1.5b-maxent-tokenwise-r1template}"

export PURE_OBJECTIVE="${PURE_OBJECTIVE:-maxent_tokenwise}"
export PURE_BETA="${PURE_BETA:-0.0}"
export PURE_MAXENT_ALPHA="${PURE_MAXENT_ALPHA:-0.005}"
export PURE_MAXENT_ALPHA_RAISE_ON_LOW_KL="${PURE_MAXENT_ALPHA_RAISE_ON_LOW_KL:-0}"
export PURE_MAXENT_ALPHA_LOWER_ON_HIGH_KL="${PURE_MAXENT_ALPHA_LOWER_ON_HIGH_KL:-1}"
export PURE_MAXENT_ALPHA_KL_THRESHOLD="${PURE_MAXENT_ALPHA_KL_THRESHOLD:-0.07}"
export PURE_MAXENT_ALPHA_KL_GAIN="${PURE_MAXENT_ALPHA_KL_GAIN:-0.5}"
export PURE_MAXENT_ALPHA_DISABLE_OUTSIDE_TRUST_ZONE="${PURE_MAXENT_ALPHA_DISABLE_OUTSIDE_TRUST_ZONE:-1}"

exec "$ROOT_DIR/archive/trl/ops/run_drgrpo_1p5b_pure_single_node.sh" "$@"
