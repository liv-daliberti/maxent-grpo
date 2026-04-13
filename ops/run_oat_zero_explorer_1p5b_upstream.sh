#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/oat_zero_explorer_1p5b_${RUN_STAMP}}"
export OAT_ZERO_WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
export OAT_ZERO_WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-qwen2.5-Math-1.5b-r1-zero-readmeflash-explorer-${RUN_STAMP}}"
export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_EVAL_STEPS}"
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"
export OAT_ZERO_SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-1}"
export OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"
export OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"

# Reuse the exact README-flash runtime and launcher, but switch the learner onto
# the DrX objective: an entropic lift over per-candidate Dr.GRPO surrogate
# utilities with a prompt-local candidate trust region.
export OAT_ZERO_OBJECTIVE="${OAT_ZERO_OBJECTIVE:-maxent_listwise}"
export OAT_ZERO_BETA="${OAT_ZERO_BETA:-0}"
export OAT_ZERO_MAXENT_TAU="${OAT_ZERO_MAXENT_TAU:-0.2}"
# Keep tau fixed for the clean explorer comparison.
export OAT_ZERO_MAXENT_TAU_LEARNABLE="${OAT_ZERO_MAXENT_TAU_LEARNABLE:-0}"
export OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED:-0}"
export OAT_ZERO_MAXENT_TAU_LR="${OAT_ZERO_MAXENT_TAU_LR:-0.02}"
export OAT_ZERO_MAXENT_TAU_MIN="${OAT_ZERO_MAXENT_TAU_MIN:-0.0}"
export OAT_ZERO_MAXENT_TAU_MAX="${OAT_ZERO_MAXENT_TAU_MAX:-0}"
export OAT_ZERO_MAXENT_TAU_WARMUP_STEPS="${OAT_ZERO_MAXENT_TAU_WARMUP_STEPS:-0}"
export OAT_ZERO_IGNORE_NO_EOS="${OAT_ZERO_IGNORE_NO_EOS:-0}"
export OAT_ZERO_MAXENT_Q_TEMPERATURE="${OAT_ZERO_MAXENT_Q_TEMPERATURE:-1.0}"
export OAT_ZERO_MAXENT_Q_EPSILON="${OAT_ZERO_MAXENT_Q_EPSILON:-1e-6}"
export OAT_ZERO_MAXENT_CANDIDATE_KL_COEF="${OAT_ZERO_MAXENT_CANDIDATE_KL_COEF:-0.0}"
export OAT_ZERO_MAXENT_COMPETITIVE_MODE_TAU="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_TAU:-0.05}"
export OAT_ZERO_MAXENT_COMPETITIVE_MODE_GAP="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_GAP:-0.10}"
export OAT_ZERO_MAXENT_COMPETITIVE_MODE_TOP_K="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_TOP_K:-3}"
export OAT_ZERO_MAXENT_COMPETITIVE_MODE_BUDGET_MAX="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_BUDGET_MAX:-0.10}"
export OAT_ZERO_MAXENT_COMPETITIVE_MODE_BUDGET_SCALE="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_BUDGET_SCALE:-0.05}"
export OAT_ZERO_MAXENT_COMPETITIVE_MODE_INTRA_TAU="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_INTRA_TAU:-0.01}"
export OAT_ZERO_MAXENT_PROMPT_SELECT_MIN_ALPHA_FRAC="${OAT_ZERO_MAXENT_PROMPT_SELECT_MIN_ALPHA_FRAC:-0.5}"
export OAT_ZERO_MAXENT_COMPETITIVE_MODE_POSITIVE_ONLY="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_POSITIVE_ONLY:-1}"
export OAT_ZERO_MAXENT_VERIFIED_DISTINCT_BONUS_COEF="${OAT_ZERO_MAXENT_VERIFIED_DISTINCT_BONUS_COEF:-0.5}"
export OAT_ZERO_MAXENT_VERIFIED_DISTINCT_MIN_MODES="${OAT_ZERO_MAXENT_VERIFIED_DISTINCT_MIN_MODES:-2}"
export OAT_ZERO_MAXENT_VERIFIED_DISTINCT_REWARD_THRESHOLD="${OAT_ZERO_MAXENT_VERIFIED_DISTINCT_REWARD_THRESHOLD:-0.999}"
export OAT_ZERO_MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_LEN_DELTA="${OAT_ZERO_MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_LEN_DELTA:-24}"
export OAT_ZERO_MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_FORMAT_DROP="${OAT_ZERO_MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_FORMAT_DROP:-0.0}"
# Thread the semantic-cluster threshold through the wrapper so explorer runs can
# actually override it instead of silently using the dataclass default.
export OAT_ZERO_MAXENT_SEMANTIC_SIMILARITY_THRESHOLD="${OAT_ZERO_MAXENT_SEMANTIC_SIMILARITY_THRESHOLD:-0.90}"
export OAT_ZERO_MAXENT_SEMANTIC_EMBEDDING_MAX_TOKENS="${OAT_ZERO_MAXENT_SEMANTIC_EMBEDDING_MAX_TOKENS:-512}"
# Default to unclipped Dr.GRPO candidate utilities for the cleaner semantic
# cluster objective. The exact launcher still supports sequence_clipped as an
# explicit override.
export OAT_ZERO_MAXENT_EXACT_DRX_WEIGHT_SOURCE="${OAT_ZERO_MAXENT_EXACT_DRX_WEIGHT_SOURCE:-unclipped}"
# Keep the explorer comparison aligned with baseline Dr.GRPO unless a run
# explicitly opts into a different actor memory split.
export OAT_ZERO_VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.35}"
# Keep baseline-style grad clipping on by default for cleaner comparisons.
# If smaller-GPU runs regress into the old DeepSpeed norm hang, callers can
# still override this back to 0.
export OAT_ZERO_MAX_NORM="${OAT_ZERO_MAX_NORM:-1.0}"
# Explorer now runs on the node302 README stack, so bias the defaults toward the
# local A100 memory envelope instead of the older 24 GB A5000 fallback. Keep a
# synchronized token budget on by default so rare 3k-token rows do not force one
# rank into a pathological 4-row backward chunk while the others wait in NCCL.
export OAT_ZERO_MAXENT_LOGPROB_CHUNK_SIZE="${OAT_ZERO_MAXENT_LOGPROB_CHUNK_SIZE:-1}"
export OAT_ZERO_MAXENT_BACKWARD_CHUNK_SIZE="${OAT_ZERO_MAXENT_BACKWARD_CHUNK_SIZE:-4}"
export OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET="${OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET:-2048}"
# Keep projection/reference sequence scoring length-normalized by default.
# The `sequence_clipped` exact-DrX utility source now uses raw completion-level
# sequence scores internally, regardless of these flags.
export OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF:-1}"
export OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY:-1}"
# Keep the informative-group metric split available. In the exact weighted
# DrX path, neutral groups naturally reduce to uniform candidate weights and
# zero Dr.GRPO advantages, so this flag now mainly affects diagnostics rather
# than the optimized objective.
export OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS="${OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS:-1}"
export OAT_ZERO_MAXENT_USE_CLIP_OBJECTIVE="${OAT_ZERO_MAXENT_USE_CLIP_OBJECTIVE:-0}"
export OAT_ZERO_MAXENT_CLIP_OBJECTIVE_COEF="${OAT_ZERO_MAXENT_CLIP_OBJECTIVE_COEF:-0.0}"
# The pure DrX recipe keeps only the informative-group weighted Dr.GRPO token
# branch active by default. Neutral groups reduce to no update unless these
# sidecar knobs are explicitly re-enabled for a separate ablation.
export OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS="${OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS:-0}"
export OAT_ZERO_MAXENT_CLIP_MODE="${OAT_ZERO_MAXENT_CLIP_MODE:-none}"
export OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY="${OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY:-0}"
export OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY:-1}"
export OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF="${OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF:-0.0}"
export OAT_ZERO_MAXENT_NEUTRAL_PROJECTION_COEF="${OAT_ZERO_MAXENT_NEUTRAL_PROJECTION_COEF:-0.0}"
export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS:-0}"
export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL:-1}"
export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS:-16}"
export OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE="${OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE:-model}"
# If tau adaptation is ever re-enabled, use rollout-side diagnostics rather than
# a target H(w*) controller. The explorer default keeps tau fixed, so these stay
# unset unless a run explicitly opts in.
export OAT_ZERO_MAXENT_TAU_ADAPTATION_METRIC="${OAT_ZERO_MAXENT_TAU_ADAPTATION_METRIC:-semantic_entropy_mu}"
export OAT_ZERO_MAXENT_TAU_TARGET_METRIC="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC:-}"
export OAT_ZERO_MAXENT_TAU_TARGET_METRIC_START="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_START:-}"
export OAT_ZERO_MAXENT_TAU_TARGET_METRIC_PEAK="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_PEAK:-}"
export OAT_ZERO_MAXENT_TAU_TARGET_METRIC_PEAK_STEP="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_PEAK_STEP:-0}"
export OAT_ZERO_MAXENT_TAU_TARGET_METRIC_FINAL="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_FINAL:-}"
export OAT_ZERO_MAXENT_TAU_TARGET_METRIC_HORIZON="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_HORIZON:-0}"

export OAT_ZERO_NUM_SAMPLES="${OAT_ZERO_NUM_SAMPLES:-8}"
export OAT_ZERO_TRAIN_BATCH_SIZE="${OAT_ZERO_TRAIN_BATCH_SIZE:-128}"
# Keep the active explorer closer to the canonical README batch geometry while
# still preserving whole prompt groups locally for the listwise objective.
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-8}"
export OAT_ZERO_ROLLOUT_BATCH_SIZE="${OAT_ZERO_ROLLOUT_BATCH_SIZE:-128}"
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-16}"
export OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE="${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE:-128}"

# vLLM's actor-side memory pool on the canonical README stack does not support
# expandable segments. Use a split-size cap instead so the learner still gets a
# fragmentation guard without enabling unsupported allocator behavior.
if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] || [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:"* ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
fi

exec "$ROOT_DIR/ops/run_oat_zero_exact_1p5b_upstream.sh" "$@"
