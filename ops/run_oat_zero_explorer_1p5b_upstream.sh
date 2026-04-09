#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/oat_zero_explorer_1p5b_${RUN_STAMP}}"
export OAT_ZERO_WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
export OAT_ZERO_WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-qwen2.5-Math-1.5b-r1-zero-readmeflash-explorer-${RUN_STAMP}}"
export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-${OAT_ZERO_EVAL_STEPS}}"
export OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"
export OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"
export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"
export OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"

# Reuse the exact README-flash runtime and launcher, but switch the learner onto
# the DrX objective: an entropic lift over per-candidate Dr.GRPO surrogate
# utilities with a prompt-local candidate trust region.
export OAT_ZERO_OBJECTIVE="${OAT_ZERO_OBJECTIVE:-maxent_listwise}"
export OAT_ZERO_BETA="${OAT_ZERO_BETA:-0}"
export OAT_ZERO_MAXENT_TAU="${OAT_ZERO_MAXENT_TAU:-0.2}"
# In the practical DrX recipe, tau is the only inner sharpness knob. Learn it
# against the target weight-entropy schedule instead of freezing it at the
# initial value. The legacy reward-softmax q temperature is kept at 1.0 for
# compatibility but is inactive in the Dr.GRPO-utility lift path.
export OAT_ZERO_MAXENT_TAU_LEARNABLE="${OAT_ZERO_MAXENT_TAU_LEARNABLE:-1}"
export OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED:-0}"
export OAT_ZERO_MAXENT_TAU_LR="${OAT_ZERO_MAXENT_TAU_LR:-0.02}"
export OAT_ZERO_MAXENT_TAU_MIN="${OAT_ZERO_MAXENT_TAU_MIN:-0.01}"
export OAT_ZERO_MAXENT_TAU_MAX="${OAT_ZERO_MAXENT_TAU_MAX:-0}"
export OAT_ZERO_MAXENT_TAU_WARMUP_STEPS="${OAT_ZERO_MAXENT_TAU_WARMUP_STEPS:-0}"
export OAT_ZERO_IGNORE_NO_EOS="${OAT_ZERO_IGNORE_NO_EOS:-0}"
export OAT_ZERO_MAXENT_Q_TEMPERATURE="${OAT_ZERO_MAXENT_Q_TEMPERATURE:-1.0}"
export OAT_ZERO_MAXENT_Q_EPSILON="${OAT_ZERO_MAXENT_Q_EPSILON:-1e-6}"
export OAT_ZERO_MAXENT_CANDIDATE_KL_COEF="${OAT_ZERO_MAXENT_CANDIDATE_KL_COEF:-0.0}"
# Default to the cleaner exact DrX split: sequence-clipped candidate utilities
# choose prompt-local weights, while the outer optimizer keeps the token clip.
export OAT_ZERO_MAXENT_EXACT_DRX_WEIGHT_SOURCE="${OAT_ZERO_MAXENT_EXACT_DRX_WEIGHT_SOURCE:-sequence_clipped}"
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
export OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET="${OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET:-4096}"
# Keep the candidate-level scoring signal utility-dense rather than length-biased.
# The shared exact launcher already defaults these to 1; keep the explorer recipe
# aligned so the next DrX run length-normalizes both policy and reference seq scores.
export OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF:-1}"
export OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY:-1}"
# Keep the informative-group metric split available. In the exact weighted
# DrX path, neutral groups naturally reduce to uniform candidate weights and
# zero Dr.GRPO advantages, so this flag now mainly affects diagnostics rather
# than the optimized objective.
export OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS="${OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS:-1}"
export OAT_ZERO_MAXENT_USE_CLIP_OBJECTIVE="${OAT_ZERO_MAXENT_USE_CLIP_OBJECTIVE:-0}"
export OAT_ZERO_MAXENT_CLIP_OBJECTIVE_COEF="${OAT_ZERO_MAXENT_CLIP_OBJECTIVE_COEF:-0.0}"
# The exact DrX recipe no longer uses the legacy sequence-side auxiliary. The
# optimized path is the weighted Dr.GRPO token surrogate itself.
export OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS="${OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS:-0}"
export OAT_ZERO_MAXENT_CLIP_MODE="${OAT_ZERO_MAXENT_CLIP_MODE:-none}"
export OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY="${OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY:-0}"
export OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY:-1}"
export OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF="${OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF:-0.0}"
export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS:-0}"
export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL:-1}"
export OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS:-16}"
export OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE="${OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE:-model}"
# Gentle sharp -> loose -> sharper schedule for the learnable-tau controller.
# The defaults stay well below the 8-way maximum entropy ln(8) ~= 2.08 while
# still giving the policy room to broaden early before tightening again.
export OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_START="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_START:-1.75}"
export OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK:-1.95}"
export OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP:-16}"
export OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_FINAL="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_FINAL:-1.70}"
export OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_HORIZON="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_HORIZON:-96}"

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
