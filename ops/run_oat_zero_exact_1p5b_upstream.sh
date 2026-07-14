#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

CANONICAL_SOURCE_ROOT="${ROOT_DIR}/src"
CANONICAL_PYTHON_BIN="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"
CANONICAL_PYTHON_LIB_DIR="${ROOT_DIR}/var/seed_paper_eval/paper310/lib"
TRAINER_MODULE="${OAT_ZERO_TRAINER_MODULE:-oat_drgrpo.train_zero_math}"

SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:-$CANONICAL_SOURCE_ROOT}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/oat_zero_exact_1p5b_${RUN_STAMP}}"
RESUME_DIR="${OAT_ZERO_RESUME_DIR:-}"
RESUME_TAG="${OAT_ZERO_RESUME_TAG:-}"
WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
# Canonical tasks are the exact multi-answer ModeBench pair. OAT_ZERO_TASK
# picks the domain; both datasets are generated deterministically on first use.
TASK="${OAT_ZERO_TASK:-countdown}"
case "$TASK" in
  countdown)
    DEFAULT_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
    ;;
  graph_coloring)
    DEFAULT_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
    ;;
  *)
    echo "Unknown OAT_ZERO_TASK=${TASK}; use countdown or graph_coloring." >&2
    exit 1
    ;;
esac
WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-qwen2.5-1.5b-instruct-${TASK}-exact-${RUN_STAMP}}"
USE_WB="${OAT_ZERO_USE_WB:-1}"
PROMPT_DATA="${OAT_ZERO_PROMPT_DATA:-$DEFAULT_DATA_ROOT/train}"
EVAL_DATA="${OAT_ZERO_EVAL_DATA:-$DEFAULT_DATA_ROOT/eval}"
PRETRAIN="${OAT_ZERO_PRETRAIN:-Qwen/Qwen2.5-1.5B-Instruct}"
VERIFIER_VERSION="${OAT_ZERO_VERIFIER_VERSION:-fast}"
INPUT_KEY="${OAT_ZERO_INPUT_KEY:-problem}"
OUTPUT_KEY="${OAT_ZERO_OUTPUT_KEY:-answer}"
MAX_TRAIN="${OAT_ZERO_MAX_TRAIN:-9999999}"
MAX_QUERIES="${OAT_ZERO_MAX_QUERIES:-$MAX_TRAIN}"
PROMPT_MAX_LENGTH="${OAT_ZERO_PROMPT_MAX_LENGTH:-256}"
GENERATE_MAX_LENGTH="${OAT_ZERO_GENERATE_MAX_LENGTH:-256}"
SAMPLING_TEMPERATURE="${OAT_ZERO_TEMPERATURE:-1}"
SAMPLING_TOP_P="${OAT_ZERO_TOP_P:-1}"
EVAL_BATCH_SIZE="${OAT_ZERO_EVAL_BATCH_SIZE:-200}"
EVAL_TEMPERATURE="${OAT_ZERO_EVAL_TEMPERATURE:-0}"
EVAL_GENERATE_MAX_LENGTH="${OAT_ZERO_EVAL_GENERATE_MAX_LENGTH:-256}"
EVAL_MODE_COVERAGE_K="${OAT_ZERO_EVAL_MODE_COVERAGE_K:-32}"
EVAL_MODE_COVERAGE_TEMPERATURE="${OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE:-1.0}"
TEST_SPLIT="${OAT_ZERO_TEST_SPLIT:-all}"
N_GPU="${OAT_ZERO_N_GPU:-8}"
NUM_GPUS_PER_ACTOR="${OAT_ZERO_NUM_GPUS_PER_ACTOR:-1}"
PROMPT_TEMPLATE="${OAT_ZERO_PROMPT_TEMPLATE:-qwen_math}"
OBJECTIVE="${OAT_ZERO_OBJECTIVE:-grpo}"
CRITIC_TYPE="${OAT_ZERO_CRITIC_TYPE:-drgrpo}"
XDR_TAU="${OAT_ZERO_XDR_TAU:-inf}"
XDR_MODE_ADAPTIVE="${OAT_ZERO_XDR_MODE_ADAPTIVE:-0}"
SEED_ENTROPY_ALPHA="${OAT_ZERO_SEED_ENTROPY_ALPHA:-0.0}"
SEMANTIC_ENTROPY_LAMBDA="${OAT_ZERO_SEMANTIC_ENTROPY_LAMBDA:-0.05}"
POLICY_ENTROPY_COEF="${OAT_ZERO_POLICY_ENTROPY_COEF:-0.0}"
SEED="${OAT_ZERO_SEED:-42}"
if [[ -n "${OAT_ZERO_RND_SEED:-}" ]]; then
  RND_SEED="${OAT_ZERO_RND_SEED}"
elif [[ -n "${OAT_ZERO_SEED:-}" ]]; then
  RND_SEED=0
else
  RND_SEED=1
fi
COLLOCATE="${OAT_ZERO_COLLOCATE:-1}"
ZERO_STAGE="${OAT_ZERO_ZERO_STAGE:-2}"
ADAM_OFFLOAD="${OAT_ZERO_ADAM_OFFLOAD:-0}"
ACTIVATION_OFFLOADING="${OAT_ZERO_ACTIVATION_OFFLOADING:-0}"
DISABLE_TRACE_CACHE="${OAT_ZERO_DISABLE_TRACE_CACHE:-0}"
GRAD_ACCUM_DTYPE="${OAT_ZERO_GRAD_ACCUM_DTYPE:-}"
VLLM_SLEEP="${OAT_ZERO_VLLM_SLEEP:-1}"
LEARNING_RATE="${OAT_ZERO_LEARNING_RATE:-0.000001}"
NUM_PPO_EPOCHS="${OAT_ZERO_NUM_PPO_EPOCHS:-1}"
BETA="${OAT_ZERO_BETA:-0}"
NUM_PROMPT_EPOCH="${OAT_ZERO_NUM_PROMPT_EPOCH:-20}"
NUM_SAMPLES="${OAT_ZERO_NUM_SAMPLES:-8}"
TRAIN_BATCH_SIZE="${OAT_ZERO_TRAIN_BATCH_SIZE:-128}"
TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-1}"
ROLLOUT_BATCH_SIZE="${OAT_ZERO_ROLLOUT_BATCH_SIZE:-128}"
ROLLOUT_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-16}"
PI_BUFFER_MAXLEN_PER_DEVICE="${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE:-128}"
VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.35}"
EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-$EVAL_STEPS}"
SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"
SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-1}"
SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"
ALLOW_NO_SAVE="${OAT_ZERO_ALLOW_NO_SAVE:-0}"
REQUIRE_FULL_EVAL_CHECKPOINTS="${OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS:-1}"
MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"
MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"
ALLOW_NO_EVAL="${OAT_ZERO_ALLOW_NO_EVAL:-0}"
MAX_EVAL_STEPS_ALLOWED="${OAT_ZERO_MAX_EVAL_STEPS_ALLOWED:-1000}"
MAX_NORM="${OAT_ZERO_MAX_NORM:-1.0}"
IGNORE_NO_EOS="${OAT_ZERO_IGNORE_NO_EOS:-0}"
MAXENT_TAU="${OAT_ZERO_MAXENT_TAU:-0.3}"
MAXENT_Q_TEMPERATURE="${OAT_ZERO_MAXENT_Q_TEMPERATURE:-2.0}"
MAXENT_Q_EPSILON="${OAT_ZERO_MAXENT_Q_EPSILON:-1e-6}"
MAXENT_CANDIDATE_KL_COEF="${OAT_ZERO_MAXENT_CANDIDATE_KL_COEF:-0.0}"
MAXENT_SEMANTIC_SIMILARITY_THRESHOLD="${OAT_ZERO_MAXENT_SEMANTIC_SIMILARITY_THRESHOLD:-0.75}"
MAXENT_SEMANTIC_EMBEDDING_SIMILARITY_THRESHOLD="${OAT_ZERO_MAXENT_SEMANTIC_EMBEDDING_SIMILARITY_THRESHOLD:-0.9}"
MAXENT_SEMANTIC_CLUSTER_METHOD="${OAT_ZERO_MAXENT_SEMANTIC_CLUSTER_METHOD:-default}"
MAXENT_SEMANTIC_EMBEDDING_MAX_TOKENS="${OAT_ZERO_MAXENT_SEMANTIC_EMBEDDING_MAX_TOKENS:-512}"
MAXENT_SEMANTIC_CLUSTER_MAX_TOKENS="${OAT_ZERO_MAXENT_SEMANTIC_CLUSTER_MAX_TOKENS:-0}"
MAXENT_SEMANTIC_SPECTRAL_MAX_CLUSTERS="${OAT_ZERO_MAXENT_SEMANTIC_SPECTRAL_MAX_CLUSTERS:-0}"
MAXENT_SEMANTIC_SPECTRAL_EIGENGAP_MIN="${OAT_ZERO_MAXENT_SEMANTIC_SPECTRAL_EIGENGAP_MIN:-0.05}"
MAXENT_SEMANTIC_CORRECTNESS_TARGET_FRAC="${OAT_ZERO_MAXENT_SEMANTIC_CORRECTNESS_TARGET_FRAC:-0.5}"
MAXENT_SEMANTIC_CORRECTNESS_SHARPNESS="${OAT_ZERO_MAXENT_SEMANTIC_CORRECTNESS_SHARPNESS:-4.0}"
MAXENT_SEMANTIC_CORRECTNESS_ANSWER_LEVEL="${OAT_ZERO_MAXENT_SEMANTIC_CORRECTNESS_ANSWER_LEVEL:-0}"
MAXENT_SEMANTIC_CORRECTNESS_MIN_ANSWER_COUNT="${OAT_ZERO_MAXENT_SEMANTIC_CORRECTNESS_MIN_ANSWER_COUNT:-1}"
MAXENT_SEMANTIC_REMIX_MODE="${OAT_ZERO_MAXENT_SEMANTIC_REMIX_MODE:-competitive}"
MAXENT_REWARD_SHAPING_ALPHA="${OAT_ZERO_MAXENT_REWARD_SHAPING_ALPHA:-0.0}"
MAXENT_TIEBREAK_ANCHOR="${OAT_ZERO_MAXENT_TIEBREAK_ANCHOR:-hybrid}"
MAXENT_TIEBREAK_CLIP_MAX="${OAT_ZERO_MAXENT_TIEBREAK_CLIP_MAX:-1.0}"
MAXENT_COMPETITIVE_MODE_TAU="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_TAU:-0.05}"
MAXENT_COMPETITIVE_MODE_GAP="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_GAP:-0.10}"
MAXENT_COMPETITIVE_MODE_TOP_K="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_TOP_K:-3}"
MAXENT_COMPETITIVE_MODE_BUDGET_MAX="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_BUDGET_MAX:-0.10}"
MAXENT_COMPETITIVE_MODE_BUDGET_SCALE="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_BUDGET_SCALE:-0.05}"
MAXENT_COMPETITIVE_MODE_INTRA_TAU="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_INTRA_TAU:-0.01}"
MAXENT_PROMPT_SELECT_MIN_ALPHA_FRAC="${OAT_ZERO_MAXENT_PROMPT_SELECT_MIN_ALPHA_FRAC:-0.5}"
MAXENT_COMPETITIVE_MODE_POSITIVE_ONLY="${OAT_ZERO_MAXENT_COMPETITIVE_MODE_POSITIVE_ONLY:-1}"
MAXENT_CORRECTNESS_SCHEDULE_ENABLED="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_ENABLED:-1}"
MAXENT_CORRECTNESS_SCHEDULE_EMA_DECAY="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_EMA_DECAY:-0.997}"
MAXENT_CORRECTNESS_SCHEDULE_LOW="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_LOW:-0.45}"
MAXENT_CORRECTNESS_SCHEDULE_HIGH="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_HIGH:-0.90}"
MAXENT_CORRECTNESS_SCHEDULE_BUDGET_MAX_EARLY="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_BUDGET_MAX_EARLY:-0.18}"
MAXENT_CORRECTNESS_SCHEDULE_BUDGET_MAX_LATE="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_BUDGET_MAX_LATE:-0.06}"
MAXENT_CORRECTNESS_SCHEDULE_PROMPT_SELECT_MIN_ALPHA_FRAC_EARLY="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_PROMPT_SELECT_MIN_ALPHA_FRAC_EARLY:-0.20}"
MAXENT_CORRECTNESS_SCHEDULE_PROMPT_SELECT_MIN_ALPHA_FRAC_LATE="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_PROMPT_SELECT_MIN_ALPHA_FRAC_LATE:-0.50}"
MAXENT_CORRECTNESS_SCHEDULE_MODE_TAU_EARLY="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_MODE_TAU_EARLY:-0.08}"
MAXENT_CORRECTNESS_SCHEDULE_MODE_TAU_LATE="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_MODE_TAU_LATE:-0.03}"
MAXENT_CORRECTNESS_SCHEDULE_INTRA_TAU_EARLY="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_INTRA_TAU_EARLY:-0.03}"
MAXENT_CORRECTNESS_SCHEDULE_INTRA_TAU_LATE="${OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_INTRA_TAU_LATE:-0.005}"
MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_LEN_DELTA="${OAT_ZERO_MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_LEN_DELTA:-24}"
MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_FORMAT_DROP="${OAT_ZERO_MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_FORMAT_DROP:-0.0}"
DEEPSPEED_ALLGATHER_BUCKET_SIZE="${OAT_ZERO_DEEPSPEED_ALLGATHER_BUCKET_SIZE:-}"
DEEPSPEED_REDUCE_BUCKET_SIZE="${OAT_ZERO_DEEPSPEED_REDUCE_BUCKET_SIZE:-}"
DEEPSPEED_OVERLAP_COMM="${OAT_ZERO_DEEPSPEED_OVERLAP_COMM:-}"
DEEPSPEED_CONTIGUOUS_GRADIENTS="${OAT_ZERO_DEEPSPEED_CONTIGUOUS_GRADIENTS:-}"
DEEPSPEED_REDUCE_SCATTER="${OAT_ZERO_DEEPSPEED_REDUCE_SCATTER:-}"
DEEPSPEED_USE_MULTI_RANK_BUCKET_ALLREDUCE="${OAT_ZERO_DEEPSPEED_USE_MULTI_RANK_BUCKET_ALLREDUCE:-}"
DEEPSPEED_ALLGATHER_PARTITIONS="${OAT_ZERO_DEEPSPEED_ALLGATHER_PARTITIONS:-}"
MAXENT_EXACT_DRX_WEIGHT_SOURCE="${OAT_ZERO_MAXENT_EXACT_DRX_WEIGHT_SOURCE:-unclipped}"
MAXENT_LOGPROB_CHUNK_SIZE="${OAT_ZERO_MAXENT_LOGPROB_CHUNK_SIZE:-2}"
MAXENT_BACKWARD_CHUNK_SIZE="${OAT_ZERO_MAXENT_BACKWARD_CHUNK_SIZE:-4}"
MAXENT_BACKWARD_TOKEN_BUDGET="${OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET:-4096}"
# Projection/reference sequence scoring can still use length normalization, but
# the exact-DrX `sequence_clipped` utility source is evaluated on raw
# completion-level sequence scores inside the learner.
MAXENT_LENGTH_NORMALIZE_REF="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF:-1}"
MAXENT_LENGTH_NORMALIZE_POLICY="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY:-1}"
MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS="${OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS:-1}"
MAXENT_USE_CLIP_OBJECTIVE="${OAT_ZERO_MAXENT_USE_CLIP_OBJECTIVE:-1}"
MAXENT_CLIP_OBJECTIVE_COEF="${OAT_ZERO_MAXENT_CLIP_OBJECTIVE_COEF:-1.0}"
MAXENT_CLIP_PRESERVE_REWARD_MASS="${OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS:-0}"
MAXENT_CLIP_MODE="${OAT_ZERO_MAXENT_CLIP_MODE:-sequence}"
MAXENT_TOKEN_CLIP_PRIMARY="${OAT_ZERO_MAXENT_TOKEN_CLIP_PRIMARY:-0}"
MAXENT_DRGRPO_TOKEN_PRIMARY="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY:-0}"
MAXENT_DRGRPO_TOKEN_ADVANTAGE_SOURCE="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_ADVANTAGE_SOURCE:-weighted}"
MAXENT_DRGRPO_TOKEN_LENGTH_NORMALIZER="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_LENGTH_NORMALIZER:-max_length}"
MAXENT_SEQUENCE_AUX_COEF="${OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF:-1.0}"
MAXENT_SEQUENCE_AUX_GROUP_FILTER="${OAT_ZERO_MAXENT_SEQUENCE_AUX_GROUP_FILTER:-all}"
MAXENT_SEQUENCE_AUX_MAX_EXPECTED_LEN_DROP="${OAT_ZERO_MAXENT_SEQUENCE_AUX_MAX_EXPECTED_LEN_DROP:-1000000000}"
MAXENT_SEQUENCE_AUX_MAX_EXPECTED_LEN_GAIN="${OAT_ZERO_MAXENT_SEQUENCE_AUX_MAX_EXPECTED_LEN_GAIN:-1000000000}"
MAXENT_SEQUENCE_AUX_MAX_EXPECTED_FORMAT_DROP="${OAT_ZERO_MAXENT_SEQUENCE_AUX_MAX_EXPECTED_FORMAT_DROP:-1.0}"
MAXENT_SEQUENCE_AUX_MIN_EXPECTED_CORRECTNESS_DELTA="${OAT_ZERO_MAXENT_SEQUENCE_AUX_MIN_EXPECTED_CORRECTNESS_DELTA:--1.0}"
MAXENT_NEUTRAL_PROJECTION_COEF="${OAT_ZERO_MAXENT_NEUTRAL_PROJECTION_COEF:-0.0}"
MAXENT_BRANCH_GRAD_DIAGNOSTICS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS:-0}"
MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL:-1}"
MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS:-0}"
MAXENT_REFERENCE_LOGPROBS_SOURCE="${OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE:-model}"
MAXENT_TAU_ADAPTATION_METRIC="${OAT_ZERO_MAXENT_TAU_ADAPTATION_METRIC:-semantic_entropy_mu}"
MAXENT_TAU_TARGET_METRIC="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC:-}"
MAXENT_TAU_TARGET_METRIC_START="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_START:-}"
MAXENT_TAU_TARGET_METRIC_PEAK="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_PEAK:-}"
MAXENT_TAU_TARGET_METRIC_PEAK_STEP="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_PEAK_STEP:-0}"
MAXENT_TAU_TARGET_METRIC_FINAL="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_FINAL:-}"
MAXENT_TAU_TARGET_METRIC_HORIZON="${OAT_ZERO_MAXENT_TAU_TARGET_METRIC_HORIZON:-0}"
MAXENT_TAU_LEARNABLE="${OAT_ZERO_MAXENT_TAU_LEARNABLE:-0}"
MAXENT_TAU_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED:-0}"
MAXENT_TAU_LR="${OAT_ZERO_MAXENT_TAU_LR:-0.0}"
MAXENT_TAU_MIN="${OAT_ZERO_MAXENT_TAU_MIN:-0.0}"
MAXENT_TAU_MAX="${OAT_ZERO_MAXENT_TAU_MAX:-0.0}"
MAXENT_TAU_WARMUP_STEPS="${OAT_ZERO_MAXENT_TAU_WARMUP_STEPS:--1}"
MAXENT_BETA_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_BETA_CONTROLLER_ENABLED:-0}"
KL_TARGET="${OAT_ZERO_KL_TARGET:-0.0}"
KL_HORIZON="${OAT_ZERO_KL_HORIZON:-0}"
KL_CTL_STEP_SIZE="${OAT_ZERO_KL_CTL_STEP_SIZE:-0.0}"
FLASH_ATTN_VERSION="${OAT_ZERO_FLASH_ATTN_VERSION:-2.7.4.post1}"
FLASH_ATTN_MAX_JOBS="${OAT_ZERO_FLASH_ATTN_MAX_JOBS:-8}"
FLASH_ATTN_CACHE_ROOT="${OAT_ZERO_FLASH_ATTN_CACHE_ROOT:-$ROOT_DIR/var/cache/oat_zero_exact_flash_attn}"
ENABLE_FLASH_ATTN="${OAT_ZERO_ENABLE_FLASH_ATTN:-1}"
SMOKE_MODE="${OAT_ZERO_SMOKE:-0}"
SMOKE_SUCCESS_PATTERN="${OAT_ZERO_SMOKE_SUCCESS_PATTERN:-weights @version=0 broadcasted to actors}"
SMOKE_TIMEOUT_SECONDS="${OAT_ZERO_SMOKE_TIMEOUT_SECONDS:-3000}"
BOOTSTRAP_ONLY="${OAT_ZERO_BOOTSTRAP_ONLY:-0}"
SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-20000}"
EXPECTED_PYTHON_PREFIX="${OAT_ZERO_EXPECTED_PYTHON_PREFIX:-3.10.}"
EXPECTED_TORCH_VERSION="${OAT_ZERO_EXPECTED_TORCH_VERSION:-2.6.0}"
EXPECTED_TRANSFORMERS_VERSION="${OAT_ZERO_EXPECTED_TRANSFORMERS_VERSION:-4.51.3}"
EXPECTED_VLLM_VERSION="${OAT_ZERO_EXPECTED_VLLM_VERSION:-0.8.4}"
EXPECTED_OAT_VERSION="${OAT_ZERO_EXPECTED_OAT_VERSION:-0.1.3.post1}"
EXPECTED_DEEPSPEED_VERSION="${OAT_ZERO_EXPECTED_DEEPSPEED_VERSION:-0.16.8}"
EXPECTED_MATH_VERIFY_VERSION="${OAT_ZERO_EXPECTED_MATH_VERIFY_VERSION:-0.7.0}"
EXPECTED_FIRE_VERSION="${OAT_ZERO_EXPECTED_FIRE_VERSION:-0.7.0}"

choose_local_root() {
  if [[ -n "${OAT_ZERO_LOCAL_ROOT:-}" ]]; then
    printf '%s\n' "$OAT_ZERO_LOCAL_ROOT"
    return
  fi
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    printf '%s\n' "${SLURM_TMPDIR%/}/maxent-grpo-oat-zero"
    return
  fi
  printf '/tmp/%s/maxent-grpo-oat-zero\n' "${USER:-unknown}"
}

LOCAL_ROOT="$(choose_local_root)"
PYTHON_BIN="${OAT_ZERO_PYTHON:-$CANONICAL_PYTHON_BIN}"
PYTHON_LIB_DIR="${OAT_ZERO_PYTHON_LIB_DIR:-$CANONICAL_PYTHON_LIB_DIR}"
LOCAL_CACHE_ROOT="${OAT_ZERO_LOCAL_CACHE_ROOT:-$LOCAL_ROOT/cache}"
LOCAL_JOB_ROOT="${OAT_ZERO_LOCAL_JOB_ROOT:-$LOCAL_ROOT/job_${SLURM_JOB_ID:-manual}}"

if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/train_zero_math.py" ]]; then
  echo "Missing trainer source tree at $SOURCE_ROOT/oat_drgrpo" >&2
  exit 1
fi

# Auto-generate the canonical ModeBench dataset when the task-default root is
# in use and the data has not been materialized yet. Generation is
# deterministic given the seed, so regenerated data is identical across runs.
if [[ "$PROMPT_DATA" == "$DEFAULT_DATA_ROOT/train" ]] \
  && { [[ ! -f "$DEFAULT_DATA_ROOT/train/dataset_dict.json" ]] \
    || [[ ! -f "$DEFAULT_DATA_ROOT/eval/dataset_dict.json" ]]; }; then
  echo "[oat-zero-exact] generating ${TASK} dataset at ${DEFAULT_DATA_ROOT}"
  case "$TASK" in
    countdown)
      "$PYTHON_BIN" "$ROOT_DIR/ops/make_exact_countdown_mode_data.py" \
        --output-root "$DEFAULT_DATA_ROOT" \
        --train-size "${OAT_ZERO_COUNTDOWN_MODE_TRAIN_SIZE:-384}" \
        --eval-size "${OAT_ZERO_COUNTDOWN_MODE_EVAL_SIZE:-128}" \
        --number-count "${OAT_ZERO_COUNTDOWN_MODE_NUMBER_COUNT:-3}" \
        --max-value "${OAT_ZERO_COUNTDOWN_MODE_MAX_VALUE:-12}" \
        --multi-min-modes "${OAT_ZERO_COUNTDOWN_MODE_MULTI_MIN_MODES:-2}" \
        --multi-max-modes "${OAT_ZERO_COUNTDOWN_MODE_MULTI_MAX_MODES:-8}" \
        --seed "${OAT_ZERO_COUNTDOWN_MODE_DATA_SEED:-0}" \
        --overwrite
      ;;
    graph_coloring)
      "$PYTHON_BIN" "$ROOT_DIR/ops/make_exact_answer_mode_data.py" \
        --output-root "$DEFAULT_DATA_ROOT" \
        --train-size "${OAT_ZERO_ANSWER_MODE_TRAIN_SIZE:-192}" \
        --eval-size "${OAT_ZERO_ANSWER_MODE_EVAL_SIZE:-96}" \
        --seed "${OAT_ZERO_ANSWER_MODE_DATA_SEED:-0}" \
        --overwrite
      ;;
  esac
fi

if [[ ! -d "$PROMPT_DATA" ]]; then
  echo "Missing prompt dataset at $PROMPT_DATA" >&2
  exit 1
fi

if [[ ! -d "$EVAL_DATA" ]]; then
  echo "Missing eval dataset at $EVAL_DATA" >&2
  exit 1
fi

if [[ -n "$RESUME_DIR" ]] && [[ ! -d "$RESUME_DIR" ]]; then
  echo "Missing resume checkpoint directory at $RESUME_DIR" >&2
  exit 1
fi

if [[ -z "$RESUME_DIR" ]] && [[ -n "$RESUME_TAG" ]]; then
  echo "OAT_ZERO_RESUME_TAG requires OAT_ZERO_RESUME_DIR to be set" >&2
  exit 1
fi

case "$OBJECTIVE" in
  grpo|maxent_listwise) ;;
  *)
    echo "OAT_ZERO_OBJECTIVE must be one of: grpo, maxent_listwise" >&2
    exit 1
    ;;
esac

case "$CRITIC_TYPE" in
  ppo|grpo|drgrpo) ;;
  *)
    echo "OAT_ZERO_CRITIC_TYPE must be one of: ppo, grpo, drgrpo" >&2
    exit 1
    ;;
esac

case "$MAXENT_TIEBREAK_ANCHOR" in
  hybrid|behavior|reference) ;;
  *)
    echo "OAT_ZERO_MAXENT_TIEBREAK_ANCHOR must be one of: hybrid, behavior, reference" >&2
    exit 1
    ;;
esac

case "$MAXENT_SEMANTIC_REMIX_MODE" in
  competitive|correctness_conditioned|anchor_rare) ;;
  *)
    echo "OAT_ZERO_MAXENT_SEMANTIC_REMIX_MODE must be one of: competitive, correctness_conditioned, anchor_rare" >&2
    exit 1
    ;;
esac

case "$MAXENT_REFERENCE_LOGPROBS_SOURCE" in
  model|behavior) ;;
  *)
    echo "OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE must be one of: model, behavior" >&2
    exit 1
    ;;
esac

case "$MAXENT_BETA_CONTROLLER_ENABLED" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_BETA_CONTROLLER_ENABLED must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$IGNORE_NO_EOS" in
  0|1) ;;
  *)
    echo "OAT_ZERO_IGNORE_NO_EOS must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_COMPETITIVE_MODE_POSITIVE_ONLY" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_COMPETITIVE_MODE_POSITIVE_ONLY must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_SEMANTIC_CORRECTNESS_ANSWER_LEVEL" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_SEMANTIC_CORRECTNESS_ANSWER_LEVEL must be 0 or 1" >&2
    exit 1
    ;;
esac

if ! [[ "$MAXENT_SEMANTIC_CORRECTNESS_MIN_ANSWER_COUNT" =~ ^[0-9]+$ ]] || (( MAXENT_SEMANTIC_CORRECTNESS_MIN_ANSWER_COUNT < 1 )); then
  echo "OAT_ZERO_MAXENT_SEMANTIC_CORRECTNESS_MIN_ANSWER_COUNT must be positive" >&2
  exit 1
fi

case "$MAXENT_CORRECTNESS_SCHEDULE_ENABLED" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_CORRECTNESS_SCHEDULE_ENABLED must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$RND_SEED" in
  0|1) ;;
  *)
    echo "OAT_ZERO_RND_SEED must be 0 or 1" >&2
    exit 1
    ;;
esac

if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
  echo "OAT_ZERO_SEED must be an integer" >&2
  exit 1
fi

for int_var_name in \
  MAX_TRAIN \
  MAX_QUERIES \
  PROMPT_MAX_LENGTH \
  GENERATE_MAX_LENGTH \
  EVAL_BATCH_SIZE \
  EVAL_GENERATE_MAX_LENGTH
do
  int_var_value="${!int_var_name}"
  if ! [[ "$int_var_value" =~ ^[0-9]+$ ]] || (( int_var_value <= 0 )); then
    echo "OAT_ZERO_${int_var_name} must be a positive integer" >&2
    exit 1
  fi
done

case "$MAXENT_TAU_LEARNABLE" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_TAU_LEARNABLE must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_TAU_CONTROLLER_ENABLED" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_BRANCH_GRAD_DIAGNOSTICS" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_CLIP_MODE" in
  sequence|token|none) ;;
  *)
    echo "OAT_ZERO_MAXENT_CLIP_MODE must be one of: sequence, token, none" >&2
    exit 1
    ;;
esac

case "$MAXENT_EXACT_DRX_WEIGHT_SOURCE" in
  sequence_clipped|clipped|unclipped|local_linear) ;;
  *)
    echo "OAT_ZERO_MAXENT_EXACT_DRX_WEIGHT_SOURCE must be one of: sequence_clipped, clipped, unclipped, local_linear" >&2
    exit 1
    ;;
esac

case "$ALLOW_NO_EVAL" in
  0|1) ;;
  *)
    echo "OAT_ZERO_ALLOW_NO_EVAL must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$SAVE_CKPT" in
  0|1) ;;
  *)
    echo "OAT_ZERO_SAVE_CKPT must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$SAVE_INITIAL_MODEL" in
  0|1) ;;
  *)
    echo "OAT_ZERO_SAVE_INITIAL_MODEL must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$ALLOW_NO_SAVE" in
  0|1) ;;
  *)
    echo "OAT_ZERO_ALLOW_NO_SAVE must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$REQUIRE_FULL_EVAL_CHECKPOINTS" in
  0|1) ;;
  *)
    echo "OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$USE_WB" in
  0|1) ;;
  *)
    echo "OAT_ZERO_USE_WB must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$COLLOCATE" in
  0|1) ;;
  *)
    echo "OAT_ZERO_COLLOCATE must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$ZERO_STAGE" in
  0|1|2|3) ;;
  *)
    echo "OAT_ZERO_ZERO_STAGE must be 0, 1, 2, or 3" >&2
    exit 1
    ;;
esac

case "$ADAM_OFFLOAD" in
  0|1) ;;
  *)
    echo "OAT_ZERO_ADAM_OFFLOAD must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$ACTIVATION_OFFLOADING" in
  0|1) ;;
  *)
    echo "OAT_ZERO_ACTIVATION_OFFLOADING must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$DISABLE_TRACE_CACHE" in
  0|1) ;;
  *)
    echo "OAT_ZERO_DISABLE_TRACE_CACHE must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$VLLM_SLEEP" in
  0|1) ;;
  *)
    echo "OAT_ZERO_VLLM_SLEEP must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$EVAL_STEPS" in
  ''|*[!0-9]*)
    echo "OAT_ZERO_EVAL_STEPS must be a positive integer" >&2
    exit 1
    ;;
esac

case "$MAX_EVAL_STEPS_ALLOWED" in
  ''|*[!0-9]*)
    echo "OAT_ZERO_MAX_EVAL_STEPS_ALLOWED must be a positive integer" >&2
    exit 1
    ;;
esac

if (( EVAL_STEPS == 0 )) && [[ "$ALLOW_NO_EVAL" != "1" ]]; then
  echo "Eval must stay enabled: OAT_ZERO_EVAL_STEPS must be > 0. To override intentionally, set OAT_ZERO_ALLOW_NO_EVAL=1." >&2
  exit 1
fi

if (( EVAL_STEPS > MAX_EVAL_STEPS_ALLOWED )) && [[ "$ALLOW_NO_EVAL" != "1" ]]; then
  echo "Eval is effectively disabled: OAT_ZERO_EVAL_STEPS=${EVAL_STEPS} exceeds OAT_ZERO_MAX_EVAL_STEPS_ALLOWED=${MAX_EVAL_STEPS_ALLOWED}. To override intentionally, set OAT_ZERO_ALLOW_NO_EVAL=1." >&2
  exit 1
fi

if ! [[ "$SAVE_STEPS" =~ ^-?[0-9]+$ ]]; then
  echo "OAT_ZERO_SAVE_STEPS must be an integer" >&2
  exit 1
fi

if ! [[ "$SAVE_FROM" =~ ^[0-9]+$ ]]; then
  echo "OAT_ZERO_SAVE_FROM must be a non-negative integer" >&2
  exit 1
fi

if ! [[ "$MAX_SAVE_NUM" =~ ^[0-9]+$ ]] || (( MAX_SAVE_NUM <= 0 )); then
  echo "OAT_ZERO_MAX_SAVE_NUM must be a positive integer" >&2
  exit 1
fi

if ! [[ "$MAX_SAVE_MEM" =~ ^[0-9]+$ ]] || (( MAX_SAVE_MEM <= 0 )); then
  echo "OAT_ZERO_MAX_SAVE_MEM must be a positive integer number of GB" >&2
  exit 1
fi

if ! [[ "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL" =~ ^[0-9]+$ ]] || (( MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL <= 0 )); then
  echo "OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL must be a positive integer" >&2
  exit 1
fi

if ! [[ "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS" =~ ^[0-9]+$ ]]; then
  echo "OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS must be a non-negative integer" >&2
  exit 1
fi

if (( SAVE_STEPS <= 0 )) && [[ "$ALLOW_NO_SAVE" != "1" ]]; then
  echo "Checkpoint saving must stay enabled: OAT_ZERO_SAVE_STEPS must be > 0. To override intentionally, set OAT_ZERO_ALLOW_NO_SAVE=1." >&2
  exit 1
fi

if [[ "$REQUIRE_FULL_EVAL_CHECKPOINTS" == "1" ]] && [[ "$ALLOW_NO_EVAL" != "1" ]] && [[ "$ALLOW_NO_SAVE" != "1" ]]; then
  if (( SAVE_STEPS != EVAL_STEPS )); then
    echo "Full checkpoint saves must happen at every eval: OAT_ZERO_SAVE_STEPS=${SAVE_STEPS} must equal OAT_ZERO_EVAL_STEPS=${EVAL_STEPS}. To override intentionally, set OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS=0." >&2
    exit 1
  fi
  if (( SAVE_FROM > EVAL_STEPS )); then
    echo "Full checkpoint saves must start no later than the first eval: OAT_ZERO_SAVE_FROM=${SAVE_FROM} must be <= OAT_ZERO_EVAL_STEPS=${EVAL_STEPS}. To override intentionally, set OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS=0." >&2
    exit 1
  fi
  if [[ "$SAVE_CKPT" != "1" ]]; then
    echo "Full checkpoints must be enabled at eval boundaries: OAT_ZERO_SAVE_CKPT must be 1. To override intentionally, set OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS=0." >&2
    exit 1
  fi
fi

ROW_SHARDED_EXACT_DRX=0
if [[ "$OBJECTIVE" == "maxent_listwise" ]]; then
  if (( NUM_SAMPLES <= 1 )); then
    echo "Listwise MaxEnt requires OAT_ZERO_NUM_SAMPLES > 1" >&2
    exit 1
  fi
  if (( TRAIN_BATCH_SIZE_PER_DEVICE <= 0 )); then
    echo "Listwise MaxEnt requires OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE > 0" >&2
    exit 1
  fi
  if (( TRAIN_BATCH_SIZE_PER_DEVICE < NUM_SAMPLES )); then
    ROW_SHARDED_EXACT_DRX=1
  fi
  if (( TRAIN_BATCH_SIZE_PER_DEVICE % NUM_SAMPLES != 0 )) && (( ROW_SHARDED_EXACT_DRX == 0 )); then
    echo "Listwise MaxEnt requires OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE to be divisible by OAT_ZERO_NUM_SAMPLES, unless enabling the row-sharded exact DrX path with OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE < OAT_ZERO_NUM_SAMPLES." >&2
    exit 1
  fi
  if (( ROW_SHARDED_EXACT_DRX == 1 )) && (( N_GPU != NUM_SAMPLES )); then
    echo "The row-sharded exact DrX path currently requires OAT_ZERO_N_GPU == OAT_ZERO_NUM_SAMPLES so each rank owns exactly one candidate row." >&2
    exit 1
  fi
fi

mkdir -p \
  "$SAVE_PATH" \
  "$ROOT_DIR/var/artifacts/logs" \
  "$ROOT_DIR/var/cache" \
  "$ROOT_DIR/var/tmp" \
  "$LOCAL_ROOT" \
  "$LOCAL_CACHE_ROOT" \
  "$LOCAL_JOB_ROOT"

runtime_probe() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import importlib.metadata as md
import json
import sys

packages = [
    "torch",
    "transformers",
    "vllm",
    "oat-llm",
    "deepspeed",
    "flash-attn",
    "math-verify",
    "fire",
]
data = {"python": sys.version.split()[0]}
for name in packages:
    try:
        data[name] = md.version(name)
    except Exception:
        data[name] = "MISSING"
print(json.dumps(data, sort_keys=True))
PY
}

runtime_matches_expected() {
  local python_bin="$1"
  [[ -x "$python_bin" ]] || return 1
  OAT_ZERO_EXPECTED_PYTHON_PREFIX="$EXPECTED_PYTHON_PREFIX" \
  OAT_ZERO_EXPECTED_TORCH_VERSION="$EXPECTED_TORCH_VERSION" \
  OAT_ZERO_EXPECTED_TRANSFORMERS_VERSION="$EXPECTED_TRANSFORMERS_VERSION" \
  OAT_ZERO_EXPECTED_VLLM_VERSION="$EXPECTED_VLLM_VERSION" \
  OAT_ZERO_EXPECTED_OAT_VERSION="$EXPECTED_OAT_VERSION" \
  OAT_ZERO_EXPECTED_DEEPSPEED_VERSION="$EXPECTED_DEEPSPEED_VERSION" \
  OAT_ZERO_EXPECTED_MATH_VERIFY_VERSION="$EXPECTED_MATH_VERIFY_VERSION" \
  OAT_ZERO_EXPECTED_FIRE_VERSION="$EXPECTED_FIRE_VERSION" \
  "$python_bin" - <<'PY'
import importlib.metadata as md
import os
import sys

expected = {
    "torch": os.environ["OAT_ZERO_EXPECTED_TORCH_VERSION"],
    "transformers": os.environ["OAT_ZERO_EXPECTED_TRANSFORMERS_VERSION"],
    "vllm": os.environ["OAT_ZERO_EXPECTED_VLLM_VERSION"],
    "oat-llm": os.environ["OAT_ZERO_EXPECTED_OAT_VERSION"],
    "deepspeed": os.environ["OAT_ZERO_EXPECTED_DEEPSPEED_VERSION"],
    "math-verify": os.environ["OAT_ZERO_EXPECTED_MATH_VERIFY_VERSION"],
    "fire": os.environ["OAT_ZERO_EXPECTED_FIRE_VERSION"],
}
python_prefix = os.environ["OAT_ZERO_EXPECTED_PYTHON_PREFIX"]
if not sys.version.startswith(python_prefix):
    raise SystemExit(1)
for name, version in expected.items():
    if md.version(name) != version:
        raise SystemExit(1)
PY
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing canonical OAT python at $PYTHON_BIN" >&2
  exit 1
fi

if ! runtime_matches_expected "$PYTHON_BIN"; then
  echo "[oat-zero-exact] runtime mismatch: expected the canonical README-flash paper310 stack." >&2
  runtime_probe "$PYTHON_BIN" >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TMPDIR="${TMPDIR:-$LOCAL_JOB_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LOCAL_CACHE_ROOT/triton_${SLURM_JOB_ID:-manual}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$LOCAL_CACHE_ROOT/torch_extensions_${SLURM_JOB_ID:-manual}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$LOCAL_CACHE_ROOT/torchinductor_${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${LOCAL_CACHE_ROOT}/xdg_${SLURM_JOB_ID:-manual}"
export PIP_CACHE_DIR="${LOCAL_CACHE_ROOT}/pip_${SLURM_JOB_ID:-manual}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
mkdir -p \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR"

if [[ -d "$PYTHON_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[oat-zero-exact] probing python libdir"
python_libdir="$("$PYTHON_BIN" - <<'PY' | tail -n 1 | tr -d '\r'
import sysconfig

print(sysconfig.get_config_var("LIBDIR") or "")
PY
)"
if [[ -n "$python_libdir" ]]; then
  export LD_LIBRARY_PATH="${python_libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[oat-zero-exact] probing torch cuda version"
torch_cuda_version="$("$PYTHON_BIN" - <<'PY' | tail -n 1 | tr -d '\r'
import torch

print(torch.version.cuda or "")
PY
)"
echo "[oat-zero-exact] torch_cuda_version=${torch_cuda_version:-cpu}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  candidate_nvcc=""
  if [[ -n "${OAT_ZERO_CUDA_HOME:-}" && -x "${OAT_ZERO_CUDA_HOME}/bin/nvcc" ]]; then
    candidate_nvcc="${OAT_ZERO_CUDA_HOME}/bin/nvcc"
  fi
  if [[ -z "$candidate_nvcc" ]] && [[ -n "$torch_cuda_version" ]] && [[ -x "/usr/local/cuda-${torch_cuda_version}/bin/nvcc" ]]; then
    candidate_nvcc="/usr/local/cuda-${torch_cuda_version}/bin/nvcc"
  fi
  if [[ -z "$candidate_nvcc" ]] && command -v nvcc >/dev/null 2>&1; then
    candidate_nvcc="$(command -v nvcc)"
  fi
  if [[ -z "$candidate_nvcc" ]]; then
    for fallback_nvcc in \
      /usr/local/cuda-12.4/bin/nvcc \
      /usr/local/cuda/bin/nvcc \
      /usr/local/cuda-13.1/bin/nvcc
    do
      if [[ -x "$fallback_nvcc" ]]; then
        candidate_nvcc="$fallback_nvcc"
        break
      fi
    done
  fi
  if [[ -n "$candidate_nvcc" ]]; then
    export CUDA_HOME="$(cd "$(dirname "$candidate_nvcc")/.." && pwd)"
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:${PATH}"
  if [[ -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

echo "[oat-zero-exact] probing torch runtime key"
flash_attn_runtime_key="$("$PYTHON_BIN" - <<'PY' | tail -n 1 | tr -d '\r'
import re
import sys

import torch

torch_version = re.sub(r"[^0-9A-Za-z._-]+", "_", torch.__version__)
cuda_version = re.sub(r"[^0-9A-Za-z._-]+", "_", torch.version.cuda or "cpu")
print(f"py{sys.version_info.major}{sys.version_info.minor}_torch{torch_version}_cu{cuda_version}")
PY
)"
FLASH_ATTN_OVERLAY_DIR="${OAT_ZERO_FLASH_ATTN_OVERLAY_DIR:-$FLASH_ATTN_CACHE_ROOT/$flash_attn_runtime_key}"

prepend_pythonpath() {
  local new_path="$1"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${new_path}:$PYTHONPATH"
  else
    export PYTHONPATH="${new_path}"
  fi
}

prepend_pythonpath "$SOURCE_ROOT"

flash_attn_importable() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import flash_attn
PY
}

flash_attn_location() {
  "$PYTHON_BIN" - <<'PY' | tail -n 1 | tr -d '\r'
import flash_attn

print(flash_attn.__file__)
PY
}

bootstrap_flash_attn() {
  local install_lock="$FLASH_ATTN_OVERLAY_DIR.install.lock"

  mkdir -p "$FLASH_ATTN_OVERLAY_DIR"
  prepend_pythonpath "$FLASH_ATTN_OVERLAY_DIR"

  if flash_attn_importable; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[oat-zero-exact] flash-attn bootstrap requires a visible GPU." >&2
    return 1
  fi

  if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "[oat-zero-exact] missing CUDA_HOME nvcc for flash-attn build." >&2
    return 1
  fi

  exec 9>"$install_lock"
  flock 9

  if flash_attn_importable; then
    return 0
  fi

  echo "[oat-zero-exact] flash-attn build using CUDA_HOME=${CUDA_HOME}"
  "${CUDA_HOME}/bin/nvcc" --version | tail -n 1
  echo "[oat-zero-exact] installing flash-attn==${FLASH_ATTN_VERSION} into ${FLASH_ATTN_OVERLAY_DIR}"
  MAX_JOBS="$FLASH_ATTN_MAX_JOBS" \
    "$PYTHON_BIN" -m pip install \
      --upgrade \
      --no-build-isolation \
      --no-deps \
      --target "$FLASH_ATTN_OVERLAY_DIR" \
      "flash-attn==${FLASH_ATTN_VERSION}"

  flash_attn_importable
}

if [[ "$ENABLE_FLASH_ATTN" == "1" ]]; then
  echo "[oat-zero-exact] checking flash-attn availability"
  if ! flash_attn_importable; then
    bootstrap_flash_attn
  fi
  flash_attn_path="$(flash_attn_location)"
  flash_attn_flag=(--flash-attn)
else
  echo "[oat-zero-exact] flash-attn disabled"
  flash_attn_path="disabled"
  flash_attn_flag=(--no-flash-attn)
fi

echo "[oat-zero-exact] host=$(hostname)"
echo "[oat-zero-exact] source_root=${SOURCE_ROOT}"
echo "[oat-zero-exact] python=${PYTHON_BIN}"
echo "[oat-zero-exact] runtime_probe=$(runtime_probe "$PYTHON_BIN")"
echo "[oat-zero-exact] local_root=${LOCAL_ROOT}"
echo "[oat-zero-exact] local_job_root=${LOCAL_JOB_ROOT}"
echo "[oat-zero-exact] save_path=${SAVE_PATH}"
echo "[oat-zero-exact] resume_dir=${RESUME_DIR:-<unset>}"
echo "[oat-zero-exact] resume_tag=${RESUME_TAG:-<unset>}"
echo "[oat-zero-exact] wb_project=${WB_PROJECT}"
echo "[oat-zero-exact] wb_run_name=${WB_RUN_NAME}"
echo "[oat-zero-exact] wandb_run_id=${OAT_ZERO_WANDB_RUN_ID:-${WANDB_RUN_ID:-<unset>}}"
echo "[oat-zero-exact] use_wb=${USE_WB}"
echo "[oat-zero-exact] prompt_data=${PROMPT_DATA}"
echo "[oat-zero-exact] eval_data=${EVAL_DATA}"
echo "[oat-zero-exact] pretrain=${PRETRAIN}"
echo "[oat-zero-exact] verifier_version=${VERIFIER_VERSION}"
echo "[oat-zero-exact] input_key=${INPUT_KEY}"
echo "[oat-zero-exact] output_key=${OUTPUT_KEY}"
echo "[oat-zero-exact] max_train=${MAX_TRAIN}"
echo "[oat-zero-exact] max_queries=${MAX_QUERIES}"
echo "[oat-zero-exact] prompt_template=${PROMPT_TEMPLATE}"
echo "[oat-zero-exact] num_gpus_per_actor=${NUM_GPUS_PER_ACTOR}"
echo "[oat-zero-exact] prompt_max_length=${PROMPT_MAX_LENGTH}"
echo "[oat-zero-exact] generate_max_length=${GENERATE_MAX_LENGTH}"
echo "[oat-zero-exact] sampling_temperature=${SAMPLING_TEMPERATURE}"
echo "[oat-zero-exact] sampling_top_p=${SAMPLING_TOP_P}"
echo "[oat-zero-exact] objective=${OBJECTIVE}"
echo "[oat-zero-exact] policy_entropy_coef=${POLICY_ENTROPY_COEF}"
echo "[oat-zero-exact] xdr_tau=${XDR_TAU}"
echo "[oat-zero-exact] seed_entropy_alpha=${SEED_ENTROPY_ALPHA}"
echo "[oat-zero-exact] collocate=${COLLOCATE}"
echo "[oat-zero-exact] zero_stage=${ZERO_STAGE}"
echo "[oat-zero-exact] adam_offload=${ADAM_OFFLOAD}"
echo "[oat-zero-exact] activation_offloading=${ACTIVATION_OFFLOADING}"
echo "[oat-zero-exact] disable_trace_cache=${DISABLE_TRACE_CACHE}"
echo "[oat-zero-exact] grad_accum_dtype=${GRAD_ACCUM_DTYPE:-<unset>}"
echo "[oat-zero-exact] vllm_sleep=${VLLM_SLEEP}"
echo "[oat-zero-exact] beta=${BETA}"
echo "[oat-zero-exact] num_samples=${NUM_SAMPLES}"
echo "[oat-zero-exact] train_batch_size=${TRAIN_BATCH_SIZE}"
echo "[oat-zero-exact] train_batch_size_per_device=${TRAIN_BATCH_SIZE_PER_DEVICE}"
echo "[oat-zero-exact] rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
echo "[oat-zero-exact] rollout_batch_size_per_device=${ROLLOUT_BATCH_SIZE_PER_DEVICE}"
echo "[oat-zero-exact] pi_buffer_maxlen_per_device=${PI_BUFFER_MAXLEN_PER_DEVICE}"
echo "[oat-zero-exact] vllm_gpu_ratio=${VLLM_GPU_RATIO}"
echo "[oat-zero-exact] max_norm=${MAX_NORM}"
echo "[oat-zero-exact] ds_allgather_bucket_size=${DEEPSPEED_ALLGATHER_BUCKET_SIZE:-<unset>}"
echo "[oat-zero-exact] ds_reduce_bucket_size=${DEEPSPEED_REDUCE_BUCKET_SIZE:-<unset>}"
echo "[oat-zero-exact] ds_overlap_comm=${DEEPSPEED_OVERLAP_COMM:-<unset>}"
echo "[oat-zero-exact] ds_contiguous_gradients=${DEEPSPEED_CONTIGUOUS_GRADIENTS:-<unset>}"
echo "[oat-zero-exact] ds_reduce_scatter=${DEEPSPEED_REDUCE_SCATTER:-<unset>}"
echo "[oat-zero-exact] ds_use_multi_rank_bucket_allreduce=${DEEPSPEED_USE_MULTI_RANK_BUCKET_ALLREDUCE:-<unset>}"
echo "[oat-zero-exact] ds_allgather_partitions=${DEEPSPEED_ALLGATHER_PARTITIONS:-<unset>}"
echo "[oat-zero-exact] pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "[oat-zero-exact] eval_steps=${EVAL_STEPS}"
echo "[oat-zero-exact] eval_batch_size=${EVAL_BATCH_SIZE}"
echo "[oat-zero-exact] eval_temperature=${EVAL_TEMPERATURE}"
echo "[oat-zero-exact] eval_generate_max_length=${EVAL_GENERATE_MAX_LENGTH}"
echo "[oat-zero-exact] eval_mode_coverage_k=${EVAL_MODE_COVERAGE_K}"
echo "[oat-zero-exact] eval_mode_coverage_temperature=${EVAL_MODE_COVERAGE_TEMPERATURE}"
echo "[oat-zero-exact] test_split=${TEST_SPLIT}"
echo "[oat-zero-exact] save_steps=${SAVE_STEPS}"
echo "[oat-zero-exact] save_from=${SAVE_FROM}"
echo "[oat-zero-exact] save_ckpt=${SAVE_CKPT}"
echo "[oat-zero-exact] save_initial_model=${SAVE_INITIAL_MODEL}"
echo "[oat-zero-exact] allow_no_save=${ALLOW_NO_SAVE}"
echo "[oat-zero-exact] require_full_eval_checkpoints=${REQUIRE_FULL_EVAL_CHECKPOINTS}"
echo "[oat-zero-exact] max_save_num=${MAX_SAVE_NUM}"
echo "[oat-zero-exact] max_save_mem=${MAX_SAVE_MEM}"
echo "[oat-zero-exact] allow_no_eval=${ALLOW_NO_EVAL}"
echo "[oat-zero-exact] max_eval_steps_allowed=${MAX_EVAL_STEPS_ALLOWED}"
echo "[oat-zero-exact] critic_type=${CRITIC_TYPE}"
echo "[oat-zero-exact] maxent_tau=${MAXENT_TAU}"
echo "[oat-zero-exact] maxent_tau_learnable=${MAXENT_TAU_LEARNABLE}"
echo "[oat-zero-exact] maxent_tau_controller_enabled=${MAXENT_TAU_CONTROLLER_ENABLED}"
echo "[oat-zero-exact] maxent_tau_adaptation_metric=${MAXENT_TAU_ADAPTATION_METRIC}"
echo "[oat-zero-exact] ignore_no_eos=${IGNORE_NO_EOS}"
echo "[oat-zero-exact] maxent_tau_target_metric=${MAXENT_TAU_TARGET_METRIC:-<unset>}"
echo "[oat-zero-exact] maxent_tau_target_metric_start=${MAXENT_TAU_TARGET_METRIC_START:-<unset>}"
echo "[oat-zero-exact] maxent_tau_target_metric_peak=${MAXENT_TAU_TARGET_METRIC_PEAK:-<unset>}"
echo "[oat-zero-exact] maxent_tau_target_metric_peak_step=${MAXENT_TAU_TARGET_METRIC_PEAK_STEP}"
echo "[oat-zero-exact] maxent_tau_target_metric_final=${MAXENT_TAU_TARGET_METRIC_FINAL:-<unset>}"
echo "[oat-zero-exact] maxent_tau_target_metric_horizon=${MAXENT_TAU_TARGET_METRIC_HORIZON}"
echo "[oat-zero-exact] maxent_q_temperature=${MAXENT_Q_TEMPERATURE}"
echo "[oat-zero-exact] maxent_q_epsilon=${MAXENT_Q_EPSILON}"
echo "[oat-zero-exact] maxent_candidate_kl_coef=${MAXENT_CANDIDATE_KL_COEF}"
echo "[oat-zero-exact] maxent_semantic_similarity_threshold=${MAXENT_SEMANTIC_SIMILARITY_THRESHOLD}"
echo "[oat-zero-exact] maxent_semantic_embedding_similarity_threshold=${MAXENT_SEMANTIC_EMBEDDING_SIMILARITY_THRESHOLD}"
echo "[oat-zero-exact] maxent_semantic_cluster_method=${MAXENT_SEMANTIC_CLUSTER_METHOD}"
echo "[oat-zero-exact] maxent_semantic_embedding_max_tokens=${MAXENT_SEMANTIC_EMBEDDING_MAX_TOKENS}"
echo "[oat-zero-exact] maxent_semantic_cluster_max_tokens=${MAXENT_SEMANTIC_CLUSTER_MAX_TOKENS}"
echo "[oat-zero-exact] maxent_semantic_spectral_max_clusters=${MAXENT_SEMANTIC_SPECTRAL_MAX_CLUSTERS}"
echo "[oat-zero-exact] maxent_semantic_spectral_eigengap_min=${MAXENT_SEMANTIC_SPECTRAL_EIGENGAP_MIN}"
echo "[oat-zero-exact] maxent_semantic_correctness_target_frac=${MAXENT_SEMANTIC_CORRECTNESS_TARGET_FRAC}"
echo "[oat-zero-exact] maxent_semantic_correctness_sharpness=${MAXENT_SEMANTIC_CORRECTNESS_SHARPNESS}"
echo "[oat-zero-exact] maxent_semantic_correctness_answer_level=${MAXENT_SEMANTIC_CORRECTNESS_ANSWER_LEVEL}"
echo "[oat-zero-exact] maxent_semantic_correctness_min_answer_count=${MAXENT_SEMANTIC_CORRECTNESS_MIN_ANSWER_COUNT}"
echo "[oat-zero-exact] maxent_semantic_remix_mode=${MAXENT_SEMANTIC_REMIX_MODE}"
echo "[oat-zero-exact] semantic_entropy_lambda=${SEMANTIC_ENTROPY_LAMBDA}"
echo "[oat-zero-exact] maxent_reward_shaping_alpha=${MAXENT_REWARD_SHAPING_ALPHA}"
echo "[oat-zero-exact] maxent_tiebreak_anchor=${MAXENT_TIEBREAK_ANCHOR}"
echo "[oat-zero-exact] maxent_tiebreak_clip_max=${MAXENT_TIEBREAK_CLIP_MAX}"
echo "[oat-zero-exact] maxent_competitive_mode_tau=${MAXENT_COMPETITIVE_MODE_TAU}"
echo "[oat-zero-exact] maxent_competitive_mode_gap=${MAXENT_COMPETITIVE_MODE_GAP}"
echo "[oat-zero-exact] maxent_competitive_mode_top_k=${MAXENT_COMPETITIVE_MODE_TOP_K}"
echo "[oat-zero-exact] maxent_competitive_mode_budget_max=${MAXENT_COMPETITIVE_MODE_BUDGET_MAX}"
echo "[oat-zero-exact] maxent_competitive_mode_budget_scale=${MAXENT_COMPETITIVE_MODE_BUDGET_SCALE}"
echo "[oat-zero-exact] maxent_competitive_mode_intra_tau=${MAXENT_COMPETITIVE_MODE_INTRA_TAU}"
echo "[oat-zero-exact] maxent_prompt_select_min_alpha_frac=${MAXENT_PROMPT_SELECT_MIN_ALPHA_FRAC}"
echo "[oat-zero-exact] maxent_competitive_mode_positive_only=${MAXENT_COMPETITIVE_MODE_POSITIVE_ONLY}"
echo "[oat-zero-exact] maxent_correctness_schedule_enabled=${MAXENT_CORRECTNESS_SCHEDULE_ENABLED}"
echo "[oat-zero-exact] maxent_correctness_schedule_ema_decay=${MAXENT_CORRECTNESS_SCHEDULE_EMA_DECAY}"
echo "[oat-zero-exact] maxent_correctness_schedule_low=${MAXENT_CORRECTNESS_SCHEDULE_LOW}"
echo "[oat-zero-exact] maxent_correctness_schedule_high=${MAXENT_CORRECTNESS_SCHEDULE_HIGH}"
echo "[oat-zero-exact] maxent_correctness_schedule_budget_max_early=${MAXENT_CORRECTNESS_SCHEDULE_BUDGET_MAX_EARLY}"
echo "[oat-zero-exact] maxent_correctness_schedule_budget_max_late=${MAXENT_CORRECTNESS_SCHEDULE_BUDGET_MAX_LATE}"
echo "[oat-zero-exact] maxent_correctness_schedule_prompt_select_min_alpha_frac_early=${MAXENT_CORRECTNESS_SCHEDULE_PROMPT_SELECT_MIN_ALPHA_FRAC_EARLY}"
echo "[oat-zero-exact] maxent_correctness_schedule_prompt_select_min_alpha_frac_late=${MAXENT_CORRECTNESS_SCHEDULE_PROMPT_SELECT_MIN_ALPHA_FRAC_LATE}"
echo "[oat-zero-exact] maxent_correctness_schedule_mode_tau_early=${MAXENT_CORRECTNESS_SCHEDULE_MODE_TAU_EARLY}"
echo "[oat-zero-exact] maxent_correctness_schedule_mode_tau_late=${MAXENT_CORRECTNESS_SCHEDULE_MODE_TAU_LATE}"
echo "[oat-zero-exact] maxent_correctness_schedule_intra_tau_early=${MAXENT_CORRECTNESS_SCHEDULE_INTRA_TAU_EARLY}"
echo "[oat-zero-exact] maxent_correctness_schedule_intra_tau_late=${MAXENT_CORRECTNESS_SCHEDULE_INTRA_TAU_LATE}"
echo "[oat-zero-exact] maxent_semantic_guard_max_expected_len_delta=${MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_LEN_DELTA}"
echo "[oat-zero-exact] maxent_semantic_guard_max_expected_format_drop=${MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_FORMAT_DROP}"
echo "[oat-zero-exact] maxent_exact_drx_weight_source=${MAXENT_EXACT_DRX_WEIGHT_SOURCE}"
echo "[oat-zero-exact] maxent_logprob_chunk_size=${MAXENT_LOGPROB_CHUNK_SIZE}"
echo "[oat-zero-exact] maxent_backward_chunk_size=${MAXENT_BACKWARD_CHUNK_SIZE}"
echo "[oat-zero-exact] maxent_backward_token_budget=${MAXENT_BACKWARD_TOKEN_BUDGET}"
echo "[oat-zero-exact] maxent_length_normalize_ref=${MAXENT_LENGTH_NORMALIZE_REF}"
echo "[oat-zero-exact] maxent_length_normalize_policy=${MAXENT_LENGTH_NORMALIZE_POLICY}"
echo "[oat-zero-exact] maxent_listwise_skip_zero_variance_groups=${MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS}"
echo "[oat-zero-exact] maxent_use_clip_objective=${MAXENT_USE_CLIP_OBJECTIVE}"
echo "[oat-zero-exact] maxent_clip_objective_coef=${MAXENT_CLIP_OBJECTIVE_COEF}"
echo "[oat-zero-exact] maxent_clip_preserve_reward_mass=${MAXENT_CLIP_PRESERVE_REWARD_MASS}"
echo "[oat-zero-exact] maxent_clip_mode=${MAXENT_CLIP_MODE}"
echo "[oat-zero-exact] maxent_token_clip_primary=${MAXENT_TOKEN_CLIP_PRIMARY}"
echo "[oat-zero-exact] maxent_drgrpo_token_primary=${MAXENT_DRGRPO_TOKEN_PRIMARY}"
echo "[oat-zero-exact] maxent_drgrpo_token_advantage_source=${MAXENT_DRGRPO_TOKEN_ADVANTAGE_SOURCE}"
echo "[oat-zero-exact] maxent_drgrpo_token_length_normalizer=${MAXENT_DRGRPO_TOKEN_LENGTH_NORMALIZER}"
echo "[oat-zero-exact] row_sharded_exact_drx=${ROW_SHARDED_EXACT_DRX}"
echo "[oat-zero-exact] maxent_sequence_aux_coef=${MAXENT_SEQUENCE_AUX_COEF}"
echo "[oat-zero-exact] maxent_sequence_aux_group_filter=${MAXENT_SEQUENCE_AUX_GROUP_FILTER}"
echo "[oat-zero-exact] maxent_sequence_aux_max_expected_len_drop=${MAXENT_SEQUENCE_AUX_MAX_EXPECTED_LEN_DROP}"
echo "[oat-zero-exact] maxent_sequence_aux_max_expected_len_gain=${MAXENT_SEQUENCE_AUX_MAX_EXPECTED_LEN_GAIN}"
echo "[oat-zero-exact] maxent_sequence_aux_max_expected_format_drop=${MAXENT_SEQUENCE_AUX_MAX_EXPECTED_FORMAT_DROP}"
echo "[oat-zero-exact] maxent_sequence_aux_min_expected_correctness_delta=${MAXENT_SEQUENCE_AUX_MIN_EXPECTED_CORRECTNESS_DELTA}"
echo "[oat-zero-exact] maxent_neutral_projection_coef=${MAXENT_NEUTRAL_PROJECTION_COEF}"
echo "[oat-zero-exact] maxent_branch_grad_diagnostics=${MAXENT_BRANCH_GRAD_DIAGNOSTICS}"
echo "[oat-zero-exact] maxent_branch_grad_diagnostics_interval=${MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL}"
echo "[oat-zero-exact] maxent_branch_grad_diagnostics_max_steps=${MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS}"
echo "[oat-zero-exact] maxent_reference_logprobs_source=${MAXENT_REFERENCE_LOGPROBS_SOURCE}"
echo "[oat-zero-exact] shm_size_mb=${SHM_SIZE_MB}"
echo "[oat-zero-exact] cuda_home=${CUDA_HOME:-unset}"
echo "[oat-zero-exact] enable_flash_attn=${ENABLE_FLASH_ATTN}"
echo "[oat-zero-exact] flash_attn_path=${flash_attn_path}"
echo "[oat-zero-exact] smoke_mode=${SMOKE_MODE}"
echo "[oat-zero-exact] bootstrap_only=${BOOTSTRAP_ONLY}"

if [[ "$BOOTSTRAP_ONLY" == "1" ]]; then
  exit 0
fi

save_initial_model_snapshot() {
  local save_dir="$SAVE_PATH/saved_models/step_00000"
  if [[ -f "$save_dir/config.json" ]]; then
    echo "[oat-zero-exact] initial model snapshot already exists at $save_dir"
    return 0
  fi

  echo "[oat-zero-exact] saving initial model snapshot to $save_dir"
  PRETRAIN="$PRETRAIN" INITIAL_SAVE_DIR="$save_dir" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

pretrain = os.environ["PRETRAIN"]
save_dir = Path(os.environ["INITIAL_SAVE_DIR"])
save_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(pretrain)
model = AutoModelForCausalLM.from_pretrained(
    pretrain,
    low_cpu_mem_usage=True,
    torch_dtype="auto",
)
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)
print(save_dir)
PY
}

if [[ "$SAVE_INITIAL_MODEL" == "1" ]]; then
  save_initial_model_snapshot
fi

maxent_length_normalize_ref_flag=(--no-maxent-length-normalize-ref)
if [[ "$MAXENT_LENGTH_NORMALIZE_REF" == "1" ]]; then
  maxent_length_normalize_ref_flag=(--maxent-length-normalize-ref)
fi

maxent_length_normalize_policy_flag=(--no-maxent-length-normalize-policy)
if [[ "$MAXENT_LENGTH_NORMALIZE_POLICY" == "1" ]]; then
  maxent_length_normalize_policy_flag=(--maxent-length-normalize-policy)
fi

maxent_skip_zero_variance_flag=(--no-maxent-listwise-skip-zero-variance-groups)
if [[ "$MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS" == "1" ]]; then
  maxent_skip_zero_variance_flag=(--maxent-listwise-skip-zero-variance-groups)
fi

maxent_competitive_mode_positive_only_flag=(--no-maxent-competitive-mode-positive-only)
if [[ "$MAXENT_COMPETITIVE_MODE_POSITIVE_ONLY" == "1" ]]; then
  maxent_competitive_mode_positive_only_flag=(--maxent-competitive-mode-positive-only)
fi

maxent_correctness_schedule_enabled_flag=(--no-maxent-correctness-schedule-enabled)
if [[ "$MAXENT_CORRECTNESS_SCHEDULE_ENABLED" == "1" ]]; then
  maxent_correctness_schedule_enabled_flag=(--maxent-correctness-schedule-enabled)
fi

maxent_use_clip_objective_flag=(--no-maxent-use-clip-objective)
if [[ "$MAXENT_USE_CLIP_OBJECTIVE" == "1" ]]; then
  maxent_use_clip_objective_flag=(--maxent-use-clip-objective)
fi

maxent_clip_preserve_reward_mass_flag=(--no-maxent-clip-preserve-reward-mass)
if [[ "$MAXENT_CLIP_PRESERVE_REWARD_MASS" == "1" ]]; then
  maxent_clip_preserve_reward_mass_flag=(--maxent-clip-preserve-reward-mass)
fi

maxent_token_clip_primary_flag=(--no-maxent-token-clip-primary)
if [[ "$MAXENT_TOKEN_CLIP_PRIMARY" == "1" ]]; then
  maxent_token_clip_primary_flag=(--maxent-token-clip-primary)
fi

maxent_drgrpo_token_primary_flag=(--no-maxent-drgrpo-token-primary)
if [[ "$MAXENT_DRGRPO_TOKEN_PRIMARY" == "1" ]]; then
  maxent_drgrpo_token_primary_flag=(--maxent-drgrpo-token-primary)
fi

maxent_semantic_correctness_answer_level_flag=(--no-maxent-semantic-correctness-answer-level)
if [[ "$MAXENT_SEMANTIC_CORRECTNESS_ANSWER_LEVEL" == "1" ]]; then
  maxent_semantic_correctness_answer_level_flag=(--maxent-semantic-correctness-answer-level)
fi

maxent_branch_grad_diagnostics_flag=(--no-maxent-branch-grad-diagnostics)
if [[ "$MAXENT_BRANCH_GRAD_DIAGNOSTICS" == "1" ]]; then
  maxent_branch_grad_diagnostics_flag=(--maxent-branch-grad-diagnostics)
fi

maxent_beta_controller_flag=(--no-maxent-beta-controller-enabled)
if [[ "$MAXENT_BETA_CONTROLLER_ENABLED" == "1" ]]; then
  maxent_beta_controller_flag=(--maxent-beta-controller-enabled)
fi

maxent_tau_learnable_flag=(--no-maxent-tau-learnable)
if [[ "$MAXENT_TAU_LEARNABLE" == "1" ]]; then
  maxent_tau_learnable_flag=(--maxent-tau-learnable)
fi

maxent_tau_controller_flag=(--no-maxent-tau-controller-enabled)
if [[ "$MAXENT_TAU_CONTROLLER_ENABLED" == "1" ]]; then
  maxent_tau_controller_flag=(--maxent-tau-controller-enabled)
fi

ignore_no_eos_flag=(--no-ignore-no-eos)
if [[ "$IGNORE_NO_EOS" == "1" ]]; then
  ignore_no_eos_flag=(--ignore-no-eos)
fi

objective_args=(
  --objective "$OBJECTIVE"
  --beta "$BETA"
  --policy-entropy-coef "$POLICY_ENTROPY_COEF"
  --xdr-tau "$XDR_TAU"
  --seed-entropy-alpha "$SEED_ENTROPY_ALPHA"
)
if [[ "$XDR_MODE_ADAPTIVE" == "1" ]]; then
  objective_args+=(--xdr-mode-adaptive)
fi

if [[ "$OBJECTIVE" == "maxent_listwise" ]]; then
  objective_args+=(
    --maxent-tau "$MAXENT_TAU"
    "${maxent_tau_learnable_flag[@]}"
    "${maxent_tau_controller_flag[@]}"
    --maxent-q-temperature "$MAXENT_Q_TEMPERATURE"
    --maxent-q-epsilon "$MAXENT_Q_EPSILON"
    --maxent-candidate-kl-coef "$MAXENT_CANDIDATE_KL_COEF"
    --maxent-semantic-similarity-threshold "$MAXENT_SEMANTIC_SIMILARITY_THRESHOLD"
    --maxent-semantic-embedding-similarity-threshold "$MAXENT_SEMANTIC_EMBEDDING_SIMILARITY_THRESHOLD"
    --maxent-semantic-cluster-method "$MAXENT_SEMANTIC_CLUSTER_METHOD"
    --maxent-semantic-embedding-max-tokens "$MAXENT_SEMANTIC_EMBEDDING_MAX_TOKENS"
    --maxent-semantic-cluster-max-tokens "$MAXENT_SEMANTIC_CLUSTER_MAX_TOKENS"
    --maxent-semantic-spectral-max-clusters "$MAXENT_SEMANTIC_SPECTRAL_MAX_CLUSTERS"
    --maxent-semantic-spectral-eigengap-min "$MAXENT_SEMANTIC_SPECTRAL_EIGENGAP_MIN"
    --maxent-semantic-correctness-target-frac "$MAXENT_SEMANTIC_CORRECTNESS_TARGET_FRAC"
    --maxent-semantic-correctness-sharpness "$MAXENT_SEMANTIC_CORRECTNESS_SHARPNESS"
    "${maxent_semantic_correctness_answer_level_flag[@]}"
    --maxent-semantic-correctness-min-answer-count "$MAXENT_SEMANTIC_CORRECTNESS_MIN_ANSWER_COUNT"
    --maxent-semantic-remix-mode "$MAXENT_SEMANTIC_REMIX_MODE"
    --semantic-entropy-lambda "$SEMANTIC_ENTROPY_LAMBDA"
    --maxent-reward-shaping-alpha "$MAXENT_REWARD_SHAPING_ALPHA"
    --maxent-tiebreak-anchor "$MAXENT_TIEBREAK_ANCHOR"
    --maxent-tiebreak-clip-max "$MAXENT_TIEBREAK_CLIP_MAX"
    --maxent-competitive-mode-tau "$MAXENT_COMPETITIVE_MODE_TAU"
    --maxent-competitive-mode-gap "$MAXENT_COMPETITIVE_MODE_GAP"
    --maxent-competitive-mode-top-k "$MAXENT_COMPETITIVE_MODE_TOP_K"
    --maxent-competitive-mode-budget-max "$MAXENT_COMPETITIVE_MODE_BUDGET_MAX"
    --maxent-competitive-mode-budget-scale "$MAXENT_COMPETITIVE_MODE_BUDGET_SCALE"
    --maxent-competitive-mode-intra-tau "$MAXENT_COMPETITIVE_MODE_INTRA_TAU"
    --maxent-prompt-select-min-alpha-frac "$MAXENT_PROMPT_SELECT_MIN_ALPHA_FRAC"
    "${maxent_competitive_mode_positive_only_flag[@]}"
    "${maxent_correctness_schedule_enabled_flag[@]}"
    --maxent-correctness-schedule-ema-decay "$MAXENT_CORRECTNESS_SCHEDULE_EMA_DECAY"
    --maxent-correctness-schedule-low "$MAXENT_CORRECTNESS_SCHEDULE_LOW"
    --maxent-correctness-schedule-high "$MAXENT_CORRECTNESS_SCHEDULE_HIGH"
    --maxent-correctness-schedule-budget-max-early "$MAXENT_CORRECTNESS_SCHEDULE_BUDGET_MAX_EARLY"
    --maxent-correctness-schedule-budget-max-late "$MAXENT_CORRECTNESS_SCHEDULE_BUDGET_MAX_LATE"
    --maxent-correctness-schedule-prompt-select-min-alpha-frac-early "$MAXENT_CORRECTNESS_SCHEDULE_PROMPT_SELECT_MIN_ALPHA_FRAC_EARLY"
    --maxent-correctness-schedule-prompt-select-min-alpha-frac-late "$MAXENT_CORRECTNESS_SCHEDULE_PROMPT_SELECT_MIN_ALPHA_FRAC_LATE"
    --maxent-correctness-schedule-mode-tau-early "$MAXENT_CORRECTNESS_SCHEDULE_MODE_TAU_EARLY"
    --maxent-correctness-schedule-mode-tau-late "$MAXENT_CORRECTNESS_SCHEDULE_MODE_TAU_LATE"
    --maxent-correctness-schedule-intra-tau-early "$MAXENT_CORRECTNESS_SCHEDULE_INTRA_TAU_EARLY"
    --maxent-correctness-schedule-intra-tau-late "$MAXENT_CORRECTNESS_SCHEDULE_INTRA_TAU_LATE"
    --maxent-semantic-guard-max-expected-len-delta "$MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_LEN_DELTA"
    --maxent-semantic-guard-max-expected-format-drop "$MAXENT_SEMANTIC_GUARD_MAX_EXPECTED_FORMAT_DROP"
    --maxent-exact-drx-weight-source "$MAXENT_EXACT_DRX_WEIGHT_SOURCE"
    --maxent-clip-objective-coef "$MAXENT_CLIP_OBJECTIVE_COEF"
    --maxent-clip-mode "$MAXENT_CLIP_MODE"
    --maxent-reference-logprobs-source "$MAXENT_REFERENCE_LOGPROBS_SOURCE"
    --maxent-tau-adaptation-metric "$MAXENT_TAU_ADAPTATION_METRIC"
    --maxent-tau-lr "$MAXENT_TAU_LR"
    --maxent-tau-min "$MAXENT_TAU_MIN"
    --maxent-tau-max "$MAXENT_TAU_MAX"
    --maxent-tau-warmup-steps "$MAXENT_TAU_WARMUP_STEPS"
    --maxent-tau-target-metric-horizon "$MAXENT_TAU_TARGET_METRIC_HORIZON"
    --kl_target "$KL_TARGET"
    --kl_horizon "$KL_HORIZON"
    --kl_ctl_step_size "$KL_CTL_STEP_SIZE"
    "${maxent_length_normalize_ref_flag[@]}"
    "${maxent_length_normalize_policy_flag[@]}"
    "${maxent_skip_zero_variance_flag[@]}"
    "${maxent_use_clip_objective_flag[@]}"
    "${maxent_clip_preserve_reward_mass_flag[@]}"
    "${maxent_token_clip_primary_flag[@]}"
    "${maxent_drgrpo_token_primary_flag[@]}"
    --maxent-drgrpo-token-advantage-source "$MAXENT_DRGRPO_TOKEN_ADVANTAGE_SOURCE"
    --maxent-drgrpo-token-length-normalizer "$MAXENT_DRGRPO_TOKEN_LENGTH_NORMALIZER"
    --maxent-sequence-aux-coef "$MAXENT_SEQUENCE_AUX_COEF"
    --maxent-sequence-aux-group-filter "$MAXENT_SEQUENCE_AUX_GROUP_FILTER"
    --maxent-sequence-aux-max-expected-len-drop "$MAXENT_SEQUENCE_AUX_MAX_EXPECTED_LEN_DROP"
    --maxent-sequence-aux-max-expected-len-gain "$MAXENT_SEQUENCE_AUX_MAX_EXPECTED_LEN_GAIN"
    --maxent-sequence-aux-max-expected-format-drop "$MAXENT_SEQUENCE_AUX_MAX_EXPECTED_FORMAT_DROP"
    --maxent-sequence-aux-min-expected-correctness-delta "$MAXENT_SEQUENCE_AUX_MIN_EXPECTED_CORRECTNESS_DELTA"
    --maxent-neutral-projection-coef "$MAXENT_NEUTRAL_PROJECTION_COEF"
    "${maxent_branch_grad_diagnostics_flag[@]}"
    --maxent-branch-grad-diagnostics-interval "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL"
    --maxent-branch-grad-diagnostics-max-steps "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS"
    "${maxent_beta_controller_flag[@]}"
  )
  if [[ -n "$MAXENT_TAU_TARGET_METRIC" ]]; then
    objective_args+=(
      --maxent-tau-target-metric "$MAXENT_TAU_TARGET_METRIC"
    )
  fi
  if [[ -n "$MAXENT_TAU_TARGET_METRIC_START" ]]; then
    objective_args+=(
      --maxent-tau-target-metric-start "$MAXENT_TAU_TARGET_METRIC_START"
    )
  fi
  if [[ -n "$MAXENT_TAU_TARGET_METRIC_PEAK" ]]; then
    objective_args+=(
      --maxent-tau-target-metric-peak "$MAXENT_TAU_TARGET_METRIC_PEAK"
      --maxent-tau-target-metric-peak-step "$MAXENT_TAU_TARGET_METRIC_PEAK_STEP"
    )
  fi
  if [[ -n "$MAXENT_TAU_TARGET_METRIC_FINAL" ]]; then
    objective_args+=(
      --maxent-tau-target-metric-final "$MAXENT_TAU_TARGET_METRIC_FINAL"
    )
  fi
  if [[ "$MAXENT_LOGPROB_CHUNK_SIZE" -gt 0 ]]; then
    objective_args+=(
      --maxent-logprob-chunk-size "$MAXENT_LOGPROB_CHUNK_SIZE"
    )
  fi
  if [[ "$MAXENT_BACKWARD_CHUNK_SIZE" -gt 0 ]]; then
    objective_args+=(
      --maxent-backward-chunk-size "$MAXENT_BACKWARD_CHUNK_SIZE"
    )
  fi
  if [[ -n "${OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET+x}" ]] || [[ "$MAXENT_BACKWARD_TOKEN_BUDGET" -gt 0 ]]; then
    objective_args+=(
      --maxent-backward-token-budget "$MAXENT_BACKWARD_TOKEN_BUDGET"
    )
  fi
fi

seed_args=()
if [[ "$RND_SEED" == "1" ]]; then
  seed_args+=(--rnd-seed)
else
  seed_args+=(--no-rnd-seed --seed "$SEED")
fi

cmd=(
  "$PYTHON_BIN" -m "$TRAINER_MODULE"
  --critic_type "$CRITIC_TYPE"
  --gpus "$N_GPU"
  --num_gpus_per_actor "$NUM_GPUS_PER_ACTOR"
  --enable_prefix_caching
  --vllm_gpu_ratio "$VLLM_GPU_RATIO"
  "${flash_attn_flag[@]}"
  --shm_size_mb "$SHM_SIZE_MB"
  --gradient-checkpointing
  --bf16
  "${seed_args[@]}"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler constant
  --num_ppo_epochs "$NUM_PPO_EPOCHS"
  --max_norm "$MAX_NORM"
  "${ignore_no_eos_flag[@]}"
  "${objective_args[@]}"
  --oracle_type reward
  --oracle math
  --pretrain "$PRETRAIN"
  --prompt_template "$PROMPT_TEMPLATE"
  --verifier_version "$VERIFIER_VERSION"
  --zero-stage "$ZERO_STAGE"
  --ref_offload
  --prompt_data "$PROMPT_DATA"
  --train_split train
  --input_key "$INPUT_KEY"
  --output_key "$OUTPUT_KEY"
  --max-train "$MAX_TRAIN"
  --max_queries "$MAX_QUERIES"
  --num_prompt_epoch "$NUM_PROMPT_EPOCH"
  --prompt_max_length "$PROMPT_MAX_LENGTH"
  --num_samples "$NUM_SAMPLES"
  --temperature "$SAMPLING_TEMPERATURE"
  --top_p "$SAMPLING_TOP_P"
  --generate_max_length "$GENERATE_MAX_LENGTH"
  --save_path "$SAVE_PATH"
  --save_steps "$SAVE_STEPS"
  --save_from "$SAVE_FROM"
  --max_save_num "$MAX_SAVE_NUM"
  --max_save_mem "$MAX_SAVE_MEM"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --train_batch_size_per_device "$TRAIN_BATCH_SIZE_PER_DEVICE"
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
  --rollout_batch_size_per_device "$ROLLOUT_BATCH_SIZE_PER_DEVICE"
  --pi_buffer_maxlen_per_device "$PI_BUFFER_MAXLEN_PER_DEVICE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --eval_steps "$EVAL_STEPS"
  --eval_temperature "$EVAL_TEMPERATURE"
  --eval_generate_max_length "$EVAL_GENERATE_MAX_LENGTH"
  --eval_mode_coverage_k "$EVAL_MODE_COVERAGE_K"
  --eval_mode_coverage_temperature "$EVAL_MODE_COVERAGE_TEMPERATURE"
  --eval_data "$EVAL_DATA"
  --eval_input_key input
  --test_split "$TEST_SPLIT"
)

if [[ -n "$RESUME_DIR" ]]; then
  cmd+=(
    --resume_dir "$RESUME_DIR"
  )
  if [[ -n "$RESUME_TAG" ]]; then
    cmd+=(
      --resume_tag "$RESUME_TAG"
    )
  fi
fi

if [[ "$VLLM_SLEEP" == "1" ]]; then
  cmd+=(--vllm_sleep)
fi

if [[ "$COLLOCATE" == "1" ]]; then
  cmd+=(--collocate)
fi

if [[ "$ADAM_OFFLOAD" == "1" ]]; then
  cmd+=(--adam_offload)
fi

if [[ "$ACTIVATION_OFFLOADING" == "1" ]]; then
  cmd+=(--activation_offloading)
fi

if [[ "$DISABLE_TRACE_CACHE" == "1" ]]; then
  cmd+=(--disable_trace_cache)
fi

if [[ "$SAVE_CKPT" == "1" ]]; then
  cmd+=(--save-ckpt)
fi

if [[ -n "$GRAD_ACCUM_DTYPE" ]]; then
  cmd+=(--grad_accum_dtype "$GRAD_ACCUM_DTYPE")
fi

if [[ "$USE_WB" == "1" ]]; then
  cmd+=(
    --use-wb
    --wb_project "$WB_PROJECT"
    --wb-run-name "$WB_RUN_NAME"
  )
fi

echo "[oat-zero-exact] ${cmd[*]}"

if [[ "$SMOKE_MODE" != "1" ]]; then
  exec "${cmd[@]}"
fi

RUN_LOG="${ROOT_DIR}/var/artifacts/logs/${WB_RUN_NAME}.inner.log"
rm -f "$RUN_LOG"
touch "$RUN_LOG"

setsid bash -lc 'exec "$@"' _ "${cmd[@]}" \
  > >(tee -a "$RUN_LOG") \
  2> >(tee -a "$RUN_LOG" >&2) &
child_pid=$!
child_pgid=$child_pid

start_ts="$(date +%s)"
smoke_success=0

while kill -0 "$child_pid" >/dev/null 2>&1; do
  if rg -q --fixed-strings "$SMOKE_SUCCESS_PATTERN" "$RUN_LOG"; then
    smoke_success=1
    break
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts >= SMOKE_TIMEOUT_SECONDS )); then
    break
  fi
  sleep 10
done

if (( smoke_success == 1 )); then
  echo "[oat-zero-exact] smoke success pattern observed; stopping exact upstream run cleanly."
  kill -INT -- "-${child_pgid}" >/dev/null 2>&1 || true
  wait "$child_pid" || true
  exit 0
fi

if kill -0 "$child_pid" >/dev/null 2>&1; then
  echo "[oat-zero-exact] smoke run timed out before success pattern." >&2
  kill -TERM -- "-${child_pgid}" >/dev/null 2>&1 || true
  wait "$child_pid" || true
  exit 1
fi

wait "$child_pid"
