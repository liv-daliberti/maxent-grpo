#!/usr/bin/env bash
# Submit the paper's matched comparative on exact multi-answer ModeBench data
# (including the externally executed Python-factor domain) or an explicitly
# supplied MATH artifact: Dr.GRPO (baseline) vs
# xDr.GRPO (tau sweep), with optional entropy-
# feedback tau, Token-MaxEnt, SEED, and semantic adaptive controls. All arms
# share the same data, model, prompt format, rollout
# budget G, and optimization settings; only the aggregation weight / extra
# regularizer differs. Dr.GRPO is the xdr tau=inf endpoint of the same code
# path (the grpo arm simply leaves OAT_ZERO_XDR_TAU=inf).
#
# After the training jobs finish, evaluate with:
#   ops/run_countdown_comparative_eval.sh   (see header there)
# and analyze with:
#   ops/analyze_countdown_comparative.py
set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STAMP_PREFIX="${RUN_STAMP_PREFIX:-countdown_comparative_$(date +%Y%m%d_%H%M%S)}"
TASK="${OAT_ZERO_COMPARATIVE_TASK:-countdown}"
# Model scale for every arm (run_experiment.sh preset name).
MODEL="${OAT_ZERO_COMPARATIVE_MODEL:-qwen2.5-0.5b-instruct}"
# countdown data preset: easy3 (3 numbers, 2-8 modes, 384/128 — calibrated for
# 0.5B) or countdown4 (4 numbers, 2-8 modes, 1024/256 — the larger pool for
# 3B-scale models; a 0.5B gets no reward signal on it).
DATA_PRESET="${OAT_ZERO_COMPARATIVE_DATA_PRESET:-easy3}"
TRAIN_SEEDS_CSV="${OAT_ZERO_TRAIN_SEEDS:-43,44,45}"
XDR_TAUS_CSV="${OAT_ZERO_XDR_TAUS:-0.05,0.1,0.25,0.5,1,2}"
INCLUDE_TOKEN_ENTROPY_ARM="${OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM:-1}"
INCLUDE_SEED_ARM="${OAT_ZERO_INCLUDE_SEED_ARM:-1}"
# Mode-adaptive tempering arm: tau_x = tau0/log(1+kappa_x) from observed
# distinct correct modes per group.
INCLUDE_XDR_ADAPT_ARM="${OAT_ZERO_INCLUDE_XDR_ADAPT_ARM:-0}"
XDR_ADAPT_TAU0="${OAT_ZERO_XDR_ADAPT_TAU0:-0.05}"
# Label-free controller: calibrate a token-entropy target during a fixed-tau
# warmup, then lower candidate-aggregation tau only below that target.
INCLUDE_XDR_TAU_CONTROL_ARM="${OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM:-0}"
INCLUDE_XDR_SAC_DUAL_ARM="${OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM:-0}"
INCLUDE_MAXENT_ARM="${OAT_ZERO_INCLUDE_MAXENT_ARM:-0}"
INCLUDE_MAXENT_CONTROL_ARM="${OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM:-0}"
INCLUDE_MAXENT_DUAL_ARM="${OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM:-0}"
INCLUDE_MAXENT_INVERSE_ARM="${OAT_ZERO_INCLUDE_MAXENT_INVERSE_ARM:-0}"
INCLUDE_MAXENT_INVERSE_CANONICAL_ARM="${OAT_ZERO_INCLUDE_MAXENT_INVERSE_CANONICAL_ARM:-0}"
INCLUDE_MAXENT_INVERSE_CANONICAL_REPLAY_ARM="${OAT_ZERO_INCLUDE_MAXENT_INVERSE_CANONICAL_REPLAY_ARM:-0}"
INCLUDE_OPEN_SET_SPLIT_CANONICAL_ARM="${OAT_ZERO_INCLUDE_OPEN_SET_SPLIT_CANONICAL_ARM:-0}"
INCLUDE_VERIFIED_FIRST_SPLIT_CANONICAL_ARM="${OAT_ZERO_INCLUDE_VERIFIED_FIRST_SPLIT_CANONICAL_ARM:-0}"
INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM="${OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM:-0}"
INCLUDE_VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL_ARM="${OAT_ZERO_INCLUDE_VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL_ARM:-0}"
INCLUDE_VERIFIED_COUNTERFACTUAL_CANONICAL_ARM="${OAT_ZERO_INCLUDE_VERIFIED_COUNTERFACTUAL_CANONICAL_ARM:-0}"
INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM="${OAT_ZERO_INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM:-0}"
INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM="${OAT_ZERO_INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM:-0}"
INCLUDE_MAXENT_LENGTH_DUAL_ARM="${OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM:-0}"
INCLUDE_DIAYN_ARM="${OAT_ZERO_INCLUDE_DIAYN_ARM:-0}"
INCLUDE_OUTCOME_COLLISION_ARM="${OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM:-0}"
INCLUDE_OUTCOME_COLLISION_OUTSIDE_CENTERING_ARM="${OAT_ZERO_INCLUDE_OUTCOME_COLLISION_OUTSIDE_CENTERING_ARM:-0}"
INCLUDE_SEMANTIC_SHANNON_ARM="${OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM:-0}"
INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM="${OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM:-0}"
INCLUDE_QUALITY_GATED_SEMANTIC_NOVELTY_ARM="${OAT_ZERO_INCLUDE_QUALITY_GATED_SEMANTIC_NOVELTY_ARM:-0}"
INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM="${OAT_ZERO_INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM:-0}"
INCLUDE_SIGNAL_FIRST_SEMANTIC_BALANCE_ARM="${OAT_ZERO_INCLUDE_SIGNAL_FIRST_SEMANTIC_BALANCE_ARM:-0}"
INCLUDE_ONLINE_CANONICAL_MAXENT_ARM="${OAT_ZERO_INCLUDE_ONLINE_CANONICAL_MAXENT_ARM:-0}"
INCLUDE_ONLINE_CANONICAL_HAARNOJA_ARM="${OAT_ZERO_INCLUDE_ONLINE_CANONICAL_HAARNOJA_ARM:-0}"
INCLUDE_ONLINE_CANONICAL_POLICY_ENTROPY_ARM="${OAT_ZERO_INCLUDE_ONLINE_CANONICAL_POLICY_ENTROPY_ARM:-0}"
SIGNAL_FIRST_XDR_TAU="${OAT_ZERO_SIGNAL_FIRST_XDR_TAU:-0.05}"
XDR_TAU_CONTROL_BASE="${OAT_ZERO_XDR_TAU_CONTROL_BASE:-0.05}"
XDR_TAU_CONTROL_RATIO="${OAT_ZERO_XDR_TAU_CONTROL_RATIO:-0.8}"
XDR_TAU_CONTROL_WARMUP_STEPS="${OAT_ZERO_XDR_TAU_CONTROL_WARMUP_STEPS:-64}"
XDR_TAU_CONTROL_MIN="${OAT_ZERO_XDR_TAU_CONTROL_MIN:-0.005}"
XDR_TAU_CONTROL_EMA_DECAY="${OAT_ZERO_XDR_TAU_CONTROL_EMA_DECAY:-0.9}"
XDR_TAU_CONTROL_GAIN="${OAT_ZERO_XDR_TAU_CONTROL_GAIN:-20.0}"
XDR_SAC_DUAL_BASE="${OAT_ZERO_XDR_SAC_DUAL_BASE:-0.05}"
XDR_SAC_DUAL_RATIO="${OAT_ZERO_XDR_SAC_DUAL_RATIO:-0.8}"
XDR_SAC_DUAL_WARMUP_STEPS="${OAT_ZERO_XDR_SAC_DUAL_WARMUP_STEPS:-64}"
XDR_SAC_DUAL_MIN_TAU="${OAT_ZERO_XDR_SAC_DUAL_MIN_TAU:-0.005}"
XDR_SAC_DUAL_MAX_TAU="${OAT_ZERO_XDR_SAC_DUAL_MAX_TAU:-0.5}"
XDR_SAC_DUAL_ALPHA_LR="${OAT_ZERO_XDR_SAC_DUAL_ALPHA_LR:-0.003}"
MAXENT_ALPHA="${OAT_ZERO_MAXENT_ALPHA:-0.05}"
MAXENT_OBJECTIVE="${OAT_ZERO_MAXENT_OBJECTIVE:-sequence}"
MAXENT_FIXED_ALPHA="${OAT_ZERO_MAXENT_FIXED_ALPHA:-$MAXENT_ALPHA}"
MAXENT_CONTROL_BASE_ALPHA="${OAT_ZERO_MAXENT_CONTROL_BASE_ALPHA:-$MAXENT_ALPHA}"
MAXENT_DUAL_BASE_ALPHA="${OAT_ZERO_MAXENT_DUAL_BASE_ALPHA:-$MAXENT_ALPHA}"
MAXENT_CONTROL_RATIO="${OAT_ZERO_MAXENT_CONTROL_RATIO:-0.8}"
MAXENT_CONTROL_TARGET_ENTROPY="${OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY:-0.0}"
MAXENT_CONTROL_WARMUP_STEPS="${OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS:-64}"
MAXENT_CONTROL_MAX_ALPHA="${OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA:-0.5}"
MAXENT_CONTROL_EMA_DECAY="${OAT_ZERO_MAXENT_CONTROL_EMA_DECAY:-0.9}"
MAXENT_CONTROL_GAIN="${OAT_ZERO_MAXENT_CONTROL_GAIN:-1.0}"
MAXENT_DUAL_RATIO="${OAT_ZERO_MAXENT_DUAL_RATIO:-0.8}"
MAXENT_DUAL_TARGET_ENTROPY="${OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY:-0.0}"
MAXENT_DUAL_WARMUP_STEPS="${OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS:-64}"
MAXENT_DUAL_MIN_ALPHA="${OAT_ZERO_MAXENT_DUAL_MIN_ALPHA:-0.005}"
MAXENT_DUAL_MAX_ALPHA="${OAT_ZERO_MAXENT_DUAL_MAX_ALPHA:-0.5}"
MAXENT_DUAL_ALPHA_LR="${OAT_ZERO_MAXENT_DUAL_ALPHA_LR:-0.003}"
MAXENT_DUAL_EMA_DECAY="${OAT_ZERO_MAXENT_DUAL_EMA_DECAY:-0.7}"
MAXENT_INVERSE_BASE_ALPHA="${OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA:-$MAXENT_ALPHA}"
MAXENT_INVERSE_WARMUP_STEPS="${OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS:-64}"
MAXENT_INVERSE_EMA_DECAY="${OAT_ZERO_MAXENT_INVERSE_EMA_DECAY:-0.9}"
MAXENT_LENGTH_TARGET="${OAT_ZERO_MAXENT_LENGTH_TARGET:-0.0}"
MAXENT_LENGTH_LAMBDA_INIT="${OAT_ZERO_MAXENT_LENGTH_LAMBDA_INIT:-0.0}"
MAXENT_LENGTH_LAMBDA_MAX="${OAT_ZERO_MAXENT_LENGTH_LAMBDA_MAX:-0.02}"
MAXENT_LENGTH_EMA_DECAY="${OAT_ZERO_MAXENT_LENGTH_EMA_DECAY:-0.9}"
MAXENT_LENGTH_DUAL_LR="${OAT_ZERO_MAXENT_LENGTH_DUAL_LR:-0.0002}"
DIAYN_NUM_OPTIONS="${OAT_ZERO_DIAYN_NUM_OPTIONS:-4}"
DIAYN_MI_BETA="${OAT_ZERO_DIAYN_MI_BETA:-0.1}"
DIAYN_MI_EMA_DECAY="${OAT_ZERO_DIAYN_MI_EMA_DECAY:-0.9}"
DIAYN_MI_SMOOTHING="${OAT_ZERO_DIAYN_MI_SMOOTHING:-1.0}"
DIAYN_MI_BONUS_CLIP="${OAT_ZERO_DIAYN_MI_BONUS_CLIP:-5.0}"
DIAYN_MI_LEAVE_ONE_OUT="${OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT:-0}"
OUTCOME_COLLISION_COEF="${OAT_ZERO_OUTCOME_COLLISION_COEF:-0.1}"
SEMANTIC_SHANNON_COEF="${OAT_ZERO_SEMANTIC_SHANNON_COEF:-0.1}"
SEMANTIC_SHANNON_SURPRISAL_CLIP="${OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP:-5.0}"
SEMANTIC_SHANNON_PSEUDOCOUNT="${OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT:-1.0}"
SEMANTIC_SHANNON_QUALITY_GATED_CAP="${OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_CAP:-0.05}"
SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP:-0.05}"
SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS="${OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS:-64}"
SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY="${OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY:-0.9}"
ONLINE_CANONICAL_BANK_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA:-0.10}"
ONLINE_CANONICAL_NOVELTY_BETA="${OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA:-0.50}"
ONLINE_CANONICAL_BANK_PSEUDOCOUNT="${OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT:-1.0}"
ONLINE_CANONICAL_BANK_SURPRISAL_CLIP="${OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP:-5.0}"
ONLINE_CANONICAL_DUAL_TARGET_RATIO="${OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO:-0.8}"
ONLINE_CANONICAL_DUAL_MIN_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA:-$ONLINE_CANONICAL_BANK_ALPHA}"
ONLINE_CANONICAL_DUAL_MAX_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA:-0.5}"
ONLINE_CANONICAL_DUAL_ALPHA_LR="${OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR:-0.003}"
ONLINE_CANONICAL_DUAL_EMA_DECAY="${OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY:-0.9}"
ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS="${OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS:-64}"
ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY="${OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY:-0.9}"
ONLINE_CANONICAL_REPLAY_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA:-0.1}"
ONLINE_CANONICAL_REPLAY_OBJECTIVE="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE:-bank_balance}"
ONLINE_CANONICAL_REPLAY_CAPACITY="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY:-16}"
ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS:-0}"
ONLINE_CANONICAL_REPLAY_WARMUP_STEPS="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS:-64}"
ONLINE_CANONICAL_REPLAY_EMA_DECAY="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY:-0.9}"
ONLINE_CANONICAL_REPLAY_MASS_ALPHA="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA:-0.1}"
ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS:-64}"
ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY="${OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY:-0.9}"
DRGRPO_VARIANT="${OAT_ZERO_DRGRPO_VARIANT:-grpo}"
ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS:-256}"
ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS:-3}"
ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE:-1.0}"
ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS="${OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS:-0}"
ONLINE_CANONICAL_KEY_MODE="${OAT_ZERO_ONLINE_CANONICAL_KEY_MODE:-modebench_outcome}"
VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE="${OAT_ZERO_VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE:-16}"
VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS="${OAT_ZERO_VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS:-2}"
VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP="${OAT_ZERO_VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP:-2.0}"
MATH_STRATEGY_ENDPOINT="${OAT_ZERO_MATH_STRATEGY_ENDPOINT:-}"
MATH_STRATEGY_MODEL="${OAT_ZERO_MATH_STRATEGY_MODEL:-qwen2.5-72b}"
MATH_STRATEGY_TIMEOUT_SECONDS="${OAT_ZERO_MATH_STRATEGY_TIMEOUT_SECONDS:-600}"
MATH_STRATEGY_WORKERS="${OAT_ZERO_MATH_STRATEGY_WORKERS:-4}"
MATH_STRATEGY_MAX_ITEM_CHARS="${OAT_ZERO_MATH_STRATEGY_MAX_ITEM_CHARS:-4000}"
MATH_STRATEGY_GATE_TASK_REWARD="${OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD:-0}"
VERIFIED_DISCOVERY_TRACKING="${OAT_ZERO_VERIFIED_DISCOVERY_TRACKING:-1}"
SEED_ENTROPY_ALPHA="${OAT_ZERO_SEED_ENTROPY_ALPHA:-1.0}"
# Stable recipe from the stabilization sweep (stab_sweep_1432): at lr 5e-6
# the policy's success rate collapses to zero within a run; at 2e-7 it rises
# and holds. The 3x prompt budget compensates the smaller steps.
LEARNING_RATE="${OAT_ZERO_LEARNING_RATE:-0.0000002}"
MAX_TRAIN="${OAT_ZERO_MAX_TRAIN:-9216}"
MAX_QUERIES="${OAT_ZERO_MAX_QUERIES:-100000}"
# G = 16 rollouts per prompt for every arm (G=8 leaves most groups without a
# correct sample at this model scale, starving the group-relative signal).
NUM_SAMPLES="${OAT_ZERO_NUM_SAMPLES:-16}"
TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node105,node302}"
TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:1}"
TRAIN_CPUS_PER_TASK="${OAT_ZERO_TRAIN_CPUS_PER_TASK:-16}"
TRAIN_MEMORY="${OAT_ZERO_TRAIN_MEMORY:-96G}"
TRAIN_PARTITION="${OAT_ZERO_TRAIN_PARTITION:-}"
TRAIN_ACCOUNT="${OAT_ZERO_TRAIN_ACCOUNT:-}"
# Short walltime: these 0.5B arms finish in a few hours, and a tight limit
# lets jobs backfill ahead of maintenance reservations instead of pending on
# the slurm template's 24h default.
TRAIN_TIME_LIMIT="${OAT_ZERO_TRAIN_TIME_LIMIT:-08:00:00}"
REBUILD_DATA="${OAT_ZERO_COMPARATIVE_REBUILD:-0}"
REQUIRE_EXISTING_DATA="${OAT_ZERO_REQUIRE_EXISTING_DATA:-0}"
if [[ "$REQUIRE_EXISTING_DATA" != "0" && "$REQUIRE_EXISTING_DATA" != "1" ]]; then
  echo "OAT_ZERO_REQUIRE_EXISTING_DATA must be 0 or 1" >&2
  exit 1
fi
if [[ "$VERIFIED_DISCOVERY_TRACKING" != "0" && "$VERIFIED_DISCOVERY_TRACKING" != "1" ]]; then
  echo "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING must be 0 or 1" >&2
  exit 1
fi
if [[ "$DRGRPO_VARIANT" != "grpo" && "$DRGRPO_VARIANT" != "grpo_compute_matched" ]]; then
  echo "OAT_ZERO_DRGRPO_VARIANT must be grpo or grpo_compute_matched" >&2
  exit 1
fi
# Optional comma-separated arm labels for incremental matched extensions. This
# lets a new treatment reuse already-landed controls without resubmitting them.
# Labels are the manifest names, e.g. xdr_tau_control or xdr_tau0p05.
ONLY_ARMS_CSV="${OAT_ZERO_ONLY_ARMS:-}"
SBATCH_HOLD="${OAT_ZERO_SBATCH_HOLD:-0}"
if [[ "$SBATCH_HOLD" != "0" && "$SBATCH_HOLD" != "1" ]]; then
  echo "OAT_ZERO_SBATCH_HOLD must be 0 or 1" >&2
  exit 1
fi

arm_enabled() {
  local arm="$1"
  [[ -z "$ONLY_ARMS_CSV" || ",${ONLY_ARMS_CSV}," == *",${arm},"* ]]
}

case "$TASK" in
  countdown)
    case "$DATA_PRESET" in
      easy3)
        DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_countdown_easy3_probe}"
        CD_TRAIN_SIZE="${OAT_ZERO_COUNTDOWN_MODE_TRAIN_SIZE:-384}"
        CD_EVAL_SIZE="${OAT_ZERO_COUNTDOWN_MODE_EVAL_SIZE:-128}"
        CD_NUMBER_COUNT="${OAT_ZERO_COUNTDOWN_MODE_NUMBER_COUNT:-3}"
        ;;
      countdown4)
        DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_countdown4_probe}"
        CD_TRAIN_SIZE="${OAT_ZERO_COUNTDOWN_MODE_TRAIN_SIZE:-1024}"
        CD_EVAL_SIZE="${OAT_ZERO_COUNTDOWN_MODE_EVAL_SIZE:-256}"
        CD_NUMBER_COUNT="${OAT_ZERO_COUNTDOWN_MODE_NUMBER_COUNT:-4}"
        ;;
      *)
        echo "Unknown OAT_ZERO_COMPARATIVE_DATA_PRESET=${DATA_PRESET}; use easy3 or countdown4." >&2
        exit 1
        ;;
    esac
    ;;
  graph_coloring)
    DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_answer_mode_probe}"
    ;;
  python_factor)
    DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/python_factor_modebench_v1}"
    ;;
  math)
    # MATH data are provenance-pinned external artifacts. Never synthesize or
    # rebuild them through the ModeBench generators below.
    DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/oat_drgrpo_math_paper}"
    ;;
  *)
    echo "Unknown OAT_ZERO_COMPARATIVE_TASK=${TASK}; use countdown, graph_coloring, python_factor, or math." >&2
    exit 1
    ;;
esac

if [[ "$REBUILD_DATA" == "1" ]] || [[ ! -f "$DATA_ROOT/train/dataset_dict.json" ]] || [[ ! -f "$DATA_ROOT/eval/dataset_dict.json" ]]; then
  if [[ "$REQUIRE_EXISTING_DATA" == "1" ]]; then
    echo "Frozen campaign requires the already-verified dataset: $DATA_ROOT" >&2
    exit 1
  fi
  if [[ "${OAT_ZERO_COMPARATIVE_CONFIG_ONLY:-0}" == "1" ]]; then
    echo "Configuration-only mode will not create or rebuild data: $DATA_ROOT" >&2
    exit 1
  fi
  case "$TASK" in
    countdown)
      "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
        "$ROOT_DIR/ops/make_exact_countdown_mode_data.py" \
        --output-root "$DATA_ROOT" \
        --train-size "$CD_TRAIN_SIZE" \
        --eval-size "$CD_EVAL_SIZE" \
        --number-count "$CD_NUMBER_COUNT" \
        --max-value "${OAT_ZERO_COUNTDOWN_MODE_MAX_VALUE:-12}" \
        --multi-min-modes "${OAT_ZERO_COUNTDOWN_MODE_MULTI_MIN_MODES:-2}" \
        --multi-max-modes "${OAT_ZERO_COUNTDOWN_MODE_MULTI_MAX_MODES:-8}" \
        --seed "${OAT_ZERO_COUNTDOWN_MODE_DATA_SEED:-0}" \
        --overwrite
      ;;
    graph_coloring)
      "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
        "$ROOT_DIR/ops/make_exact_answer_mode_data.py" \
        --output-root "$DATA_ROOT" \
        --train-size "${OAT_ZERO_ANSWER_MODE_TRAIN_SIZE:-192}" \
        --eval-size "${OAT_ZERO_ANSWER_MODE_EVAL_SIZE:-96}" \
        --seed "${OAT_ZERO_ANSWER_MODE_DATA_SEED:-0}" \
        --overwrite
      ;;
    python_factor)
      "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
        "$ROOT_DIR/ops/make_python_factor_mode_data.py" \
        --output-root "$DATA_ROOT" \
        --train-size "${OAT_ZERO_PYTHON_FACTOR_TRAIN_SIZE:-384}" \
        --eval-size "${OAT_ZERO_PYTHON_FACTOR_EVAL_SIZE:-128}" \
        --case-count "${OAT_ZERO_PYTHON_FACTOR_CASE_COUNT:-4}" \
        --max-value "${OAT_ZERO_PYTHON_FACTOR_MAX_VALUE:-96}" \
        --min-modes "${OAT_ZERO_PYTHON_FACTOR_MIN_MODES:-16}" \
        --seed "${OAT_ZERO_PYTHON_FACTOR_DATA_SEED:-5100}" \
        --overwrite
      ;;
    math)
      echo "MATH artifacts are missing and must be restored with ops/math500/import_oat_math.py: $DATA_ROOT" >&2
      exit 1
      ;;
  esac
fi

# Arm label -> shared variant plus arm-specific environment.
submit_arm() {
  local arm="$1"
  local variant="$2"
  local seed="$3"
  local xdr_tau="${4:-}"
  local seed_alpha="${5:-}"
  local run_stamp="${STAMP_PREFIX}_${arm}_s${seed}"
  local local_root="/tmp/${USER}/maxent-grpo-oat-zero-${run_stamp}"
  local export_vars
  export_vars="ALL"
  export_vars+=",RUN_STAMP=${run_stamp}"
  export_vars+=",OAT_ZERO_SOURCE_ROOT=${CAMPAIGN_SOURCE_ROOT}"
  export_vars+=",OAT_ZERO_SEED=${seed}"
  export_vars+=",OAT_ZERO_VARIANT=${variant}"
  export_vars+=",OAT_ZERO_MODEL=${MODEL}"
  export_vars+=",OAT_ZERO_COMPARATIVE_TASK=${TASK}"
  if [[ -n "${OAT_ZERO_PRETRAIN:-}" ]]; then
    export_vars+=",OAT_ZERO_PRETRAIN=${OAT_ZERO_PRETRAIN}"
  fi
  if [[ -n "${VLLM_USE_V1:-}" ]]; then
    export_vars+=",VLLM_USE_V1=${VLLM_USE_V1}"
  fi
  if [[ -n "${OAT_ZERO_VLLM_SLEEP:-}" ]]; then
    export_vars+=",OAT_ZERO_VLLM_SLEEP=${OAT_ZERO_VLLM_SLEEP}"
  fi
  if [[ -n "${HF_HUB_OFFLINE:-}" ]]; then
    export_vars+=",HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
  fi
  if [[ -n "${TRANSFORMERS_OFFLINE:-}" ]]; then
    export_vars+=",TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
  fi
  if [[ -n "${OAT_ZERO_OPS_SNAPSHOT_ROOT:-}" ]]; then
    export_vars+=",OAT_ZERO_OPS_SNAPSHOT_ROOT=${OAT_ZERO_OPS_SNAPSHOT_ROOT}"
  fi
  export_vars+=",OAT_ZERO_DATA_ROOT=${DATA_ROOT}"
  export_vars+=",OAT_ZERO_REQUIRE_EXISTING_DATA=${REQUIRE_EXISTING_DATA}"
  export_vars+=",OAT_ZERO_NUM_SAMPLES=${NUM_SAMPLES}"
  export_vars+=",OAT_ZERO_TRAIN_BATCH_SIZE=${OAT_ZERO_TRAIN_BATCH_SIZE:-$NUM_SAMPLES}"
  export_vars+=",OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-$NUM_SAMPLES}"
  export_vars+=",OAT_ZERO_ROLLOUT_BATCH_SIZE=${OAT_ZERO_ROLLOUT_BATCH_SIZE:-1}"
  export_vars+=",OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-1}"
  export_vars+=",OAT_ZERO_N_GPU=${OAT_ZERO_N_GPU:-1}"
  export_vars+=",OAT_ZERO_NUM_GPUS_PER_ACTOR=${OAT_ZERO_NUM_GPUS_PER_ACTOR:-1}"
  export_vars+=",OAT_ZERO_ZERO_STAGE=${OAT_ZERO_ZERO_STAGE:-2}"
  export_vars+=",OAT_ZERO_VLLM_GPU_RATIO=${OAT_ZERO_VLLM_GPU_RATIO:-0.25}"
  export_vars+=",OAT_ZERO_ENABLE_FLASH_ATTN=${OAT_ZERO_ENABLE_FLASH_ATTN:-0}"
  export_vars+=",OAT_ZERO_ADAM_OFFLOAD=${OAT_ZERO_ADAM_OFFLOAD:-0}"
  export_vars+=",OAT_ZERO_ACTIVATION_OFFLOADING=${OAT_ZERO_ACTIVATION_OFFLOADING:-0}"
  export_vars+=",OAT_ZERO_COLLOCATE=${OAT_ZERO_COLLOCATE:-1}"
  export_vars+=",OAT_ZERO_NUM_PPO_EPOCHS=${OAT_ZERO_NUM_PPO_EPOCHS:-1}"
  export_vars+=",OAT_ZERO_MAX_NORM=${OAT_ZERO_MAX_NORM:-1.0}"
  export_vars+=",OAT_ZERO_BETA=${OAT_ZERO_BETA:-0}"
  export_vars+=",OAT_ZERO_IGNORE_NO_EOS=${OAT_ZERO_IGNORE_NO_EOS:-0}"
  export_vars+=",OAT_ZERO_INPUT_KEY=${OAT_ZERO_INPUT_KEY:-problem}"
  export_vars+=",OAT_ZERO_OUTPUT_KEY=${OAT_ZERO_OUTPUT_KEY:-answer}"
  export_vars+=",OAT_ZERO_EVAL_INPUT_KEY=${OAT_ZERO_EVAL_INPUT_KEY:-${OAT_ZERO_INPUT_KEY:-problem}}"
  export_vars+=",OAT_ZERO_EVAL_OUTPUT_KEY=${OAT_ZERO_EVAL_OUTPUT_KEY:-${OAT_ZERO_OUTPUT_KEY:-answer}}"
  export_vars+=",OAT_ZERO_MAX_TRAIN=${MAX_TRAIN}"
  export_vars+=",OAT_ZERO_MAX_QUERIES=${MAX_QUERIES}"
  export_vars+=",OAT_ZERO_MAX_PROMPT_EPOCHS=${OAT_ZERO_MAX_PROMPT_EPOCHS:-5}"
  export_vars+=",OAT_ZERO_NUM_PROMPT_EPOCH=${OAT_ZERO_NUM_PROMPT_EPOCH:-10}"
  export_vars+=",OAT_ZERO_EVAL_STEPS=${OAT_ZERO_EVAL_STEPS:-64}"
  export_vars+=",OAT_ZERO_SYNC_PARAMS_EVERY=${OAT_ZERO_SYNC_PARAMS_EVERY:-1}"
  # The runtime derives a quarter-epoch prompt interval from the actual data.
  # Forward an explicit interval only to request still-more-frequent evals;
  # requests looser than quarter-epoch are capped by run_experiment.sh.
  if [[ -n "${OAT_ZERO_EVAL_PROMPT_INTERVAL:-}" ]]; then
    export_vars+=",OAT_ZERO_EVAL_PROMPT_INTERVAL=${OAT_ZERO_EVAL_PROMPT_INTERVAL}"
  fi
  export_vars+=",OAT_ZERO_ALLOW_SPARSE_EVAL=${OAT_ZERO_ALLOW_SPARSE_EVAL:-0}"
  export_vars+=",OAT_ZERO_SAVE_STEPS=${OAT_ZERO_SAVE_STEPS:-64}"
  export_vars+=",OAT_ZERO_SAVE_FROM=${OAT_ZERO_SAVE_FROM:-64}"
  export_vars+=",OAT_ZERO_SAVE_CKPT=${OAT_ZERO_SAVE_CKPT:-0}"
  export_vars+=",OAT_ZERO_MAX_SAVE_NUM=${OAT_ZERO_MAX_SAVE_NUM:-4}"
  export_vars+=",OAT_ZERO_USE_WB=0"
  export_vars+=",OAT_ZERO_PROMPT_TEMPLATE=${OAT_ZERO_PROMPT_TEMPLATE:-qwen_boxed}"
  export_vars+=",OAT_ZERO_CANONICAL_GRAPH_ACTIONS=${OAT_ZERO_CANONICAL_GRAPH_ACTIONS:-0}"
  export_vars+=",OAT_ZERO_CANONICAL_ACTION_TASK=${OAT_ZERO_CANONICAL_ACTION_TASK:-none}"
  export_vars+=",OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT=${OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT:-3}"
  export_vars+=",OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=${OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING:-0}"
  export_vars+=",OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=${OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING:-0}"
  if [[ "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" || "$variant" == "verified_route_successor" ]]; then
    # Counterfactual support proposals execute inside the replicated free-form
    # path.  Pin both flags by arm so a domain-level default cannot silently
    # make a valid proposal arm fail only after Slurm releases it.
    export_vars+=",OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1"
    export_vars+=",OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1"
  else
    export_vars+=",OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=${OAT_ZERO_REPLICATED_FREEFORM_SAMPLING:-0}"
    export_vars+=",OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=${OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC:-0}"
  fi
  export_vars+=",OAT_ZERO_VLLM_SLEEP_LEVEL=${OAT_ZERO_VLLM_SLEEP_LEVEL:-1}"
  export_vars+=",OAT_ZERO_PROMPT_MAX_LENGTH=${OAT_ZERO_PROMPT_MAX_LENGTH:-256}"
  export_vars+=",OAT_ZERO_GENERATE_MAX_LENGTH=${OAT_ZERO_GENERATE_MAX_LENGTH:-192}"
  export_vars+=",OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=${OAT_ZERO_EVAL_GENERATE_MAX_LENGTH:-192}"
  export_vars+=",OAT_ZERO_EVAL_BATCH_SIZE=${OAT_ZERO_EVAL_BATCH_SIZE:-64}"
  export_vars+=",OAT_ZERO_TEST_SPLIT=${OAT_ZERO_TEST_SPLIT:-all}"
  export_vars+=",OAT_ZERO_VERIFIER_VERSION=${OAT_ZERO_VERIFIER_VERSION:-fast}"
  export_vars+=",OAT_ZERO_EVAL_MODE_COVERAGE_K=${OAT_ZERO_EVAL_MODE_COVERAGE_K:-0}"
  export_vars+=",OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=${OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE:-1.0}"
  export_vars+=",OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=${OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS:-4}"
  export_vars+=",OAT_ZERO_EVAL_MODE_COVERAGE_SEED=${OAT_ZERO_EVAL_MODE_COVERAGE_SEED:-1001}"
  export_vars+=",OAT_ZERO_TEMPERATURE=${OAT_ZERO_TEMPERATURE:-1.0}"
  export_vars+=",OAT_ZERO_TOP_P=${OAT_ZERO_TOP_P:-1.0}"
  export_vars+=",OAT_ZERO_EVAL_TEMPERATURE=${OAT_ZERO_EVAL_TEMPERATURE:-0}"
  export_vars+=",OAT_ZERO_SYNC_PARAMS_EVERY=${OAT_ZERO_SYNC_PARAMS_EVERY:-1}"
  if [[ -n "${OAT_ZERO_MAX_MODEL_LEN:-}" ]]; then
    export_vars+=",OAT_ZERO_MAX_MODEL_LEN=${OAT_ZERO_MAX_MODEL_LEN}"
  fi
  export_vars+=",OAT_ZERO_LOCAL_ROOT=${local_root}"
  if [[ -n "${OAT_ZERO_PROTOCOL_IDENTITY:-}" ]]; then
    export_vars+=",OAT_ZERO_PROTOCOL_IDENTITY=${OAT_ZERO_PROTOCOL_IDENTITY}"
  fi
  # Recovery semantics are scientific provenance, not scheduler-local
  # convenience.  Pin them in SubmitLine instead of relying on --export=ALL so
  # a held-job audit can prove both the branch and checkpoint lifecycle.
  export_vars+=",OAT_ZERO_AUTO_RESUME=${OAT_ZERO_AUTO_RESUME:-1}"
  export_vars+=",OAT_ZERO_WATCHDOG_REQUEUE=${OAT_ZERO_WATCHDOG_REQUEUE:-1}"
  export_vars+=",OAT_ZERO_WATCHDOG_MAX_RESTARTS=${OAT_ZERO_WATCHDOG_MAX_RESTARTS:-6}"
  export_vars+=",OAT_ZERO_RESUME_STEPS=${OAT_ZERO_RESUME_STEPS:--1}"
  export_vars+=",OAT_ZERO_RESUME_FROM=${OAT_ZERO_RESUME_FROM:-0}"
  export_vars+=",OAT_ZERO_MAX_RESUME_NUM=${OAT_ZERO_MAX_RESUME_NUM:-1}"
  export_vars+=",OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=${OAT_ZERO_PRUNE_RESUME_ON_SUCCESS:-1}"
  export_vars+=",OAT_ZERO_EXPORT_STEPS=${OAT_ZERO_EXPORT_STEPS:-0}"
  export_vars+=",OAT_ZERO_EXPORT_FROM=${OAT_ZERO_EXPORT_FROM:-0}"
  export_vars+=",OAT_ZERO_MAX_EXPORT_NUM=${OAT_ZERO_MAX_EXPORT_NUM:-1}"
  if [[ -n "${OAT_ZERO_INITIAL_RESUME_DIR:-}" ]]; then
    export_vars+=",OAT_ZERO_INITIAL_RESUME_DIR=${OAT_ZERO_INITIAL_RESUME_DIR}"
    export_vars+=",OAT_ZERO_INITIAL_RESUME_TAG=${OAT_ZERO_INITIAL_RESUME_TAG:-}"
  fi
  if [[ -n "${OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES:-}" ]]; then
    export_vars+=",OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES=${OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES}"
  fi
  # Pin every method knob on every arm: --export=ALL would otherwise let a
  # stray export silently change an arm and void the matched design.
  export_vars+=",OAT_ZERO_RND_SEED=0"
  export_vars+=",OAT_ZERO_LEARNING_RATE=${LEARNING_RATE}"
  if [[ "$variant" == "xdr_adapt" ]]; then
    export_vars+=",OAT_ZERO_XDR_MODE_ADAPTIVE=1"
  else
    export_vars+=",OAT_ZERO_XDR_MODE_ADAPTIVE=0"
  fi
  export_vars+=",OAT_ZERO_XDR_TAU_CONTROL_TARGET_RATIO=0.0"
  export_vars+=",OAT_ZERO_XDR_SAC_DUAL_TARGET_RATIO=0.0"
  if [[ "$variant" == "signal_first_semantic_balance" ]]; then
    export_vars+=",OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS=1"
  else
    export_vars+=",OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS=0"
  fi
  case "$variant" in
    maxent)
      export_vars+=",OAT_ZERO_MAXENT_ALPHA=${MAXENT_FIXED_ALPHA}"
      ;;
    maxent_control)
      export_vars+=",OAT_ZERO_MAXENT_ALPHA=${MAXENT_CONTROL_BASE_ALPHA}"
      ;;
    maxent_dual)
      export_vars+=",OAT_ZERO_MAXENT_ALPHA=${MAXENT_DUAL_BASE_ALPHA}"
      ;;
    maxent_inverse|maxent_inverse_canonical|maxent_inverse_canonical_replay|open_set_split_canonical)
      export_vars+=",OAT_ZERO_MAXENT_ALPHA=${MAXENT_INVERSE_BASE_ALPHA}"
      ;;
    maxent_length_dual)
      export_vars+=",OAT_ZERO_MAXENT_ALPHA=${MAXENT_ALPHA}"
      ;;
    *)
      export_vars+=",OAT_ZERO_MAXENT_ALPHA=0.0"
      ;;
  esac
  export_vars+=",OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO=0.0"
  export_vars+=",OAT_ZERO_MAXENT_DUAL_TARGET_RATIO=0.0"
  if [[ "$variant" == "maxent_inverse" || "$variant" == "maxent_inverse_canonical" || "$variant" == "maxent_inverse_canonical_replay" || "$variant" == "open_set_split_canonical" ]]; then
    export_vars+=",OAT_ZERO_MAXENT_INVERSE_ADAPTATION=1"
    export_vars+=",OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS=${MAXENT_INVERSE_WARMUP_STEPS}"
    export_vars+=",OAT_ZERO_MAXENT_INVERSE_EMA_DECAY=${MAXENT_INVERSE_EMA_DECAY}"
  else
    export_vars+=",OAT_ZERO_MAXENT_INVERSE_ADAPTATION=0"
    export_vars+=",OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS=64"
    export_vars+=",OAT_ZERO_MAXENT_INVERSE_EMA_DECAY=0.9"
  fi
  if [[ "$variant" == "diayn" ]]; then
    export_vars+=",OAT_ZERO_DIAYN_NUM_OPTIONS=${DIAYN_NUM_OPTIONS}"
    export_vars+=",OAT_ZERO_DIAYN_MI_BETA=${DIAYN_MI_BETA}"
    export_vars+=",OAT_ZERO_DIAYN_MI_EMA_DECAY=${DIAYN_MI_EMA_DECAY}"
    export_vars+=",OAT_ZERO_DIAYN_MI_SMOOTHING=${DIAYN_MI_SMOOTHING}"
    export_vars+=",OAT_ZERO_DIAYN_MI_BONUS_CLIP=${DIAYN_MI_BONUS_CLIP}"
    export_vars+=",OAT_ZERO_DIAYN_MI_CORRECT_ONLY=1"
    export_vars+=",OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT=${DIAYN_MI_LEAVE_ONE_OUT}"
  else
    export_vars+=",OAT_ZERO_DIAYN_NUM_OPTIONS=0"
    export_vars+=",OAT_ZERO_DIAYN_MI_BETA=0.0"
    export_vars+=",OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT=0"
  fi
  if [[ "$variant" == "outcome_collision" ]]; then
    export_vars+=",OAT_ZERO_OUTCOME_COLLISION_COEF=${OUTCOME_COLLISION_COEF}"
  elif [[ "$variant" == "outcome_collision_outside_centering" ]]; then
    export_vars+=",OAT_ZERO_OUTCOME_COLLISION_COEF=${OUTCOME_COLLISION_COEF}"
  else
    export_vars+=",OAT_ZERO_OUTCOME_COLLISION_COEF=0.0"
  fi
  if [[ "$variant" == "outcome_collision_outside_centering" ]]; then
    export_vars+=",OAT_ZERO_OUTCOME_COLLISION_OUTSIDE_CENTERING=1"
  else
    export_vars+=",OAT_ZERO_OUTCOME_COLLISION_OUTSIDE_CENTERING=0"
  fi
  if [[ "$variant" == "semantic_shannon" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_COEF=${SEMANTIC_SHANNON_COEF}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=${SEMANTIC_SHANNON_SURPRISAL_CLIP}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=${SEMANTIC_SHANNON_PSEUDOCOUNT}"
  elif [[ "$variant" == "semantic_shannon_advantage" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_COEF=${SEMANTIC_SHANNON_COEF}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=${SEMANTIC_SHANNON_SURPRISAL_CLIP}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=${SEMANTIC_SHANNON_PSEUDOCOUNT}"
  elif [[ "$variant" == "quality_gated_semantic_novelty" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_COEF=${SEMANTIC_SHANNON_COEF}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=${SEMANTIC_SHANNON_SURPRISAL_CLIP}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=${SEMANTIC_SHANNON_PSEUDOCOUNT}"
  elif [[ "$variant" == "success_conditioned_signed_semantic_shannon" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_COEF=${SEMANTIC_SHANNON_COEF}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=${SEMANTIC_SHANNON_SURPRISAL_CLIP}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=${SEMANTIC_SHANNON_PSEUDOCOUNT}"
  elif [[ "$variant" == "signal_first_semantic_balance" || "$variant" == "open_set_split_canonical" || "$variant" == "verified_first_split_canonical" || "$variant" == "verified_first_global_replay_canonical" || "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_COEF=${SEMANTIC_SHANNON_COEF}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=${SEMANTIC_SHANNON_SURPRISAL_CLIP}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=${SEMANTIC_SHANNON_PSEUDOCOUNT}"
  else
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0"
  fi
  if [[ "$variant" == "semantic_shannon_advantage" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1"
  elif [[ "$variant" == "quality_gated_semantic_novelty" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1"
  elif [[ "$variant" == "success_conditioned_signed_semantic_shannon" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1"
  elif [[ "$variant" == "signal_first_semantic_balance" || "$variant" == "open_set_split_canonical" || "$variant" == "verified_first_split_canonical" || "$variant" == "verified_first_global_replay_canonical" || "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1"
  else
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0"
  fi
  if [[ "$variant" == "quality_gated_semantic_novelty" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=1"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_CAP=${SEMANTIC_SHANNON_QUALITY_GATED_CAP}"
  else
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=0"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_CAP=0.05"
  fi
  if [[ "$variant" == "success_conditioned_signed_semantic_shannon" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP=${SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP}"
  elif [[ "$variant" == "signal_first_semantic_balance" || "$variant" == "open_set_split_canonical" || "$variant" == "verified_first_split_canonical" || "$variant" == "verified_first_global_replay_canonical" || "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP=${SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP}"
  else
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=0"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP=0.05"
  fi
  if [[ "$variant" == "open_set_split_canonical" || "$variant" == "verified_first_split_canonical" || "$variant" == "verified_first_global_replay_canonical" || "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" ]]; then
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=1"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS=${SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS}"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY=${SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY}"
  else
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=0"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS=64"
    export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY=0.9"
  fi
  if [[ "$variant" == "online_canonical_maxent" || "$variant" == "online_canonical_haarnoja" || "$variant" == "online_canonical_policy_entropy" || "$variant" == "maxent_inverse_canonical" || "$variant" == "maxent_inverse_canonical_replay" || "$variant" == "open_set_split_canonical" || "$variant" == "verified_first_split_canonical" || "$variant" == "verified_first_global_replay_canonical" || "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" || "$variant" == "verified_route_successor" ]]; then
  export_vars+=",OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=${ONLINE_CANONICAL_BANK_ALPHA}"
  export_vars+=",OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=${ONLINE_CANONICAL_NOVELTY_BETA}"
  if [[ -n "${OAT_ZERO_EXPECT_ONLINE_CANONICAL_NOVELTY_BETA:-}" ]]; then
    export_vars+=",OAT_ZERO_EXPECT_ONLINE_CANONICAL_NOVELTY_BETA=${OAT_ZERO_EXPECT_ONLINE_CANONICAL_NOVELTY_BETA}"
  fi
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=${ONLINE_CANONICAL_BANK_PSEUDOCOUNT}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=${ONLINE_CANONICAL_BANK_SURPRISAL_CLIP}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=${ONLINE_CANONICAL_KEY_MODE}"
    if [[ "$variant" == "online_canonical_haarnoja" ]]; then
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=${ONLINE_CANONICAL_DUAL_TARGET_RATIO}"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=${ONLINE_CANONICAL_DUAL_MIN_ALPHA}"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=${ONLINE_CANONICAL_DUAL_MAX_ALPHA}"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=${ONLINE_CANONICAL_DUAL_ALPHA_LR}"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=${ONLINE_CANONICAL_DUAL_EMA_DECAY}"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0"
    elif [[ "$variant" == "online_canonical_policy_entropy" ]]; then
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=1"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS=${ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS}"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY=${ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY}"
    else
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0"
    fi
  else
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=${ONLINE_CANONICAL_KEY_MODE}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0"
  fi
  if [[ "$variant" == "maxent_inverse_canonical_replay" || "$variant" == "open_set_split_canonical" || "$variant" == "verified_first_split_canonical" || "$variant" == "verified_first_global_replay_canonical" || "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" || "$variant" == "verified_route_successor" ]]; then
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY=1"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=${ONLINE_CANONICAL_REPLAY_ALPHA}"
    if [[ "$variant" == "verified_route_successor" ]]; then
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout"
    elif [[ "$variant" == "open_set_split_canonical" || "$variant" == "verified_first_split_canonical" || "$variant" == "verified_first_global_replay_canonical" || "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" ]]; then
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout"
    else
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=${ONLINE_CANONICAL_REPLAY_OBJECTIVE}"
    fi
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=${ONLINE_CANONICAL_REPLAY_CAPACITY}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=${ONLINE_CANONICAL_REPLAY_WARMUP_STEPS}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=${ONLINE_CANONICAL_REPLAY_EMA_DECAY}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=${ONLINE_CANONICAL_REPLAY_MASS_ALPHA}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=${ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=${ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY}"
    if [[ "$variant" == "verified_first_global_replay_canonical" || "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" || "$variant" == "verified_route_successor" ]]; then
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1"
    else
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=0"
    fi
    if [[ "$variant" == "verified_first_bootstrap_local_canonical" || "$variant" == "verified_counterfactual_canonical" ]]; then
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=${ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS}"
    else
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0"
    fi
  else
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY=0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=bank_balance"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.9"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.1"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.9"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0"
  fi
  if [[ "$variant" == "verified_counterfactual_canonical" || "$variant" == "verified_entropy_gated_singleton_escape_canonical" || "$variant" == "verified_route_successor" ]]; then
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS=${ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=${ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS}"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=${ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE}"
    if [[ "$variant" == "verified_entropy_gated_singleton_escape_canonical" ]]; then
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ENTROPY_GATE=1"
    elif [[ "$variant" == "verified_route_successor" ]]; then
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ENTROPY_GATE=0"
    else
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=0"
      export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ENTROPY_GATE=0"
    fi
  else
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ENTROPY_GATE=0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS=256"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0"
  fi
  export_vars+=",OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=${ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS}"
  export_vars+=",OAT_ZERO_VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE=${VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE}"
  export_vars+=",OAT_ZERO_VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS=${VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS}"
  export_vars+=",OAT_ZERO_VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP=${VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP}"
  if [[ "$variant" == "grpo_compute_matched" ]]; then
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY=1"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1"
  elif [[ "$variant" == "verified_route_successor" ]]; then
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=verified_route"
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0"
  else
    export_vars+=",OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0"
  fi
  export_vars+=",OAT_ZERO_MATH_STRATEGY_ENDPOINT=${MATH_STRATEGY_ENDPOINT}"
  export_vars+=",OAT_ZERO_MATH_STRATEGY_MODEL=${MATH_STRATEGY_MODEL}"
  export_vars+=",OAT_ZERO_MATH_STRATEGY_TIMEOUT_SECONDS=${MATH_STRATEGY_TIMEOUT_SECONDS}"
  export_vars+=",OAT_ZERO_MATH_STRATEGY_WORKERS=${MATH_STRATEGY_WORKERS}"
  export_vars+=",OAT_ZERO_MATH_STRATEGY_MAX_ITEM_CHARS=${MATH_STRATEGY_MAX_ITEM_CHARS}"
  export_vars+=",OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=${MATH_STRATEGY_GATE_TASK_REWARD}"
  export_vars+=",OAT_ZERO_MAXENT_OBJECTIVE=${MAXENT_OBJECTIVE}"
  export_vars+=",OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=${VERIFIED_DISCOVERY_TRACKING}"
  export_vars+=",OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=0.0"
  export_vars+=",OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0.0"
  if [[ "$variant" == "xdr_tau_control" ]]; then
    export_vars+=",OAT_ZERO_XDR_TAU_CONTROL_RATIO=${XDR_TAU_CONTROL_RATIO}"
    export_vars+=",OAT_ZERO_XDR_TAU_CONTROL_WARMUP_STEPS=${XDR_TAU_CONTROL_WARMUP_STEPS}"
    export_vars+=",OAT_ZERO_XDR_TAU_CONTROL_MIN=${XDR_TAU_CONTROL_MIN}"
    export_vars+=",OAT_ZERO_XDR_TAU_CONTROL_EMA_DECAY=${XDR_TAU_CONTROL_EMA_DECAY}"
    export_vars+=",OAT_ZERO_XDR_TAU_CONTROL_GAIN=${XDR_TAU_CONTROL_GAIN}"
  else
    export_vars+=",OAT_ZERO_XDR_TAU_CONTROL_RATIO=0.0"
  fi
  if [[ "$variant" == "xdr_sac_dual" ]]; then
    export_vars+=",OAT_ZERO_XDR_SAC_DUAL_RATIO=${XDR_SAC_DUAL_RATIO}"
    export_vars+=",OAT_ZERO_XDR_SAC_DUAL_WARMUP_STEPS=${XDR_SAC_DUAL_WARMUP_STEPS}"
    export_vars+=",OAT_ZERO_XDR_SAC_DUAL_MIN_TAU=${XDR_SAC_DUAL_MIN_TAU}"
    export_vars+=",OAT_ZERO_XDR_SAC_DUAL_MAX_TAU=${XDR_SAC_DUAL_MAX_TAU}"
    export_vars+=",OAT_ZERO_XDR_SAC_DUAL_ALPHA_LR=${XDR_SAC_DUAL_ALPHA_LR}"
  else
    export_vars+=",OAT_ZERO_XDR_SAC_DUAL_RATIO=0.0"
  fi
  if [[ "$variant" == "maxent_control" ]]; then
    export_vars+=",OAT_ZERO_MAXENT_CONTROL_RATIO=${MAXENT_CONTROL_RATIO}"
    export_vars+=",OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=${MAXENT_CONTROL_TARGET_ENTROPY}"
    export_vars+=",OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=${MAXENT_CONTROL_WARMUP_STEPS}"
    export_vars+=",OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=${MAXENT_CONTROL_MAX_ALPHA}"
    export_vars+=",OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=${MAXENT_CONTROL_EMA_DECAY}"
    export_vars+=",OAT_ZERO_MAXENT_CONTROL_GAIN=${MAXENT_CONTROL_GAIN}"
  fi
  if [[ "$variant" == "maxent_dual" ]]; then
    export_vars+=",OAT_ZERO_MAXENT_DUAL_RATIO=${MAXENT_DUAL_RATIO}"
    export_vars+=",OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=${MAXENT_DUAL_TARGET_ENTROPY}"
    export_vars+=",OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=${MAXENT_DUAL_WARMUP_STEPS}"
    export_vars+=",OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=${MAXENT_DUAL_MIN_ALPHA}"
    export_vars+=",OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=${MAXENT_DUAL_MAX_ALPHA}"
    export_vars+=",OAT_ZERO_MAXENT_DUAL_ALPHA_LR=${MAXENT_DUAL_ALPHA_LR}"
    export_vars+=",OAT_ZERO_MAXENT_DUAL_EMA_DECAY=${MAXENT_DUAL_EMA_DECAY}"
  fi
  if [[ "$variant" == "maxent" || "$variant" == "maxent_control" || "$variant" == "maxent_dual" || "$variant" == "maxent_length_dual" ]]; then
    export_vars+=",OAT_ZERO_MAXENT_LENGTH_TARGET=${MAXENT_LENGTH_TARGET}"
    export_vars+=",OAT_ZERO_MAXENT_LENGTH_LAMBDA_INIT=${MAXENT_LENGTH_LAMBDA_INIT}"
    export_vars+=",OAT_ZERO_MAXENT_LENGTH_LAMBDA_MAX=${MAXENT_LENGTH_LAMBDA_MAX}"
    export_vars+=",OAT_ZERO_MAXENT_LENGTH_EMA_DECAY=${MAXENT_LENGTH_EMA_DECAY}"
    export_vars+=",OAT_ZERO_MAXENT_LENGTH_DUAL_LR=${MAXENT_LENGTH_DUAL_LR}"
  else
    export_vars+=",OAT_ZERO_MAXENT_LENGTH_TARGET=0.0"
  fi
  if [[ "$variant" == "grpo_entropy" ]]; then
    export_vars+=",OAT_ZERO_POLICY_ENTROPY_COEF=${OAT_ZERO_TOKEN_ENTROPY_COEF:-0.01}"
  else
    export_vars+=",OAT_ZERO_POLICY_ENTROPY_COEF=0.0"
  fi
  if [[ -n "$xdr_tau" ]]; then
    export_vars+=",OAT_ZERO_XDR_TAU=${xdr_tau}"
  else
    export_vars+=",OAT_ZERO_XDR_TAU=inf"
  fi
  if [[ -n "$seed_alpha" ]]; then
    export_vars+=",OAT_ZERO_SEED_ENTROPY_ALPHA=${seed_alpha}"
  else
    export_vars+=",OAT_ZERO_SEED_ENTROPY_ALPHA=0.0"
  fi

  local -a sbatch_args
  sbatch_args=(
    --parsable
    "--export=${export_vars}"
    "--nodelist=${TRAIN_NODELIST}"
    "--gres=${TRAIN_GRES}"
    "--cpus-per-task=${TRAIN_CPUS_PER_TASK}"
    "--mem=${TRAIN_MEMORY}"
    "--time=${TRAIN_TIME_LIMIT}"
  )
  if [[ "$SBATCH_HOLD" == "1" ]]; then
    sbatch_args+=(--hold)
  fi
  if [[ -n "$TRAIN_PARTITION" ]]; then
    sbatch_args+=("--partition=${TRAIN_PARTITION}")
  fi
  if [[ -n "$TRAIN_ACCOUNT" ]]; then
    sbatch_args+=("--account=${TRAIN_ACCOUNT}")
  fi
  local slurm_entrypoint="${OAT_ZERO_OPS_SNAPSHOT_ROOT:-$ROOT_DIR/ops}/slurm/train_node302.slurm"
  sbatch "${sbatch_args[@]}" "$slurm_entrypoint"
}

IFS=',' read -r -a train_seeds <<< "$TRAIN_SEEDS_CSV"
IFS=',' read -r -a xdr_taus <<< "$XDR_TAUS_CSV"

echo "[comparative] stamp_prefix=${STAMP_PREFIX}"
echo "[comparative] task=${TASK}"
echo "[comparative] model=${MODEL}"
echo "[comparative] data_preset=${DATA_PRESET}"
echo "[comparative] data_root=${DATA_ROOT}"
echo "[comparative] train_seeds=${TRAIN_SEEDS_CSV}"
echo "[comparative] xdr_taus=${XDR_TAUS_CSV}"
echo "[comparative] num_samples=${NUM_SAMPLES}"
echo "[comparative] max_train_rows=${MAX_TRAIN} prompt_epochs=${OAT_ZERO_NUM_PROMPT_EPOCH:-10} max_queries=${MAX_QUERIES}"
echo "[comparative] eval_prompt_interval=${OAT_ZERO_EVAL_PROMPT_INTERVAL:-runtime-derived-quarter-epoch}"
echo "[comparative] allow_sparse_eval=${OAT_ZERO_ALLOW_SPARSE_EVAL:-0}"
echo "[comparative] train_memory=${TRAIN_MEMORY}"
echo "[comparative] train_nodelist=${TRAIN_NODELIST}"
echo "[comparative] train_gres=${TRAIN_GRES}"
echo "[comparative] train_cpus_per_task=${TRAIN_CPUS_PER_TASK}"
echo "[comparative] train_partition=${TRAIN_PARTITION:-template-default}"
echo "[comparative] train_account=${TRAIN_ACCOUNT:-template-default}"
echo "[comparative] include_token_entropy_arm=${INCLUDE_TOKEN_ENTROPY_ARM}"
echo "[comparative] include_xdr_tau_control_arm=${INCLUDE_XDR_TAU_CONTROL_ARM}"
echo "[comparative] include_xdr_sac_dual_arm=${INCLUDE_XDR_SAC_DUAL_ARM}"
echo "[comparative] include_maxent_arm=${INCLUDE_MAXENT_ARM}"
echo "[comparative] include_maxent_control_arm=${INCLUDE_MAXENT_CONTROL_ARM}"
echo "[comparative] include_maxent_dual_arm=${INCLUDE_MAXENT_DUAL_ARM}"
echo "[comparative] include_maxent_inverse_arm=${INCLUDE_MAXENT_INVERSE_ARM} reference_alpha=${MAXENT_INVERSE_BASE_ALPHA} projection=none warmup_steps=${MAXENT_INVERSE_WARMUP_STEPS} ema_decay=${MAXENT_INVERSE_EMA_DECAY}"
echo "[comparative] include_maxent_inverse_canonical_arm=${INCLUDE_MAXENT_INVERSE_CANONICAL_ARM} reference_alpha=${MAXENT_INVERSE_BASE_ALPHA} projection=none bank_alpha=${ONLINE_CANONICAL_BANK_ALPHA} novelty_beta=${ONLINE_CANONICAL_NOVELTY_BETA}"
echo "[comparative] include_maxent_inverse_canonical_replay_arm=${INCLUDE_MAXENT_INVERSE_CANONICAL_REPLAY_ARM} direct_reference_alpha=${MAXENT_INVERSE_BASE_ALPHA} replay_reference_alpha=${ONLINE_CANONICAL_REPLAY_ALPHA} replay_objective=${ONLINE_CANONICAL_REPLAY_OBJECTIVE} replay_capacity=${ONLINE_CANONICAL_REPLAY_CAPACITY} replay_projection=none"
echo "[comparative] include_open_set_split_canonical_arm=${INCLUDE_OPEN_SET_SPLIT_CANONICAL_ARM} semantic_reference_coefficient=${SEMANTIC_SHANNON_COEF} replay_balance_reference_alpha=${ONLINE_CANONICAL_REPLAY_ALPHA} projection=none gold_support_feedback=none"
echo "[comparative] include_verified_first_split_canonical_arm=${INCLUDE_VERIFIED_FIRST_SPLIT_CANONICAL_ARM} direct_token_entropy=off semantic_reference_coefficient=${SEMANTIC_SHANNON_COEF} replay_balance_reference_alpha=${ONLINE_CANONICAL_REPLAY_ALPHA} projection=none gold_support_feedback=none"
echo "[comparative] include_verified_first_global_replay_canonical_arm=${INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM} direct_token_entropy=off global_verified_groups_per_step=1 projection=none gold_support_feedback=none"
echo "[comparative] include_verified_first_bootstrap_local_canonical_arm=${INCLUDE_VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL_ARM} direct_token_entropy=off global_verified_groups_per_step=1 global_bootstrap_steps=${ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS} then=prompt_local projection=none gold_support_feedback=none"
echo "[comparative] include_verified_counterfactual_canonical_arm=${INCLUDE_VERIFIED_COUNTERFACTUAL_CANONICAL_ARM} primary_actuator=validator_preserving_transform_of_model_verified_response fallback_groups_per_eligible_prompt=up_to_${ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS} fallback_width=num_samples original_prompt_temperature_sweep=${ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE},+0.2/attempt transform_and_fallback_rows_to_ppo=0 desired_mode_count=none evaluation_feedback=none"
echo "[comparative] include_verified_entropy_gated_singleton_escape_canonical_arm=${INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM} trigger=model_entropy_below_self_warmup_reference bank_support_exactly_one=1 max_admitted_alternates=1 projection=none gold_support_feedback=none evaluation_feedback=none"
echo "[comparative] include_verified_route_successor_arm=${INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM} route_recurrence_min_prompts=${VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS} fixed_control_groups=${ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS} control_rows_to_ppo=0"
echo "[comparative] drgrpo_variant=${DRGRPO_VARIANT}"
echo "[comparative] maxent_arm_alphas=fixed:${MAXENT_FIXED_ALPHA},control:${MAXENT_CONTROL_BASE_ALPHA},dual:${MAXENT_DUAL_BASE_ALPHA}"
echo "[comparative] maxent_objective=${MAXENT_OBJECTIVE}"
echo "[comparative] include_maxent_length_dual_arm=${INCLUDE_MAXENT_LENGTH_DUAL_ARM}"
echo "[comparative] include_diayn_arm=${INCLUDE_DIAYN_ARM} options=${DIAYN_NUM_OPTIONS} beta=${DIAYN_MI_BETA} leave_one_out=${DIAYN_MI_LEAVE_ONE_OUT}"
echo "[comparative] include_outcome_collision_arm=${INCLUDE_OUTCOME_COLLISION_ARM} coefficient=${OUTCOME_COLLISION_COEF}"
echo "[comparative] include_outcome_collision_outside_centering_arm=${INCLUDE_OUTCOME_COLLISION_OUTSIDE_CENTERING_ARM} coefficient=${OUTCOME_COLLISION_COEF}"
echo "[comparative] include_semantic_shannon_arm=${INCLUDE_SEMANTIC_SHANNON_ARM} coefficient=${SEMANTIC_SHANNON_COEF} surprisal_clip=${SEMANTIC_SHANNON_SURPRISAL_CLIP} pseudocount=${SEMANTIC_SHANNON_PSEUDOCOUNT}"
echo "[comparative] include_semantic_shannon_advantage_arm=${INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM} coefficient=${SEMANTIC_SHANNON_COEF} surprisal_clip=${SEMANTIC_SHANNON_SURPRISAL_CLIP} pseudocount=${SEMANTIC_SHANNON_PSEUDOCOUNT}"
echo "[comparative] include_quality_gated_semantic_novelty_arm=${INCLUDE_QUALITY_GATED_SEMANTIC_NOVELTY_ARM} coefficient=${SEMANTIC_SHANNON_COEF} surprisal_clip=${SEMANTIC_SHANNON_SURPRISAL_CLIP} pseudocount=${SEMANTIC_SHANNON_PSEUDOCOUNT} cap=${SEMANTIC_SHANNON_QUALITY_GATED_CAP}"
echo "[comparative] include_success_conditioned_signed_semantic_shannon_arm=${INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM} coefficient=${SEMANTIC_SHANNON_COEF} surprisal_clip=${SEMANTIC_SHANNON_SURPRISAL_CLIP} pseudocount=${SEMANTIC_SHANNON_PSEUDOCOUNT} symmetric_cap=${SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP}"
echo "[comparative] include_signal_first_semantic_balance_arm=${INCLUDE_SIGNAL_FIRST_SEMANTIC_BALANCE_ARM} xdr_tau=${SIGNAL_FIRST_XDR_TAU} task_only_xdr_weights=1"
echo "[comparative] include_online_canonical_maxent_arm=${INCLUDE_ONLINE_CANONICAL_MAXENT_ARM} alpha=${ONLINE_CANONICAL_BANK_ALPHA} novelty_beta=${ONLINE_CANONICAL_NOVELTY_BETA} pseudocount=${ONLINE_CANONICAL_BANK_PSEUDOCOUNT} surprisal_clip=${ONLINE_CANONICAL_BANK_SURPRISAL_CLIP} key_mode=${ONLINE_CANONICAL_KEY_MODE}"
echo "[comparative] include_online_canonical_haarnoja_arm=${INCLUDE_ONLINE_CANONICAL_HAARNOJA_ARM} normalized_target=${ONLINE_CANONICAL_DUAL_TARGET_RATIO} alpha_bounds=[${ONLINE_CANONICAL_DUAL_MIN_ALPHA},${ONLINE_CANONICAL_DUAL_MAX_ALPHA}] alpha_lr=${ONLINE_CANONICAL_DUAL_ALPHA_LR} ema_decay=${ONLINE_CANONICAL_DUAL_EMA_DECAY}"
echo "[comparative] include_online_canonical_policy_entropy_arm=${INCLUDE_ONLINE_CANONICAL_POLICY_ENTROPY_ARM} reference_alpha=${ONLINE_CANONICAL_BANK_ALPHA} projection=none warmup_steps=${ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS} ema_decay=${ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY}"
echo "[comparative] only_arms=${ONLY_ARMS_CSV:-all-configured-arms}"
echo "[comparative] sbatch_hold=${SBATCH_HOLD}"

if [[ "${OAT_ZERO_COMPARATIVE_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "[comparative] configuration only; no jobs submitted"
  exit 0
fi

mkdir -p "$ROOT_DIR/var/artifacts/logs"

# Freeze the Python objective before submission, not when a queued job finally
# starts. Every arm in a campaign—and every same-stamp recovery—therefore
# imports identical source even if the shared working tree changes meanwhile.
CAMPAIGN_SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/${STAMP_PREFIX}"
CAMPAIGN_SOURCE_ROOT="${OAT_ZERO_CAMPAIGN_SOURCE_ROOT:-${CAMPAIGN_SOURCE_PARENT}/src}"
if [[ ! -f "${CAMPAIGN_SOURCE_ROOT}/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$CAMPAIGN_SOURCE_PARENT"
  source_snapshot_tmp="$(mktemp -d "${CAMPAIGN_SOURCE_PARENT}/.source.XXXXXX")"
  mkdir -p "$source_snapshot_tmp/src"
  cp -a "$ROOT_DIR/src/." "$source_snapshot_tmp/src/"
  mv "$source_snapshot_tmp/src" "$CAMPAIGN_SOURCE_ROOT"
  rmdir "$source_snapshot_tmp"
fi
echo "[comparative] source_snapshot=${CAMPAIGN_SOURCE_ROOT}"

manifest="$ROOT_DIR/var/artifacts/${STAMP_PREFIX}_comparative_jobs.tsv"
if [[ -f "$manifest" ]]; then
  if [[ "${OAT_ZERO_APPEND_MANIFEST:-0}" != "1" ]]; then
    echo "Manifest ${manifest} already exists; pick a fresh RUN_STAMP_PREFIX" >&2
    echo "instead of resubmitting under an old stamp (mixed-environment runs" >&2
    echo "would be indistinguishable in the analysis)." >&2
    exit 1
  fi
  echo "[comparative] appending same-stamp recovery jobs to ${manifest}"
else
  printf "arm\tseed\tjob_id\trun_stamp\n" > "$manifest"
fi

for seed in "${train_seeds[@]}"; do
  if arm_enabled grpo; then
    job_id="$(submit_arm grpo "$DRGRPO_VARIANT" "$seed")"
    printf "grpo\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_grpo_s${seed}" >> "$manifest"
    echo "[comparative] grpo seed=${seed} job=${job_id}"
  fi

  for tau in "${xdr_taus[@]}"; do
    arm="xdr_tau${tau//./p}"
    if arm_enabled "$arm"; then
      job_id="$(submit_arm "$arm" xdr "$seed" "$tau")"
      printf "%s\t%s\t%s\t%s\n" "$arm" "$seed" "$job_id" "${STAMP_PREFIX}_${arm}_s${seed}" >> "$manifest"
      echo "[comparative] ${arm} seed=${seed} job=${job_id}"
    fi
  done

  if [[ "$INCLUDE_XDR_TAU_CONTROL_ARM" == "1" ]] && arm_enabled xdr_tau_control; then
    job_id="$(submit_arm xdr_tau_control xdr_tau_control "$seed" "$XDR_TAU_CONTROL_BASE")"
    printf "xdr_tau_control\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_xdr_tau_control_s${seed}" >> "$manifest"
    echo "[comparative] xdr_tau_control seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_XDR_SAC_DUAL_ARM" == "1" ]] && arm_enabled xdr_sac_dual; then
    job_id="$(submit_arm xdr_sac_dual xdr_sac_dual "$seed" "$XDR_SAC_DUAL_BASE")"
    printf "xdr_sac_dual\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_xdr_sac_dual_s${seed}" >> "$manifest"
    echo "[comparative] xdr_sac_dual seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_MAXENT_ARM" == "1" ]] && arm_enabled maxent; then
    job_id="$(submit_arm maxent maxent "$seed")"
    printf "maxent\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_maxent_s${seed}" >> "$manifest"
    echo "[comparative] maxent seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_MAXENT_CONTROL_ARM" == "1" ]] && arm_enabled maxent_control; then
    job_id="$(submit_arm maxent_control maxent_control "$seed")"
    printf "maxent_control\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_maxent_control_s${seed}" >> "$manifest"
    echo "[comparative] maxent_control seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_MAXENT_DUAL_ARM" == "1" ]] && arm_enabled maxent_dual; then
    job_id="$(submit_arm maxent_dual maxent_dual "$seed")"
    printf "maxent_dual\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_maxent_dual_s${seed}" >> "$manifest"
    echo "[comparative] maxent_dual seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_MAXENT_INVERSE_ARM" == "1" ]] && arm_enabled maxent_inverse; then
    job_id="$(submit_arm maxent_inverse maxent_inverse "$seed")"
    printf "maxent_inverse\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_maxent_inverse_s${seed}" >> "$manifest"
    echo "[comparative] maxent_inverse seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_MAXENT_INVERSE_CANONICAL_ARM" == "1" ]] && arm_enabled maxent_inverse_canonical; then
    job_id="$(submit_arm maxent_inverse_canonical maxent_inverse_canonical "$seed")"
    printf "maxent_inverse_canonical\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_maxent_inverse_canonical_s${seed}" >> "$manifest"
    echo "[comparative] maxent_inverse_canonical seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_MAXENT_INVERSE_CANONICAL_REPLAY_ARM" == "1" ]] && arm_enabled maxent_inverse_canonical_replay; then
    job_id="$(submit_arm maxent_inverse_canonical_replay maxent_inverse_canonical_replay "$seed")"
    printf "maxent_inverse_canonical_replay\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_maxent_inverse_canonical_replay_s${seed}" >> "$manifest"
    echo "[comparative] maxent_inverse_canonical_replay seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_OPEN_SET_SPLIT_CANONICAL_ARM" == "1" ]] && arm_enabled open_set_split_canonical; then
    job_id="$(submit_arm open_set_split_canonical open_set_split_canonical "$seed")"
    printf "open_set_split_canonical\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_open_set_split_canonical_s${seed}" >> "$manifest"
    echo "[comparative] open_set_split_canonical seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_VERIFIED_FIRST_SPLIT_CANONICAL_ARM" == "1" ]] && arm_enabled verified_first_split_canonical; then
    job_id="$(submit_arm verified_first_split_canonical verified_first_split_canonical "$seed")"
    printf "verified_first_split_canonical\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_verified_first_split_canonical_s${seed}" >> "$manifest"
    echo "[comparative] verified_first_split_canonical seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM" == "1" ]] && arm_enabled verified_first_global_replay_canonical; then
    job_id="$(submit_arm verified_first_global_replay_canonical verified_first_global_replay_canonical "$seed")"
    printf "verified_first_global_replay_canonical\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_verified_first_global_replay_canonical_s${seed}" >> "$manifest"
    echo "[comparative] verified_first_global_replay_canonical seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL_ARM" == "1" ]] && arm_enabled verified_first_bootstrap_local_canonical; then
    job_id="$(submit_arm verified_first_bootstrap_local_canonical verified_first_bootstrap_local_canonical "$seed")"
    printf "verified_first_bootstrap_local_canonical\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_verified_first_bootstrap_local_canonical_s${seed}" >> "$manifest"
    echo "[comparative] verified_first_bootstrap_local_canonical seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_VERIFIED_COUNTERFACTUAL_CANONICAL_ARM" == "1" ]] && arm_enabled verified_counterfactual_canonical; then
    job_id="$(submit_arm verified_counterfactual_canonical verified_counterfactual_canonical "$seed")"
    printf "verified_counterfactual_canonical\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_verified_counterfactual_canonical_s${seed}" >> "$manifest"
    echo "[comparative] verified_counterfactual_canonical seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM" == "1" ]] && arm_enabled verified_entropy_gated_singleton_escape_canonical; then
    job_id="$(submit_arm verified_entropy_gated_singleton_escape_canonical verified_entropy_gated_singleton_escape_canonical "$seed")"
    printf "verified_entropy_gated_singleton_escape_canonical\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_verified_entropy_gated_singleton_escape_canonical_s${seed}" >> "$manifest"
    echo "[comparative] verified_entropy_gated_singleton_escape_canonical seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM" == "1" ]] && arm_enabled verified_route_successor; then
    job_id="$(submit_arm verified_route_successor verified_route_successor "$seed")"
    printf "verified_route_successor\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_verified_route_successor_s${seed}" >> "$manifest"
    echo "[comparative] verified_route_successor seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_MAXENT_LENGTH_DUAL_ARM" == "1" ]] && arm_enabled maxent_length_dual; then
    job_id="$(submit_arm maxent_length_dual maxent_length_dual "$seed")"
    printf "maxent_length_dual\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_maxent_length_dual_s${seed}" >> "$manifest"
    echo "[comparative] maxent_length_dual seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_TOKEN_ENTROPY_ARM" == "1" ]] && arm_enabled grpo_entropy; then
    job_id="$(submit_arm grpo_entropy grpo_entropy "$seed")"
    printf "grpo_entropy\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_grpo_entropy_s${seed}" >> "$manifest"
    echo "[comparative] grpo_entropy seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_XDR_ADAPT_ARM" == "1" ]] && arm_enabled xdr_adapt; then
    job_id="$(submit_arm xdr_adapt xdr_adapt "$seed" "$XDR_ADAPT_TAU0")"
    printf "xdr_adapt\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_xdr_adapt_s${seed}" >> "$manifest"
    echo "[comparative] xdr_adapt seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_SEED_ARM" == "1" ]] && arm_enabled seed; then
    job_id="$(submit_arm seed seed "$seed" "" "$SEED_ENTROPY_ALPHA")"
    printf "seed\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_seed_s${seed}" >> "$manifest"
    echo "[comparative] seed seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_DIAYN_ARM" == "1" ]] && arm_enabled diayn; then
    job_id="$(submit_arm diayn diayn "$seed")"
    printf "diayn\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_diayn_s${seed}" >> "$manifest"
    echo "[comparative] diayn seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_OUTCOME_COLLISION_ARM" == "1" ]] && arm_enabled outcome_collision; then
    job_id="$(submit_arm outcome_collision outcome_collision "$seed")"
    printf "outcome_collision\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_outcome_collision_s${seed}" >> "$manifest"
    echo "[comparative] outcome_collision seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_OUTCOME_COLLISION_OUTSIDE_CENTERING_ARM" == "1" ]] && arm_enabled outcome_collision_outside_centering; then
    job_id="$(submit_arm outcome_collision_outside_centering outcome_collision_outside_centering "$seed")"
    printf "outcome_collision_outside_centering\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_outcome_collision_outside_centering_s${seed}" >> "$manifest"
    echo "[comparative] outcome_collision_outside_centering seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_SEMANTIC_SHANNON_ARM" == "1" ]] && arm_enabled semantic_shannon; then
    job_id="$(submit_arm semantic_shannon semantic_shannon "$seed")"
    printf "semantic_shannon\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_semantic_shannon_s${seed}" >> "$manifest"
    echo "[comparative] semantic_shannon seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM" == "1" ]] && arm_enabled semantic_shannon_advantage; then
    job_id="$(submit_arm semantic_shannon_advantage semantic_shannon_advantage "$seed")"
    printf "semantic_shannon_advantage\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_semantic_shannon_advantage_s${seed}" >> "$manifest"
    echo "[comparative] semantic_shannon_advantage seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_QUALITY_GATED_SEMANTIC_NOVELTY_ARM" == "1" ]] && arm_enabled quality_gated_semantic_novelty; then
    job_id="$(submit_arm quality_gated_semantic_novelty quality_gated_semantic_novelty "$seed")"
    printf "quality_gated_semantic_novelty\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_quality_gated_semantic_novelty_s${seed}" >> "$manifest"
    echo "[comparative] quality_gated_semantic_novelty seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM" == "1" ]] && arm_enabled success_conditioned_signed_semantic_shannon; then
    job_id="$(submit_arm success_conditioned_signed_semantic_shannon success_conditioned_signed_semantic_shannon "$seed")"
    printf "success_conditioned_signed_semantic_shannon\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_success_conditioned_signed_semantic_shannon_s${seed}" >> "$manifest"
    echo "[comparative] success_conditioned_signed_semantic_shannon seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_SIGNAL_FIRST_SEMANTIC_BALANCE_ARM" == "1" ]] && arm_enabled signal_first_semantic_balance; then
    job_id="$(submit_arm signal_first_semantic_balance signal_first_semantic_balance "$seed" "$SIGNAL_FIRST_XDR_TAU")"
    printf "signal_first_semantic_balance\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_signal_first_semantic_balance_s${seed}" >> "$manifest"
    echo "[comparative] signal_first_semantic_balance seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_ONLINE_CANONICAL_MAXENT_ARM" == "1" ]] && arm_enabled online_canonical_maxent; then
    job_id="$(submit_arm online_canonical_maxent online_canonical_maxent "$seed")"
    printf "online_canonical_maxent\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_online_canonical_maxent_s${seed}" >> "$manifest"
    echo "[comparative] online_canonical_maxent seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_ONLINE_CANONICAL_HAARNOJA_ARM" == "1" ]] && arm_enabled online_canonical_haarnoja; then
    job_id="$(submit_arm online_canonical_haarnoja online_canonical_haarnoja "$seed")"
    printf "online_canonical_haarnoja\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_online_canonical_haarnoja_s${seed}" >> "$manifest"
    echo "[comparative] online_canonical_haarnoja seed=${seed} job=${job_id}"
  fi
  if [[ "$INCLUDE_ONLINE_CANONICAL_POLICY_ENTROPY_ARM" == "1" ]] && arm_enabled online_canonical_policy_entropy; then
    job_id="$(submit_arm online_canonical_policy_entropy online_canonical_policy_entropy "$seed")"
    printf "online_canonical_policy_entropy\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_online_canonical_policy_entropy_s${seed}" >> "$manifest"
    echo "[comparative] online_canonical_policy_entropy seed=${seed} job=${job_id}"
  fi
done

echo "[comparative] manifest=${manifest}"
echo "[comparative] next: RUN_STAMP_PREFIX=${STAMP_PREFIX} ops/run_countdown_comparative_eval.sh (on a GPU node)"
