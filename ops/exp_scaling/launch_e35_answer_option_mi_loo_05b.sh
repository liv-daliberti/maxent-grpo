#!/usr/bin/env bash
# Launch E35's three-seed Dr.GRPO versus leave-one-out answer-option MI cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|full) ;;
  *)
    echo "Usage: $0 {config|full}" >&2
    exit 1
    ;;
esac

PROTOCOL="$ROOT_DIR/paper/preregistration/e35_answer_option_mi_loo_05b.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
GRAPH_PREFIX=gce35_answer_option_mi_loo_05b_v2
COUNTDOWN_PREFIX=cde35_answer_option_mi_loo_05b_v2

for required in \
  "$PROTOCOL" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E35 prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E35 protocol is not frozen" >&2
  exit 1
fi

if [[ "$phase" == full ]]; then
  for manifest in \
    "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
    "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"; do
    if [[ -e "$manifest" ]]; then
      echo "Fresh E35 prefix required; manifest already exists: $manifest" >&2
      exit 1
    fi
  done
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=3501,3502,3503
export OAT_ZERO_ONLY_ARMS=grpo,diayn
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0
export OAT_ZERO_INCLUDE_DIAYN_ARM=1

export OAT_ZERO_DIAYN_NUM_OPTIONS=4
export OAT_ZERO_DIAYN_MI_BETA=0.10
export OAT_ZERO_DIAYN_MI_EMA_DECAY=0.9
export OAT_ZERO_DIAYN_MI_SMOOTHING=1.0
export OAT_ZERO_DIAYN_MI_BONUS_CLIP=5.0
export OAT_ZERO_DIAYN_MI_CORRECT_ONLY=1
export OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT=1

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=5
export OAT_ZERO_NUM_PROMPT_EPOCH=5
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1

export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
export OAT_ZERO_TEST_SPLIT=multi_answer
export OAT_ZERO_VERIFIER_VERSION=fast
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_GENERATE_MAX_LENGTH=192
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
export OAT_ZERO_MAX_MODEL_LEN=512
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=350100
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1

export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=0
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_RESUME_STEPS=-1
export OAT_ZERO_MAX_RESUME_NUM=1
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=1
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E35_TRAIN_NODELIST:-node105,node202,node203,node204,node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E35_TRAIN_GRES:-gpu:1}"
export OAT_ZERO_TRAIN_CPUS_PER_TASK="${OAT_ZERO_E35_TRAIN_CPUS_PER_TASK:-8}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E35_TRAIN_MEMORY:-64G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E35_TRAIN_TIME_LIMIT:-08:00:00}"

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
fi

export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
"$ROOT_DIR/ops/submit_countdown_comparative.sh"

export OAT_ZERO_COMPARATIVE_TASK=countdown
export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
"$ROOT_DIR/ops/submit_countdown_comparative.sh"

echo "[e35] ${phase} complete for the three-seed matched cohort"
