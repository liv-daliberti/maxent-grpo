#!/usr/bin/env bash
# Launch E25's 3B free-form conditional-token base-preserving dual treatment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PHASE="${1:-}"
case "$PHASE" in
  config|full|release-staged) ;;
  *) echo "Usage: $0 {config|full|release-staged}" >&2; exit 2 ;;
esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/e25_modebench_freeform_dual_3b.md"
MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
E22_MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_HASH=217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448
EXECUTION_HASH=05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_conditional_token_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e21_math_conditional_token_ops_${EXECUTION_HASH}/ops"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
GRAPH_PREFIX=gce25_freeform_conditional_dual_3b_v2
COUNTDOWN_PREFIX=cde25_freeform_conditional_dual_3b_v2
GRAPH_TARGET=1.622718550885717
COUNTDOWN_TARGET=1.347109432487438

for required in \
  "$PYTHON_BIN" \
  "$PROTOCOL" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors.index.json" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh" \
  "$OPS_ROOT/slurm/train_node302.slurm" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E25 prerequisite: $required" >&2
    exit 1
  fi
done

if ! grep -q '^\*\*Status: FROZEN' "$PROTOCOL"; then
  echo "E25 protocol is not frozen" >&2
  exit 1
fi

for tokenizer_file in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  if ! cmp -s "$E22_MODEL_ROOT/$tokenizer_file" "$MODEL_ROOT/$tokenizer_file"; then
    echo "E25 tokenizer drift: $tokenizer_file differs from E22-v2" >&2
    exit 1
  fi
done

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_PYTHON="$PYTHON_BIN"
export OAT_ZERO_PYTHON_LIB_DIR="$ROOT_DIR/var/seed_paper_eval/paper310/lib"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$PROTOCOL"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=maxent_dual
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0

export OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean
export OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_DUAL_RATIO=1.0
export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=64
export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00060
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010
export OAT_ZERO_MAXENT_LENGTH_TARGET=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
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
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
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
export OAT_ZERO_EVAL_BATCH_SIZE=32
export OAT_ZERO_SYNC_PARAMS_EVERY=1

export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0
export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=1
export OAT_ZERO_ACTIVATION_OFFLOADING=1
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1
unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E25_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E25_TRAIN_GRES:-gpu:a100:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E25_TRAIN_PARTITION:-mltheory}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E25_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E25_TRAIN_MEMORY:-96G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E25_TRAIN_TIME_LIMIT:-168:00:00}"

submit_task() {
  local task="$1"
  if [[ "$task" == graph_coloring ]]; then
    export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
    export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
    export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$GRAPH_TARGET"
    export OAT_ZERO_MAX_TRAIN=192
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
    export OAT_ZERO_SAVE_STEPS=48
    export OAT_ZERO_SAVE_FROM=48
  else
    export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
    export OAT_ZERO_COMPARATIVE_TASK=countdown
    export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
    export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$COUNTDOWN_TARGET"
    export OAT_ZERO_MAX_TRAIN=384
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
    export OAT_ZERO_SAVE_STEPS=96
    export OAT_ZERO_SAVE_FROM=96
  fi
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$PHASE" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=1
  submit_task graph_coloring
  submit_task countdown
  echo "[e25] both 3B free-form dual configurations passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
if [[ "$PHASE" == full ]]; then
  for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX"; do
    manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
    if [[ -e "$manifest" ]]; then
      echo "E25 manifest already exists; refusing duplicate cohort: $manifest" >&2
      exit 1
    fi
  done
  submit_task graph_coloring
  submit_task countdown
else
  echo "[e25] auditing the existing held cohort; no duplicate jobs submitted"
fi

job_ids=()
for spec in \
  "$GRAPH_PREFIX|$GRAPH_TARGET|192|48" \
  "$COUNTDOWN_PREFIX|$COUNTDOWN_TARGET|384|96"; do
  IFS='|' read -r prefix target max_train save_steps <<< "$spec"
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(
    awk -F '\t' 'NR > 1 && $1 == "maxent_dual" && $2 ~ /^(43|44|45)$/ && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if (( ${#task_jobs[@]} != 3 )); then
    echo "E25 ${prefix} cohort incomplete (${#task_jobs[@]}/3); jobs remain held" >&2
    exit 1
  fi
  for job_id in "${task_jobs[@]}"; do
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' \
      'Reason=JobHeldUser' \
      'Partition=mltheory' \
      'Account=mltheory' \
      'ReqNodeList=node302' \
      'TresPerNode=gres/gpu:a100:1' \
      'mem=96G' \
      'OAT_ZERO_MODEL=qwen2.5-3b-instruct' \
      'OAT_ZERO_VARIANT=maxent_dual' \
      'OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean' \
      'OAT_ZERO_MAXENT_ALPHA=0.000075' \
      "OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=${target}" \
      'OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075' \
      'OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00060' \
      'OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010' \
      'OAT_ZERO_CANONICAL_ACTION_TASK=none' \
      'OAT_ZERO_NUM_SAMPLES=16' \
      'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1' \
      'OAT_ZERO_N_GPU=1' \
      'OAT_ZERO_ADAM_OFFLOAD=1' \
      'OAT_ZERO_ACTIVATION_OFFLOADING=1' \
      "OAT_ZERO_MAX_TRAIN=${max_train}" \
      "OAT_ZERO_SAVE_STEPS=${save_steps}"; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E25 held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  done
  job_ids+=("${task_jobs[@]}")
done

if (( ${#job_ids[@]} != 6 )); then
  echo "E25 cohort incomplete (${#job_ids[@]}/6); jobs remain held" >&2
  exit 1
fi
scontrol release "${job_ids[@]}"
echo "[e25] released six 3B free-form dual jobs: ${job_ids[*]}"
