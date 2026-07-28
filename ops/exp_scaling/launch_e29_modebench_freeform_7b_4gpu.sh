#!/usr/bin/env bash
# Launch E29's matched free-form 7B Graph/Countdown cohort on one four-GPU node.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PHASE="${1:-}"
case "$PHASE" in
  config|full|recover48) ;;
  *) echo "Usage: $0 {config|full|recover48}" >&2; exit 2 ;;
esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/e29_modebench_freeform_7b_4gpu.md"
MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_REFERENCE="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_HASH=4e9bcb75c635d6e746eaf6389f31de8b0d321b6d8f79b895bb0dc4a438b44a86
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/e29_freeform_7b_4gpu_${SOURCE_HASH}"
SOURCE_ROOT="$SNAPSHOT_PARENT/src"
OPS_ROOT="$SNAPSHOT_PARENT/ops"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
GRAPH_PREFIX=gce29_freeform_7b_4gpu_v5_buffer_restore
COUNTDOWN_PREFIX=cde29_freeform_7b_4gpu_v5_buffer_restore
GRAPH_TARGET=1.622718550885717
COUNTDOWN_TARGET=1.347109432487438

for required in \
  "$PYTHON_BIN" "$PROTOCOL" "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E29 prerequisite: $required" >&2
    exit 1
  fi
done

if ! grep -q '^\*\*Status: FROZEN USER-REQUESTED SCALE EXTENSION' "$PROTOCOL"; then
  echo "E29 protocol is not frozen" >&2
  exit 1
fi
for tokenizer_file in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  if ! cmp -s "$TOKENIZER_REFERENCE/$tokenizer_file" "$MODEL_ROOT/$tokenizer_file"; then
    echo "E29 tokenizer drift: $tokenizer_file differs from the matched cohorts" >&2
    exit 1
  fi
done

if [[ "$PHASE" == full && ! -e "$SNAPSHOT_PARENT" ]]; then
  mkdir "$SNAPSHOT_PARENT"
  snapshot_tmp="$(mktemp -d "$SNAPSHOT_PARENT/.snapshot.XXXXXX")"
  mkdir -p "$snapshot_tmp/src" "$snapshot_tmp/ops"
  cp -a "$ROOT_DIR/src/." "$snapshot_tmp/src/"
  cp -a "$ROOT_DIR/ops/." "$snapshot_tmp/ops/"
  mv "$snapshot_tmp/src" "$SOURCE_ROOT"
  mv "$snapshot_tmp/ops" "$OPS_ROOT"
  rmdir "$snapshot_tmp"
fi

if [[ "$PHASE" == config ]]; then
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
fi
observed_source_hash="$(
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
    "$SOURCE_ROOT" "$ROOT_DIR"
)"
if [[ "$observed_source_hash" != "$SOURCE_HASH" ]]; then
  echo "E29 source mismatch: expected=$SOURCE_HASH observed=$observed_source_hash" >&2
  exit 1
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_PYTHON="$PYTHON_BIN"
export OAT_ZERO_PYTHON_LIB_DIR="$ROOT_DIR/var/seed_paper_eval/paper310/lib"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$PROTOCOL"
export OAT_ZERO_7B_ANALYTICAL_APPROVED=1
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-7b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=grpo,maxent_dual
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
export OAT_ZERO_NUM_PROMPT_EPOCH=10
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=4
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1
export OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1

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
export OAT_ZERO_VLLM_SLEEP_LEVEL=2
unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_STALE_SECONDS=2700
export OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600
export OAT_ZERO_WATCHDOG_POLL_SECONDS=30
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E29_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E29_TRAIN_GRES:-gpu:a100:4}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E29_TRAIN_PARTITION:-mltheory}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E29_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E29_TRAIN_MEMORY:-256G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E29_TRAIN_TIME_LIMIT:-168:00:00}"
export OAT_ZERO_SBATCH_HOLD=1

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

if [[ "$PHASE" == recover48 ]]; then
  export OAT_ZERO_APPEND_MANIFEST=1
  export OAT_ZERO_SBATCH_HOLD=1
  export OAT_ZERO_VLLM_GPU_RATIO=0.40
  export OAT_ZERO_TRAIN_NODELIST=node103,node104,node205,node206,node207,node208
  export OAT_ZERO_TRAIN_GRES=gpu:a6000:4
  export OAT_ZERO_TRAIN_PARTITION=lowprio
  export OAT_ZERO_TRAIN_ACCOUNT=allcs

  export OAT_ZERO_TRAIN_SEEDS=43
  export OAT_ZERO_ONLY_ARMS=maxent_dual
  submit_task graph_coloring
  export OAT_ZERO_TRAIN_SEEDS=45
  export OAT_ZERO_ONLY_ARMS=grpo,maxent_dual
  submit_task graph_coloring

  export OAT_ZERO_TRAIN_SEEDS=44,45
  export OAT_ZERO_ONLY_ARMS=grpo,maxent_dual
  submit_task countdown

  mapfile -t recovery_jobs < <(
    {
      tail -n 3 "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
      tail -n 4 "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
    } | awk -F '\t' '$3 ~ /^[0-9]+$/ {print $3}'
  )
  if (( ${#recovery_jobs[@]} != 7 )); then
    echo "E29 48GB recovery incomplete (${#recovery_jobs[@]}/7)" >&2
    exit 1
  fi
  for job_id in "${recovery_jobs[@]}"; do
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=lowprio' \
      'Account=allcs' 'TresPerNode=gres/gpu:a6000:4' \
      'OAT_ZERO_VLLM_GPU_RATIO=0.40' 'OAT_ZERO_VLLM_SLEEP_LEVEL=2'; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E29 48GB recovery audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  done
  echo "[e29] audited held 48GB recovery jobs: ${recovery_jobs[*]}"
  exit 0
fi

if [[ "$PHASE" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  submit_task graph_coloring
  submit_task countdown
  echo "[e29] both matched 7B four-GPU task configurations passed"
  exit 0
fi
unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY

for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  if [[ -e "$manifest" ]]; then
    echo "E29 manifest already exists; refusing duplicate cohort: $manifest" >&2
    exit 1
  fi
done
submit_task graph_coloring
submit_task countdown

job_ids=()
for spec in \
  "$GRAPH_PREFIX|$GRAPH_TARGET|192|48" \
  "$COUNTDOWN_PREFIX|$COUNTDOWN_TARGET|384|96"; do
  IFS='|' read -r prefix target max_train save_steps <<< "$spec"
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(
    awk -F '\t' 'NR > 1 && $1 ~ /^(grpo|maxent_dual)$/ && $2 ~ /^(43|44|45)$/ && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if (( ${#task_jobs[@]} != 6 )); then
    echo "E29 ${prefix} cohort incomplete (${#task_jobs[@]}/6); jobs remain held" >&2
    exit 1
  fi
  for job_id in "${task_jobs[@]}"; do
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=mltheory' \
      'Account=mltheory' 'ReqNodeList=node302' \
      'TresPerNode=gres/gpu:a100:4' 'mem=256G' \
      'OAT_ZERO_MODEL=qwen2.5-7b-instruct' \
      'OAT_ZERO_CANONICAL_ACTION_TASK=none' \
      'OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1' \
      'OAT_ZERO_NUM_SAMPLES=16' 'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4' \
      'OAT_ZERO_ROLLOUT_BATCH_SIZE=4' 'OAT_ZERO_N_GPU=4' \
      'OAT_ZERO_NUM_GPUS_PER_ACTOR=1' \
      'OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1' 'OAT_ZERO_SYNC_PARAMS_EVERY=1' \
      'OAT_ZERO_VLLM_SLEEP_LEVEL=2' \
      "OAT_ZERO_MAX_TRAIN=${max_train}" "OAT_ZERO_SAVE_STEPS=${save_steps}"; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E29 held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  done
  job_ids+=("${task_jobs[@]}")
done

if (( ${#job_ids[@]} != 12 )); then
  echo "E29 cohort incomplete (${#job_ids[@]}/12); jobs remain held" >&2
  exit 1
fi
echo "[e29] audited 12-job matched free-form 7B cohort; jobs remain held: ${job_ids[*]}"
echo "[e29] graph_manifest=$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
echo "[e29] countdown_manifest=$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
