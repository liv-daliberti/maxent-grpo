#!/usr/bin/env bash
# Launch the E23 canonical Countdown 7B four-GPU empty-shard fix cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PHASE="${1:-full}"
case "$PHASE" in
  config|full) ;;
  *) echo "Usage: $0 {config|full}" >&2; exit 2 ;;
esac

PROTOCOL="$ROOT_DIR/paper/preregistration/e23_canonical_maxent_7b_countdown.md"
SOURCE_HASH=044f6df047788dc8b67bbe224281a403c6d5eab04de89881f5a17cfe5c147cf9
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/e23_e24_canonical_4gpu_fix_${SOURCE_HASH}"
SOURCE_ROOT="$SNAPSHOT_PARENT/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e23_e24_canonical_4gpu_ops_72abacc11178931c3d3923c691fa303d06e45aec0dfef686e87ec607ef4bd808/ops"
SUBMITTER="$ROOT_DIR/ops/submit_countdown_comparative.sh"
MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
E16_MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
STAMP=cde23_canonical_maxent_7b_v6_4xa100_evalsync_fix
MANIFEST="$ROOT_DIR/var/artifacts/${STAMP}_comparative_jobs.tsv"

for required in \
  "$PROTOCOL" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$SUBMITTER" \
  "$OPS_ROOT/slurm/train_node302.slurm" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$PYTHON_BIN"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E23 prerequisite: $required" >&2
    exit 1
  fi
done

observed_source_hash="$(
  PYTHONPATH="$SNAPSHOT_PARENT" "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
    "$SOURCE_ROOT" "$ROOT_DIR"
)"
if [[ "$observed_source_hash" != "$SOURCE_HASH" ]]; then
  echo "Frozen E23 source mismatch: expected=$SOURCE_HASH observed=$observed_source_hash" >&2
  exit 1
fi

for tokenizer_file in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  if ! cmp -s "$E16_MODEL_ROOT/$tokenizer_file" "$MODEL_ROOT/$tokenizer_file"; then
    echo "E23 tokenizer drift: $tokenizer_file differs from E16" >&2
    exit 1
  fi
done

export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_PYTHON="$PYTHON_BIN"
export OAT_ZERO_PYTHON_LIB_DIR="$ROOT_DIR/var/seed_paper_eval/paper310/lib"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$PROTOCOL"
export OAT_ZERO_7B_ANALYTICAL_APPROVED=1

export RUN_STAMP_PREFIX="$STAMP"
export OAT_ZERO_COMPARATIVE_TASK=countdown
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-7b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
export OAT_ZERO_PROMPT_TEMPLATE=qwen_countdown_digits
export OAT_ZERO_CANONICAL_ACTION_TASK=countdown
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=maxent,maxent_control,maxent_dual
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_APPEND_MANIFEST=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_NUM_PROMPT_EPOCH=5
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=30704
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1.0
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0

export OAT_ZERO_MAXENT_ALPHA=0.10
export OAT_ZERO_MAXENT_FIXED_ALPHA=0.10
export OAT_ZERO_MAXENT_CONTROL_BASE_ALPHA=0.075
export OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.075
export OAT_ZERO_MAXENT_CONTROL_RATIO=1.0
export OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=3.9741470618167156
export OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=1
export OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=0.10
export OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=0.9
export OAT_ZERO_MAXENT_CONTROL_GAIN=4.0
export OAT_ZERO_MAXENT_DUAL_RATIO=1.0
export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=3.9741470618167156
export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=1
export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.05
export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.10
export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.005
export OAT_ZERO_MAXENT_LENGTH_TARGET=0

export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=4
export OAT_ZERO_NUM_GPUS_PER_ACTOR=4
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=1
export OAT_ZERO_ACTIVATION_OFFLOADING=1
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1
unset PYTORCH_CUDA_ALLOC_CONF

export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT=3
export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1
export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1
export OAT_ZERO_GENERATE_MAX_LENGTH=192
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_BATCH_SIZE=32
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_TEST_SPLIT=multi_answer

export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
# OAT counts these in learner updates. With rollout batch four, 96 prompts
# correspond to 24 updates. Keep sync/checkpoint aligned with evaluation.
export OAT_ZERO_SYNC_PARAMS_EVERY=24
export OAT_ZERO_SAVE_STEPS=24
export OAT_ZERO_SAVE_FROM=24
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=1
export OAT_ZERO_MAX_SAVE_MEM=2000
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_STALE_SECONDS=2700
export OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600
export OAT_ZERO_WATCHDOG_POLL_SECONDS=30
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E23_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E23_TRAIN_GRES:-gpu:a100:4}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E23_TRAIN_PARTITION:-mltheory}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E23_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E23_TRAIN_MEMORY:-192G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E23_TRAIN_TIME_LIMIT:-168:00:00}"
export OAT_ZERO_SBATCH_HOLD=1

if [[ "$PHASE" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  bash "$SUBMITTER"
  echo "[e23-v6-evalsync-fix] canonical Countdown-7B configuration passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
if [[ -e "$MANIFEST" ]]; then
  echo "E23 manifest already exists; refusing a duplicate cohort: $MANIFEST" >&2
  exit 1
fi

bash "$SUBMITTER"

mapfile -t job_ids < <(
  tail -n +2 "$MANIFEST" |
    awk -F '\t' '$1 ~ /^maxent(_control|_dual)?$/ && $2 ~ /^(43|44|45)$/ && $3 ~ /^[0-9]+$/ {print $3}'
)
if (( ${#job_ids[@]} != 9 )); then
  echo "E23 cohort incomplete (${#job_ids[@]}/9); submitted jobs remain held" >&2
  exit 1
fi

for arm in maxent maxent_control maxent_dual; do
  for seed in 43 44 45; do
    matches="$(awk -F '\t' -v arm="$arm" -v seed="$seed" '$1 == arm && $2 == seed {count++} END {print count+0}' "$MANIFEST")"
    if [[ "$matches" != 1 ]]; then
      echo "E23 manifest cell mismatch: arm=$arm seed=$seed count=$matches; jobs remain held" >&2
      exit 1
    fi
  done
done

for job_id in "${job_ids[@]}"; do
  job_dump="$(scontrol show job -o "$job_id")"
  manifest_row="$(awk -F '\t' -v job_id="$job_id" '$3 == job_id {print $1 "|" $2 "|" $4}' "$MANIFEST")"
  IFS='|' read -r arm seed run_stamp <<< "$manifest_row"
  case "$arm" in
    maxent) variant=maxent ;;
    maxent_control) variant=maxent_control ;;
    maxent_dual) variant=maxent_dual ;;
    *) echo "Unexpected E23 arm for job $job_id: $arm" >&2; exit 1 ;;
  esac
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'Account=mltheory' \
    'Partition=mltheory' \
    'ReqNodeList=node302' \
    'TresPerNode=gres/gpu:a100:4' \
    'mem=192G' \
    "RUN_STAMP=${run_stamp}" \
    "OAT_ZERO_SEED=${seed}" \
    "OAT_ZERO_VARIANT=${variant}" \
    'OAT_ZERO_MODEL=qwen2.5-7b-instruct' \
    'OAT_ZERO_CANONICAL_ACTION_TASK=countdown' \
    'OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1' \
    'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_N_GPU=4' \
    'OAT_ZERO_NUM_GPUS_PER_ACTOR=4' \
    'OAT_ZERO_ROLLOUT_BATCH_SIZE=4' \
    'OAT_ZERO_MAX_TRAIN=30704' \
    'OAT_ZERO_SYNC_PARAMS_EVERY=24' \
    'OAT_ZERO_SAVE_STEPS=24' \
    'OAT_ZERO_SAVE_FROM=24'; do
    if [[ "$job_dump" != *"$required"* ]]; then
      echo "E23 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
done

release_mode="${OAT_ZERO_7B_FIX_RELEASE_MODE:-held}"
case "$release_mode" in
  held)
    echo "[e23-v6-evalsync-fix] audited cohort remains held for canary release: ${job_ids[*]}"
    ;;
  all)
    scontrol release "${job_ids[@]}"
    echo "[e23-v6-evalsync-fix] released complete canonical Countdown-7B cohort: ${job_ids[*]}"
    ;;
  *)
    echo "OAT_ZERO_7B_FIX_RELEASE_MODE must be held or all" >&2
    exit 2
    ;;
esac
echo "[e23-v6-evalsync-fix] manifest=$MANIFEST"
