#!/usr/bin/env bash
# Launch the frozen E19 matched canonical Dr.GRPO-only 0.5B controls.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PHASE="${1:-config}"
case "$PHASE" in
  config|full|release) ;;
  *) echo "Usage: $0 {config|full|release}" >&2; exit 1 ;;
esac

PROTOCOL="$ROOT_DIR/paper/preregistration/e19_canonical_drgrpo_05b_control.md"
SOURCE_HASH=0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec
OPS_HASH=b7f01de3e9a9247afe6d95f984ce77c0e0615e4aa067a7fa77c701da15bc9597
SNAPSHOT_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e16_canonical_full_${SOURCE_HASH}"
SOURCE_ROOT="$SNAPSHOT_ROOT/src"
OPS_ROOT="$SNAPSHOT_ROOT/ops"
MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
GRAPH_PREFIX=gce19_canonical_drgrpo_05b_v1
COUNTDOWN_PREFIX=cde19_canonical_drgrpo_05b_v1
GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"

for required in \
  "$PROTOCOL" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/train.sh" \
  "$OPS_ROOT/submit_countdown_comparative.sh" \
  "$OPS_ROOT/exp_scaling/verify_e16_execution_surface.py" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E19 prerequisite: $required" >&2
    exit 1
  fi
done

observed_source_hash="$(
  PYTHONPATH="$SNAPSHOT_ROOT" "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
    "$SOURCE_ROOT" "$ROOT_DIR"
)"
if [[ "$observed_source_hash" != "$SOURCE_HASH" ]]; then
  echo "Frozen E19 source mismatch: expected=$SOURCE_HASH observed=$observed_source_hash" >&2
  exit 1
fi

observed_ops_hash="$(
  "$PYTHON_BIN" "$OPS_ROOT/exp_scaling/verify_e16_execution_surface.py" \
    --repo-root "$SNAPSHOT_ROOT" | jq -er '.sha256'
)"
if [[ "$observed_ops_hash" != "$OPS_HASH" ]]; then
  echo "Frozen E19 execution mismatch: expected=$OPS_HASH observed=$observed_ops_hash" >&2
  exit 1
fi

if ! grep -Fqx '  --beta "${OAT_ZERO_BETA:-0}"' "$OPS_ROOT/train.sh"; then
  echo "Frozen E19 trainer no longer guarantees beta=0 by default" >&2
  exit 1
fi

if [[ "$PHASE" == full || "$PHASE" == release ]]; then
  if ! grep -q '^\*\*Status: FROZEN' "$PROTOCOL"; then
    echo "E19 protocol is not frozen; refusing cohort mutation" >&2
    exit 1
  fi
fi

if [[ "$PHASE" == full ]]; then
  for manifest in "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST"; do
    if [[ -e "$manifest" ]]; then
      echo "E19 manifest already exists; refusing a duplicate cohort: $manifest" >&2
      exit 1
    fi
  done
fi

if [[ "$PHASE" == release ]]; then
  for manifest in "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST"; do
    if [[ ! -f "$manifest" ]]; then
      echo "E19 held-cohort manifest is missing: $manifest" >&2
      exit 1
    fi
  done
fi

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

export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=grpo
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
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_APPEND_MANIFEST=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_NUM_PROMPT_EPOCH=5
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1.0
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_XDR_TAU=inf
export OAT_ZERO_XDR_MODE_ADAPTIVE=0
export OAT_ZERO_XDR_TAU_CONTROL_RATIO=0
export OAT_ZERO_XDR_SAC_DUAL_RATIO=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_SEED_ENTROPY_ALPHA=0
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_MAXENT_CONTROL_RATIO=0
export OAT_ZERO_MAXENT_DUAL_RATIO=0
export OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=0
export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0
export OAT_ZERO_MAXENT_LENGTH_TARGET=0

export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=0
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1

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
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_TEST_SPLIT=multi_answer

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=5
export OAT_ZERO_MAX_SAVE_MEM=2000
export OAT_ZERO_SYNC_PARAMS_EVERY=1
export OAT_ZERO_AUTO_RESUME=0
export OAT_ZERO_WATCHDOG_REQUEUE=0
export OAT_ZERO_WATCHDOG_STALE_SECONDS=2700
export OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600
export OAT_ZERO_WATCHDOG_POLL_SECONDS=60

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E19_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E19_TRAIN_GRES:-gpu:a100:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E19_TRAIN_PARTITION:-mltheory}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E19_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E19_TRAIN_MEMORY:-64G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E19_TRAIN_TIME_LIMIT:-08:00:00}"
export OAT_ZERO_SBATCH_HOLD=1

submit_task() {
  local task="$1" prefix="$2" data_root="$3" prompt_template="$4"
  local budget="$5" updates="$6" eval_interval="$7"
  export RUN_STAMP_PREFIX="$prefix"
  export OAT_ZERO_COMPARATIVE_TASK="$task"
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$data_root"
  export OAT_ZERO_PROMPT_TEMPLATE="$prompt_template"
  export OAT_ZERO_CANONICAL_ACTION_TASK="$task"
  export OAT_ZERO_MAX_TRAIN="$budget"
  export OAT_ZERO_MAX_QUERIES="$budget"
  export OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES="$updates"
  export OAT_ZERO_EVAL_PROMPT_INTERVAL="$eval_interval"
  export OAT_ZERO_SAVE_STEPS="$eval_interval"
  export OAT_ZERO_SAVE_FROM="$eval_interval"
  unset OAT_ZERO_EVAL_STEPS
  if [[ "$task" == graph_coloring ]]; then
    export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=1
  else
    export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
  fi
  bash "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$PHASE" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  submit_task graph_coloring "$GRAPH_PREFIX" \
    "$ROOT_DIR/var/data/exact_answer_mode_probe" qwen_graph_digits 15344 960 48
  submit_task countdown "$COUNTDOWN_PREFIX" \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits 30704 1920 96
  echo "[e19] both matched 0.5B Dr.GRPO configurations passed; no jobs submitted"
  exit 0
fi

if [[ "$PHASE" == full ]]; then
  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  submit_task graph_coloring "$GRAPH_PREFIX" \
    "$ROOT_DIR/var/data/exact_answer_mode_probe" qwen_graph_digits 15344 960 48
  submit_task countdown "$COUNTDOWN_PREFIX" \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits 30704 1920 96
fi

mapfile -t job_ids < <(
  tail -n +2 "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" |
    awk -F '\t' '$3 ~ /^[0-9]+$/ {print $3}'
)
if (( ${#job_ids[@]} != 6 )); then
  echo "E19 cohort incomplete (${#job_ids[@]}/6); jobs remain held" >&2
  exit 1
fi
if (( $(printf '%s\n' "${job_ids[@]}" | sort -u | wc -l) != 6 )); then
  echo "E19 manifests contain duplicate job IDs; jobs remain held" >&2
  exit 1
fi

for job_id in "${job_ids[@]}"; do
  resolved="$(scontrol show job -dd "$job_id")"
  if [[ "$resolved" != *"JobState=PENDING"* || "$resolved" != *"Reason=JobHeldUser"* ]]; then
    echo "E19 job $job_id is not pending on the user hold; refusing cohort release" >&2
    exit 1
  fi
  for required_export in \
    OAT_ZERO_VARIANT=grpo \
    OAT_ZERO_XDR_TAU=inf \
    OAT_ZERO_MAXENT_ALPHA=0.0 \
    OAT_ZERO_POLICY_ENTROPY_COEF=0.0 \
    OAT_ZERO_SEED_ENTROPY_ALPHA=0.0 \
    OAT_ZERO_XDR_TAU_CONTROL_RATIO=0.0 \
    OAT_ZERO_XDR_SAC_DUAL_RATIO=0.0 \
    OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO=0.0 \
    OAT_ZERO_MAXENT_DUAL_TARGET_RATIO=0.0; do
    if [[ "$resolved" != *"$required_export"* ]]; then
      echo "E19 job $job_id missing frozen export $required_export; cohort remains held" >&2
      exit 1
    fi
  done
done

scontrol release "${job_ids[@]}"
echo "[e19] released complete six-job cohort: ${job_ids[*]}"
