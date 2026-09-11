#!/usr/bin/env bash
# Launch the frozen E18 matched canonical Dr.GRPO-only 3B controls.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PHASE="${1:-full}"
case "$PHASE" in
  config|full|countdown-config|countdown-retry|graph-seed43-config|graph-seed43-retry|graph-seed44-config|graph-seed44-retry|graph-seed45-config|graph-seed45-retry|countdown-seed43-config|countdown-seed43-retry|countdown-seed44-config|countdown-seed44-retry) ;;
  *) echo "Usage: $0 {config|full|countdown-config|countdown-retry|graph-seed43-config|graph-seed43-retry|graph-seed44-config|graph-seed44-retry|graph-seed45-config|graph-seed45-retry|countdown-seed43-config|countdown-seed43-retry|countdown-seed44-config|countdown-seed44-retry}" >&2; exit 1 ;;
esac

PROTOCOL="$ROOT_DIR/paper/preregistration/e18_canonical_drgrpo_3b_control.md"
SOURCE_HASH=0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/e16_canonical_full_${SOURCE_HASH}"
SOURCE_ROOT="$SNAPSHOT_PARENT/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e17_canonical_3b_ops_a8414bcd1932787ee1cea7e8c0b23868282f45903eeb9d495b23e4649f397f0c/ops"
MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"

for required in \
  "$PROTOCOL" \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E18 prerequisite: $required" >&2
    exit 1
  fi
done

observed_source_hash="$(
  PYTHONPATH="$SNAPSHOT_PARENT" "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
    "$SOURCE_ROOT" "$ROOT_DIR"
)"
if [[ "$observed_source_hash" != "$SOURCE_HASH" ]]; then
  echo "Frozen E18 source mismatch: expected=$SOURCE_HASH observed=$observed_source_hash" >&2
  exit 1
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

export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct
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
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1.0
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_MAXENT_LENGTH_TARGET=0

export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO="${OAT_ZERO_E18_VLLM_GPU_RATIO:-0.25}"
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=1
export OAT_ZERO_ACTIVATION_OFFLOADING=1
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
export OAT_ZERO_EVAL_BATCH_SIZE=32
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_TEST_SPLIT=multi_answer

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=1
export OAT_ZERO_MAX_SAVE_MEM=2000
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=0
export OAT_ZERO_WATCHDOG_STALE_SECONDS=0

export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E18_TRAIN_NODELIST:-node103,node104,node208}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E18_TRAIN_GRES:-gpu:a6000:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E18_TRAIN_PARTITION:-lowprio}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E18_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E18_TRAIN_MEMORY:-96G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E18_TRAIN_TIME_LIMIT:-168:00:00}"
export OAT_ZERO_SBATCH_HOLD=1

submit_task() {
  local task="$1" prefix="$2" data_root="$3" prompt_template="$4"
  local budget="$5" eval_interval="$6"
  export RUN_STAMP_PREFIX="$prefix"
  export OAT_ZERO_COMPARATIVE_TASK="$task"
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$data_root"
  export OAT_ZERO_PROMPT_TEMPLATE="$prompt_template"
  export OAT_ZERO_CANONICAL_ACTION_TASK="$task"
  export OAT_ZERO_MAX_TRAIN="$budget"
  export OAT_ZERO_EVAL_PROMPT_INTERVAL="$eval_interval"
  export OAT_ZERO_SYNC_PARAMS_EVERY="$eval_interval"
  export OAT_ZERO_SAVE_STEPS="$eval_interval"
  export OAT_ZERO_SAVE_FROM="$eval_interval"
  if [[ "$task" == graph_coloring ]]; then
    export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=1
  else
    export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
  fi
  bash "$OPS_ROOT/submit_countdown_comparative.sh"
}

recover_one() {
  local mode="$1" task="$2" prefix="$3" data_root="$4"
  local prompt_template="$5" budget="$6" eval_interval="$7" seed="$8"
  local fresh_start="${9:-0}" prior_job_id="${10:-}"
  local manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  export OAT_ZERO_TRAIN_SEEDS="$seed"
  if [[ "$fresh_start" == "1" ]]; then
    # A model-only or partially written ZeRO checkpoint is not resumable.
    # Start from the pinned pretrained model instead of resetting optimizer
    # state halfway through the registered trajectory.
    export OAT_ZERO_AUTO_RESUME=0
    unset OAT_ZERO_RESUME_DIR OAT_ZERO_RESUME_TAG
  fi
  if [[ "$mode" == config ]]; then
    export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
    submit_task "$task" "$prefix" "$data_root" "$prompt_template" \
      "$budget" "$eval_interval"
    echo "[e18] ${task} seed-${seed} recovery configuration passed; fresh_start=${fresh_start}; no job submitted"
    return 0
  fi

  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  export OAT_ZERO_APPEND_MANIFEST=1
  if [[ ! -f "$manifest" ]]; then
    echo "E18 ${task} recovery requires the original manifest: $manifest" >&2
    return 1
  fi
  local manifest_lines_before
  manifest_lines_before="$(wc -l < "$manifest")"
  submit_task "$task" "$prefix" "$data_root" "$prompt_template" \
    "$budget" "$eval_interval"
  mapfile -t job_ids < <(
    tail -n "+$((manifest_lines_before + 1))" "$manifest" |
      awk -F '\t' -v seed="$seed" \
        '$1 == "grpo" && $2 == seed && $3 ~ /^[0-9]+$/ {print $3}'
  )
  if (( ${#job_ids[@]} != 1 )); then
    echo "E18 ${task} seed-${seed} recovery incomplete; any new job remains held" >&2
    return 1
  fi

  local job_id="${job_ids[0]}" job_dump
  job_dump="$(scontrol show job -o "$job_id")"
  # Slurm may canonicalize node103,node104,node208 to node[103-104,208].
  # The immutable SubmitLine retains the literal request, so accept either.
  if [[ "$job_dump" != *"ReqNodeList=${OAT_ZERO_TRAIN_NODELIST}"* ]] \
    && [[ "$job_dump" != *"--nodelist=${OAT_ZERO_TRAIN_NODELIST}"* ]]; then
    echo "E18 ${task} seed-${seed} held-job audit failed: missing requested node list" >&2
    return 1
  fi
  for required in \
    'JobState=PENDING' \
    "${OAT_ZERO_TRAIN_GRES}" \
    "RUN_STAMP=${prefix}_grpo_s${seed}"; do
    if [[ "$job_dump" != *"$required"* ]]; then
      echo "E18 ${task} seed-${seed} held-job audit failed: missing $required" >&2
      return 1
    fi
  done
  if [[ "$fresh_start" == "1" ]]; then
    scontrol update JobId="$job_id" Comment="E18-fresh-start-no-auto-resume"
    job_dump="$(scontrol show job -o "$job_id")"
    if [[ "$job_dump" != *'Comment=E18-fresh-start-no-auto-resume'* ]]; then
      echo "E18 ${task} seed-${seed} fresh-start audit failed; replacement remains held" >&2
      return 1
    fi
  fi
  if [[ -n "$prior_job_id" ]]; then
    local prior_dump
    prior_dump="$(scontrol show job -o "$prior_job_id")"
    if [[ "$prior_dump" != *"RUN_STAMP=${prefix}_grpo_s${seed}"* ]] \
      || [[ ! "$prior_dump" =~ JobState=(PENDING|RUNNING|SUSPENDED|COMPLETING|CONFIGURING|REQUEUED) ]]; then
      echo "E18 ${task} seed-${seed} prior-writer audit failed for ${prior_job_id}; replacement remains held" >&2
      return 1
    fi
    scontrol update JobId="$prior_job_id" Requeue=0
    scancel "$prior_job_id"
    echo "[e18] retired non-resumable prior writer: $prior_job_id"
  fi
  scontrol release "$job_id"
  echo "[e18] released ${task} seed-${seed} recovery job: $job_id"
}

if [[ "$PHASE" == graph-seed43-config || "$PHASE" == graph-seed43-retry \
  || "$PHASE" == graph-seed44-config || "$PHASE" == graph-seed44-retry \
  || "$PHASE" == graph-seed45-config || "$PHASE" == graph-seed45-retry ]]; then
  graph_seed="${PHASE#graph-seed}"
  graph_seed="${graph_seed%%-*}"
  recover_one "${PHASE##*-}" graph_coloring gce18_canonical_drgrpo_3b_v1 \
    "$ROOT_DIR/var/data/exact_answer_mode_probe" qwen_graph_digits 15344 48 \
    "$graph_seed"
  exit $?
fi

if [[ "$PHASE" == countdown-seed43-config || "$PHASE" == countdown-seed43-retry ]]; then
  recover_one "${PHASE##*-}" countdown cde18_canonical_drgrpo_3b_v1 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits 30704 96 43 \
    1 30020609
  exit $?
fi

if [[ "$PHASE" == countdown-seed44-config || "$PHASE" == countdown-seed44-retry ]]; then
  recover_one "${PHASE##*-}" countdown cde18_canonical_drgrpo_3b_v1 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits 30704 96 44
  exit $?
fi

if [[ "$PHASE" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  submit_task graph_coloring gce18_canonical_drgrpo_3b_v1 \
    "$ROOT_DIR/var/data/exact_answer_mode_probe" qwen_graph_digits 15344 48
  submit_task countdown cde18_canonical_drgrpo_3b_v1 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits 30704 96
  echo "[e18] both matched Dr.GRPO configurations passed; no jobs submitted"
  exit 0
fi

if [[ "$PHASE" == countdown-config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  submit_task countdown cde18_canonical_drgrpo_3b_v1 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits 30704 96
  echo "[e18] Countdown matched Dr.GRPO recovery configuration passed; no jobs submitted"
  exit 0
fi

if [[ "$PHASE" == countdown-retry ]]; then
  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  export OAT_ZERO_APPEND_MANIFEST=1
  countdown_manifest="$ROOT_DIR/var/artifacts/cde18_canonical_drgrpo_3b_v1_comparative_jobs.tsv"
  if [[ ! -f "$countdown_manifest" ]]; then
    echo "E18 Countdown recovery requires the original manifest: $countdown_manifest" >&2
    exit 1
  fi
  manifest_lines_before="$(wc -l < "$countdown_manifest")"
  submit_task countdown cde18_canonical_drgrpo_3b_v1 \
    "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits 30704 96
  mapfile -t job_ids < <(
    tail -n "+$((manifest_lines_before + 1))" "$countdown_manifest" |
      awk -F '\t' '$1 == "grpo" && $3 ~ /^[0-9]+$/ {print $3}'
  )
  if (( ${#job_ids[@]} != 3 )); then
    echo "E18 Countdown recovery incomplete (${#job_ids[@]}/3); new jobs remain held" >&2
    exit 1
  fi
  scontrol release "${job_ids[@]}"
  echo "[e18] released three Countdown recovery jobs: ${job_ids[*]}"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
submit_task graph_coloring gce18_canonical_drgrpo_3b_v1 \
  "$ROOT_DIR/var/data/exact_answer_mode_probe" qwen_graph_digits 15344 48
submit_task countdown cde18_canonical_drgrpo_3b_v1 \
  "$ROOT_DIR/var/data/exact_countdown_easy3_probe" qwen_countdown_digits 30704 96

graph_manifest="$ROOT_DIR/var/artifacts/gce18_canonical_drgrpo_3b_v1_comparative_jobs.tsv"
countdown_manifest="$ROOT_DIR/var/artifacts/cde18_canonical_drgrpo_3b_v1_comparative_jobs.tsv"
mapfile -t job_ids < <(tail -n +2 "$graph_manifest" "$countdown_manifest" | awk -F '\t' '$3 ~ /^[0-9]+$/ {print $3}')
if (( ${#job_ids[@]} != 6 )); then
  echo "E18 cohort incomplete (${#job_ids[@]}/6); jobs remain held" >&2
  exit 1
fi
scontrol release "${job_ids[@]}"
echo "[e18] released complete six-job cohort: ${job_ids[*]}"
