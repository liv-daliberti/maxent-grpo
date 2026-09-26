#!/usr/bin/env bash
# Repair all 3B ModeBench free-form trajectories after stale-actor resumes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PHASE="${1:-}"
case "$PHASE" in
  config|stage|release-staged) ;;
  *) echo "Usage: $0 {config|stage|release-staged}" >&2; exit 2 ;;
esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/modebench_freeform_resume_repair_20260722.md"
MODEL_ROOT="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
SOURCE_HASH=33fbde7130221b188ad251696345e37fd227f7ef3ce21c19cabcab1f80637359
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/modebench_freeform_3b_resume_repair_v2_${SOURCE_HASH}"
SOURCE_ROOT="$SNAPSHOT_PARENT/src"
OPS_ROOT="$SNAPSHOT_PARENT/ops"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"

E25_GRAPH_PREFIX=gce25_freeform_conditional_dual_3b_repair_v2
E25_COUNTDOWN_PREFIX=cde25_freeform_conditional_dual_3b_repair_v2
E28_GRAPH_PREFIX=gce28_freeform_drgrpo_3b_repair_v2
E28_COUNTDOWN_PREFIX=cde28_freeform_drgrpo_3b_repair_v2
GRAPH_TARGET=1.622718550885717
COUNTDOWN_TARGET=1.347109432487438

for required in \
  "$PYTHON_BIN" "$PROTOCOL" "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" "$MODEL_ROOT/model.safetensors.index.json" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing repair prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN REMEDIATION ADDENDUM' "$PROTOCOL"; then
  echo "Resume-repair protocol is not frozen" >&2
  exit 1
fi

if [[ "$PHASE" == stage && ! -e "$SNAPSHOT_PARENT" ]]; then
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

for required in \
  "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh" \
  "$OPS_ROOT/run_experiment.sh" \
  "$OPS_ROOT/train.sh" \
  "$OPS_ROOT/slurm/train_node302.slurm"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing immutable repair runtime: $required" >&2
    exit 1
  fi
done
observed_source_hash="$(
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
    "$SOURCE_ROOT" "$ROOT_DIR"
)"
if [[ "$observed_source_hash" != "$SOURCE_HASH" ]]; then
  echo "Repair source mismatch: expected=$SOURCE_HASH observed=$observed_source_hash" >&2
  exit 1
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_PYTHON="$PYTHON_BIN"
export OAT_ZERO_PYTHON_LIB_DIR="$ROOT_DIR/var/seed_paper_eval/paper310/lib"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$PROTOCOL"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
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
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_REPAIR_TRAIN_NODELIST:-node302}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_REPAIR_TRAIN_GRES:-gpu:a100:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_REPAIR_TRAIN_PARTITION:-mltheory}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_REPAIR_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_REPAIR_TRAIN_MEMORY:-96G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_REPAIR_TRAIN_TIME_LIMIT:-168:00:00}"
export OAT_ZERO_SBATCH_HOLD=1

configure_task() {
  local task="$1" prefix="$2"
  export RUN_STAMP_PREFIX="$prefix"
  if [[ "$task" == graph_coloring ]]; then
    export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
    export OAT_ZERO_MAX_TRAIN=192
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
    export OAT_ZERO_RESUME_STEPS=192
    export OAT_ZERO_RESUME_FROM=192
  else
    export OAT_ZERO_COMPARATIVE_TASK=countdown
    export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
    export OAT_ZERO_MAX_TRAIN=384
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
    export OAT_ZERO_RESUME_STEPS=384
    export OAT_ZERO_RESUME_FROM=384
  fi
}

submit_fresh_cohort() {
  local task="$1" prefix="$2" arm="$3"
  configure_task "$task" "$prefix"
  export OAT_ZERO_TRAIN_SEEDS=43,44,45
  export OAT_ZERO_ONLY_ARMS="$arm"
  export OAT_ZERO_APPEND_MANIFEST=0
  unset OAT_ZERO_INITIAL_RESUME_DIR OAT_ZERO_INITIAL_RESUME_TAG
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

submit_resumed_countdown_seed() {
  local seed="$1" attempt="$2" tag="$3" append="$4"
  configure_task countdown "$E28_COUNTDOWN_PREFIX"
  export OAT_ZERO_TRAIN_SEEDS="$seed"
  export OAT_ZERO_ONLY_ARMS=grpo
  export OAT_ZERO_APPEND_MANIFEST="$append"
  export OAT_ZERO_INITIAL_RESUME_DIR="$ROOT_DIR/var/data/xdr_qwen25_3b_instruct_grpo_cde28_freeform_drgrpo_3b_v1_grpo_s${seed}/${attempt}/checkpoints"
  export OAT_ZERO_INITIAL_RESUME_TAG="$tag"
  if [[ ! -d "$OAT_ZERO_INITIAL_RESUME_DIR/$OAT_ZERO_INITIAL_RESUME_TAG" ]]; then
    echo "Missing clean E28 Countdown bootstrap: $OAT_ZERO_INITIAL_RESUME_DIR/$OAT_ZERO_INITIAL_RESUME_TAG" >&2
    exit 1
  fi
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

prefixes=(
  "$E25_GRAPH_PREFIX" "$E25_COUNTDOWN_PREFIX"
  "$E28_GRAPH_PREFIX" "$E28_COUNTDOWN_PREFIX"
)

if [[ "$PHASE" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
  submit_fresh_cohort graph_coloring "$E25_GRAPH_PREFIX" maxent_dual
  submit_fresh_cohort countdown "$E25_COUNTDOWN_PREFIX" maxent_dual
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
  submit_fresh_cohort graph_coloring "$E28_GRAPH_PREFIX" grpo
  submit_resumed_countdown_seed 43 'debug_0721T18:03:35' step_00096 0
  echo "[repair] all fresh/restart configurations passed; no jobs submitted"
  exit 0
fi

if [[ "$PHASE" == stage ]]; then
  for prefix in "${prefixes[@]}"; do
    manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
    if [[ -e "$manifest" ]]; then
      echo "Repair manifest already exists; refusing duplicate cohort: $manifest" >&2
      exit 1
    fi
  done
  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
  submit_fresh_cohort graph_coloring "$E25_GRAPH_PREFIX" maxent_dual
  submit_fresh_cohort countdown "$E25_COUNTDOWN_PREFIX" maxent_dual
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
  submit_fresh_cohort graph_coloring "$E28_GRAPH_PREFIX" grpo
  submit_resumed_countdown_seed 43 'debug_0721T18:03:35' step_00096 0
  submit_resumed_countdown_seed 44 'debug_0721T18:03:34' step_00096 1
  submit_resumed_countdown_seed 45 'debug_0721T23:20:55' step_00864 1
  echo "[repair] staged twelve held jobs; run release-staged after audit"
fi

job_ids=()
for spec in \
  "$E25_GRAPH_PREFIX|maxent_dual|fresh" \
  "$E25_COUNTDOWN_PREFIX|maxent_dual|fresh" \
  "$E28_GRAPH_PREFIX|grpo|fresh" \
  "$E28_COUNTDOWN_PREFIX|grpo|resumed"; do
  IFS='|' read -r prefix arm start_kind <<< "$spec"
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  if [[ ! -f "$manifest" ]]; then
    echo "Missing staged repair manifest: $manifest" >&2
    exit 1
  fi
  mapfile -t task_jobs < <(
    awk -F '\t' -v arm="$arm" \
      'NR > 1 && $1 == arm && $2 ~ /^(43|44|45)$/ && $3 ~ /^[0-9]+$/ {print $3}' \
      "$manifest"
  )
  if (( ${#task_jobs[@]} != 3 )); then
    echo "Repair cohort $prefix incomplete (${#task_jobs[@]}/3); jobs remain held" >&2
    exit 1
  fi
  for job_id in "${task_jobs[@]}"; do
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=mltheory' \
      'Account=mltheory' 'ReqNodeList=node302' \
      'TresPerNode=gres/gpu:a100:1' 'mem=96G' \
      'OAT_ZERO_MODEL=qwen2.5-3b-instruct' \
      "OAT_ZERO_VARIANT=${arm}" \
      "RUN_STAMP=${prefix}_${arm}_s" \
      "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
      'OAT_ZERO_AUTO_RESUME=1' \
      'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0'; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "Repair held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
    if [[ "$start_kind" == fresh && "$job_record" == *'OAT_ZERO_INITIAL_RESUME_DIR='* ]]; then
      echo "Fresh repair job $job_id unexpectedly carries an external resume path" >&2
      exit 1
    fi
    job_ids+=("$job_id")
  done
done

if (( ${#job_ids[@]} != 12 )); then
  echo "Repair cohort incomplete (${#job_ids[@]}/12); jobs remain held" >&2
  exit 1
fi
if (( $(printf '%s\n' "${job_ids[@]}" | sort -u | wc -l) != 12 )); then
  echo "Repair manifests contain duplicate job IDs; jobs remain held" >&2
  exit 1
fi

for seed in 43 44 45; do
  manifest="$ROOT_DIR/var/artifacts/${E28_COUNTDOWN_PREFIX}_comparative_jobs.tsv"
  job_id="$(awk -F '\t' -v seed="$seed" '$1 == "grpo" && $2 == seed {print $3}' "$manifest")"
  job_record="$(scontrol show job "$job_id" -o)"
  expected_tag=step_00096
  [[ "$seed" == 45 ]] && expected_tag=step_00864
  for required in \
    "OAT_ZERO_INITIAL_RESUME_DIR=${ROOT_DIR}/var/data/xdr_qwen25_3b_instruct_grpo_cde28_freeform_drgrpo_3b_v1_grpo_s${seed}/" \
    "OAT_ZERO_INITIAL_RESUME_TAG=${expected_tag}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E28 Countdown repair audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
done

if [[ "$PHASE" == stage ]]; then
  echo "[repair] held-job audit passed: ${job_ids[*]}"
  exit 0
fi

scontrol release "${job_ids[@]}"
echo "[repair] released twelve clean 3B replacement jobs: ${job_ids[*]}"
