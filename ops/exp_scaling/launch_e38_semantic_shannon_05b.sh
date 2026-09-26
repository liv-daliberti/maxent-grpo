#!/usr/bin/env bash
# Launch the treatment-only E38 semantic-Shannon extension to E37.
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

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e38_semantic_shannon_05b.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
GRAPH_PREFIX=gce38_semantic_shannon_05b_v1
COUNTDOWN_PREFIX=cde38_semantic_shannon_05b_v1
IDENTITY="$ROOT_DIR/var/artifacts/e38_semantic_shannon_05b_v1_identity.json"
E37_IDENTITY="$ROOT_DIR/var/artifacts/e37_outcome_collision_05b_v1_identity.json"
E37_GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/gce37_outcome_collision_05b_v1_comparative_jobs.tsv"
E37_COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/cde37_outcome_collision_05b_v1_comparative_jobs.tsv"
EXPECTED_JOBS_PER_TASK=3

for required in \
  "$PYTHON_BIN" \
  "$PROTOCOL" \
  "$E37_IDENTITY" \
  "$E37_GRAPH_MANIFEST" \
  "$E37_COUNTDOWN_MANIFEST" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E38 prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E38 protocol is not frozen" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/src/oat_drgrpo/semantic_shannon.py" ]] || \
   ! grep -q 'semantic_shannon_coef' "$ROOT_DIR/src/oat_drgrpo/args.py" || \
   ! grep -q 'semantic_shannon_tracker_state' \
     "$ROOT_DIR/src/oat_drgrpo/learner/run.py"; then
  echo "Current source lacks the checkpointed E38 semantic-Shannon mechanism" >&2
  exit 1
fi
"$PYTHON_BIN" - "$E37_IDENTITY" "$E37_GRAPH_MANIFEST" \
  "$E37_COUNTDOWN_MANIFEST" <<'PY'
import csv
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_evaluation = {
    "draws": 4,
    "k": 8,
    "pass_at_1": "greedy",
    "prompt": "neutral_qwen_boxed",
    "seeds": [370100, 370101, 370102, 370103],
    "step_zero_pairing": "exact",
}
if (
    identity.get("schema") != "e37_outcome_collision_05b_v1"
    or identity.get("arms") != ["grpo", "outcome_collision"]
    or identity.get("seeds") != [43, 44, 45]
    or identity.get("num_samples") != 16
    or identity.get("prompt_epochs") != 10
    or identity.get("evaluation") != expected_evaluation
):
    raise SystemExit("E37 identity does not match the frozen E38 comparator")

expected_rows = {
    (arm, str(seed))
    for seed in (43, 44, 45)
    for arm in ("grpo", "outcome_collision")
}
for manifest_name in sys.argv[2:]:
    with pathlib.Path(manifest_name).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    observed = {(row["arm"], row["seed"]) for row in rows}
    if len(rows) != 6 or observed != expected_rows:
        raise SystemExit(f"E37 comparator manifest is not the frozen six-run cohort: {manifest_name}")
PY

for manifest in \
  "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
  "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"; do
  if [[ "$phase" == full && -e "$manifest" ]]; then
    echo "Fresh E38 prefix required; manifest already exists: $manifest" >&2
    exit 1
  fi
done

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | cut -d' ' -f1
  )
}

freeze_execution() {
  local staging
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e38_semantic_shannon_05b_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E38 source snapshot hash mismatch" >&2
    exit 1
  fi

  local ops_staging ops_input
  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e38-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  cp "$ROOT_DIR/ops/repo_env.sh" "$ops_input/repo_env.sh"
  cp "$ROOT_DIR/ops/run_experiment.sh" "$ops_input/run_experiment.sh"
  cp "$ROOT_DIR/ops/train.sh" "$ops_input/train.sh"
  cp "$ROOT_DIR/ops/resolve_eval_cadence.py" "$ops_input/resolve_eval_cadence.py"
  cp "$ROOT_DIR/ops/submit_countdown_comparative.sh" "$ops_input/submit_countdown_comparative.sh"
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e38_semantic_shannon_05b_ops_${EXECUTION_HASH}/ops"
  if [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
    mkdir -p "$(dirname "$OPS_ROOT")"
    ops_staging="$(mktemp -d "$(dirname "$OPS_ROOT")/.ops.XXXXXX")"
    mv "$ops_input" "$ops_staging/ops"
    mv "$ops_staging/ops" "$OPS_ROOT"
    rmdir "$ops_staging"
  else
    find "$ops_input" -type f -delete
    rmdir "$ops_input/slurm" "$ops_input"
  fi
  if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
    echo "E38 execution snapshot hash mismatch" >&2
    exit 1
  fi
}

if [[ "$phase" == full ]]; then
  freeze_execution
else
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  EXECUTION_HASH=config-only-current-tree
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=semantic_shannon
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
export OAT_ZERO_INCLUDE_DIAYN_ARM=0
export OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM=0
export OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM=1

export OAT_ZERO_OUTCOME_COLLISION_COEF=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0
export OAT_ZERO_MAXENT_OBJECTIVE=sequence
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_SEED_ENTROPY_ALPHA=0
export OAT_ZERO_DIAYN_NUM_OPTIONS=0
export OAT_ZERO_DIAYN_MI_BETA=0
export OAT_ZERO_DIAYN_MI_LEAVE_ONE_OUT=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=10
export OAT_ZERO_NUM_PROMPT_EPOCH=10
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
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=370100
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1

export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0
export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0
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

# Match E37's optimizer-resumable, one-checkpoint-per-pass lifecycle.
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E38_TRAIN_NODELIST:-node105,node202,node203,node204}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E38_TRAIN_GRES:-gpu:a5000:1}"
export OAT_ZERO_TRAIN_CPUS_PER_TASK="${OAT_ZERO_E38_TRAIN_CPUS_PER_TASK:-4}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E38_TRAIN_PARTITION:-all}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E38_TRAIN_ACCOUNT:-mltheory}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E38_TRAIN_MEMORY:-32G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E38_TRAIN_TIME_LIMIT:-24:00:00}"

write_identity() {
  local protocol_hash launcher_hash e37_identity_hash e37_graph_manifest_hash
  local e37_countdown_manifest_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  e37_identity_hash="$(sha256sum "$E37_IDENTITY" | cut -d' ' -f1)"
  e37_graph_manifest_hash="$(sha256sum "$E37_GRAPH_MANIFEST" | cut -d' ' -f1)"
  e37_countdown_manifest_hash="$(sha256sum "$E37_COUNTDOWN_MANIFEST" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$IDENTITY" "$protocol_hash" "$launcher_hash" \
    "$SOURCE_HASH" "$EXECUTION_HASH" "$e37_identity_hash" \
    "$e37_graph_manifest_hash" "$e37_countdown_manifest_hash" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e38_semantic_shannon_05b_v1",
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "e37_comparators": {
        "identity": "e37_outcome_collision_05b_v1_identity.json",
        "identity_sha256": sys.argv[6],
        "graph_prefix": "gce37_outcome_collision_05b_v1",
        "graph_manifest_sha256": sys.argv[7],
        "countdown_prefix": "cde37_outcome_collision_05b_v1",
        "countdown_manifest_sha256": sys.argv[8],
        "arms": ["grpo", "outcome_collision"],
    },
    "model": "Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
    "tasks": {
        "graph_coloring": {"prompt_pool": 192},
        "countdown": {"prompt_pool": 384},
    },
    "new_arms": ["semantic_shannon"],
    "semantic_shannon": {
        "coefficient": 0.10,
        "surprisal_clip": 5.0,
        "pseudocount": 1.0,
        "group_leave_one_out": True,
        "predictive_support": "explicit_outcomes_plus_one_unseen_bucket",
        "prompt_identity": "sha256_unpadded_prompt_token_ids",
        "history_update": "after_group_scoring",
        "invalid_outcome": "shared_INVALID",
    },
    "seeds": [43, 44, 45],
    "num_samples": 16,
    "prompt_epochs": 10,
    "evaluation": {
        "prompt": "neutral_qwen_boxed",
        "k": 8,
        "draws": 4,
        "seeds": [370100, 370101, 370102, 370103],
        "pass_at_1": "greedy",
        "step_zero_pairing_to_e37": "exact",
    },
    "resume": {
        "semantic_history_checkpointed": True,
        "actor_sync_before_eval_or_rollout": True,
        "checkpoints_per_prompt_epoch": 1,
        "keep": 2,
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

submit_task() {
  local task="$1"
  if [[ "$task" == graph_coloring ]]; then
    export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
    export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
    export OAT_ZERO_MAX_TRAIN=192
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
    export OAT_ZERO_SAVE_STEPS=192
    export OAT_ZERO_SAVE_FROM=192
    export OAT_ZERO_RESUME_STEPS=192
    export OAT_ZERO_RESUME_FROM=192
  else
    export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
    export OAT_ZERO_COMPARATIVE_TASK=countdown
    export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
    export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
    export OAT_ZERO_MAX_TRAIN=384
    export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
    export OAT_ZERO_SAVE_STEPS=384
    export OAT_ZERO_SAVE_FROM=384
    export OAT_ZERO_RESUME_STEPS=384
    export OAT_ZERO_RESUME_FROM=384
  fi
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_task graph_coloring
  submit_task countdown
  echo "[e38] both matched 0.5B configurations passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1

# Keep a failed held submission recoverable and prevent a partial E38 cohort
# from consuming scheduler resources. Exact-prefix artifacts are moved aside
# with a failure suffix so a reviewed retry can still use the frozen prefix.
COHORT_RELEASED=0
cleanup_partial_cohort() {
  local status="$?"
  trap - EXIT
  if [[ "$COHORT_RELEASED" != "1" ]]; then
    local -a cleanup_ids=()
    local -a manifest_ids=()
    local manifest failure_stamp artifact
    for manifest in \
      "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
      "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"; do
      if [[ -f "$manifest" ]]; then
        mapfile -t manifest_ids < <(
          awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
        )
        cleanup_ids+=("${manifest_ids[@]}")
      fi
    done
    if [[ "${#cleanup_ids[@]}" -gt 0 ]]; then
      scancel "${cleanup_ids[@]}" 2>/dev/null || true
      echo "[e38] cancelled incomplete held cohort: ${cleanup_ids[*]}" >&2
    fi
    failure_stamp="$(date +%Y%m%d_%H%M%S)"
    for artifact in \
      "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
      "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv" \
      "$IDENTITY"; do
      if [[ -e "$artifact" ]]; then
        mv "$artifact" "${artifact}.failed_${failure_stamp}"
      fi
    done
  fi
  exit "$status"
}
trap cleanup_partial_cohort EXIT

write_identity
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
submit_task graph_coloring
submit_task countdown

job_ids=()
for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest")
  if [[ "${#task_jobs[@]}" -ne "$EXPECTED_JOBS_PER_TASK" ]]; then
    mapfile -t submitted_jobs < <(
      awk -F '\t' 'FNR > 1 && $3 ~ /^[0-9]+$/ {print $3}' \
        "$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv" \
        "$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
    )
    scancel "${submitted_jobs[@]}" 2>/dev/null || true
    echo "E38 ${prefix} has ${#task_jobs[@]} jobs; expected $EXPECTED_JOBS_PER_TASK" >&2
    exit 1
  fi
  job_ids+=("${task_jobs[@]}")
done

for job_id in "${job_ids[@]}"; do
  if ! scontrol update JobId="$job_id" Partition="$OAT_ZERO_TRAIN_PARTITION"; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "E38 could not normalize held job $job_id to ${OAT_ZERO_TRAIN_PARTITION}" >&2
    exit 1
  fi
done

for spec in "$GRAPH_PREFIX|192|48" "$COUNTDOWN_PREFIX|384|96"; do
  IFS='|' read -r prefix resume_steps eval_interval <<< "$spec"
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest")
  for job_id in "${task_jobs[@]}"; do
    arm="$(awk -F '\t' -v job_id="$job_id" '$3 == job_id {print $1}' "$manifest")"
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' 'Reason=JobHeldUser' \
      'OAT_ZERO_VARIANT=semantic_shannon' \
      'OAT_ZERO_OUTCOME_COLLISION_COEF=0.0' \
      'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10' \
      'OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0' \
      'OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0' \
      'OAT_ZERO_MAX_PROMPT_EPOCHS=10' \
      'OAT_ZERO_NUM_PROMPT_EPOCH=10' \
      'OAT_ZERO_NUM_SAMPLES=16' \
      'OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed' \
      'OAT_ZERO_DIAYN_NUM_OPTIONS=0' \
      'OAT_ZERO_DIAYN_MI_BETA=0.0' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_SEED=370100' \
      "OAT_ZERO_EVAL_PROMPT_INTERVAL=${eval_interval}" \
      "OAT_ZERO_RESUME_STEPS=${resume_steps}" \
      "OAT_ZERO_RESUME_FROM=${resume_steps}" \
      'OAT_ZERO_MAX_RESUME_NUM=2' \
      'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0' \
      'OAT_ZERO_EXPORT_STEPS=0' \
      'OAT_ZERO_MAX_EXPORT_NUM=1' \
      "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
      "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
      "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
      if [[ "$job_record" != *"$required"* ]]; then
        scancel "${job_ids[@]}" 2>/dev/null || true
        echo "E38 held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
    if [[ "$arm" != semantic_shannon ]]; then
      scancel "${job_ids[@]}" 2>/dev/null || true
      echo "E38 manifest has unexpected arm for $job_id: $arm" >&2
      exit 1
    fi
  done
done

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  if [[ "$job_record" != *"Partition=${OAT_ZERO_TRAIN_PARTITION}"* ]] || \
     [[ "$job_record" != *"--nodelist=${OAT_ZERO_TRAIN_NODELIST}"* ]] || \
     [[ "$job_record" != *"gres/gpu:a5000=1"* ]] || \
     [[ "$job_record" != *"NumCPUs=${OAT_ZERO_TRAIN_CPUS_PER_TASK}"* ]] || \
     [[ "$job_record" != *"MinMemoryNode=${OAT_ZERO_TRAIN_MEMORY}"* ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "E38 held job $job_id does not attest the frozen placement" >&2
    exit 1
  fi
done

scontrol release "${job_ids[@]}"
COHORT_RELEASED=1
trap - EXIT
echo "[e38] released ${#job_ids[@]} semantic-Shannon 0.5B jobs: ${job_ids[*]}"
