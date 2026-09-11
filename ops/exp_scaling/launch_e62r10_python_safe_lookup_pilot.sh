#!/usr/bin/env bash
# Configure or atomically submit E62-R10's same-seed Python mechanism pilot.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|pilot) ;;
  *)
    echo "Usage: $0 {config|pilot}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e62r10_python_safe_lookup_counterfactual_05b.md"
AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e62r10_python_safe_lookup_pilot.py"
PLOTTER="$ROOT_DIR/ops/exp_scaling/plot_e62r10_python_safe_lookup_pilot.py"
WATCHER="$ROOT_DIR/ops/exp_scaling/watch_e62r10_python_safe_lookup_pilot.sh"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/python_factor_modebench_v1"
PREFIX=pye62r10_safe_lookup_05b_1ep_pilot
CONTROL=verified_first_bootstrap_local_canonical
TREATMENT=verified_counterfactual_canonical
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
IDENTITY="$ROOT_DIR/var/artifacts/e62r10_safe_lookup_python_pilot_identity.json"
AUDIT_OUT="$ROOT_DIR/var/artifacts/e62r10_python_pilot_audit_latest.json"

for required in \
  "$PROTOCOL" "$AUDITOR" "$PLOTTER" "$WATCHER" \
  "$MODEL_ROOT/config.json" "$DATA_ROOT/identity.json" \
  "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E62 Python pilot prerequisite: $required" >&2
    exit 1
  fi
done
if [[ "$phase" == pilot ]]; then
  for fresh in "$MANIFEST" "$IDENTITY"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E62 pilot artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
fi

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
  local staging ops_input ops_staging
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e62r10_safe_lookup_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E62-R10 source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e62r10-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  for file in \
    repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
    submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$ops_input/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e62r10_safe_lookup_ops_${EXECUTION_HASH}/ops"
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
    echo "E62-R10 execution snapshot hash mismatch" >&2
    exit 1
  fi
}

write_identity() {
  "$PYTHON_BIN" - \
    "$IDENTITY" "$PROTOCOL" "$0" "$AUDITOR" "$PLOTTER" "$WATCHER" \
    "$MANIFEST" "$SOURCE_HASH" "$EXECUTION_HASH" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[7])
with manifest.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
jobs = {
    row["arm"]: int(row["job_id"])
    for row in rows
    if row.get("arm") in {
        "verified_first_bootstrap_local_canonical",
        "verified_counterfactual_canonical",
    }
}
payload = {
    "schema": "e62r10_safe_lookup_python_pilot_v1",
    "protocol_sha256": digest(sys.argv[2]),
    "launcher_sha256": digest(sys.argv[3]),
    "auditor_sha256": digest(sys.argv[4]),
    "plotter_sha256": digest(sys.argv[5]),
    "watcher_sha256": digest(sys.argv[6]),
    "manifest_sha256": digest(sys.argv[7]),
    "manifest_relative_path": str(manifest.relative_to(path.parents[2])),
    "source_hash": sys.argv[8],
    "execution_surface_hash": sys.argv[9],
    "jobs": jobs,
    "attempt_selection": "exact_manifest_job_ids",
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "domain": "python_factor",
    "arms": [
        "verified_first_bootstrap_local_canonical",
        "verified_counterfactual_canonical",
    ],
    "seed": 9011,
    "prompt_pool": 384,
    "eval_pool": 128,
    "max_updates": 384,
    "checkpoint_interval_updates": 96,
    "num_neutral_samples": 16,
    "optimizer_layout": {
        "learner_world_size": 1,
        "logical_batch_size": 16,
        "physical_microbatch_per_device": 4,
        "gradient_accumulation_steps": 4,
    },
    "proposal_compute": {
        "primary_actuator": "prompt_local_verified_output_lookup",
        "transform_admission": "independent_validator_roundtrip",
        "transform_rows_sent_to_ppo": 0,
        "groups_per_eligible_prompt_max": 3,
        "samples_per_group": 16,
        "fallback_prompt": "untouched_original_task",
        "sampling_temperature_schedule": [1.0, 1.2, 1.4],
        "anchor_usage": "transform_source_and_known_outcome_filter",
        "request_rng": "explicit_disjoint_neutral_and_proposal_streams",
        "conditioned_rows_sent_to_ppo": 0,
    },
    "coefficient_projection": None,
    "information_firewall": {
        "gold_support_feedback": False,
        "evaluation_feedback": False,
        "desired_entropy": None,
        "desired_mode_count": None,
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

if [[ "$phase" == config ]]; then
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  EXECUTION_HASH=config-only-current-tree
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
else
  freeze_execution
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export RUN_STAMP_PREFIX="$PREFIX"
export OAT_ZERO_COMPARATIVE_TASK=python_factor
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=9011
export OAT_ZERO_ONLY_ARMS="$CONTROL,$TREATMENT"
export OAT_ZERO_XDR_TAUS=""

for flag in \
  TOKEN_ENTROPY SEED XDR_ADAPT XDR_TAU_CONTROL XDR_SAC_DUAL \
  MAXENT MAXENT_CONTROL MAXENT_DUAL MAXENT_LENGTH_DUAL DIAYN \
  OUTCOME_COLLISION OUTCOME_COLLISION_OUTSIDE_CENTERING \
  SEMANTIC_SHANNON SEMANTIC_SHANNON_ADVANTAGE \
  QUALITY_GATED_SEMANTIC_NOVELTY \
  SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON \
  SIGNAL_FIRST_SEMANTIC_BALANCE ONLINE_CANONICAL_MAXENT \
  ONLINE_CANONICAL_HAARNOJA ONLINE_CANONICAL_POLICY_ENTROPY \
  MAXENT_INVERSE MAXENT_INVERSE_CANONICAL \
  MAXENT_INVERSE_CANONICAL_REPLAY OPEN_SET_SPLIT_CANONICAL \
  VERIFIED_FIRST_SPLIT_CANONICAL \
  VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL \
  VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL \
  VERIFIED_COUNTERFACTUAL_CANONICAL; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL_ARM=1
export OAT_ZERO_INCLUDE_VERIFIED_COUNTERFACTUAL_CANONICAL_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS=64
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS=256
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=1
export OAT_ZERO_NUM_PROMPT_EPOCH=1
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1
export OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1
export OAT_ZERO_VLLM_SLEEP_LEVEL=1
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
export OAT_ZERO_MAX_MODEL_LEN=768
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=620100
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
export OAT_ZERO_SYNC_PARAMS_EVERY=1
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
export OAT_ZERO_SAVE_STEPS=96
export OAT_ZERO_SAVE_FROM=96
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_RESUME_STEPS=96
export OAT_ZERO_RESUME_FROM=96
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=0-12:00:00
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E62_PYTHON_NODELIST:-node020,node022,node023,node024,node026}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E62_PYTHON_GRES:-gpu:rtx_3090:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E62_PYTHON_PARTITION:-lowprio}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E62_PYTHON_ACCOUNT:-allcs}"

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e62] Python pilot configuration passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
declare -a job_ids=()
released=0
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" ]]; then
    for job_id in "${job_ids[@]}"; do
      if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        scancel "$job_id" 2>/dev/null || true
      fi
    done
  fi
  exit "$status"
}
trap cleanup_held EXIT

"$OPS_ROOT/submit_countdown_comparative.sh"
mapfile -t job_ids < <(
  awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#job_ids[@]}" -ne 2 ]]; then
  echo "E62 Python pilot has ${#job_ids[@]} jobs; expected two" >&2
  exit 1
fi
write_identity
for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=1' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=1' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=64' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout' \
    'OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1' \
    'OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0' \
    'OAT_ZERO_SAVE_STEPS=96' \
    'OAT_ZERO_RESUME_STEPS=96' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_TRAIN_BATCH_SIZE=16' \
    'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4' \
    'OAT_ZERO_ROLLOUT_BATCH_SIZE=1' \
    'OAT_ZERO_N_GPU=1' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E62 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
done
treatment_job="$(awk -F $'\t' -v arm="$TREATMENT" '$1 == arm {print $3}' "$MANIFEST")"
treatment_record="$(scontrol show job "$treatment_job" -o)"
for required in \
  'OAT_ZERO_VARIANT=verified_counterfactual_canonical' \
  'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1' \
  'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS=256' \
  'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3' \
  'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0'; do
  if [[ "$treatment_record" != *"$required"* ]]; then
    echo "E62 treatment audit failed: missing $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" "$AUDITOR" --out "$AUDIT_OUT"
for job_id in "${job_ids[@]}"; do
  scontrol update "JobId=$job_id" Requeue=1
  scontrol release "$job_id"
done
released=1
trap - EXIT
echo "[e62r10] released Python mechanism pilot jobs ${job_ids[*]}"
echo "[e62r10] identity=$IDENTITY"
