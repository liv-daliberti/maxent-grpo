#!/usr/bin/env bash
# Launch E50's matched executable Python-factor extension.
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
BASE_PROTOCOL="$ROOT_DIR/paper/preregistration/e50_python_factor_extension_20260724.md"
PROTOCOL="$ROOT_DIR/paper/preregistration/e50_python_factor_non_mltheory_placement_amendment_20260724.md"
PARENT_IDENTITY="$ROOT_DIR/var/artifacts/e50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1_identity.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/python_factor_modebench_v1"
DATA_IDENTITY="$DATA_ROOT/identity.json"
PREFIX=pye50_uncapped_normalized_canonical_haarnoja_05b_50ep_v3_allcs
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
IDENTITY="$ROOT_DIR/var/artifacts/${PREFIX}_identity.json"
EXPECTED_JOBS=6
PROMPT_POOL=384
EVAL_INTERVAL=96
PROMPT_EPOCHS=50

for required in \
  "$PYTHON_BIN" \
  "$BASE_PROTOCOL" \
  "$PROTOCOL" \
  "$PARENT_IDENTITY" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors" \
  "$DATA_IDENTITY" \
  "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E50 Python prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$BASE_PROTOCOL"; then
  echo "E50 Python base extension protocol is not frozen" >&2
  exit 1
fi
if ! grep -q '^\*\*Status: FROZEN BEFORE RETRY' "$PROTOCOL"; then
  echo "E50 Python placement amendment is not frozen" >&2
  exit 1
fi
for contract in \
  'src/oat_drgrpo/python_modebench.py:PYTHON_FACTOR_VERSION = "factor-v1"' \
  'src/oat_drgrpo/python_modebench_process.py:validate_python_factor_function_external' \
  'src/oat_drgrpo/python_modebench_worker.py:execute_python_factor_candidate' \
  'src/oat_drgrpo/math_grader.py:PYTHON_FACTOR_VERIFIER' \
  'src/oat_drgrpo/online_canonical_controller.py:verified_bank_entropy_over_log_support_v1' \
  'src/oat_drgrpo/learner/run.py:online_canonical_alpha_controller_state' \
  'ops/submit_countdown_comparative.sh:python_factor)'; do
  path="${contract%%:*}"
  needle="${contract#*:}"
  if ! grep -q "$needle" "$ROOT_DIR/$path"; then
    echo "Current source lacks E50 Python contract: $path :: $needle" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$DATA_IDENTITY" "$PARENT_IDENTITY" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
parent = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_data = {
    "schema": "python_factor_modebench_v1",
    "seed": 5100,
    "train_rows": 384,
    "eval_rows": 128,
    "train_rows_sha256": "bcfa9accfa3b5c7dd312c85683157af070e7f2741385e9b5a882fd57da037cf7",
    "eval_rows_sha256": "be0f621c5a0ae84ca45ef4ab866ae12f4c64472eeed1a0ae4666918ab794d183",
    "case_count": 4,
    "max_value": 96,
    "minimum_exact_mode_count": 16,
    "maximum_exact_mode_count": 3600,
    "external_verifier": "isolated_python_jsonl_worker",
    "support": "finite_exact",
}
for key, value in expected_data.items():
    if data.get(key) != value:
        raise SystemExit(
            f"Python dataset identity drift for {key}: "
            f"{data.get(key)!r} != {value!r}"
        )
if parent.get("schema") != "e50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1":
    raise SystemExit("parent E50 identity schema drift")
if parent.get("seeds") != [43, 44, 45] or parent.get("prompt_epochs") != 50:
    raise SystemExit("parent E50 matched-cohort contract drift")
PY

if [[ "$phase" == full && -e "$MANIFEST" ]]; then
  echo "Fresh E50 Python prefix required; manifest exists: $MANIFEST" >&2
  exit 1
fi
if [[ "$phase" == full && -e "$IDENTITY" ]]; then
  echo "Fresh E50 Python identity required; file exists: $IDENTITY" >&2
  exit 1
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
  local staging ops_staging ops_input
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e50_python_factor_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E50 Python source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e50-python-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  for file in repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$ops_input/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e50_python_factor_ops_${EXECUTION_HASH}/ops"
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
    echo "E50 Python execution snapshot hash mismatch" >&2
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
export OAT_ZERO_COMPARATIVE_TASK=python_factor
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja
export OAT_ZERO_XDR_TAUS=""
export RUN_STAMP_PREFIX="$PREFIX"

for flag in \
  TOKEN_ENTROPY SEED XDR_ADAPT XDR_TAU_CONTROL XDR_SAC_DUAL \
  MAXENT MAXENT_CONTROL MAXENT_DUAL MAXENT_LENGTH_DUAL DIAYN \
  OUTCOME_COLLISION OUTCOME_COLLISION_OUTSIDE_CENTERING \
  SEMANTIC_SHANNON SEMANTIC_SHANNON_ADVANTAGE \
  QUALITY_GATED_SEMANTIC_NOVELTY \
  SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON \
  SIGNAL_FIRST_SEMANTIC_BALANCE ONLINE_CANONICAL_MAXENT; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_ONLINE_CANONICAL_HAARNOJA_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80
export OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=inf
export OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003
export OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90
export OAT_ZERO_OUTCOME_COLLISION_COEF=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0
export OAT_ZERO_MAXENT_OBJECTIVE=sequence
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_SEED_ENTROPY_ALPHA=0
export OAT_ZERO_DIAYN_NUM_OPTIONS=0
export OAT_ZERO_DIAYN_MI_BETA=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN="$PROMPT_POOL"
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS="$PROMPT_EPOCHS"
export OAT_ZERO_NUM_PROMPT_EPOCH="$PROMPT_EPOCHS"
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
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=440100
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_EVAL_PROMPT_INTERVAL="$EVAL_INTERVAL"
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
export OAT_ZERO_SAVE_STEPS="$PROMPT_POOL"
export OAT_ZERO_SAVE_FROM="$PROMPT_POOL"
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_RESUME_STEPS="$PROMPT_POOL"
export OAT_ZERO_RESUME_FROM="$PROMPT_POOL"
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E50_PYTHON_TRAIN_NODELIST:-node020,node024,node025,node026}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E50_PYTHON_TRAIN_GRES:-gpu:rtx_3090:1}"
export OAT_ZERO_TRAIN_CPUS_PER_TASK="${OAT_ZERO_E50_PYTHON_TRAIN_CPUS_PER_TASK:-8}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E50_PYTHON_TRAIN_PARTITION:-all}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E50_PYTHON_TRAIN_ACCOUNT:-allcs}"
export OAT_ZERO_TRAIN_MEMORY="${OAT_ZERO_E50_PYTHON_TRAIN_MEMORY:-64G}"
export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E50_PYTHON_TRAIN_TIME_LIMIT:-7-00:00:00}"

write_identity() {
  local base_protocol_hash protocol_hash launcher_hash parent_hash data_identity_hash
  base_protocol_hash="$(sha256sum "$BASE_PROTOCOL" | cut -d' ' -f1)"
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  parent_hash="$(sha256sum "$PARENT_IDENTITY" | cut -d' ' -f1)"
  data_identity_hash="$(sha256sum "$DATA_IDENTITY" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$IDENTITY" "$base_protocol_hash" "$protocol_hash" \
    "$launcher_hash" "$parent_hash" "$data_identity_hash" \
    "$SOURCE_HASH" "$EXECUTION_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e50_python_factor_extension_v3_allcs",
    "base_protocol_sha256": sys.argv[2],
    "placement_amendment_sha256": sys.argv[3],
    "launcher_sha256": sys.argv[4],
    "parent_e50_identity_sha256": sys.argv[5],
    "dataset_identity_sha256": sys.argv[6],
    "source_hash": sys.argv[7],
    "execution_surface_hash": sys.argv[8],
    "model": "Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
    "task": {
        "name": "python_factor",
        "prompt_pool": 384,
        "evaluation_prompts": 128,
        "validator": "isolated_python_factor_worker",
        "canonical_key": "executed_return_vector",
        "exact_modes_per_prompt": [16, 3600],
    },
    "arms": ["grpo", "online_canonical_haarnoja"],
    "seeds": [43, 44, 45],
    "num_samples": 16,
    "prompt_epochs": 50,
    "evaluation_interval_passes": 0.25,
    "relationship_to_parent": "postfreeze_matched_domain_extension",
    "retry_of": [
        "aborted_held_v1_pre_release",
        "aborted_held_v2_scheduler_partition_remap",
    ],
    "online_canonical_haarnoja": {
        "initial_alpha": 0.10,
        "alpha_lower_bound": 0.10,
        "alpha_upper_bound": None,
        "upper_projection": False,
        "normalized_entropy_target": 0.80,
        "alpha_lr": 0.003,
        "ema_decay": 0.90,
        "novelty_beta": 0.50,
        "pseudocount": 1.0,
        "surprisal_clip": 5.0,
    },
    "grpo": {
        "objective": "ordinary_drgrpo",
        "verified_discovery_tracking": "passive_zero_influence",
    },
    "placement": {
        "node": os.environ["OAT_ZERO_TRAIN_NODELIST"],
        "gpu_per_job": os.environ["OAT_ZERO_TRAIN_GRES"],
        "partition": os.environ["OAT_ZERO_TRAIN_PARTITION"],
        "account": os.environ["OAT_ZERO_TRAIN_ACCOUNT"],
        "cpus_per_job": int(os.environ["OAT_ZERO_TRAIN_CPUS_PER_TASK"]),
        "memory": os.environ["OAT_ZERO_TRAIN_MEMORY"],
        "time_limit": os.environ["OAT_ZERO_TRAIN_TIME_LIMIT"],
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

submit_python() {
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_python
  echo "[e50-python] matched 50-pass configuration passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
COHORT_RELEASED=0
job_ids=()
cleanup_partial_cohort() {
  local status="$?"
  trap - EXIT
  if [[ "$COHORT_RELEASED" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "[e50-python] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_partial_cohort EXIT

write_identity
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
submit_python

mapfile -t job_ids < <(
  awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#job_ids[@]}" -ne "$EXPECTED_JOBS" ]]; then
  echo "E50 Python extension has ${#job_ids[@]} jobs; expected $EXPECTED_JOBS" >&2
  exit 1
fi
"$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    (arm, str(seed))
    for arm in ("grpo", "online_canonical_haarnoja")
    for seed in (43, 44, 45)
}
observed = {(row["arm"], row["seed"]) for row in rows}
if len(rows) != 6 or observed != expected:
    raise SystemExit(f"E50 Python held manifest mismatch: {observed!r}")
PY

for job_id in "${job_ids[@]}"; do
  scontrol update \
    "JobId=${job_id}" \
    "Account=${OAT_ZERO_TRAIN_ACCOUNT}" \
    "Partition=${OAT_ZERO_TRAIN_PARTITION}" \
    QOS=none \
    "NodeList=${OAT_ZERO_TRAIN_NODELIST}"
done

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'OAT_ZERO_COMPARATIVE_TASK=python_factor' \
    "OAT_ZERO_DATA_ROOT=${DATA_ROOT}" \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=50' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=50' \
    'OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed' \
    'OAT_ZERO_TEST_SPLIT=multi_answer' \
    'OAT_ZERO_VERIFIER_VERSION=fast' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_SEED=440100' \
    'OAT_ZERO_EVAL_PROMPT_INTERVAL=96' \
    'OAT_ZERO_RESUME_STEPS=384' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E50 Python held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  if [[ "$job_record" == *'OAT_ZERO_VARIANT=grpo'* ]]; then
    for required in \
      'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0' \
      'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0' \
      'OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0'; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E50 Python Dr.GRPO audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  else
    for required in \
      'OAT_ZERO_VARIANT=online_canonical_haarnoja' \
      'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10' \
      'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50' \
      'OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80' \
      'OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10' \
      'OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=inf' \
      'OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003' \
      'OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90'; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E50 Python Haarnoja audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  fi
  if [[ "$job_record" != *"--nodelist=${OAT_ZERO_TRAIN_NODELIST}"* ]] || \
     [[ "$job_record" != *"gres/gpu:rtx_3090:1"* ]] || \
     [[ "$job_record" != *"NumCPUs=${OAT_ZERO_TRAIN_CPUS_PER_TASK}"* ]] || \
     [[ "$job_record" != *"MinMemoryNode=${OAT_ZERO_TRAIN_MEMORY}"* ]] || \
     [[ "$job_record" != *"TimeLimit=${OAT_ZERO_TRAIN_TIME_LIMIT}"* ]]; then
    echo "E50 Python held job $job_id does not attest frozen placement/time" >&2
    exit 1
  fi
  if [[ "$job_record" != *"Account=${OAT_ZERO_TRAIN_ACCOUNT}"* ]] || \
     [[ "$job_record" != *"Partition=${OAT_ZERO_TRAIN_PARTITION}"* ]]; then
    echo "E50 Python held job $job_id does not attest account/partition" >&2
    exit 1
  fi
done

scontrol release "${job_ids[@]}"
COHORT_RELEASED=1
trap - EXIT
echo "[e50-python] released ${#job_ids[@]} matched 50-pass jobs: ${job_ids[*]}"
echo "[e50-python] identity=$IDENTITY"
