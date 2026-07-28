#!/usr/bin/env bash
# Launch E45-MIR's matched Dr.GRPO/online-canonical-MaxEnt cohort.
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
PROTOCOL="$ROOT_DIR/paper/preregistration/e45_mathir_online_growing_support_05b.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/mathir_algebra_v0_probe"
DATA_IDENTITY="$DATA_ROOT/identity.json"
BOOTSTRAP_PROBE="$ROOT_DIR/var/artifacts/e45_mathir_bootstrap_probe_v1.json"
# v1 was cancelled while held: Slurm 25.11 renders the typed GRES count with
# a colon, and the original audit expected the older equals rendering.
PREFIX=mie45_ogs_mathir_05b_v2
IDENTITY="$ROOT_DIR/var/artifacts/e45_ogs_mathir_05b_v2_identity.json"
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
EXPECTED_JOBS=6

for required in \
  "$PYTHON_BIN" \
  "$PROTOCOL" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors" \
  "$DATA_IDENTITY" \
  "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json" \
  "$BOOTSTRAP_PROBE" \
  "$ROOT_DIR/src/oat_drgrpo/mathir.py"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E45-MIR prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E45-MIR protocol is not frozen" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/src/oat_drgrpo/online_canonical_bank.py" ]] || \
   ! grep -q 'validate_mathir_algebra(candidate, spec)' \
     "$ROOT_DIR/src/oat_drgrpo/math_grader.py" || \
   ! grep -q 'online_canonical_bank_state' \
     "$ROOT_DIR/src/oat_drgrpo/learner/run.py" || \
   ! grep -q 'online_canonical_advantage_applied_after_task_centering' \
     "$ROOT_DIR/src/oat_drgrpo/learner/grpo.py"; then
  echo "Current source lacks the frozen E45-MIR execution/bank mechanism" >&2
  exit 1
fi

"$PYTHON_BIN" - "$DATA_IDENTITY" "$BOOTSTRAP_PROBE" "$MODEL_ROOT/config.json" <<'PY'
import hashlib
import json
import pathlib
import sys

data_identity_path = pathlib.Path(sys.argv[1])
probe_path = pathlib.Path(sys.argv[2])
model_config_path = pathlib.Path(sys.argv[3])
data = json.loads(data_identity_path.read_text(encoding="utf-8"))
probe = json.loads(probe_path.read_text(encoding="utf-8"))
expected = {
    "schema": "mathir_algebra_linear_v0_dataset_v1",
    "train_rows": 384,
    "eval_rows": 128,
    "train_rows_sha256": "4725b97a6db30148989331a1b6735fa9a71e6381faf532260bce1ad74bd6aa09",
    "eval_rows_sha256": "24e73de2feed8aff73a67d9264621ac18254eb2fc76a8157f392d27297ccc11f",
    "families": ["ax_plus_b_eq_c", "x_over_a_plus_b_eq_c"],
    "support": "open_growing",
}
for key, value in expected.items():
    if data.get(key) != value:
        raise SystemExit(f"E45-MIR dataset identity mismatch for {key}")
data_identity_hash = hashlib.sha256(data_identity_path.read_bytes()).hexdigest()
model_config_hash = hashlib.sha256(model_config_path.read_bytes()).hexdigest()
if probe.get("schema") != "e45_mathir_bootstrap_probe_v1":
    raise SystemExit("E45-MIR bootstrap schema mismatch")
if probe.get("data_identity_sha256") != data_identity_hash:
    raise SystemExit("E45-MIR bootstrap used different data")
if probe.get("model_config_sha256") != model_config_hash:
    raise SystemExit("E45-MIR bootstrap used different model")
if probe.get("temperature") != 0.5 or probe.get("top_p") != 0.9:
    raise SystemExit("E45-MIR bootstrap sampling mismatch")
if probe.get("samples_per_row") != 16 or not probe.get("passed"):
    raise SystemExit("E45-MIR bootstrap gate did not pass")
if probe.get("total_correct", 0) <= 0:
    raise SystemExit("E45-MIR bootstrap has no task reward")
if sum(
    family.get("nonseed_correct", 0)
    for family in probe.get("family_summary", {}).values()
) != 0:
    raise SystemExit("E45-MIR bootstrap discovery target was already present")
PY

if [[ "$phase" == full && -e "$MANIFEST" ]]; then
  echo "Fresh E45-MIR prefix required; manifest exists: $MANIFEST" >&2
  exit 1
fi
if [[ "$phase" == full && -e "$IDENTITY" ]]; then
  echo "Fresh E45-MIR identity required; file exists: $IDENTITY" >&2
  exit 1
fi

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f \
      ! -path '*/__pycache__/*' \
      ! -name '*.pyc' \
      -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | cut -d' ' -f1
  )
}

freeze_execution() {
  local staging ops_staging ops_input
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e45_ogs_mathir_05b_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E45-MIR source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e45-mir-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  cp "$ROOT_DIR/ops/repo_env.sh" "$ops_input/repo_env.sh"
  cp "$ROOT_DIR/ops/run_experiment.sh" "$ops_input/run_experiment.sh"
  cp "$ROOT_DIR/ops/train.sh" "$ops_input/train.sh"
  cp "$ROOT_DIR/ops/resolve_eval_cadence.py" "$ops_input/resolve_eval_cadence.py"
  cp "$ROOT_DIR/ops/submit_countdown_comparative.sh" \
    "$ops_input/submit_countdown_comparative.sh"
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e45_ogs_mathir_05b_ops_${EXECUTION_HASH}/ops"
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
    echo "E45-MIR execution snapshot hash mismatch" >&2
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
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_maxent
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
export OAT_ZERO_INCLUDE_OUTCOME_COLLISION_OUTSIDE_CENTERING_ARM=0
export OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM=0
export OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM=0
export OAT_ZERO_INCLUDE_QUALITY_GATED_SEMANTIC_NOVELTY_ARM=0
export OAT_ZERO_INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM=0
export OAT_ZERO_INCLUDE_SIGNAL_FIRST_SEMANTIC_BALANCE_ARM=0
export OAT_ZERO_INCLUDE_ONLINE_CANONICAL_MAXENT_ARM=1

export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_OUTCOME_COLLISION_COEF=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0
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
export OAT_ZERO_MAX_TRAIN=384
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
export OAT_ZERO_PROMPT_MAX_LENGTH=224
export OAT_ZERO_GENERATE_MAX_LENGTH=64
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=64
export OAT_ZERO_MAX_MODEL_LEN=384
export OAT_ZERO_TEMPERATURE=0.5
export OAT_ZERO_TOP_P=0.9
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=16
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=0.5
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=450100
export OAT_ZERO_EVAL_BATCH_SIZE=32
export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
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

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_SAVE_STEPS=384
export OAT_ZERO_SAVE_FROM=384
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_RESUME_STEPS=384
export OAT_ZERO_RESUME_FROM=384
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_NODELIST=node302
export OAT_ZERO_TRAIN_GRES=gpu:a100:1
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_PARTITION=all
export OAT_ZERO_TRAIN_ACCOUNT=mltheory
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=12:00:00

write_identity() {
  local protocol_hash launcher_hash data_hash probe_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  data_hash="$(sha256sum "$DATA_IDENTITY" | cut -d' ' -f1)"
  probe_hash="$(sha256sum "$BOOTSTRAP_PROBE" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$IDENTITY" "$protocol_hash" "$launcher_hash" \
    "$SOURCE_HASH" "$EXECUTION_HASH" "$data_hash" "$probe_hash" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e45_ogs_mathir_05b_v2",
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "data_identity_sha256": sys.argv[6],
    "bootstrap_probe_sha256": sys.argv[7],
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "task": {
        "name": "mathir_algebra",
        "version": "linear-v0",
        "prompt_pool": 384,
        "eval_pool": 128,
        "support": "open_growing",
        "public_seed_per_family": 1,
        "validator": "executed_state_transformations",
    },
    "arms": ["grpo", "online_canonical_maxent"],
    "online_canonical_maxent": {
        "entropy_alpha": 0.10,
        "novelty_beta": 0.50,
        "pseudocount": 1.0,
        "surprisal_clip": 5.0,
        "key_mode": "modebench_outcome",
        "admission": "validator_bound_fail_closed",
        "group_snapshot": True,
        "update_after_group": True,
        "advantage_placement": "after_task_drgrpo_centering",
        "bank_checkpointed": True,
        "gold_outcome_catalogue": False,
        "haarnoja_alpha_control": False,
    },
    "seeds": [43, 44, 45],
    "num_samples": 16,
    "prompt_epochs": 10,
    "sampling": {"temperature": 0.5, "top_p": 0.9},
    "placement": {"node": "node302", "gpu": "a100", "gpus_per_job": 1},
    "evaluation": {
        "prompt": "neutral_qwen_boxed",
        "k": 16,
        "draws": 4,
        "seeds": [450100, 450101, 450102, 450103],
        "open_support_coverage_denominator": None,
        "primary": "distinct_nonseed_correct",
        "step_zero_pairing": "exact",
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

submit_cohort() {
  export RUN_STAMP_PREFIX="$PREFIX"
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_cohort
  echo "[e45-mir] matched configuration passed; no jobs submitted"
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
    echo "[e45-mir] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_partial_cohort EXIT

write_identity
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
submit_cohort

mapfile -t job_ids < <(
  awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#job_ids[@]}" -ne "$EXPECTED_JOBS" ]]; then
  echo "E45-MIR cohort has ${#job_ids[@]} jobs; expected $EXPECTED_JOBS" >&2
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
    for arm in ("grpo", "online_canonical_maxent")
    for seed in (43, 44, 45)
}
observed = {(row["arm"], row["seed"]) for row in rows}
if len(rows) != 6 or observed != expected:
    raise SystemExit(
        f"E45-MIR held manifest mismatch: rows={len(rows)} "
        f"observed={sorted(observed)}"
    )
PY

for job_id in "${job_ids[@]}"; do
  arm="$(awk -F '\t' -v job_id="$job_id" '$3 == job_id {print $1}' "$MANIFEST")"
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=10' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=10' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed' \
    'OAT_ZERO_TEST_SPLIT=multi_answer' \
    'OAT_ZERO_VERIFIER_VERSION=fast' \
    'OAT_ZERO_TEMPERATURE=0.5' \
    'OAT_ZERO_TOP_P=0.9' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=16' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_SEED=450100' \
    'OAT_ZERO_EVAL_PROMPT_INTERVAL=96' \
    'OAT_ZERO_RESUME_STEPS=384' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E45-MIR held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  if [[ "$arm" == online_canonical_maxent ]]; then
    arm_requirements=(
      'OAT_ZERO_VARIANT=online_canonical_maxent'
      'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10'
      'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50'
      'OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0'
      'OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0'
      'OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome'
    )
  elif [[ "$arm" == grpo ]]; then
    arm_requirements=(
      'OAT_ZERO_VARIANT=grpo'
      'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0'
      'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0'
      'OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome'
    )
  else
    echo "E45-MIR manifest has unexpected arm for $job_id: $arm" >&2
    exit 1
  fi
  for required in "${arm_requirements[@]}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E45-MIR arm audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  if [[ "$job_record" != *"--nodelist=node302"* ]] || \
     [[ "$job_record" != *"gres/gpu:a100:1"* ]] || \
     [[ "$job_record" != *"NumCPUs=8"* ]] || \
     [[ "$job_record" != *"MinMemoryNode=64G"* ]]; then
    echo "E45-MIR held job $job_id does not attest one-A100 node302 placement" >&2
    exit 1
  fi
done

scontrol release "${job_ids[@]}"
COHORT_RELEASED=1
trap - EXIT
echo "[e45-mir] released ${#job_ids[@]} matched jobs: ${job_ids[*]}"
echo "[e45-mir] identity=$IDENTITY"
