#!/usr/bin/env bash
# Replace Gate 2's pre-optimizer-invalid free-form-MATH endpoint job once.
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
ORIGINAL_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_identity.json"
AMENDMENT="$ROOT_DIR/paper/preregistration/e69_gate2_math_endpoint_startup_repair_20260728.md"
MATH_DATA="$ROOT_DIR/var/data/math12k_384_route_dev128_v1"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
INVALID_JOB=30159730
PREFIX=mde69_gate2_math_endpoint_repair
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
REPAIR_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_math_endpoint_repair_identity.json"
INVALID_LOG="$ROOT_DIR/var/artifacts/logs/xdr_train-${INVALID_JOB}.out"

for required in \
  "$ORIGINAL_IDENTITY" "$AMENDMENT" \
  "$MATH_DATA/train/dataset_dict.json" "$MATH_DATA/eval/dataset_dict.json" \
  "$MODEL_ROOT/config.json" "$INVALID_LOG"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E69 Gate 2 repair prerequisite: $required" >&2
    exit 1
  fi
done

readarray -t frozen_hashes < <(
  "$PYTHON_BIN" - "$ORIGINAL_IDENTITY" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if identity.get("schema") != "e69_gate2_compute_matched_screen_v1":
    raise SystemExit("unexpected E69 Gate 2 identity schema")
jobs = identity.get("jobs", {}).get("math_dev", [])
invalid = [row for row in jobs if int(row["job_id"]) == 30159730]
if (
    len(invalid) != 1
    or invalid[0]["arm"] != "verified_first_global_replay_canonical"
    or int(invalid[0]["seed"]) != 43
):
    raise SystemExit("original Gate 2 identity does not bind invalid MATH cell")
print(identity["source_hash"])
print(identity["execution_surface_hash"])
PY
)
SOURCE_HASH="${frozen_hashes[0]}"
EXECUTION_HASH="${frozen_hashes[1]}"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_ops_${EXECUTION_HASH}/ops"

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
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E69 Gate 2 repair source snapshot mismatch" >&2
  exit 1
fi
if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "E69 Gate 2 repair execution snapshot mismatch" >&2
  exit 1
fi

if [[ "$phase" == full ]]; then
  for fresh in "$MANIFEST" "$REPAIR_IDENTITY"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E69 Gate 2 repair artifact required: $fresh" >&2
      exit 1
    fi
  done
  if ! grep -q \
    'semantic_shannon_separate_advantage requires a positive semantic_shannon_coef' \
    "$INVALID_LOG"; then
    echo "Invalid job log does not contain the frozen startup failure" >&2
    exit 1
  fi
  if find "$ROOT_DIR/var/data" -maxdepth 2 -type d \
      -path "*mde69_gate2_compute_matched*job${INVALID_JOB}" \
      -print -quit | grep -q .; then
    echo "Invalid Gate 2 job unexpectedly has a run directory" >&2
    exit 1
  fi
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export RUN_STAMP_PREFIX="$PREFIX"
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATH_DATA"
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=43
export OAT_ZERO_ONLY_ARMS=verified_first_global_replay_canonical
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
  VERIFIED_FIRST_SPLIT_CANONICAL VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL \
  VERIFIED_COUNTERFACTUAL_CANONICAL VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL \
  VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL \
  VERIFIED_ROUTE_SUCCESSOR; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM=1

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
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_verified_answer
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0
export OAT_ZERO_VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE=16
export OAT_ZERO_VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS=2
export OAT_ZERO_VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP=2.0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=6
export OAT_ZERO_NUM_PROMPT_EPOCH=6
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
export OAT_ZERO_PROMPT_TEMPLATE=qwen_math
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEST_SPLIT=math
export OAT_ZERO_VERIFIER_VERSION=math_verify
export OAT_ZERO_PROMPT_MAX_LENGTH=1024
export OAT_ZERO_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_MAX_MODEL_LEN=2048
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690205
export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_ALLOW_SPARSE_EVAL=1
export OAT_ZERO_SYNC_PARAMS_EVERY=1
export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1
export OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1
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
export OAT_ZERO_RESUME_STEPS=384
export OAT_ZERO_RESUME_FROM=384
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=7-00:00:00
export OAT_ZERO_TRAIN_NODELIST=node302
export OAT_ZERO_TRAIN_GRES=gpu:a100:1
export OAT_ZERO_TRAIN_PARTITION=lowprio
export OAT_ZERO_TRAIN_ACCOUNT=mltheory

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e69-gate2-repair] configuration passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$ORIGINAL_IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
released=0
replacement_job=""
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && -n "$replacement_job" ]]; then
    scancel "$replacement_job" 2>/dev/null || true
    echo "[e69-gate2-repair] cancelled incomplete held repair job" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

"$OPS_ROOT/submit_countdown_comparative.sh"
mapfile -t jobs < <(
  awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#jobs[@]}" -ne 1 ]]; then
  echo "E69 Gate 2 repair manifest must contain exactly one job" >&2
  exit 1
fi
replacement_job="${jobs[0]}"
record="$(scontrol show job "$replacement_job" -o)"
for required in \
  'JobState=PENDING' 'Reason=JobHeldUser' \
  'OAT_ZERO_VARIANT=verified_first_global_replay_canonical' \
  'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10' \
  'OAT_ZERO_MAX_PROMPT_EPOCHS=6' \
  'OAT_ZERO_NUM_SAMPLES=16' \
  'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3' \
  "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
  "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
  "OAT_ZERO_PROTOCOL_IDENTITY=${ORIGINAL_IDENTITY}"; do
  if [[ "$record" != *"$required"* ]]; then
    echo "E69 Gate 2 repair held-job audit missing: $required" >&2
    exit 1
  fi
done

export SOURCE_HASH EXECUTION_HASH replacement_job
"$PYTHON_BIN" - \
  "$REPAIR_IDENTITY" "$ORIGINAL_IDENTITY" "$AMENDMENT" "$0" "$MANIFEST" \
  "$INVALID_LOG" <<'PY'
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
rows = list(csv.DictReader(pathlib.Path(sys.argv[5]).open(), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit("repair manifest must have exactly one row")
row = rows[0]
payload = {
    "schema": "e69_gate2_math_endpoint_startup_repair_v1",
    "original_identity_sha256": digest(sys.argv[2]),
    "amendment_sha256": digest(sys.argv[3]),
    "launcher_sha256": digest(sys.argv[4]),
    "manifest_sha256": digest(sys.argv[5]),
    "invalid_log_sha256": digest(sys.argv[6]),
    "source_hash": os.environ["SOURCE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_HASH"],
    "invalid_job": {
        "job_id": 30159730,
        "optimizer_records": 0,
        "failure": (
            "semantic_shannon_separate_advantage requires a positive "
            "semantic_shannon_coef"
        ),
    },
    "replacement": {
        "arm": row["arm"],
        "seed": int(row["seed"]),
        "job_id": int(row["job_id"]),
        "run_stamp": row["run_stamp"],
    },
    "only_change": {
        "name": "OAT_ZERO_SEMANTIC_SHANNON_COEF",
        "invalid": "0",
        "replacement": "0.10",
        "meaning": "restore the prospectively named E66 endpoint-only arm",
    },
    "attempt_selection": "exclude_invalid_preoptimizer_job_use_exact_replacement",
    "terminal_outcomes_observed_before_repair": False,
    "math500_sealed": True,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

scontrol release "$replacement_job"
released=1
trap - EXIT
echo "[e69-gate2-repair] released replacement job $replacement_job"
echo "[e69-gate2-repair] identity=$REPAIR_IDENTITY"
