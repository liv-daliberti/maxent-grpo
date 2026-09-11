#!/usr/bin/env bash
# Configure or atomically submit E69-R3's paired MATH-dev evaluation repair.
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
PYTHON_LIB_DIR="$(cd "$(dirname "$PYTHON_BIN")/../lib" && pwd)"
export LD_LIBRARY_PATH="$PYTHON_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ORIGINAL_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_identity.json"
R1_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_r1_execution_repair_identity.json"
R2_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_r2_route_endpoint_bookkeeping_repair_identity.json"
R3_PROTOCOL="$ROOT_DIR/paper/preregistration/e69_gate2_r3_math_dev_evaluation_split_repair_20260729.md"
R3_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_r3_math_dev_evaluation_split_repair_identity.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
MATH_DATA="$ROOT_DIR/var/data/math12k_384_route_dev128_v1"
PREFIX=mde69_gate2_r3_eval_split_repair
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
STARTUP_MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_startup_rejected_jobs.tsv"
STARTUP_ACCOUNTING="$ROOT_DIR/var/artifacts/e69_gate2_r3_startup_rejected_accounting.txt"
OLD_MATH_JOBS=(30159729 30160101)
OLD_TRANSITION_JOB=30170234

for required in \
  "$ORIGINAL_IDENTITY" "$R1_IDENTITY" "$R2_IDENTITY" "$R3_PROTOCOL" \
  "$MODEL_ROOT/config.json" \
  "$MATH_DATA/train/dataset_dict.json" \
  "$MATH_DATA/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E69-R3 prerequisite: $required" >&2
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

hash_source_tree() {
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

readarray -t parent_values < <(
  "$PYTHON_BIN" - "$R1_IDENTITY" "$R2_IDENTITY" <<'PY'
import json
import pathlib
import sys

r1 = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
r2 = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if r1.get("schema") != "e69_gate2_r1_execution_repair_v1":
    raise SystemExit("unexpected E69-R1 identity schema")
if r2.get("schema") != "e69_gate2_r2_route_endpoint_bookkeeping_repair_v1":
    raise SystemExit("unexpected E69-R2 identity schema")
print(r2["source_hash"])
print(r2["execution_surface_hash"])
for row in r1["mappings"]["graph_coloring"]:
    print(int(row["replacement_job_id"]))
print(int(r2["mapping"]["replacement_job_id"]))
PY
)
PARENT_SOURCE_HASH="${parent_values[0]}"
EXECUTION_HASH="${parent_values[1]}"
GRAPH_JOBS=("${parent_values[@]:2:4}")
PYTHON_JOB="${parent_values[6]}"
PARENT_SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_r2_${PARENT_SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_ops_${EXECUTION_HASH}/ops"

if [[ "$(hash_source_tree "$PARENT_SOURCE_ROOT")" != "$PARENT_SOURCE_HASH" ]]; then
  echo "E69-R3 parent source snapshot mismatch" >&2
  exit 1
fi
if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "E69-R3 execution snapshot mismatch" >&2
  exit 1
fi

SOURCE_HASH="$(hash_source_tree "$ROOT_DIR/src")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_r3_${SOURCE_HASH}/src"
if [[ ! -d "$SOURCE_ROOT" ]]; then
  staging="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e69-r3-source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  find "$staging/src" -type d -name __pycache__ -prune -exec rm -rf {} +
  mv "$staging" "$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_r3_${SOURCE_HASH}"
fi
if [[ "$(hash_source_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E69-R3 source snapshot mismatch" >&2
  exit 1
fi
mapfile -t changed_source_files < <(
  "$PYTHON_BIN" - "$PARENT_SOURCE_ROOT" "$SOURCE_ROOT" <<'PY'
import hashlib
import pathlib
import sys

def hashes(root):
    root = pathlib.Path(root)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }

parent = hashes(sys.argv[1])
repaired = hashes(sys.argv[2])
for relative in sorted(set(parent) | set(repaired)):
    if parent.get(relative) != repaired.get(relative):
        print(relative)
PY
)
expected_changed=(
  oat_drgrpo/args.py
  oat_drgrpo/learner/init.py
)
if [[ "${changed_source_files[*]}" != "${expected_changed[*]}" ]]; then
  echo "E69-R3 source scope drift: ${changed_source_files[*]:-none}" >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$SOURCE_ROOT" "$MATH_DATA" <<'PY'
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from datasets import load_from_disk
from oat_drgrpo.args import ZeroMathArgs, validate_zero_math_args

data = load_from_disk(str(pathlib.Path(sys.argv[2]) / "eval"))
if list(data) != ["math_dev"] or len(data["math_dev"]) != 128:
    raise SystemExit("E69-R3 sealed MATH-dev evaluation split drift")
args = ZeroMathArgs(
    online_canonical_replay=True,
    online_canonical_key_mode="math_verified_answer",
    prompt_template="qwen_math",
    verifier_version="math_verify",
    test_split="math_dev",
)
validate_zero_math_args(args)
PY
if [[ "$(hash_source_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E69-R3 source snapshot changed during validation" >&2
  exit 1
fi

if [[ "$phase" == full ]]; then
  if [[ -e "$MANIFEST" && ! -e "$R3_IDENTITY" ]]; then
    if [[ -e "$STARTUP_MANIFEST" || -e "$STARTUP_ACCOUNTING" ]]; then
      echo "Ambiguous E69-R3 startup-rejection provenance" >&2
      exit 1
    fi
    "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import pathlib
import sys

rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t"))
if (
    len(rows) != 2
    or {int(row["job_id"]) for row in rows} != {30172460, 30172461}
    or {row["arm"] for row in rows}
    != {"grpo", "verified_first_global_replay_canonical"}
):
    raise SystemExit("unexpected E69-R3 startup-rejected manifest")
PY
    sacct -X -j 30172460,30172461 \
      -o JobIDRaw,State,Elapsed,Restarts,NodeList,Reason,ExitCode -n -P \
      > "$STARTUP_ACCOUNTING"
    "$PYTHON_BIN" - "$STARTUP_ACCOUNTING" <<'PY'
import pathlib
import sys

rows = [
    line.split("|")
    for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if (
    len(rows) != 2
    or {int(row[0]) for row in rows} != {30172460, 30172461}
    or any(not row[1].startswith("CANCELLED") for row in rows)
    or any(row[2] != "00:00:00" for row in rows)
    or any(row[3] != "0" for row in rows)
):
    raise SystemExit("E69-R3 startup-rejected scheduler evidence drift")
PY
    mv "$MANIFEST" "$STARTUP_MANIFEST"
  fi
  for fresh in "$R3_IDENTITY" "$MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E69-R3 artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
  for required in "$STARTUP_MANIFEST" "$STARTUP_ACCOUNTING"; do
    if [[ ! -f "$required" ]]; then
      echo "Missing E69-R3 startup-rejection provenance: $required" >&2
      exit 1
    fi
  done
  "$PYTHON_BIN" - "$ROOT_DIR" "${OLD_MATH_JOBS[@]}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for job_id in map(int, sys.argv[2:]):
    matches = list((root / "var/data").glob(f"xdr_*/debug_job{job_id}"))
    if len(matches) != 1:
        raise SystemExit(f"E69-R3 MATH source run job{job_id} is not unique")
    if (matches[0] / "eval_mode_coverage_draws.jsonl").exists():
        raise SystemExit(
            f"E69-R3 source job{job_id} unexpectedly has evaluation outcomes"
        )
PY
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export RUN_STAMP_PREFIX="$PREFIX"
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATH_DATA"
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=43
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_DRGRPO_VARIANT=grpo_compute_matched

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
export OAT_ZERO_TEST_SPLIT=math_dev
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
export OAT_ZERO_AUTO_RESUME=0
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=7-00:00:00
export OAT_ZERO_TRAIN_NODELIST=node302
export OAT_ZERO_TRAIN_GRES=gpu:a100:1
export OAT_ZERO_TRAIN_PARTITION=lowprio
export OAT_ZERO_TRAIN_ACCOUNT=mltheory

submit_arm() {
  local arm="$1"
  local coefficient="$2"
  local append="$3"
  export OAT_ZERO_ONLY_ARMS="$arm"
  export OAT_ZERO_SEMANTIC_SHANNON_COEF="$coefficient"
  export OAT_ZERO_APPEND_MANIFEST="$append"
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_arm grpo 0 0
  submit_arm verified_first_global_replay_canonical 0.10 0
  echo "[e69-r3] paired MATH-dev configuration and repair contract passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$R3_IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
replacement_jobs=()
transition_job=""
released=0
transition_replaced=0
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "${#replacement_jobs[@]}" -gt 0 ]]; then
    scancel "${replacement_jobs[@]}" 2>/dev/null || true
  fi
  if [[ "$transition_replaced" != "1" && -n "$transition_job" ]]; then
    scancel "$transition_job" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup_held EXIT

submit_arm grpo 0 0
submit_arm verified_first_global_replay_canonical 0.10 1
mapfile -t replacement_jobs < <(
  awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#replacement_jobs[@]}" -ne 2 ]]; then
  echo "E69-R3 manifest must contain exactly two jobs" >&2
  exit 1
fi

for replacement_job in "${replacement_jobs[@]}"; do
  record="$(scontrol show job "$replacement_job" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    'ReqNodeList=node302' 'TresPerNode=gres/gpu:a100:1' \
    '--gres=gpu:a100:1' \
    'OAT_ZERO_TEST_SPLIT=math_dev' \
    'OAT_ZERO_AUTO_RESUME=0' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=6' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${R3_IDENTITY}"; do
    if [[ "$record" != *"$required"* ]]; then
      echo "E69-R3 held-job audit $replacement_job missing: $required" >&2
      exit 1
    fi
  done
done

dependency_jobs=("${GRAPH_JOBS[@]}" "$PYTHON_JOB" "${replacement_jobs[@]}")
dependency="$(IFS=:; echo "${dependency_jobs[*]}")"
transition_job="$(
  sbatch --parsable --dependency="afterany:${dependency}" \
    "$ROOT_DIR/ops/slurm/e69_gate2_to_gate3.slurm"
)"
transition_record="$(scontrol show job "$transition_job" -o)"
if [[ "$transition_record" != *'JobState=PENDING'* \
    || "$transition_record" != *'Reason=Dependency'* ]]; then
  echo "E69-R3 transition is not dependency-held: $transition_job" >&2
  exit 1
fi

export SOURCE_HASH PARENT_SOURCE_HASH EXECUTION_HASH
"$PYTHON_BIN" - \
  "$R3_IDENTITY" "$ORIGINAL_IDENTITY" "$R2_IDENTITY" "$R3_PROTOCOL" \
  "$0" "$MANIFEST" "$STARTUP_MANIFEST" "$STARTUP_ACCOUNTING" \
  "$transition_job" \
  "${replacement_jobs[0]}" "${replacement_jobs[1]}" \
  "${OLD_MATH_JOBS[0]}" "${OLD_MATH_JOBS[1]}" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

root = pathlib.Path.cwd()
identity_path = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(pathlib.Path(sys.argv[6]).open(), delimiter="\t"))
if len(rows) != 2:
    raise SystemExit("E69-R3 manifest grid mismatch")
expected = {
    ("grpo", 43, int(sys.argv[10])),
    ("verified_first_global_replay_canonical", 43, int(sys.argv[11])),
}
observed = {
    (row["arm"], int(row["seed"]), int(row["job_id"]))
    for row in rows
}
if observed != expected:
    raise SystemExit("E69-R3 replacement cells drift")

old_jobs = [int(sys.argv[12]), int(sys.argv[13])]
old_arms = {
    30159729: "grpo",
    30160101: "verified_first_global_replay_canonical",
}
excluded = []
for job_id in old_jobs:
    logs = root / f"var/artifacts/logs/xdr_train-{job_id}.out"
    raw = logs.read_bytes()
    matches = list((root / "var/data").glob(f"xdr_*/debug_job{job_id}"))
    if len(matches) != 1:
        raise SystemExit(f"E69-R3 source job{job_id} is not unique")
    eval_path = matches[0] / "eval_mode_coverage_draws.jsonl"
    if eval_path.exists():
        raise SystemExit(f"E69-R3 source job{job_id} has evaluation outcomes")
    excluded.append(
        {
            "job_id": job_id,
            "arm": old_arms[job_id],
            "log_prefix_size_bytes": len(raw),
            "log_prefix_sha256": hashlib.sha256(raw).hexdigest(),
            "evaluation_records_at_freeze": 0,
            "run_dir": str(matches[0].resolve()),
        }
    )

mapping_by_arm = {row["arm"]: row for row in rows}
payload = {
    "schema": "e69_gate2_r3_math_dev_evaluation_split_repair_v1",
    "original_identity_sha256": digest(sys.argv[2]),
    "parent_r2_identity_sha256": digest(sys.argv[3]),
    "protocol": str(pathlib.Path(sys.argv[4]).resolve()),
    "protocol_sha256": digest(sys.argv[4]),
    "launcher_sha256": digest(sys.argv[5]),
    "manifest_sha256": digest(sys.argv[6]),
    "startup_rejected_manifest_sha256": digest(sys.argv[7]),
    "startup_rejected_accounting_sha256": digest(sys.argv[8]),
    "startup_rejected_jobs": [30172460, 30172461],
    "parent_source_hash": os.environ["PARENT_SOURCE_HASH"],
    "source_hash": os.environ["SOURCE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_HASH"],
    "mappings": [
        {
            "domain": "math_dev",
            "arm": arm,
            "seed": 43,
            "invalid_job_id": invalid_job,
            "replacement_job_id": int(mapping_by_arm[arm]["job_id"]),
            "run_stamp": mapping_by_arm[arm]["run_stamp"],
        }
        for arm, invalid_job in (
            ("grpo", 30159729),
            ("verified_first_global_replay_canonical", 30160101),
        )
    ],
    "excluded_attempts": excluded,
    "only_configuration_change": {
        "name": "OAT_ZERO_TEST_SPLIT",
        "invalid": "math",
        "replacement": "math_dev",
    },
    "changed_source_files": [
        "oat_drgrpo/args.py",
        "oat_drgrpo/learner/init.py",
    ],
    "transition": {
        "job_id": int(sys.argv[9]),
        "dependency_type": "afterany",
        "dependency_jobs": [
            30168827,
            30168828,
            30168829,
            30168830,
            30170233,
            int(sys.argv[10]),
            int(sys.argv[11]),
        ],
        "fail_closed_audit": True,
    },
    "attempt_selection": (
        "replace_complete_paired_math_dev_cohort_from_initialization"
    ),
    "repair_decision_basis": (
        "sealed_eval_split_name_mismatch_and_zero_evaluation_records_only"
    ),
    "gate_outcomes_observed_before_freeze": False,
    "training_telemetry_observed_before_freeze": True,
    "outcome_tuning": False,
    "algorithm_or_gate_change": False,
    "implementation_contract_repair": True,
    "math500_sealed": True,
}
identity_path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(
    prefix=f".{identity_path.name}.",
    dir=identity_path.parent,
)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, identity_path)
PY

scancel "$OLD_TRANSITION_JOB"
transition_replaced=1
scontrol release "${replacement_jobs[@]}"
released=1
trap - EXIT
echo "[e69-r3] preserved excluded MATH-dev jobs: ${OLD_MATH_JOBS[*]}"
echo "[e69-r3] released paired replacements: ${replacement_jobs[*]}"
echo "[e69-r3] transition job: $transition_job"
echo "[e69-r3] identity=$R3_IDENTITY"
