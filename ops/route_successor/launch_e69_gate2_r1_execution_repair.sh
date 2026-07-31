#!/usr/bin/env bash
# Configure or atomically submit E69-R1's five execution-only repair jobs.
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
PLACEMENT_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_pending_placement_repair_identity.json"
GRAPH_PARENT_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_graph_a5000_preemption_repair_identity.json"
R1_PROTOCOL="$ROOT_DIR/paper/preregistration/e69_gate2_r1_execution_repair_20260728.md"
R1_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_r1_execution_repair_identity.json"
SOURCE_ACCOUNTING="$ROOT_DIR/var/artifacts/e69_gate2_r1_excluded_source_accounting.txt"
ROUTE_PARENT="$ROOT_DIR/var/artifacts/e69_gate2_route_temporal_snapshots.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA="$ROOT_DIR/var/data/exact_answer_mode_probe"
PYTHON_DATA="$ROOT_DIR/var/data/python_factor_modebench_v1"
GRAPH_PREFIX=gce69_gate2_r1_execution_repair_retry1
PYTHON_PREFIX=pye69_gate2_r1_execution_repair_retry1
GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/${PYTHON_PREFIX}_comparative_jobs.tsv"
SOURCE_GRAPH_JOBS=(30160592 30160594 30160595 30160596)
SOURCE_PYTHON_JOB=30160205
OLD_TRANSITION_JOB=30160614
MATH_JOBS=(30159729 30160101)
A5000_POOL=node105,node202,node203,node204

for required in \
  "$ORIGINAL_IDENTITY" "$PLACEMENT_IDENTITY" "$GRAPH_PARENT_IDENTITY" \
  "$R1_PROTOCOL" "$ROUTE_PARENT" "$MODEL_ROOT/config.json" \
  "$GRAPH_DATA/train/dataset_dict.json" "$GRAPH_DATA/eval/dataset_dict.json" \
  "$PYTHON_DATA/train/dataset_dict.json" "$PYTHON_DATA/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E69-R1 prerequisite: $required" >&2
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
  echo "E69-R1 source snapshot mismatch" >&2
  exit 1
fi
if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "E69-R1 execution snapshot mismatch" >&2
  exit 1
fi

if [[ "$phase" == full ]]; then
  for fresh in \
    "$R1_IDENTITY" "$SOURCE_ACCOUNTING" \
    "$GRAPH_MANIFEST" "$PYTHON_MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E69-R1 artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
  "$PYTHON_BIN" - \
    "$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_audit_latest.json" \
    "${SOURCE_GRAPH_JOBS[@]}" "$SOURCE_PYTHON_JOB" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
graph_jobs = {int(value) for value in sys.argv[2:6]}
python_job = int(sys.argv[6])
runs = {int(row["job_id"]): row for row in audit.get("physical_runs", [])}
if set(runs).isdisjoint(graph_jobs):
    raise SystemExit("E69-R1 source audit does not contain the Graph cohort")
for job_id in graph_jobs:
    if not runs.get(job_id, {}).get("terminal"):
        raise SystemExit(f"Graph source job is not terminal: {job_id}")
if runs.get(python_job, {}).get("terminal"):
    raise SystemExit("Python source job unexpectedly became terminal")
if audit.get("math500_sealed") is not True:
    raise SystemExit("MATH-500 seal is absent")
PY
  python_record="$(scontrol show job "$SOURCE_PYTHON_JOB" -o)"
  if [[ "$python_record" != *'JobState=RUNNING'* \
      && "$python_record" != *'JobState=PENDING'* ]]; then
    echo "Python source job is not active: $SOURCE_PYTHON_JOB" >&2
    exit 1
  fi
  if [[ "$python_record" != *'Restarts=17'* ]]; then
    echo "Python source restart evidence drifted from the frozen 17" >&2
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
export OAT_ZERO_INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM=1
export OAT_ZERO_INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS=64
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
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
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
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
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=3-00:00:00
export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
export OAT_ZERO_TEST_SPLIT=multi_answer
export OAT_ZERO_VERIFIER_VERSION=fast
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_GENERATE_MAX_LENGTH=192
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
export OAT_ZERO_MAX_MODEL_LEN=512
export OAT_ZERO_TRAIN_NODELIST="$A5000_POOL"
export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
export OAT_ZERO_TRAIN_PARTITION=all
export OAT_ZERO_TRAIN_ACCOUNT=mltheory

submit_domain() {
  local domain="$1"
  export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
  case "$domain" in
    graph_coloring)
      export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
      export OAT_ZERO_ONLY_ARMS="grpo,verified_first_global_replay_canonical,verified_entropy_gated_singleton_escape_canonical,verified_route_successor"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA"
      export OAT_ZERO_MAX_TRAIN=192
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=192
      export OAT_ZERO_SAVE_STEPS=192
      export OAT_ZERO_SAVE_FROM=192
      export OAT_ZERO_RESUME_STEPS=192
      export OAT_ZERO_RESUME_FROM=192
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690201
      ;;
    python_factor)
      export RUN_STAMP_PREFIX="$PYTHON_PREFIX"
      export OAT_ZERO_ONLY_ARMS=verified_route_successor
      export OAT_ZERO_COMPARATIVE_TASK=python_factor
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690203
      ;;
    *)
      echo "Unknown E69-R1 domain: $domain" >&2
      exit 1
      ;;
  esac
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_domain graph_coloring
  submit_domain python_factor
  echo "[e69-r1] Graph and Python configurations passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$R1_IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
replacement_jobs=()
transition_job=""
released=0
source_cancelled=0
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "$source_cancelled" != "1" ]]; then
    if [[ "${#replacement_jobs[@]}" -gt 0 ]]; then
      scancel "${replacement_jobs[@]}" 2>/dev/null || true
    fi
    if [[ -n "$transition_job" ]]; then
      scancel "$transition_job" 2>/dev/null || true
    fi
    echo "[e69-r1] cancelled incomplete held submission" >&2
  elif [[ "$released" != "1" ]]; then
    echo "[e69-r1] source Python is cancelled; replacements remain held" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

submit_domain graph_coloring
submit_domain python_factor
for manifest_spec in "$GRAPH_MANIFEST:4" "$PYTHON_MANIFEST:1"; do
  manifest="${manifest_spec%:*}"
  expected="${manifest_spec##*:}"
  mapfile -t jobs < <(
    awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#jobs[@]}" -ne "$expected" ]]; then
    echo "E69-R1 manifest has ${#jobs[@]} jobs; expected $expected: $manifest" >&2
    exit 1
  fi
  replacement_jobs+=("${jobs[@]}")
done
if [[ "${#replacement_jobs[@]}" -ne 5 ]]; then
  echo "E69-R1 must contain exactly five jobs" >&2
  exit 1
fi

for job_id in "${replacement_jobs[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=mltheory' \
    'ReqNodeList=node[105,202-204]' 'gres/gpu:a5000=1' \
    '--partition=all' '--account=mltheory' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=6' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=6' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${R1_IDENTITY}"; do
    if [[ "$record" != *"$required"* ]]; then
      echo "E69-R1 held-job audit $job_id missing: $required" >&2
      exit 1
    fi
  done
done

dependency_jobs=("${MATH_JOBS[@]}" "${replacement_jobs[@]}")
dependency="$(IFS=:; echo "${dependency_jobs[*]}")"
transition_job="$(
  sbatch --parsable --dependency="afterany:${dependency}" \
    "$ROOT_DIR/ops/slurm/e69_gate2_to_gate3.slurm"
)"
transition_record="$(scontrol show job "$transition_job" -o)"
if [[ "$transition_record" != *'JobState=PENDING'* \
    || "$transition_record" != *'Reason=Dependency'* ]]; then
  echo "E69-R1 transition is not dependency-held: $transition_job" >&2
  exit 1
fi

sacct -X \
  -j 30160592,30160594,30160595,30160596,30160205 \
  -o JobIDRaw,State,Elapsed,Restarts,NodeList,Reason -n -P \
  > "$SOURCE_ACCOUNTING"

export SOURCE_HASH EXECUTION_HASH
"$PYTHON_BIN" - \
  "$R1_IDENTITY" "$ORIGINAL_IDENTITY" "$PLACEMENT_IDENTITY" \
  "$GRAPH_PARENT_IDENTITY" "$R1_PROTOCOL" "$0" \
  "$GRAPH_MANIFEST" "$PYTHON_MANIFEST" "$SOURCE_ACCOUNTING" \
  "$ROUTE_PARENT" "$transition_job" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

def rows(raw):
    return list(csv.DictReader(pathlib.Path(raw).open(), delimiter="\t"))

identity_path = pathlib.Path(sys.argv[1])
graph_parent = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
placement_parent = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
graph_replacements = rows(sys.argv[7])
python_replacements = rows(sys.argv[8])
if len(graph_replacements) != 4 or len(python_replacements) != 1:
    raise SystemExit("E69-R1 replacement manifest grid mismatch")
graph_by_arm = {row["arm"]: row for row in graph_replacements}
python_row = python_replacements[0]
graph_sources = graph_parent["mappings"]
python_sources = [
    row
    for row in placement_parent["mappings"]["python_factor"]
    if row["arm"] == "verified_route_successor"
]
if len(graph_sources) != 4 or len(python_sources) != 1:
    raise SystemExit("E69-R1 source mapping grid mismatch")
if set(graph_by_arm) != {row["arm"] for row in graph_sources}:
    raise SystemExit("E69-R1 Graph arm grid drift")
if python_row["arm"] != "verified_route_successor":
    raise SystemExit("E69-R1 Python replacement arm drift")

graph_mappings = [
    {
        "arm": source["arm"],
        "seed": int(source["seed"]),
        "invalid_job_id": int(source["replacement_job_id"]),
        "replacement_job_id": int(graph_by_arm[source["arm"]]["job_id"]),
        "run_stamp": graph_by_arm[source["arm"]]["run_stamp"],
    }
    for source in graph_sources
]
python_source = python_sources[0]
python_mappings = [
    {
        "arm": "verified_route_successor",
        "seed": int(python_source["seed"]),
        "invalid_job_id": int(python_source["replacement_job_id"]),
        "replacement_job_id": int(python_row["job_id"]),
        "run_stamp": python_row["run_stamp"],
    }
]
if {row["invalid_job_id"] for row in graph_mappings} != {
    30160592, 30160594, 30160595, 30160596
}:
    raise SystemExit("E69-R1 Graph source job grid drift")
if python_mappings[0]["invalid_job_id"] != 30160205:
    raise SystemExit("E69-R1 Python source job drift")

payload = {
    "schema": "e69_gate2_r1_execution_repair_v1",
    "original_identity_sha256": digest(sys.argv[2]),
    "parent_placement_repair_identity_sha256": digest(sys.argv[3]),
    "parent_graph_preemption_repair_identity_sha256": digest(sys.argv[4]),
    "protocol": str(pathlib.Path(sys.argv[5]).resolve()),
    "protocol_sha256": digest(sys.argv[5]),
    "launcher_sha256": digest(sys.argv[6]),
    "manifest_sha256": {
        "graph_coloring": digest(sys.argv[7]),
        "python_factor": digest(sys.argv[8]),
    },
    "source_accounting_sha256": digest(sys.argv[9]),
    "parent_route_temporal_snapshot_sha256": digest(sys.argv[10]),
    "source_hash": os.environ["SOURCE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_HASH"],
    "mappings": {
        "graph_coloring": graph_mappings,
        "python_factor": python_mappings,
    },
    "excluded_source_jobs": {
        "graph_coloring": [30160592, 30160594, 30160595, 30160596],
        "python_factor": [30160205],
    },
    "observed_scheduler_restarts": {
        "30160592": 7,
        "30160594": 4,
        "30160595": 2,
        "30160596": 0,
        "30160205": 17,
    },
    "placement": {
        "nodelist": "node105,node202,node203,node204",
        "gres": "gpu:a5000:1",
        "requested_partition": "all",
        "resolved_partition": "mltheory",
        "account": "mltheory",
    },
    "transition": {
        "job_id": int(sys.argv[11]),
        "dependency_type": "afterany",
        "dependency_jobs": [
            30159729,
            30160101,
            *[row["replacement_job_id"] for row in graph_mappings],
            python_mappings[0]["replacement_job_id"],
        ],
        "fail_closed_audit": True,
    },
    "attempt_selection": (
        "replace_complete_graph_cohort_and_corrupted_python_successor"
    ),
    "repair_decision_basis": (
        "frozen_scheduler_integrity_rule_not_metric_values"
    ),
    "terminal_outcomes_observed_before_repair": True,
    "outcome_tuning": False,
    "algorithm_or_gate_change": False,
    "math_jobs_untouched": [30159729, 30160101],
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

scancel "$SOURCE_PYTHON_JOB" "$OLD_TRANSITION_JOB"
source_cancelled=1
scontrol release "${replacement_jobs[@]}"
released=1
trap - EXIT
echo "[e69-r1] excluded Python source job: $SOURCE_PYTHON_JOB"
echo "[e69-r1] released five replacements: ${replacement_jobs[*]}"
echo "[e69-r1] transition job: $transition_job"
echo "[e69-r1] identity=$R1_IDENTITY"
