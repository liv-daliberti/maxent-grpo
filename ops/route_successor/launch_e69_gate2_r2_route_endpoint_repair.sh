#!/usr/bin/env bash
# Configure or atomically submit E69-R2's prospective Python-only repair.
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
R1_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_r1_execution_repair_identity.json"
R2_PROTOCOL="$ROOT_DIR/paper/preregistration/e69_gate2_r2_route_endpoint_bookkeeping_repair_20260728.md"
R2_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_r2_route_endpoint_bookkeeping_repair_identity.json"
R2_ACCOUNTING="$ROOT_DIR/var/artifacts/e69_gate2_r2_failed_r1_python_accounting.txt"
R1_ROUTE_SNAPSHOTS="$ROOT_DIR/var/artifacts/e69_gate2_r1_route_temporal_snapshots.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
PYTHON_DATA="$ROOT_DIR/var/data/python_factor_modebench_v1"
PYTHON_PREFIX=pye69_gate2_r2_route_endpoint_repair
PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/${PYTHON_PREFIX}_comparative_jobs.tsv"
FAILED_R1_JOB=30168831
OLD_TRANSITION_JOB=30168832
MATH_JOBS=(30159729 30160101)
A5000_POOL=node105,node202,node203,node204

for required in \
  "$ORIGINAL_IDENTITY" "$R1_IDENTITY" "$R2_PROTOCOL" \
  "$R1_ROUTE_SNAPSHOTS" "$MODEL_ROOT/config.json" \
  "$PYTHON_DATA/train/dataset_dict.json" \
  "$PYTHON_DATA/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E69-R2 prerequisite: $required" >&2
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
  "$PYTHON_BIN" - "$R1_IDENTITY" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if identity.get("schema") != "e69_gate2_r1_execution_repair_v1":
    raise SystemExit("unexpected E69-R1 identity schema")
print(identity["source_hash"])
print(identity["execution_surface_hash"])
for row in identity["mappings"]["graph_coloring"]:
    print(int(row["replacement_job_id"]))
PY
)
PARENT_SOURCE_HASH="${parent_values[0]}"
EXECUTION_HASH="${parent_values[1]}"
GRAPH_JOBS=("${parent_values[@]:2}")
PARENT_SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_${PARENT_SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_ops_${EXECUTION_HASH}/ops"

if [[ "$(hash_tree "$PARENT_SOURCE_ROOT")" != "$PARENT_SOURCE_HASH" ]]; then
  echo "E69-R2 parent source snapshot mismatch" >&2
  exit 1
fi
if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "E69-R2 execution snapshot mismatch" >&2
  exit 1
fi

SOURCE_HASH="$(hash_source_tree "$ROOT_DIR/src")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_r2_${SOURCE_HASH}/src"
if [[ ! -d "$SOURCE_ROOT" ]]; then
  staging="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e69-r2-source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  find "$staging/src" -type d -name __pycache__ -prune -exec rm -rf {} +
  mv "$staging" "$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_r2_${SOURCE_HASH}"
fi
if [[ "$(hash_source_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E69-R2 source snapshot mismatch" >&2
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
if [[ "${#changed_source_files[@]}" -ne 1 \
    || "${changed_source_files[0]}" != "oat_drgrpo/verified_route_library.py" ]]; then
  echo "E69-R2 source scope drift: ${changed_source_files[*]:-none}" >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$SOURCE_ROOT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from oat_drgrpo.verified_route_library import VerifiedRouteLibrary

library = VerifiedRouteLibrary(replay_groups_per_step=1)
common = {
    "prompt_token_ids": [[1]],
    "verifier_ids": ["python_factor_function"],
    "route_signatures": [
        "python-factor-route:v1:lambda(mod(input,literal))"
    ],
    "model_mean_logprobs": [-1.0],
    "task_verified": [True],
    "active_mask": [True],
}
library.observe_neutral(
    endpoint_keys=["python_factor:2,3"],
    response_token_ids=[[22]],
    **common,
)
library.observe_neutral(
    endpoint_keys=["python_factor:3,2"],
    response_token_ids=[[11]],
    **common,
)
exemplar = library.prompt_exemplars([1])[0]
if (
    exemplar.endpoint_key != "python_factor:3,2"
    or exemplar.response_token_ids != (11,)
    or library.diagnostics().cross_prompt_neutral_reproductions != 0
):
    raise SystemExit("E69-R2 route/endpoint repair contract failed")
try:
    library.observe_neutral(
        endpoint_keys=["python_factor:7,7"],
        response_token_ids=[[11]],
        **common,
    )
except ValueError as exc:
    if str(exc) != "verified route endpoint changed for one response":
        raise
else:
    raise SystemExit("E69-R2 identical-response endpoint drift did not fail")
PY
if [[ "$(hash_source_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E69-R2 source snapshot changed during contract validation" >&2
  exit 1
fi

if [[ "$phase" == full ]]; then
  for fresh in "$R2_IDENTITY" "$R2_ACCOUNTING" "$PYTHON_MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E69-R2 artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
  failed_record="$(sacct -X -j "$FAILED_R1_JOB" -n -P -o JobIDRaw,State)"
  if [[ "$failed_record" != "$FAILED_R1_JOB|CANCELLED by "* ]]; then
    echo "E69-R2 failed Python scheduler state drift: $failed_record" >&2
    exit 1
  fi
  if ! rg -q \
    'ValueError: verified route identity changed for one source prompt' \
    "$ROOT_DIR/var/artifacts/logs/xdr_train-${FAILED_R1_JOB}.out"; then
    echo "E69-R2 failed Python invariant evidence is absent" >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$ROOT_DIR" "$R1_IDENTITY" "$FAILED_R1_JOB" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
identity = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
job_id = int(sys.argv[3])
rows = identity["mappings"]["python_factor"]
if len(rows) != 1 or int(rows[0]["replacement_job_id"]) != job_id:
    raise SystemExit("E69-R2 failed Python mapping drift")
stamp = rows[0]["run_stamp"]
matches = list(
    (root / "var/data").glob(
        f"xdr_*_{stamp}/debug_job{job_id}"
    )
)
if len(matches) != 1:
    raise SystemExit("E69-R2 failed Python run directory is not unique")
run_dir = matches[0]
checkpoints = list(run_dir.glob("checkpoints/**/mp_rank_00_model_states.pt"))
if checkpoints:
    raise SystemExit("E69-R2 failed Python unexpectedly has a checkpoint")
metrics = [
    json.loads(line)
    for line in (run_dir / "train_metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]
steps = [
    int(row.get("trainer/global_step", row.get("misc/global_step", -1)))
    for row in metrics
]
if max(steps, default=-1) != 301:
    raise SystemExit("E69-R2 failed Python terminal prefix drift")
evaluations = [
    json.loads(line)
    for line in (run_dir / "eval_mode_coverage_draws.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]
if {int(row["step"]) for row in evaluations} != {0}:
    raise SystemExit("E69-R2 observed a post-initialization Python outcome")
print(run_dir.resolve())
PY
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
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690203
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
export OAT_ZERO_AUTO_RESUME=0
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
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
export OAT_ZERO_SAVE_STEPS=384
export OAT_ZERO_SAVE_FROM=384
export OAT_ZERO_RESUME_STEPS=384
export OAT_ZERO_RESUME_FROM=384
export OAT_ZERO_TRAIN_NODELIST="$A5000_POOL"
export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
export OAT_ZERO_TRAIN_PARTITION=all
export OAT_ZERO_TRAIN_ACCOUNT=mltheory

export RUN_STAMP_PREFIX="$PYTHON_PREFIX"
export OAT_ZERO_ONLY_ARMS=verified_route_successor
export OAT_ZERO_COMPARATIVE_TASK=python_factor
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA"
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e69-r2] Python configuration and repair contract passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$R2_IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
replacement_job=""
transition_job=""
released=0
transition_replaced=0
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && -n "$replacement_job" ]]; then
    scancel "$replacement_job" 2>/dev/null || true
  fi
  if [[ "$transition_replaced" != "1" && -n "$transition_job" ]]; then
    scancel "$transition_job" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup_held EXIT

"$OPS_ROOT/submit_countdown_comparative.sh"
mapfile -t replacement_jobs < <(
  awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$PYTHON_MANIFEST"
)
if [[ "${#replacement_jobs[@]}" -ne 1 ]]; then
  echo "E69-R2 manifest must contain exactly one job" >&2
  exit 1
fi
replacement_job="${replacement_jobs[0]}"
record="$(scontrol show job "$replacement_job" -o)"
for required in \
  'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=mltheory' \
  'ReqNodeList=node[105,202-204]' 'gres/gpu:a5000=1' \
  '--partition=all' '--account=mltheory' \
  'OAT_ZERO_MAX_PROMPT_EPOCHS=6' \
  'OAT_ZERO_NUM_PROMPT_EPOCH=6' \
  'OAT_ZERO_NUM_SAMPLES=16' \
  'OAT_ZERO_AUTO_RESUME=0' \
  'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3' \
  'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
  "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
  "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
  "OAT_ZERO_PROTOCOL_IDENTITY=${R2_IDENTITY}"; do
  if [[ "$record" != *"$required"* ]]; then
    echo "E69-R2 held-job audit $replacement_job missing: $required" >&2
    exit 1
  fi
done

dependency_jobs=("${MATH_JOBS[@]}" "${GRAPH_JOBS[@]}" "$replacement_job")
dependency="$(IFS=:; echo "${dependency_jobs[*]}")"
transition_job="$(
  sbatch --parsable --dependency="afterany:${dependency}" \
    "$ROOT_DIR/ops/slurm/e69_gate2_to_gate3.slurm"
)"
transition_record="$(scontrol show job "$transition_job" -o)"
if [[ "$transition_record" != *'JobState=PENDING'* \
    || "$transition_record" != *'Reason=Dependency'* ]]; then
  echo "E69-R2 transition is not dependency-held: $transition_job" >&2
  exit 1
fi

sacct -X -j "$FAILED_R1_JOB" \
  -o JobIDRaw,State,Elapsed,Restarts,NodeList,Reason,ExitCode -n -P \
  > "$R2_ACCOUNTING"

export SOURCE_HASH PARENT_SOURCE_HASH EXECUTION_HASH
"$PYTHON_BIN" - \
  "$R2_IDENTITY" "$ORIGINAL_IDENTITY" "$R1_IDENTITY" \
  "$R2_PROTOCOL" "$0" "$PYTHON_MANIFEST" "$R2_ACCOUNTING" \
  "$R1_ROUTE_SNAPSHOTS" "$transition_job" "$replacement_job" \
  "$PARENT_SOURCE_ROOT/oat_drgrpo/verified_route_library.py" \
  "$SOURCE_ROOT/oat_drgrpo/verified_route_library.py" \
  "$ROOT_DIR/var/artifacts/logs/xdr_train-${FAILED_R1_JOB}.out" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

identity_path = pathlib.Path(sys.argv[1])
manifest_rows = list(
    csv.DictReader(pathlib.Path(sys.argv[6]).open(), delimiter="\t")
)
if len(manifest_rows) != 1:
    raise SystemExit("E69-R2 manifest grid mismatch")
row = manifest_rows[0]
if (
    row["arm"] != "verified_route_successor"
    or int(row["seed"]) != 43
    or int(row["job_id"]) != int(sys.argv[10])
):
    raise SystemExit("E69-R2 Python replacement cell drift")
r1 = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
graph_jobs = sorted(
    int(value["replacement_job_id"])
    for value in r1["mappings"]["graph_coloring"]
)
payload = {
    "schema": "e69_gate2_r2_route_endpoint_bookkeeping_repair_v1",
    "original_identity_sha256": digest(sys.argv[2]),
    "parent_r1_identity_sha256": digest(sys.argv[3]),
    "protocol": str(pathlib.Path(sys.argv[4]).resolve()),
    "protocol_sha256": digest(sys.argv[4]),
    "launcher_sha256": digest(sys.argv[5]),
    "manifest_sha256": digest(sys.argv[6]),
    "failed_r1_accounting_sha256": digest(sys.argv[7]),
    "parent_route_temporal_snapshot_sha256": digest(sys.argv[8]),
    "parent_source_hash": os.environ["PARENT_SOURCE_HASH"],
    "source_hash": os.environ["SOURCE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_HASH"],
    "old_route_library_sha256": digest(sys.argv[11]),
    "repaired_route_library_sha256": digest(sys.argv[12]),
    "failed_r1_log_sha256": digest(sys.argv[13]),
    "mapping": {
        "domain": "python_factor",
        "arm": "verified_route_successor",
        "seed": 43,
        "invalid_job_id": 30168831,
        "replacement_job_id": int(sys.argv[10]),
        "run_stamp": row["run_stamp"],
    },
    "excluded_r1_failure": {
        "job_id": 30168831,
        "last_optimizer_step": 301,
        "durable_checkpoints": 0,
        "post_initialization_evaluations": 0,
        "failure": "verified route identity changed for one source prompt",
    },
    "placement": {
        "nodelist": "node105,node202,node203,node204",
        "gres": "gpu:a5000:1",
        "requested_partition": "all",
        "resolved_partition": "mltheory",
        "account": "mltheory",
    },
    "transition": {
        "job_id": int(sys.argv[9]),
        "dependency_type": "afterany",
        "dependency_jobs": [
            30159729,
            30160101,
            *graph_jobs,
            int(sys.argv[10]),
        ],
        "fail_closed_audit": True,
    },
    "attempt_selection": "replace_only_failed_python_successor_from_initialization",
    "repair_decision_basis": "hard_invariant_exception_and_frozen_identity_semantics_only",
    "r1_failure_observed_before_repair": True,
    "repaired_outcomes_observed_before_freeze": False,
    "outcome_tuning": False,
    "algorithm_or_gate_change": False,
    "implementation_contract_repair": True,
    "minimal_scope_proof": (
        "branch_is_inert_for_any_successor_input_stream_without_a_"
        "same_prompt_same_route_different_endpoint_observation"
    ),
    "math_jobs_untouched": [30159729, 30160101],
    "graph_jobs_untouched": graph_jobs,
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
scontrol release "$replacement_job"
released=1
trap - EXIT
echo "[e69-r2] preserved failed R1 Python job: $FAILED_R1_JOB"
echo "[e69-r2] released Python replacement: $replacement_job"
echo "[e69-r2] transition job: $transition_job"
echo "[e69-r2] identity=$R2_IDENTITY"
