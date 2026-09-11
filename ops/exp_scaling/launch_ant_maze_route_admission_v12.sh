#!/usr/bin/env bash
# Freeze and submit the anchored AntMaze v12 route admission gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_admission_v12_anchored_waypoint_repair_20260730.md"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/admit_ant_maze_modebench_v12.slurm"
TRAINING_IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v11_identity.json"
CONTROLLER_RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v11.evaluation.json"
CONTROLLER_MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v11.zip"
V11_ROUTE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v11_generation_identity.json"
V11_FAILURE_LOG="$ROOT_DIR/var/artifacts/logs/antmaze-v11-route-gate-30199417.err"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_generation_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v12"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_admission_audit.json"
FILES=(make_ant_maze_mode_data_v12.py make_ant_maze_mode_data_v11.py make_ant_maze_mode_data.py audit_ant_maze_mode_data_v12.py audit_ant_maze_mode_data.py)

for required in "$PYTHON_BIN" "$PROTOCOL" "$SLURM_SCRIPT" \
  "$ROOT_DIR/var/maze_runtime/venv/bin/python" "$TRAINING_IDENTITY" \
  "$CONTROLLER_RECEIPT" "$CONTROLLER_MODEL" "$V11_ROUTE_IDENTITY" \
  "$V11_FAILURE_LOG"; do
  [[ -e "$required" ]] || { echo "Missing AntMaze v12 prerequisite: $required" >&2; exit 1; }
done
for file in "${FILES[@]}"; do
  [[ -f "$ROOT_DIR/ops/$file" ]] || { echo "Missing AntMaze v12 script: $file" >&2; exit 1; }
done
grep -Fq "ant_v11_admission_train_00 upper fixture failed" "$V11_FAILURE_LOG" || {
  echo "AntMaze v11 failure log does not contain the frozen first-map outcome" >&2
  exit 1
}

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" -m py_compile \
    "$ROOT_DIR/ops/make_ant_maze_mode_data_v12.py" \
    "$ROOT_DIR/ops/audit_ant_maze_mode_data_v12.py" \
    "$ROOT_DIR/src/oat_drgrpo/ant_maze_worker_v12.py" \
    "$ROOT_DIR/src/oat_drgrpo/maze_modebench_worker.py"
  bash -n "$SLURM_SCRIPT"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_ant_maze_route_admission_v12.py"
  echo "[antmaze-v12-route] configuration passed; no v12 route executed"
  exit 0
fi

"$PYTHON_BIN" - "$CONTROLLER_RECEIPT" "$CONTROLLER_MODEL" "$TRAINING_IDENTITY" "$V11_ROUTE_IDENTITY" <<'PYGATE'
import hashlib,json,pathlib,sys
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
receipt=json.load(open(sys.argv[1],encoding="utf-8")); training=json.load(open(sys.argv[3],encoding="utf-8")); route=json.load(open(sys.argv[4],encoding="utf-8"))
if receipt.get("status")!="pass" or receipt.get("decision")!="admitted_to_fresh_maze_route_gate_v11": raise SystemExit("v12 requires the exact passing v11 controller")
if receipt.get("hashes",{}).get("model_sha256")!=digest(sys.argv[2]): raise SystemExit("v11 model hash drift")
if training.get("job_id")!=30198291 or training.get("seed")!=73011: raise SystemExit("v11 training identity drift")
if route.get("job_id")!=30199417 or route.get("map_substitution") is not False: raise SystemExit("v11 failed-route identity drift")
PYGATE

for fresh in "$IDENTITY" "$DATA_ROOT" "$AUDIT"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh AntMaze v12 target required: $fresh" >&2; exit 1; }
done
SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"
  rmdir "$staging"
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "v12 source snapshot mismatch" >&2; exit 1; }

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-maze-v12-route.XXXXXX")"
for file in "${FILES[@]}"; do cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"; done
cp "$SLURM_SCRIPT" "$OPS_INPUT/admit_ant_maze_modebench_v12.slurm"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_ops_${OPS_HASH}"
if [[ ! -f "$OPS_ROOT/make_ant_maze_mode_data_v12.py" ]]; then
  mv "$OPS_INPUT" "$OPS_ROOT"
else
  find "$OPS_INPUT" -type f -delete
  rmdir "$OPS_INPUT"
fi
[[ "$(hash_tree "$OPS_ROOT")" == "$OPS_HASH" ]] || { echo "v12 ops snapshot mismatch" >&2; exit 1; }

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold --partition=all --account=allcs \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
    "$OPS_ROOT/admit_ant_maze_modebench_v12.slurm"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid AntMaze v12 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$OPS_HASH" "$PROTOCOL" "$0" \
  "$SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" "$CONTROLLER_RECEIPT" \
  "$CONTROLLER_MODEL" "$TRAINING_IDENTITY" "$V11_ROUTE_IDENTITY" "$V11_FAILURE_LOG" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode("ascii")).hexdigest()
path=pathlib.Path(sys.argv[1])
executor={
  "targeting_version":"initial-grid-cumulative-v12",
  "worker_source_sha256":digest(sys.argv[7]),
  "controller_receipt_sha256":digest(sys.argv[8]),
  "controller_model_sha256":digest(sys.argv[9]),
  "controller_training_identity_sha256":digest(sys.argv[10]),
  "waypoint_distance":4.0,
  "waypoint_success_threshold":0.45,
}
payload={
  "schema_version":"ant-maze-v12-route-generation-identity-v1",
  "job_id":int(sys.argv[2]),
  "source_hash":sys.argv[3],
  "execution_hash":sys.argv[4],
  "protocol_sha256":digest(sys.argv[5]),
  "launcher_sha256":digest(sys.argv[6]),
  **executor,
  "executor_identity_sha256":canonical(executor),
  "v11_route_identity_sha256":digest(sys.argv[11]),
  "v11_failure_log_sha256":digest(sys.argv[12]),
  "v11_failed_route_job_id":30199417,
  "controller_training_job_id":30198291,
  "map_count":12,
  "map_size":11,
  "reset_seed_base":107300,
  "simple_route_fixtures":24,
  "known_regression_fixtures":1,
  "fresh_simple_route_fixtures":23,
  "perturbation_replays":2400,
  "language_model_sampling":False,
  "map_substitution":False,
  "route_substitution":False,
  "reused_exact_v11_slate":True,
}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID

record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=all' 'NumCPUs=4' 'MinMemoryNode=32G' 'TimeLimit=02:00:00'; do
  [[ "$record" == *"$required"* ]] || { echo "Held Ant v12 job missing $required" >&2; exit 1; }
done
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[antmaze-v12-route] released admission job $job_id"
echo "[antmaze-v12-route] identity=$IDENTITY"
