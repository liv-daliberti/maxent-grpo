#!/usr/bin/env bash
# Freeze and submit the AntMaze v10 fresh-map route admission gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_admission_v10_20260729.md"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/admit_ant_maze_modebench_v10.slurm"
TRAINING_IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v10_identity.json"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v10_generation_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v10"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v10_admission_audit.json"
RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v10.evaluation.json"
MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v10.zip"
FILES=(
  make_ant_maze_mode_data_v10.py
  make_ant_maze_mode_data.py
  audit_ant_maze_mode_data_v10.py
  audit_ant_maze_mode_data.py
)
phase="${1:-}"

case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac

for required in "$PYTHON_BIN" "$PROTOCOL" "$SLURM_SCRIPT" \
  "$ROOT_DIR/var/maze_runtime/venv/bin/python" "$TRAINING_IDENTITY"; do
  [[ -e "$required" ]] || {
    echo "Missing AntMaze v10 route prerequisite: $required" >&2
    exit 1
  }
done
for file in "${FILES[@]}"; do
  [[ -f "$ROOT_DIR/ops/$file" ]] || {
    echo "Missing AntMaze v10 route script: $file" >&2
    exit 1
  }
done

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" -m py_compile \
    "$ROOT_DIR/ops/make_ant_maze_mode_data.py" \
    "$ROOT_DIR/ops/make_ant_maze_mode_data_v10.py" \
    "$ROOT_DIR/ops/audit_ant_maze_mode_data.py" \
    "$ROOT_DIR/ops/audit_ant_maze_mode_data_v10.py" \
    "$ROOT_DIR/src/oat_drgrpo/ant_maze_worker_v10.py" \
    "$ROOT_DIR/src/oat_drgrpo/maze_modebench_worker.py"
  bash -n "$SLURM_SCRIPT"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_ant_maze_route_admission.py" \
    "$ROOT_DIR/tests/test_ant_maze_route_admission_v10.py" \
    "$ROOT_DIR/tests/test_ant_maze_worker_v5.py" \
    "$ROOT_DIR/tests/test_ant_maze_worker_v8.py"
  echo "[antmaze-v10-route] configuration passed; no v10 route map executed"
  exit 0
fi

for required in "$RECEIPT" "$MODEL"; do
  [[ -f "$required" ]] || {
    echo "Missing admitted AntMaze v10 controller artifact: $required" >&2
    exit 1
  }
done
"$PYTHON_BIN" - "$RECEIPT" "$MODEL" "$TRAINING_IDENTITY" <<'PYGATE'
import hashlib, json, pathlib, sys
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
receipt=json.load(open(sys.argv[1],encoding="utf-8"))
identity=json.load(open(sys.argv[3],encoding="utf-8"))
if receipt.get("status") != "pass" or receipt.get("decision") != "admitted_to_fresh_maze_route_gate_v10":
    raise SystemExit("AntMaze v10 route gate requires the exact admitted controller")
if receipt.get("seed") != 73010 or receipt.get("timesteps") != 2_000_000:
    raise SystemExit("AntMaze v10 controller receipt differs from the frozen run")
checks=receipt.get("checks",{})
if not checks or not all(checks.values()):
    raise SystemExit("AntMaze v10 controller receipt has a failed or absent check")
if receipt.get("hashes",{}).get("model_sha256") != digest(sys.argv[2]):
    raise SystemExit("AntMaze v10 receipt does not bind the controller model")
if identity.get("job_id") != 30193111 or identity.get("seed") != 73010:
    raise SystemExit("AntMaze v10 training identity differs from the frozen run")
PYGATE

for fresh in "$IDENTITY" "$DATA_ROOT" "$AUDIT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh AntMaze v10 route target required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v10_route_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"
  rmdir "$staging"
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || exit 1

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-maze-v10-route.XXXXXX")"
for file in "${FILES[@]}"; do
  cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"
done
cp "$SLURM_SCRIPT" "$OPS_INPUT/admit_ant_maze_modebench_v10.slurm"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v10_route_ops_${OPS_HASH}"
if [[ ! -f "$OPS_ROOT/make_ant_maze_mode_data_v10.py" ]]; then
  mv "$OPS_INPUT" "$OPS_ROOT"
else
  find "$OPS_INPUT" -type f -delete
  rmdir "$OPS_INPUT"
fi
[[ "$(hash_tree "$OPS_ROOT")" == "$OPS_HASH" ]] || exit 1

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
    "$OPS_ROOT/admit_ant_maze_modebench_v10.slurm"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid AntMaze v10 route job ID: $job_id" >&2
  exit 1
}

cleanup() {
  local status="$?"
  trap - EXIT
  scancel "$job_id" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
job_record="$(scontrol show job "$job_id" -o)"
for required in "JobState=PENDING" "Reason=JobHeldUser" \
  "OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT" "OAT_ZERO_OPS_ROOT=$OPS_ROOT" \
  "OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY"; do
  [[ "$job_record" == *"$required"* ]] || {
    echo "Held v10 route job missing $required" >&2
    exit 1
  }
done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$OPS_HASH" \
  "$PROTOCOL" "$0" "$RECEIPT" "$MODEL" "$TRAINING_IDENTITY" <<'PYID'
import hashlib, json, os, pathlib, sys, tempfile
def digest(raw): return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema_version":"ant-maze-v10-route-generation-identity-v1",
 "job_id":int(sys.argv[2]),"source_hash":sys.argv[3],"execution_hash":sys.argv[4],
 "protocol_sha256":digest(sys.argv[5]),"launcher_sha256":digest(sys.argv[6]),
 "controller_receipt_sha256":digest(sys.argv[7]),
 "controller_model_sha256":digest(sys.argv[8]),
 "controller_training_identity_sha256":digest(sys.argv[9]),
 "controller_training_job_id":30193111,"map_count":12,"map_size":11,
 "reset_seed_base":107300,"real_route_executions":48,
 "perturbation_replays":2400,"language_model_sampling":False,
 "map_substitution":False,"frozen_before_controller_outcome":True}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID

scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[antmaze-v10-route] released admission job $job_id"
echo "[antmaze-v10-route] identity=$IDENTITY"
