#!/usr/bin/env bash
# Freeze and submit the conditional three-node AntMaze v9 exact-slate replay.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_v9_cross_node_audit_20260729.md"
EXPORTER="$ROOT_DIR/ops/export_ant_maze_v9_cross_node_specs.py"
AUDITOR="$ROOT_DIR/ops/audit_ant_maze_cross_node_v9.py"
AGGREGATOR="$ROOT_DIR/ops/aggregate_ant_maze_cross_node_v9.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/audit_ant_maze_cross_node_v9.slurm"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v9"
ROUTE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v9_generation_identity.json"
ROUTE_AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v9_admission_audit.json"
CONTROLLER_RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v9.evaluation.json"
CONTROLLER_MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v9.zip"
SPECS="$ROOT_DIR/var/artifacts/ant_maze_v9_cross_node_specs_v1.json"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_v9_cross_node_v1_identity.json"
AGGREGATE="$ROOT_DIR/var/artifacts/ant_maze_v9_cross_node_v1.json"
PREFIX="$ROOT_DIR/var/artifacts/ant_maze_v9_cross_node_v1_replica"
phase="${1:-}"

case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac

for required in "$PYTHON_BIN" "$WORKER_PYTHON" "$PROTOCOL" "$EXPORTER" \
  "$AUDITOR" "$AGGREGATOR" "$SLURM_SCRIPT"; do
  [[ -f "$required" ]] || {
    echo "Missing Ant v9 cross-node prerequisite: $required" >&2
    exit 1
  }
done

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" -m py_compile "$EXPORTER" "$AUDITOR" "$AGGREGATOR"
  "$PYTHON_BIN" "$EXPORTER" --help >/dev/null
  PYTHONPATH="$ROOT_DIR/src" "$WORKER_PYTHON" "$AUDITOR" --help >/dev/null
  "$PYTHON_BIN" "$AGGREGATOR" --help >/dev/null
  bash -n "$SLURM_SCRIPT"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_ant_maze_route_admission_v9.py" \
    "$ROOT_DIR/tests/test_ant_maze_cross_node_v9.py"
  echo "[ant-v9-cross-node] configuration passed; no route executed"
  exit 0
fi

for required in "$DATA_ROOT/identity.json" "$ROUTE_IDENTITY" "$ROUTE_AUDIT" \
  "$CONTROLLER_RECEIPT" "$CONTROLLER_MODEL"; do
  [[ -f "$required" ]] || {
    echo "Missing admitted Ant v9 route artifact: $required" >&2
    exit 1
  }
done
"$PYTHON_BIN" - "$ROUTE_IDENTITY" "$ROUTE_AUDIT" "$CONTROLLER_RECEIPT" \
  "$CONTROLLER_MODEL" <<'PYANT'
import hashlib, json, pathlib, sys
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
identity=json.load(open(sys.argv[1],encoding="utf-8"))
audit=json.load(open(sys.argv[2],encoding="utf-8"))
if identity.get("schema_version") != "ant-maze-v9-route-generation-identity-v1":
    raise SystemExit("wrong Ant v9 route identity")
if identity.get("map_substitution") is not False or identity.get("language_model_sampling") is not False:
    raise SystemExit("Ant v9 route identity violates the information boundary")
if identity.get("controller_receipt_sha256") != digest(sys.argv[3]):
    raise SystemExit("Ant v9 route identity receipt hash mismatch")
if identity.get("controller_model_sha256") != digest(sys.argv[4]):
    raise SystemExit("Ant v9 route identity model hash mismatch")
if audit.get("status") != "pass" or audit.get("decision") != "admitted_to_v9_cross_node_route_determinism_gate":
    raise SystemExit("Ant v9 route audit is not admitted")
if audit.get("real_simulator_replays") != 24 or audit.get("perturbation_replays") != 2400:
    raise SystemExit("Ant v9 route audit replay count mismatch")
PYANT

for fresh in "$SPECS" "$IDENTITY" "$AGGREGATE" \
  "${PREFIX}_0.json" "${PREFIX}_1.json" "${PREFIX}_2.json"; do
  [[ ! -e "$fresh" ]] || {
    echo "Fresh Ant v9 cross-node target required: $fresh" >&2
    exit 1
  }
done

SOURCE_HASH="$($PYTHON_BIN -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_hash"])' "$ROUTE_IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v9_route_${SOURCE_HASH}/src"
[[ -f "$SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v9.py" ]] || {
  echo "Exact Ant v9 route source snapshot is unavailable" >&2
  exit 1
}
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || {
  echo "Exact Ant v9 route source snapshot hash mismatch" >&2
  exit 1
}

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-v9-cross-node.XXXXXX")"
cp "$EXPORTER" "$EXECUTION_INPUT/export_ant_maze_v9_cross_node_specs.py"
cp "$AUDITOR" "$EXECUTION_INPUT/audit_ant_maze_cross_node_v9.py"
cp "$AGGREGATOR" "$EXECUTION_INPUT/aggregate_ant_maze_cross_node_v9.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/audit_ant_maze_cross_node_v9.slurm"
cp "$PROTOCOL" "$EXECUTION_INPUT/ant_maze_v9_cross_node_audit_20260729.md"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_v9_cross_node_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/audit_ant_maze_cross_node_v9.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || exit 1

"$PYTHON_BIN" "$EXECUTION_ROOT/export_ant_maze_v9_cross_node_specs.py" \
  --data-root "$DATA_ROOT" \
  --route-audit "$ROUTE_AUDIT" \
  --route-identity "$ROUTE_IDENTITY" \
  --output "$SPECS"
SPECS_HASH="$(sha256sum "$SPECS" | cut -d' ' -f1)"

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH,OAT_ZERO_ROUTE_IDENTITY=$ROUTE_IDENTITY,OAT_ZERO_CROSS_NODE_SPECS=$SPECS" \
  "$EXECUTION_ROOT/audit_ant_maze_cross_node_v9.slurm")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Ant v9 cross-node job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT
job_record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'NumNodes=3' 'NumTasks=3' \
  "OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT" "OAT_ZERO_CROSS_NODE_SPECS=$SPECS"; do
  [[ "$job_record" == *"$required"* ]] || {
    echo "Held Ant v9 cross-node job missing $required" >&2
    exit 1
  }
done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" \
  "$EXECUTION_ROOT/ant_maze_v9_cross_node_audit_20260729.md" "$SPECS_HASH" \
  "$ROUTE_IDENTITY" "$ROUTE_AUDIT" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema_version":"ant-maze-v9-cross-node-identity-v1","job_id":int(sys.argv[2]),
 "source_hash":sys.argv[3],"execution_hash":sys.argv[4],
 "protocol_sha256":digest(sys.argv[5]),"spec_export_sha256":sys.argv[6],
 "route_identity_sha256":digest(sys.argv[7]),"route_audit_sha256":digest(sys.argv[8]),
 "node_count":3,"maps":12,"routes_per_map":2,"repetitions_per_node":3,
 "expected_execution_count":216,"language_model_sampling":False,
 "map_substitution":False}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID

scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-v9-cross-node] released three-node audit job $job_id"
echo "[ant-v9-cross-node] identity=$IDENTITY"
