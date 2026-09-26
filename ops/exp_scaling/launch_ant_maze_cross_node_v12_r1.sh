#!/usr/bin/env bash
# Freeze and submit the executor-binding-only repair of the AntMaze v12 three-node gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_v12_cross_node_audit_r1_controller_binding_repair_20260730.md"
EXPORTER="$ROOT_DIR/ops/export_ant_maze_v12_cross_node_specs.py"
AUDITOR="$ROOT_DIR/ops/audit_ant_maze_cross_node_v12.py"
AGGREGATOR="$ROOT_DIR/ops/aggregate_ant_maze_cross_node_v12.py"
BASE_EXPORTER="$ROOT_DIR/ops/export_ant_maze_v10_cross_node_specs.py"
BASE_AUDITOR="$ROOT_DIR/ops/audit_ant_maze_cross_node_v10.py"
BASE_AGGREGATOR="$ROOT_DIR/ops/aggregate_ant_maze_cross_node_v10.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/audit_ant_maze_cross_node_v12_r1.slurm"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v12"
ROUTE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r4_generation_identity.json"
ROUTE_AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_admission_audit.json"
FAILED_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_v12_cross_node_v1_identity.json"
FAILED_LOG="$ROOT_DIR/var/artifacts/logs/antmaze-v12-crossnode-30200738.err"
SPECS="$ROOT_DIR/var/artifacts/ant_maze_v12_cross_node_specs_v1_r1.json"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_v12_cross_node_v1_r1_identity.json"
AGGREGATE="$ROOT_DIR/var/artifacts/ant_maze_v12_cross_node_v1_r1.json"
PREFIX="$ROOT_DIR/var/artifacts/ant_maze_v12_cross_node_v1_r1_replica"
for required in "$PYTHON_BIN" "$WORKER_PYTHON" "$PROTOCOL" "$EXPORTER" "$AUDITOR" "$AGGREGATOR" "$BASE_EXPORTER" "$BASE_AUDITOR" "$BASE_AGGREGATOR" "$SLURM_SCRIPT"; do
  [[ -f "$required" ]] || { echo "Missing Ant v12-r1 prerequisite: $required" >&2; exit 1; }
done
hash_tree() { local tree="$1"; (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }
if [[ "$phase" == config ]]; then
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m py_compile "$EXPORTER" "$AUDITOR" "$AGGREGATOR" "$BASE_EXPORTER" "$BASE_AUDITOR" "$BASE_AGGREGATOR"
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$EXPORTER" --help >/dev/null
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$WORKER_PYTHON" "$AUDITOR" --help >/dev/null
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$AGGREGATOR" --help >/dev/null
  bash -n "$SLURM_SCRIPT"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q "$ROOT_DIR/tests/test_ant_maze_cross_node_v10.py" "$ROOT_DIR/tests/test_ant_maze_cross_node_v12.py" "$ROOT_DIR/tests/test_ant_maze_cross_node_v12_r1.py" "$ROOT_DIR/tests/test_ant_maze_route_admission_v12.py"
  echo "[ant-v12-cross-node-r1] configuration passed; no route executed"
  exit 0
fi
for required in "$DATA_ROOT/identity.json" "$ROUTE_IDENTITY" "$ROUTE_AUDIT" "$FAILED_IDENTITY" "$FAILED_LOG"; do [[ -f "$required" ]] || { echo "Missing Ant v12-r1 artifact: $required" >&2; exit 1; }; done
"$PYTHON_BIN" - "$ROUTE_IDENTITY" "$ROUTE_AUDIT" "$FAILED_IDENTITY" "$FAILED_LOG" <<'PYANT'
import json,pathlib,sys
route=json.load(open(sys.argv[1])); audit=json.load(open(sys.argv[2])); failed=json.load(open(sys.argv[3])); log=pathlib.Path(sys.argv[4]).read_text(errors="replace")
if route.get("job_id")!=30200585 or route.get("map_substitution") is not False or route.get("language_model_sampling") is not False: raise SystemExit("wrong admitted Ant v12 route identity")
if audit.get("status")!="pass" or audit.get("decision")!="admitted_to_v12_cross_node_route_determinism_gate": raise SystemExit("v12 route audit not admitted")
if failed.get("job_id")!=30200738 or "binds the wrong controller" not in log: raise SystemExit("r1 lacks immutable cross-node failure")
PYANT
for fresh in "$SPECS" "$IDENTITY" "$AGGREGATE" "${PREFIX}_0.json" "${PREFIX}_1.json" "${PREFIX}_2.json"; do [[ ! -e "$fresh" ]] || { echo "Fresh Ant v12-r1 target required: $fresh" >&2; exit 1; }; done
SOURCE_HASH="$(jq -er '.source_hash' "$ROUTE_IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${SOURCE_HASH}/src"
[[ -f "$SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" && "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "Exact v12-r4 source snapshot mismatch" >&2; exit 1; }
EXEC_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-v12-cross-node-r1.XXXXXX")"
for file in "$EXPORTER" "$AUDITOR" "$AGGREGATOR" "$BASE_EXPORTER" "$BASE_AUDITOR" "$BASE_AGGREGATOR" "$SLURM_SCRIPT" "$PROTOCOL"; do cp "$file" "$EXEC_INPUT/$(basename "$file")"; done
EXEC_HASH="$(hash_tree "$EXEC_INPUT")"
EXEC_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_v12_cross_node_r1_ops_${EXEC_HASH}"
if [[ ! -f "$EXEC_ROOT/audit_ant_maze_cross_node_v12.py" ]]; then mv "$EXEC_INPUT" "$EXEC_ROOT"; else find "$EXEC_INPUT" -type f -delete; rmdir "$EXEC_INPUT"; fi
[[ "$(hash_tree "$EXEC_ROOT")" == "$EXEC_HASH" ]] || { echo "v12-r1 execution snapshot mismatch" >&2; exit 1; }
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$EXEC_ROOT/export_ant_maze_v12_cross_node_specs.py" --data-root "$DATA_ROOT" --route-audit "$ROUTE_AUDIT" --route-identity "$ROUTE_IDENTITY" --output "$SPECS"
[[ "$(jq -r .controller_receipt_sha256 "$SPECS")" == "$(jq -r .executor_identity_sha256 "$ROUTE_IDENTITY")" ]] || { echo "v12-r1 exporter did not bind executor identity" >&2; exit 1; }
SPECS_HASH="$(sha256sum "$SPECS" | cut -d' ' -f1)"
mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold --export="ALL,PYTHONDONTWRITEBYTECODE=1,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXEC_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXEC_HASH,OAT_ZERO_ROUTE_IDENTITY=$ROUTE_IDENTITY,OAT_ZERO_CROSS_NODE_SPECS=$SPECS" "$EXEC_ROOT/audit_ant_maze_cross_node_v12_r1.slurm")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Ant v12-r1 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }; trap cleanup EXIT
record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'NumNodes=3' 'NumTasks=3' "OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT" "OAT_ZERO_CROSS_NODE_SPECS=$SPECS"; do [[ "$record" == *"$required"* ]] || { echo "Held v12-r1 job missing $required" >&2; exit 1; }; done
"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXEC_HASH" "$PROTOCOL" "$SPECS_HASH" "$ROUTE_IDENTITY" "$ROUTE_AUDIT" "$FAILED_IDENTITY" "$FAILED_LOG" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={"schema_version":"ant-maze-v12-cross-node-r1-identity-v1","job_id":int(sys.argv[2]),"source_hash":sys.argv[3],"execution_hash":sys.argv[4],"protocol_sha256":digest(sys.argv[5]),"spec_export_sha256":sys.argv[6],"route_identity_sha256":digest(sys.argv[7]),"route_audit_sha256":digest(sys.argv[8]),"failed_job_id":30200738,"failed_identity_sha256":digest(sys.argv[9]),"failed_error_log_sha256":digest(sys.argv[10]),"repair_attempt":"executor_identity_export_binding_only_r1","node_count":3,"maps":12,"routes_per_map":2,"repetitions_per_node":3,"expected_execution_count":216,"language_model_sampling":False,"map_substitution":False,"route_substitution":False,"scientific_change":False}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-v12-cross-node-r1] released three-node audit job $job_id"
echo "[ant-v12-cross-node-r1] identity=$IDENTITY"
