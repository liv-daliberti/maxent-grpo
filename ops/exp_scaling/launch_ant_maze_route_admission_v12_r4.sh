#!/usr/bin/env bash
# Retry v12 with only cold-start timeout boundaries extended.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_admission_v12_r4_cold_start_timeout_repair_20260730.md"
R3_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r3_generation_identity.json"
R3_ERROR="$ROOT_DIR/var/artifacts/logs/antmaze-v12-route-gate-30200503.err"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r4_generation_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v12"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_admission_audit.json"
OLD_SOURCE_HASH="7f2d09ffb067a75f173541da1dbc3d56f165a9da5b687f6bff47a3b3bf7d4549"
OLD_SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${OLD_SOURCE_HASH}/src"
OLD_OPS_HASH="fa0fa619ed548019815096d4150ee083761a22a5ad8bd38b95d4aef1a2b4d394"
OLD_OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_ops_${OLD_OPS_HASH}"

for required in "$PYTHON_BIN" "$PROTOCOL" "$R3_IDENTITY" "$R3_ERROR" \
  "$OLD_SOURCE_ROOT/oat_drgrpo/maze_modebench_worker.py" \
  "$OLD_SOURCE_ROOT/oat_drgrpo/maze_modebench_process.py" \
  "$OLD_SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" \
  "$OLD_OPS_ROOT/make_ant_maze_mode_data_v12.py" \
  "$ROOT_DIR/src/oat_drgrpo/maze_modebench_worker.py" \
  "$ROOT_DIR/src/oat_drgrpo/maze_modebench_process.py" \
  "$ROOT_DIR/ops/make_ant_maze_mode_data_v12.py"; do
  [[ -f "$required" ]] || { echo "Missing AntMaze v12 r4 prerequisite: $required" >&2; exit 1; }
done
grep -Fq "ant_v12_admission_train_00 upper fixture failed" "$R3_ERROR" || {
  echo "AntMaze v12 r3 failure signature is absent" >&2
  exit 1
}
[[ ! -e "$DATA_ROOT" && ! -e "$AUDIT" ]] || {
  echo "AntMaze v12 r4 requires no executed data or audit" >&2
  exit 1
}

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

[[ "$(hash_tree "$OLD_SOURCE_ROOT")" == "$OLD_SOURCE_HASH" ]] || { echo "Sealed r3 source drift" >&2; exit 1; }
[[ "$(hash_tree "$OLD_OPS_ROOT")" == "$OLD_OPS_HASH" ]] || { echo "Sealed r3 ops drift" >&2; exit 1; }
"$PYTHON_BIN" - "$R3_IDENTITY" <<'PYR3'
import json, pathlib, subprocess, sys
identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
if identity.get("job_id") != 30200503 or identity.get("source_hash") != "7f2d09ffb067a75f173541da1dbc3d56f165a9da5b687f6bff47a3b3bf7d4549":
    raise SystemExit("AntMaze v12 r3 identity drift")
record = subprocess.run(
    ["sacct", "-X", "-j", "30200503", "--format=State,ExitCode,Elapsed", "-n", "-P"],
    check=True, text=True, capture_output=True,
).stdout
if "FAILED|1:0|00:00:27" not in record:
    raise SystemExit("AntMaze v12 r3 scheduler outcome drift")
PYR3

SOURCE_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-maze-v12-r4-src.XXXXXX")"
mkdir -p "$SOURCE_INPUT/src"
cp -a "$OLD_SOURCE_ROOT/." "$SOURCE_INPUT/src/"
cp "$ROOT_DIR/src/oat_drgrpo/maze_modebench_worker.py" \
  "$SOURCE_INPUT/src/oat_drgrpo/maze_modebench_worker.py"
cp "$ROOT_DIR/src/oat_drgrpo/maze_modebench_process.py" \
  "$SOURCE_INPUT/src/oat_drgrpo/maze_modebench_process.py"
diff -q "$OLD_SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" \
  "$SOURCE_INPUT/src/oat_drgrpo/ant_maze_worker_v12.py" >/dev/null
SOURCE_HASH="$(hash_tree "$SOURCE_INPUT/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/maze_modebench_worker.py" ]]; then
  mkdir -p "$SOURCE_PARENT"; mv "$SOURCE_INPUT/src" "$SOURCE_ROOT"; rmdir "$SOURCE_INPUT"
else
  find "$SOURCE_INPUT" -type f -delete; find "$SOURCE_INPUT" -depth -type d -empty -delete
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "R4 source snapshot mismatch" >&2; exit 1; }

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-maze-v12-r4-ops.XXXXXX")"
cp -a "$OLD_OPS_ROOT/." "$OPS_INPUT/"
cp "$ROOT_DIR/ops/make_ant_maze_mode_data_v12.py" "$OPS_INPUT/make_ant_maze_mode_data_v12.py"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_ops_${OPS_HASH}"
if [[ ! -f "$OPS_ROOT/make_ant_maze_mode_data_v12.py" ]]; then mv "$OPS_INPUT" "$OPS_ROOT"; else find "$OPS_INPUT" -type f -delete; find "$OPS_INPUT" -depth -type d -empty -delete; fi
[[ "$(hash_tree "$OPS_ROOT")" == "$OPS_HASH" ]] || { echo "R4 ops snapshot mismatch" >&2; exit 1; }
SLURM_SCRIPT="$OPS_ROOT/admit_ant_maze_modebench_v12.slurm"

"$PYTHON_BIN" -B - "$OPS_ROOT/make_ant_maze_mode_data_v12.py" "$SOURCE_ROOT/oat_drgrpo/maze_modebench_worker.py" "$SOURCE_ROOT/oat_drgrpo/maze_modebench_process.py" <<'PYSYNTAX'
import pathlib, sys
for name in sys.argv[1:]:
    path = pathlib.Path(name); compile(path.read_text(encoding="utf-8"), str(path), "exec")
PYSYNTAX
bash -n "$SLURM_SCRIPT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q "$ROOT_DIR/tests/test_ant_maze_route_admission_v12.py"

if [[ "$phase" == config ]]; then
  echo "[antmaze-v12-r4] configuration passed; no route executed"
  echo "[antmaze-v12-r4] source=$SOURCE_HASH ops=$OPS_HASH"
  exit 0
fi

[[ ! -e "$IDENTITY" ]] || { echo "Fresh AntMaze v12 r4 identity required" >&2; exit 1; }
mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold --partition=all --account=allcs \
  --export="ALL,PYTHONDONTWRITEBYTECODE=1,OAT_ZERO_MAZE_WORKER_TIMEOUT_SECONDS=80,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
  "$SLURM_SCRIPT")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid AntMaze v12 r4 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT
before="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'RunTime=00:00:00' 'NumCPUs=4' 'MinMemoryNode=32G' 'TimeLimit=02:00:00'; do [[ "$before" == *"$required"* ]] || { echo "Held r4 job missing $required" >&2; exit 1; }; done
scontrol update "JobId=$job_id" Partition=all
scontrol update "JobId=$job_id" Requeue=0
after="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=all' 'Requeue=0' 'RunTime=00:00:00'; do [[ "$after" == *"$required"* ]] || { echo "Amended r4 job missing $required" >&2; exit 1; }; done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$R3_IDENTITY" "$R3_ERROR" "$PROTOCOL" "$0" "$SOURCE_HASH" "$OPS_HASH" "$before" "$after" <<'PYID'
import hashlib, json, os, pathlib, sys, tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); r3=pathlib.Path(sys.argv[3]); payload=json.loads(r3.read_text())
payload.update({"job_id":int(sys.argv[2]),"source_hash":sys.argv[7],"execution_hash":sys.argv[8],"launcher_sha256":digest(sys.argv[6]),"cold_start_timeout_repair_protocol_sha256":digest(sys.argv[5]),"r3_identity_sha256":digest(r3),"r3_error_log_sha256":digest(sys.argv[4]),"r3_failed_job_id":30200503,"repair_attempt":"cold-start-timeout-only-r4","external_timeout_seconds":90.0,"worker_timeout_seconds":80.0,"simulator_horizon_unchanged":True,"scientific_change":False,"map_substitution":False,"route_substitution":False,"held_scheduler_record_before":sys.argv[9],"held_scheduler_record_after":sys.argv[10]})
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID
scontrol release "$job_id"
trap - EXIT
echo "[antmaze-v12-r4] released admission job $job_id"
echo "[antmaze-v12-r4] identity=$IDENTITY"
