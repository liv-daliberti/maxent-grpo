#!/usr/bin/env bash
# Repair only the held partition for the never-started AntMaze v12 route job.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_admission_v12_r1_placement_repair_20260730.md"
BASE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_generation_identity.json"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r1_generation_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v12"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_admission_audit.json"

for required in "$PYTHON_BIN" "$PROTOCOL" "$BASE_IDENTITY"; do
  [[ -f "$required" ]] || { echo "Missing AntMaze v12 r1 prerequisite: $required" >&2; exit 1; }
done
SOURCE_HASH="$(jq -er '.source_hash' "$BASE_IDENTITY")"
OPS_HASH="$(jq -er '.execution_hash' "$BASE_IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_ops_${OPS_HASH}"
SLURM_SCRIPT="$OPS_ROOT/admit_ant_maze_modebench_v12.slurm"

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

[[ -f "$SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" && -f "$SLURM_SCRIPT" ]] || {
  echo "Canceled v12 snapshots are missing" >&2
  exit 1
}
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "Canceled v12 source snapshot drift" >&2; exit 1; }
[[ "$(hash_tree "$OPS_ROOT")" == "$OPS_HASH" ]] || { echo "Canceled v12 ops snapshot drift" >&2; exit 1; }

"$PYTHON_BIN" - "$BASE_IDENTITY" <<'PYBASE'
import json,pathlib,subprocess,sys
identity=json.loads(pathlib.Path(sys.argv[1]).read_text())
if identity.get("job_id")!=30200003 or identity.get("schema_version")!="ant-maze-v12-route-generation-identity-v1": raise SystemExit("canceled v12 identity drift")
record=subprocess.run(["scontrol","show","job","30200003","-o"],check=True,text=True,capture_output=True).stdout
for required in ("JobState=CANCELLED","RunTime=00:00:00","ExitCode=0:0"):
    if required not in record: raise SystemExit(f"canceled v12 job lacks {required}")
PYBASE
[[ ! -e "$DATA_ROOT" && ! -e "$AUDIT" ]] || {
  echo "AntMaze v12 r1 requires no executed data or audit" >&2
  exit 1
}

if [[ "$phase" == config ]]; then
  bash -n "$0"
  bash -n "$SLURM_SCRIPT"
  echo "[antmaze-v12-r1] configuration passed; no route executed"
  exit 0
fi

[[ ! -e "$IDENTITY" ]] || { echo "Fresh AntMaze v12 r1 identity required" >&2; exit 1; }
mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold --partition=all --account=allcs \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
    "$SLURM_SCRIPT"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid AntMaze v12 r1 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

before="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'RunTime=00:00:00' 'NumCPUs=4' 'MinMemoryNode=32G' 'TimeLimit=02:00:00'; do
  [[ "$before" == *"$required"* ]] || { echo "Held Ant v12 r1 job missing $required" >&2; exit 1; }
done
[[ "$before" == *"OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT"* && "$before" == *"OAT_ZERO_OPS_ROOT=$OPS_ROOT"* && "$before" == *"OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY"* ]] || {
  echo "Held Ant v12 r1 exports differ from the sealed snapshots" >&2
  exit 1
}
scontrol update "JobId=$job_id" Partition=all
scontrol update "JobId=$job_id" Requeue=0
after="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=all' 'Requeue=0' 'RunTime=00:00:00'; do
  [[ "$after" == *"$required"* ]] || { echo "Amended Ant v12 r1 job missing $required" >&2; exit 1; }
done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$BASE_IDENTITY" "$PROTOCOL" "$0" "$before" "$after" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); base_path=pathlib.Path(sys.argv[3]); payload=json.loads(base_path.read_text())
payload.update({
  "job_id":int(sys.argv[2]),
  "launcher_sha256":digest(sys.argv[5]),
  "placement_repair_protocol_sha256":digest(sys.argv[4]),
  "canceled_v12_identity_sha256":digest(base_path),
  "canceled_v12_job_id":30200003,
  "placement_repair_attempt":"held_partition_all_r1",
  "placement_only":True,
  "scientific_change":False,
  "held_scheduler_record_before":sys.argv[6],
  "held_scheduler_record_after":sys.argv[7],
})
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID
scontrol release "$job_id"
trap - EXIT
echo "[antmaze-v12-r1] released admission job $job_id"
echo "[antmaze-v12-r1] identity=$IDENTITY"
