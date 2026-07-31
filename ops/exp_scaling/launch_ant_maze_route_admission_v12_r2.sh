#!/usr/bin/env bash
# Repair only the pre-execution v12 generator import error; route slate and worker stay sealed.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_admission_v12_r2_import_repair_20260730.md"
BASE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_generation_identity.json"
R1_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r1_generation_identity.json"
R1_ERROR="$ROOT_DIR/var/artifacts/logs/antmaze-v12-route-gate-30200026.err"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r2_generation_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v12"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_admission_audit.json"
SOURCE_HASH="7c073fec5485f22f932d61cf795594ea8dfc546c84c9d56cfe5a4828047d97d1"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${SOURCE_HASH}/src"
FILES=(make_ant_maze_mode_data_v12.py make_ant_maze_mode_data.py audit_ant_maze_mode_data_v12.py audit_ant_maze_mode_data.py)

for required in "$PYTHON_BIN" "$PROTOCOL" "$BASE_IDENTITY" "$R1_IDENTITY" "$R1_ERROR"; do
  [[ -f "$required" ]] || { echo "Missing AntMaze v12 r2 prerequisite: $required" >&2; exit 1; }
done
for file in "${FILES[@]}"; do
  [[ -f "$ROOT_DIR/ops/$file" ]] || { echo "Missing AntMaze v12 r2 script: $file" >&2; exit 1; }
done
grep -Fq "Ant v11 route identity schema mismatch" "$R1_ERROR" || {
  echo "AntMaze v12 r1 failure signature is absent" >&2
  exit 1
}
[[ ! -e "$DATA_ROOT" && ! -e "$AUDIT" ]] || {
  echo "AntMaze v12 r2 requires no executed data or audit" >&2
  exit 1
}

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

"$PYTHON_BIN" - "$BASE_IDENTITY" "$R1_IDENTITY" "$SOURCE_HASH" <<'PY'
import json, pathlib, sys
base = json.loads(pathlib.Path(sys.argv[1]).read_text())
r1 = json.loads(pathlib.Path(sys.argv[2]).read_text())
source_hash = sys.argv[3]
for payload, job_id in ((base, 30200003), (r1, 30200026)):
    if payload.get("job_id") != job_id:
        raise SystemExit(f"AntMaze identity job drift: expected {job_id}")
    if payload.get("schema_version") != "ant-maze-v12-route-generation-identity-v1":
        raise SystemExit("AntMaze v12 identity schema drift")
    if payload.get("source_hash") != source_hash:
        raise SystemExit("AntMaze v12 source hash drift")
    if payload.get("targeting_version") != "initial-grid-cumulative-v12":
        raise SystemExit("AntMaze v12 executor drift")
PY

[[ -f "$SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" ]] || {
  echo "Sealed AntMaze v12 source snapshot is missing" >&2
  exit 1
}
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || {
  echo "Sealed AntMaze v12 source snapshot drift" >&2
  exit 1
}

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-maze-v12-r2.XXXXXX")"
for file in "${FILES[@]}"; do cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"; done
cp "$ROOT_DIR/ops/slurm/admit_ant_maze_modebench_v12.slurm" "$OPS_INPUT/admit_ant_maze_modebench_v12.slurm"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_ops_${OPS_HASH}"
if [[ ! -f "$OPS_ROOT/make_ant_maze_mode_data_v12.py" ]]; then
  mv "$OPS_INPUT" "$OPS_ROOT"
else
  find "$OPS_INPUT" -type f -delete
  rmdir "$OPS_INPUT"
fi
[[ "$(hash_tree "$OPS_ROOT")" == "$OPS_HASH" ]] || { echo "AntMaze v12 r2 ops snapshot drift" >&2; exit 1; }
SLURM_SCRIPT="$OPS_ROOT/admit_ant_maze_modebench_v12.slurm"

PYTHONPATH="$OPS_ROOT:$SOURCE_ROOT" "$PYTHON_BIN" -m py_compile \
  "$OPS_ROOT/make_ant_maze_mode_data_v12.py" \
  "$OPS_ROOT/make_ant_maze_mode_data.py" \
  "$OPS_ROOT/audit_ant_maze_mode_data_v12.py" \
  "$OPS_ROOT/audit_ant_maze_mode_data.py" \
  "$SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" \
  "$SOURCE_ROOT/oat_drgrpo/maze_modebench_worker.py"
bash -n "$SLURM_SCRIPT"
PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
  "$ROOT_DIR/tests/test_ant_maze_route_admission_v12.py"

if [[ "$phase" == config ]]; then
  echo "[antmaze-v12-r2] configuration passed; no route executed"
  echo "[antmaze-v12-r2] sealed_source=$SOURCE_HASH fixed_ops=$OPS_HASH"
  exit 0
fi

[[ ! -e "$IDENTITY" ]] || { echo "Fresh AntMaze v12 r2 identity required" >&2; exit 1; }
mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold --partition=all --account=allcs \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
    "$SLURM_SCRIPT"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid AntMaze v12 r2 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

before="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'RunTime=00:00:00' 'NumCPUs=4' 'MinMemoryNode=32G' 'TimeLimit=02:00:00'; do
  [[ "$before" == *"$required"* ]] || { echo "Held Ant v12 r2 job missing $required" >&2; exit 1; }
done
[[ "$before" == *"OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT"* && "$before" == *"OAT_ZERO_OPS_ROOT=$OPS_ROOT"* && "$before" == *"OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY"* ]] || {
  echo "Held Ant v12 r2 exports differ from the sealed snapshots" >&2
  exit 1
}
scontrol update "JobId=$job_id" Partition=all
scontrol update "JobId=$job_id" Requeue=0
after="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=all' 'Requeue=0' 'RunTime=00:00:00'; do
  [[ "$after" == *"$required"* ]] || { echo "Amended Ant v12 r2 job missing $required" >&2; exit 1; }
done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$BASE_IDENTITY" "$R1_IDENTITY" "$R1_ERROR" "$PROTOCOL" "$0" "$OPS_HASH" "$before" "$after" <<'PYID'
import hashlib, json, os, pathlib, sys, tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path = pathlib.Path(sys.argv[1])
base_path = pathlib.Path(sys.argv[3])
payload = json.loads(base_path.read_text())
payload.update({
    "job_id": int(sys.argv[2]),
    "execution_hash": sys.argv[8],
    "launcher_sha256": digest(sys.argv[7]),
    "import_repair_protocol_sha256": digest(sys.argv[6]),
    "base_v12_identity_sha256": digest(base_path),
    "r1_identity_sha256": digest(sys.argv[4]),
    "r1_error_log_sha256": digest(sys.argv[5]),
    "r1_failed_job_id": 30200026,
    "repair_attempt": "pre-execution-import-only-r2",
    "import_only": True,
    "scientific_change": False,
    "map_substitution": False,
    "route_substitution": False,
    "held_scheduler_record_before": sys.argv[9],
    "held_scheduler_record_after": sys.argv[10],
})
path.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PYID
scontrol release "$job_id"
trap - EXIT
echo "[antmaze-v12-r2] released admission job $job_id"
echo "[antmaze-v12-r2] identity=$IDENTITY"
