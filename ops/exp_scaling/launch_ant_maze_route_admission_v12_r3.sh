#!/usr/bin/env bash
# Retry v12 with only the frozen dispatcher-order defect repaired.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_admission_v12_r3_dispatch_repair_20260730.md"
R2_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r2_generation_identity.json"
R2_ERROR="$ROOT_DIR/var/artifacts/logs/antmaze-v12-route-gate-30200147.err"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r3_generation_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v12"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_admission_audit.json"
OLD_SOURCE_HASH="7c073fec5485f22f932d61cf795594ea8dfc546c84c9d56cfe5a4828047d97d1"
OLD_SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${OLD_SOURCE_HASH}/src"

for required in "$PYTHON_BIN" "$PROTOCOL" "$R2_IDENTITY" "$R2_ERROR" \
  "$OLD_SOURCE_ROOT/oat_drgrpo/maze_modebench_worker.py" \
  "$OLD_SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" \
  "$ROOT_DIR/src/oat_drgrpo/maze_modebench_worker.py"; do
  [[ -f "$required" ]] || { echo "Missing AntMaze v12 r3 prerequisite: $required" >&2; exit 1; }
done
grep -Fq "ant_v12_admission_train_00 upper fixture failed" "$R2_ERROR" || {
  echo "AntMaze v12 r2 failure signature is absent" >&2
  exit 1
}
[[ ! -e "$DATA_ROOT" && ! -e "$AUDIT" ]] || {
  echo "AntMaze v12 r3 requires no executed data or audit" >&2
  exit 1
}

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

[[ "$(hash_tree "$OLD_SOURCE_ROOT")" == "$OLD_SOURCE_HASH" ]] || {
  echo "Sealed AntMaze v12 r2 source drift" >&2
  exit 1
}
"$PYTHON_BIN" - "$R2_IDENTITY" <<'PYR2'
import json, pathlib, subprocess, sys
identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
if identity.get("job_id") != 30200147:
    raise SystemExit("AntMaze v12 r2 job identity drift")
if identity.get("source_hash") != "7c073fec5485f22f932d61cf795594ea8dfc546c84c9d56cfe5a4828047d97d1":
    raise SystemExit("AntMaze v12 r2 source identity drift")
if identity.get("schema_version") != "ant-maze-v12-route-generation-identity-v1":
    raise SystemExit("AntMaze v12 r2 schema drift")
record = subprocess.run(
    ["sacct", "-X", "-j", "30200147", "--format=State,ExitCode", "-n", "-P"],
    check=True, text=True, capture_output=True,
).stdout
if "FAILED|1:0" not in record:
    raise SystemExit("AntMaze v12 r2 scheduler outcome drift")
PYR2

SOURCE_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-maze-v12-r3.XXXXXX")"
mkdir -p "$SOURCE_INPUT/src"
cp -a "$OLD_SOURCE_ROOT/." "$SOURCE_INPUT/src/"
cp "$ROOT_DIR/src/oat_drgrpo/maze_modebench_worker.py" \
  "$SOURCE_INPUT/src/oat_drgrpo/maze_modebench_worker.py"
diff -q "$OLD_SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" \
  "$SOURCE_INPUT/src/oat_drgrpo/ant_maze_worker_v12.py" >/dev/null
NEW_SOURCE_HASH="$(hash_tree "$SOURCE_INPUT/src")"
[[ "$NEW_SOURCE_HASH" != "$OLD_SOURCE_HASH" ]] || {
  echo "AntMaze v12 r3 dispatcher repair did not change the source hash" >&2
  exit 1
}
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${NEW_SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/maze_modebench_worker.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  mv "$SOURCE_INPUT/src" "$SOURCE_ROOT"
  rmdir "$SOURCE_INPUT"
else
  find "$SOURCE_INPUT" -type f -delete
  find "$SOURCE_INPUT" -depth -type d -empty -delete
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$NEW_SOURCE_HASH" ]] || {
  echo "AntMaze v12 r3 source snapshot mismatch" >&2
  exit 1
}

OPS_HASH="$(jq -er '.execution_hash' "$R2_IDENTITY")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_ops_${OPS_HASH}"
SLURM_SCRIPT="$OPS_ROOT/admit_ant_maze_modebench_v12.slurm"
[[ -f "$SLURM_SCRIPT" ]] || { echo "Missing sealed v12 r2 ops snapshot" >&2; exit 1; }
[[ "$(hash_tree "$OPS_ROOT")" == "$OPS_HASH" ]] || { echo "Sealed v12 r2 ops drift" >&2; exit 1; }

"$PYTHON_BIN" -B - \
  "$OPS_ROOT/make_ant_maze_mode_data_v12.py" \
  "$SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" \
  "$SOURCE_ROOT/oat_drgrpo/maze_modebench_worker.py" <<'PYSYNTAX'
import pathlib, sys
for name in sys.argv[1:]:
    path = pathlib.Path(name)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
PYSYNTAX
bash -n "$SLURM_SCRIPT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" \
  "$PYTHON_BIN" -m pytest -q \
  "$ROOT_DIR/tests/test_ant_maze_route_admission_v12.py"

if [[ "$phase" == config ]]; then
  echo "[antmaze-v12-r3] configuration passed; no route executed"
  echo "[antmaze-v12-r3] source=$NEW_SOURCE_HASH ops=$OPS_HASH"
  exit 0
fi

[[ ! -e "$IDENTITY" ]] || { echo "Fresh AntMaze v12 r3 identity required" >&2; exit 1; }
mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold --partition=all --account=allcs \
    --export="ALL,PYTHONDONTWRITEBYTECODE=1,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
    "$SLURM_SCRIPT"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid AntMaze v12 r3 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

before="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'RunTime=00:00:00' 'NumCPUs=4' 'MinMemoryNode=32G' 'TimeLimit=02:00:00'; do
  [[ "$before" == *"$required"* ]] || { echo "Held Ant v12 r3 job missing $required" >&2; exit 1; }
done
scontrol update "JobId=$job_id" Partition=all
scontrol update "JobId=$job_id" Requeue=0
after="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=all' 'Requeue=0' 'RunTime=00:00:00'; do
  [[ "$after" == *"$required"* ]] || { echo "Amended Ant v12 r3 job missing $required" >&2; exit 1; }
done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$R2_IDENTITY" "$R2_ERROR" "$PROTOCOL" "$0" "$NEW_SOURCE_HASH" "$before" "$after" <<'PYID'
import hashlib, json, os, pathlib, sys, tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path = pathlib.Path(sys.argv[1]); r2_path = pathlib.Path(sys.argv[3])
payload = json.loads(r2_path.read_text())
payload.update({
    "job_id": int(sys.argv[2]),
    "source_hash": sys.argv[7],
    "launcher_sha256": digest(sys.argv[6]),
    "dispatch_repair_protocol_sha256": digest(sys.argv[5]),
    "r2_identity_sha256": digest(r2_path),
    "r2_error_log_sha256": digest(sys.argv[4]),
    "r2_failed_job_id": 30200147,
    "repair_attempt": "dispatcher-order-only-r3",
    "dispatcher_only": True,
    "scientific_change": False,
    "map_substitution": False,
    "route_substitution": False,
    "held_scheduler_record_before": sys.argv[8],
    "held_scheduler_record_after": sys.argv[9],
})
path.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
os.replace(tmp, path)
PYID
scontrol release "$job_id"
trap - EXIT
echo "[antmaze-v12-r3] released admission job $job_id"
echo "[antmaze-v12-r3] identity=$IDENTITY"
