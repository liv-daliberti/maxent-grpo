#!/usr/bin/env bash
# Freeze and submit the AntMaze v8 fresh-map route admission gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_admission_v8_20260729.md"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/admit_ant_maze_modebench_v8.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v8_generation_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v8"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v8_admission_audit.json"
RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v8.evaluation.json"
MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v8.zip"
RECEIPT_SHA="fc2961d74a1fdfc8f3d2c1936ea61041cff65333ada0e6ebcd571b44919b18f1"
MODEL_SHA="77e780dfff1147bc2f542c1761f6efeaa2305fd5ddb625244ebf8e82b3fa871d"
FILES=(
  make_ant_maze_mode_data_v8.py
  make_ant_maze_mode_data.py
  audit_ant_maze_mode_data_v8.py
  audit_ant_maze_mode_data.py
)
phase="${1:-}"

case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac

for required in "$PYTHON_BIN" "$PROTOCOL" "$SLURM_SCRIPT" \
  "$ROOT_DIR/var/maze_runtime/venv/bin/python" "$RECEIPT" "$MODEL"; do
  [[ -e "$required" ]] || {
    echo "Missing AntMaze v8 route prerequisite: $required" >&2
    exit 1
  }
done
[[ "$(sha256sum "$RECEIPT" | cut -d' ' -f1)" == "$RECEIPT_SHA" ]] || {
  echo "Ant v8 controller receipt hash mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" == "$MODEL_SHA" ]] || {
  echo "Ant v8 controller model hash mismatch" >&2
  exit 1
}
for file in "${FILES[@]}"; do
  [[ -f "$ROOT_DIR/ops/$file" ]] || {
    echo "Missing AntMaze v8 route script: $file" >&2
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
  for file in "${FILES[@]}"; do
    PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" \
      "$ROOT_DIR/ops/$file" --help >/dev/null
  done
  bash -n "$SLURM_SCRIPT"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" \
    OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python" \
    "$PYTHON_BIN" -m pytest -q \
      "$ROOT_DIR/tests/test_ant_maze_worker_v5.py" \
      "$ROOT_DIR/tests/test_ant_maze_worker_v8.py" \
      "$ROOT_DIR/tests/test_ant_maze_route_admission.py" >/dev/null
  echo "[antmaze-v8-route] configuration passed; no admission map executed"
  exit 0
fi

for fresh in "$IDENTITY" "$DATA_ROOT" "$AUDIT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh AntMaze v8 route target required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v8_route_${SOURCE_HASH}"
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

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-maze-v8-route.XXXXXX")"
for file in "${FILES[@]}"; do
  cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"
done
cp "$SLURM_SCRIPT" "$OPS_INPUT/admit_ant_maze_modebench_v8.slurm"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v8_route_ops_${OPS_HASH}"
if [[ ! -f "$OPS_ROOT/make_ant_maze_mode_data_v8.py" ]]; then
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
    "$OPS_ROOT/admit_ant_maze_modebench_v8.slurm"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid AntMaze v8 route job ID: $job_id" >&2
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
    echo "Held v8 route job missing $required" >&2
    exit 1
  }
done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$OPS_HASH" \
  "$PROTOCOL" "$0" "$RECEIPT" "$MODEL" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "ant-maze-v8-route-generation-identity-v1",
    "job_id": int(sys.argv[2]),
    "source_hash": sys.argv[3],
    "execution_hash": sys.argv[4],
    "protocol_sha256": digest(sys.argv[5]),
    "launcher_sha256": digest(sys.argv[6]),
    "controller_receipt_sha256": digest(sys.argv[7]),
    "controller_model_sha256": digest(sys.argv[8]),
    "controller_training_job_id": 30187810,
    "map_count": 12,
    "reset_seed_base": 87300,
    "real_route_executions": 48,
    "perturbation_replays": 2400,
    "language_model_sampling": False,
    "map_substitution": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[antmaze-v8-route] released admission job $job_id"
echo "[antmaze-v8-route] identity=$IDENTITY"
