#!/usr/bin/env bash
# Freeze and submit the AntMaze v5 route admission gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_admission_v1_20260729.md"
REPAIR_PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_route_runtime_adapter_r1_20260729.md"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/admit_ant_maze_modebench_v1.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v1_generation_r1_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v1"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v1_admission_audit.json"
phase="${1:-}"

case "$phase" in
  config|run) ;;
  *)
    echo "Usage: $0 {config|run}" >&2
    exit 1
    ;;
esac

FILES=(make_ant_maze_mode_data.py audit_ant_maze_mode_data.py)
for required in \
  "$PYTHON_BIN" "$PROTOCOL" "$REPAIR_PROTOCOL" "$SLURM_SCRIPT" \
  "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  "$ROOT_DIR/var/maze_runtime/controllers/ant_heading_v5.evaluation.json"; do
  [[ -e "$required" ]] || { echo "Missing AntMaze route prerequisite: $required" >&2; exit 1; }
done
for file in "${FILES[@]}"; do
  [[ -f "$ROOT_DIR/ops/$file" ]] || { echo "Missing AntMaze route script: $file" >&2; exit 1; }
done
if [[ "$(sha256sum "$ROOT_DIR/var/maze_runtime/controllers/ant_heading_v5.evaluation.json" | cut -d' ' -f1)" != "324d3301b8e21b4dbbfcc2e6b9a87aba479bd2da5aa3a040cff24adebaf8828e" ]]; then
  echo "Ant v5 controller receipt hash mismatch" >&2
  exit 1
fi

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

if [[ "$phase" == config ]]; then
  for file in "${FILES[@]}"; do
    "$PYTHON_BIN" "$ROOT_DIR/ops/$file" --help >/dev/null
  done
  echo "[antmaze-route] configuration passed; no admission map executed"
  exit 0
fi

for fresh in "$IDENTITY" "$DATA_ROOT" "$AUDIT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh AntMaze route target required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_route_${SOURCE_HASH}"
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

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.antmaze-route-ops.XXXXXX")"
for file in "${FILES[@]}"; do
  cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"
done
cp "$SLURM_SCRIPT" "$OPS_INPUT/admit_ant_maze_modebench_v1.slurm"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_route_ops_${OPS_HASH}"
if [[ ! -f "$OPS_ROOT/make_ant_maze_mode_data.py" ]]; then
  mv "$OPS_INPUT" "$OPS_ROOT"
else
  find "$OPS_INPUT" -type f -delete
  rmdir "$OPS_INPUT"
fi
[[ "$(hash_tree "$OPS_ROOT")" == "$OPS_HASH" ]] || exit 1

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT" \
    "$OPS_ROOT/admit_ant_maze_modebench_v1.slurm"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid AntMaze route job ID: $job_id" >&2; exit 1; }

cleanup() {
  local status="$?"
  trap - EXIT
  scancel "$job_id" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$OPS_HASH" "$PROTOCOL" \
  "$REPAIR_PROTOCOL" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "ant-maze-route-generation-identity-v1",
    "job_id": int(sys.argv[2]),
    "source_hash": sys.argv[3],
    "execution_hash": sys.argv[4],
    "protocol_sha256": hashlib.sha256(pathlib.Path(sys.argv[5]).read_bytes()).hexdigest(),
    "repair_protocol_sha256": hashlib.sha256(pathlib.Path(sys.argv[6]).read_bytes()).hexdigest(),
    "failed_pre_execution_job_id": 30184765,
    "controller_receipt_sha256": "324d3301b8e21b4dbbfcc2e6b9a87aba479bd2da5aa3a040cff24adebaf8828e",
    "map_count": 12,
    "real_route_executions": 48,
    "perturbation_replays": 2400,
    "language_model_sampling": False,
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
echo "[antmaze-route] released admission job $job_id"
echo "[antmaze-route] identity=$IDENTITY"
