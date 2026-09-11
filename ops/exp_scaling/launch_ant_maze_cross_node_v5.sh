#!/usr/bin/env bash
# Freeze and submit the prospective three-node AntMaze v5 robustness audit.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_v5_cross_node_audit_v1_20260729.md"
AUDITOR="$ROOT_DIR/ops/audit_ant_maze_cross_node_v5.py"
AGGREGATOR="$ROOT_DIR/ops/aggregate_ant_maze_cross_node_v5.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/audit_ant_maze_cross_node_v5.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_v5_cross_node_v1_identity.json"
AGGREGATE="$ROOT_DIR/var/artifacts/ant_maze_v5_cross_node_v1.json"
PREFIX="$ROOT_DIR/var/artifacts/ant_maze_v5_cross_node_v1_replica"
WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
phase="${1:-}"

case "$phase" in
  config|run) ;;
  *)
    echo "Usage: $0 {config|run}" >&2
    exit 1
    ;;
esac

for required in \
  "$PYTHON_BIN" "$WORKER_PYTHON" "$PROTOCOL" "$AUDITOR" "$AGGREGATOR" \
  "$SLURM_SCRIPT" \
  "$ROOT_DIR/var/maze_runtime/controllers/ant_heading_v5.evaluation.json"; do
  [[ -f "$required" ]] || {
    echo "Missing Ant cross-node prerequisite: $required" >&2
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
  PYTHONPATH="$ROOT_DIR/src" "$WORKER_PYTHON" "$AUDITOR" --help >/dev/null
  "$PYTHON_BIN" "$AGGREGATOR" --help >/dev/null
  PYTHONPATH="$ROOT_DIR/src" OAT_ZERO_MAZE_WORKER_PYTHON="$WORKER_PYTHON" \
    "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_ant_maze_worker_v5.py" >/dev/null
  echo "[ant-cross-node] configuration passed; no route executed"
  exit 0
fi

for fresh in \
  "$IDENTITY" "$AGGREGATE" \
  "${PREFIX}_0.json" "${PREFIX}_1.json" "${PREFIX}_2.json"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh Ant cross-node target required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/ant_cross_node_${SOURCE_HASH}"
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

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-cross-node-ops.XXXXXX")"
cp "$AUDITOR" "$EXECUTION_INPUT/audit_ant_maze_cross_node_v5.py"
cp "$AGGREGATOR" "$EXECUTION_INPUT/aggregate_ant_maze_cross_node_v5.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/audit_ant_maze_cross_node_v5.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_cross_node_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/audit_ant_maze_cross_node_v5.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || exit 1

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH" \
    "$EXECUTION_ROOT/audit_ant_maze_cross_node_v5.slurm"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid Ant cross-node job ID: $job_id" >&2
  exit 1
}

cleanup() {
  local status="$?"
  trap - EXIT
  scancel "$job_id" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" \
  "$PROTOCOL" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "ant-maze-v5-cross-node-identity-v1",
    "job_id": int(sys.argv[2]),
    "source_hash": sys.argv[3],
    "execution_hash": sys.argv[4],
    "protocol_sha256": hashlib.sha256(pathlib.Path(sys.argv[5]).read_bytes()).hexdigest(),
    "failed_prior_admission_job_id": 30184769,
    "node_count": 3,
    "maps": 12,
    "routes_per_map": 2,
    "repetitions_per_node": 3,
    "expected_execution_count": 216,
    "maximum_final_distance": 0.45,
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
echo "[ant-cross-node] released three-node audit job $job_id"
echo "[ant-cross-node] identity=$IDENTITY"
