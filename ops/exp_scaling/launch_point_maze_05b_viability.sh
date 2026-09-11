#!/usr/bin/env bash
# Freeze and submit the development-only PointMaze base-model viability gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|run) ;;
  *)
    echo "Usage: $0 {config|run}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/point_maze_modebench_v1"
AUDIT="$ROOT_DIR/var/artifacts/point_maze_modebench_v1_admission_audit.json"
PROTOCOL="$ROOT_DIR/paper/preregistration/point_maze_05b_viability_v1_20260729.md"
EVALUATOR="$ROOT_DIR/ops/evaluate_modebench_base_viability.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/evaluate_point_maze_05b_viability.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/point_maze_05b_viability_v1_identity.json"
RECEIPT="$ROOT_DIR/var/artifacts/point_maze_05b_viability_v1.json"

for required in \
  "$PYTHON_BIN" "$MODEL_ROOT/config.json" "$DATA_ROOT/identity.json" \
  "$DATA_ROOT/dev/dataset_dict.json" "$AUDIT" "$PROTOCOL" "$EVALUATOR" \
  "$SLURM_SCRIPT" "$ROOT_DIR/var/maze_runtime/venv/bin/python"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing PointMaze viability prerequisite: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$AUDIT" "$DATA_ROOT/identity.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    payload = json.load(open(path, encoding="utf-8"))
    if path.endswith("admission_audit.json") and payload.get("status") != "pass":
        raise SystemExit("PointMaze admission audit is not passing")
PY

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" "$EVALUATOR" --help >/dev/null
  echo "[pointmaze-viability] configuration passed; no model sampled"
  exit 0
fi

for fresh in "$IDENTITY" "$RECEIPT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh PointMaze viability artifact required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/point_maze_viability_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"
  rmdir "$staging"
fi
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "PointMaze viability source snapshot mismatch" >&2
  exit 1
fi

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.pointmaze-viability-ops.XXXXXX")"
cp "$EVALUATOR" "$EXECUTION_INPUT/evaluate_modebench_base_viability.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/evaluate_point_maze_05b_viability.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/point_maze_viability_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/evaluate_modebench_base_viability.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
if [[ "$(hash_tree "$EXECUTION_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "PointMaze viability execution snapshot mismatch" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH" \
    "$EXECUTION_ROOT/evaluate_point_maze_05b_viability.slurm"
)"
job_id="${job_id%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Invalid PointMaze viability job ID: $job_id" >&2
  exit 1
fi

cleanup() {
  local status="$?"
  trap - EXIT
  scancel "$job_id" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

"$PYTHON_BIN" - \
  "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" "$PROTOCOL" \
  "$EVALUATOR" "$SLURM_SCRIPT" "$AUDIT" "$DATA_ROOT/identity.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "point-maze-05b-viability-identity-v1",
    "job_id": int(sys.argv[2]),
    "source_hash": sys.argv[3],
    "execution_hash": sys.argv[4],
    "protocol_sha256": digest(sys.argv[5]),
    "evaluator_sha256": digest(sys.argv[6]),
    "slurm_sha256": digest(sys.argv[7]),
    "admission_audit_sha256": digest(sys.argv[8]),
    "dataset_identity_sha256": digest(sys.argv[9]),
    "model": "Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
    "split": "dev/multi_answer",
    "evaluation_split_loaded": False,
    "sample_count": 64,
    "prefix_count": 16,
    "sampling_seed": 75001,
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
echo "[pointmaze-viability] released job $job_id"
echo "[pointmaze-viability] identity=$IDENTITY"
