#!/usr/bin/env bash
# Freeze and submit A100-only repairs for the two pre-sampling dtype failures.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PANTRY_EVAL="$ROOT_DIR/ops/evaluate_pantry_plan_interactive_viability.py"
POINT_EVAL="$ROOT_DIR/ops/evaluate_point_maze_interactive_viability.py"
PANTRY_SLURM="$ROOT_DIR/ops/slurm/evaluate_pantry_plan_interactive_viability_r1.slurm"
POINT_SLURM="$ROOT_DIR/ops/slurm/evaluate_point_maze_interactive_viability_r1.slurm"
PANTRY_PROTOCOL="$ROOT_DIR/paper/preregistration/pantry_plan_interactive_05b_viability_v1_runtime_r1_20260729.md"
POINT_PROTOCOL="$ROOT_DIR/paper/preregistration/point_maze_interactive_05b_viability_v1_runtime_r1_20260729.md"
IDENTITY="$ROOT_DIR/var/artifacts/interactive_05b_viability_runtime_r1_identity.json"
PANTRY_RECEIPT="$ROOT_DIR/var/artifacts/pantry_plan_interactive_05b_viability_v1_r1.json"
POINT_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_v1_r1.json"
PANTRY_FAILURE_LOG="$ROOT_DIR/var/artifacts/logs/pantry-interactive-05b-30185302.err"
POINT_FAILURE_LOG="$ROOT_DIR/var/artifacts/logs/point-interactive-05b-30185346.err"
phase="${1:-}"

case "$phase" in
  config|run) ;;
  *)
    echo "Usage: $0 {config|run}" >&2
    exit 1
    ;;
esac

for required in \
  "$PYTHON_BIN" "$PANTRY_EVAL" "$POINT_EVAL" "$PANTRY_SLURM" \
  "$POINT_SLURM" "$PANTRY_PROTOCOL" "$POINT_PROTOCOL" \
  "$PANTRY_FAILURE_LOG" "$POINT_FAILURE_LOG"; do
  [[ -f "$required" ]] || {
    echo "Missing interactive runtime-repair prerequisite: $required" >&2
    exit 1
  }
done
rg -q "Bfloat16 is only supported" "$PANTRY_FAILURE_LOG"
rg -q "Bfloat16 is only supported" "$POINT_FAILURE_LOG"

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

if [[ "$phase" == config ]]; then
  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$PANTRY_EVAL" --help >/dev/null
  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$POINT_EVAL" --help >/dev/null
  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_pantry_plan_interactive.py" \
    "$ROOT_DIR/tests/test_point_maze_interactive_worker.py" >/dev/null
  echo "[interactive-runtime-r1] configuration passed; no model sampled"
  exit 0
fi

for fresh in "$IDENTITY" "$PANTRY_RECEIPT" "$POINT_RECEIPT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh interactive runtime-repair target required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/interactive_runtime_r1_${SOURCE_HASH}"
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

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.interactive-runtime-r1-ops.XXXXXX")"
cp "$PANTRY_EVAL" "$EXECUTION_INPUT/evaluate_pantry_plan_interactive_viability.py"
cp "$POINT_EVAL" "$EXECUTION_INPUT/evaluate_point_maze_interactive_viability.py"
cp "$PANTRY_SLURM" "$EXECUTION_INPUT/evaluate_pantry_plan_interactive_viability_r1.slurm"
cp "$POINT_SLURM" "$EXECUTION_INPUT/evaluate_point_maze_interactive_viability_r1.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/interactive_runtime_r1_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/evaluate_pantry_plan_interactive_viability.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || exit 1

mkdir -p "$ROOT_DIR/var/artifacts/logs"
export_line="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH"
pantry_job="$(
  sbatch --parsable --hold --export="$export_line" \
    "$EXECUTION_ROOT/evaluate_pantry_plan_interactive_viability_r1.slurm"
)"
pantry_job="${pantry_job%%;*}"
point_job="$(
  sbatch --parsable --hold --export="$export_line" \
    "$EXECUTION_ROOT/evaluate_point_maze_interactive_viability_r1.slurm"
)"
point_job="${point_job%%;*}"
[[ "$pantry_job" =~ ^[0-9]+$ && "$point_job" =~ ^[0-9]+$ ]] || {
  echo "Invalid interactive runtime-repair job IDs" >&2
  exit 1
}

cleanup() {
  local status="$?"
  trap - EXIT
  scancel "$pantry_job" "$point_job" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

"$PYTHON_BIN" - "$IDENTITY" "$pantry_job" "$point_job" "$SOURCE_HASH" \
  "$EXECUTION_HASH" "$PANTRY_PROTOCOL" "$POINT_PROTOCOL" \
  "$PANTRY_FAILURE_LOG" "$POINT_FAILURE_LOG" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
sha = lambda value: hashlib.sha256(pathlib.Path(value).read_bytes()).hexdigest()
payload = {
    "schema_version": "interactive-05b-viability-runtime-r1-identity-v1",
    "jobs": {
        "pantry_plan": int(sys.argv[2]),
        "point_maze": int(sys.argv[3]),
    },
    "source_hash": sys.argv[4],
    "execution_hash": sys.argv[5],
    "protocol_sha256": {
        "pantry_plan": sha(sys.argv[6]),
        "point_maze": sha(sys.argv[7]),
    },
    "superseded_infrastructure_jobs": {
        "pantry_plan": {
            "job_id": 30185302,
            "error_log_sha256": sha(sys.argv[8]),
            "model_samples": 0,
        },
        "point_maze": {
            "job_id": 30185346,
            "error_log_sha256": sha(sys.argv[9]),
            "model_samples": 0,
        },
    },
    "sole_repair": "require A100 instead of generic GPU",
    "dtype": "bfloat16",
    "scientific_settings_changed": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

scontrol update "JobId=$pantry_job" Requeue=0
scontrol update "JobId=$point_job" Requeue=0
scontrol release "$pantry_job"
scontrol release "$point_job"
trap - EXIT
echo "[interactive-runtime-r1] pantry_job=$pantry_job point_job=$point_job"
echo "[interactive-runtime-r1] identity=$IDENTITY"
