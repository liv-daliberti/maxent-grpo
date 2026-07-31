#!/usr/bin/env bash
# Freeze, identity-bind, and submit the ConstructiveCode v3 materialization/replay gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/constructive_code_executable_slate_v3_20260729.md"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/constructive_code_v3_gate.slurm"
INDEX="$ROOT_DIR/var/artifacts/constructive_code_candidate_source_index.json"
V1_ROOT="$ROOT_DIR/var/data/constructive_code_review_slate_v1"
IMAGE="$ROOT_DIR/var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"
TESTLIB="$ROOT_DIR/third_party/testlib/testlib.h"
IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v3_gate_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/constructive_code_v3"
OUTPUTS=(
  "$ROOT_DIR/var/artifacts/constructive_code_v3_replays.jsonl"
  "$ROOT_DIR/var/artifacts/constructive_code_v3_replay_manifest.json"
  "$ROOT_DIR/var/artifacts/constructive_code_v3_checker_equivalence.json"
  "$ROOT_DIR/var/artifacts/constructive_code_v3_gate_audit.json"
  "$ROOT_DIR/var/artifacts/constructive_code_v3_run_audit.json"
)
FILES=(
  materialize_constructive_code_v3.py
  materialize_constructive_code_v2.py
  materialize_constructive_code_review_slate.py
  materialize_constructive_code_plus_suites.py
  audit_constructive_code_sources.py
  replay_constructive_code_v3.py
  replay_constructive_code_v2.py
  replay_constructive_code_review_slate.py
  audit_constructive_code_v2.py
  audit_constructive_code_checker_equivalence.py
  constructive_code_sandbox.c
)

phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac

for required in "$PYTHON_BIN" "$PROTOCOL" "$SLURM_SCRIPT" "$INDEX" \
  "$V1_ROOT/manifest.json" "$IMAGE" "$TESTLIB"; do
  [[ -e "$required" ]] || {
    echo "Missing ConstructiveCode v3 prerequisite: $required" >&2
    exit 1
  }
done
for file in "${FILES[@]}"; do
  [[ -f "$ROOT_DIR/ops/$file" ]] || {
    echo "Missing ConstructiveCode v3 source: $file" >&2
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
    [[ "$file" == *.py ]] && "$PYTHON_BIN" -m py_compile "$ROOT_DIR/ops/$file"
  done
  bash -n "$SLURM_SCRIPT"
  echo "[constructive-v3-gate] configuration passed; no source or candidate executed"
  exit 0
fi

for fresh in "$IDENTITY" "$DATA_ROOT" "${OUTPUTS[@]}"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh ConstructiveCode v3 target required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v3_source_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  source_staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$source_staging/src"
  cp -a "$ROOT_DIR/src/." "$source_staging/src/"
  mv "$source_staging/src" "$SOURCE_ROOT"
  rmdir "$source_staging"
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || {
  echo "ConstructiveCode v3 source snapshot hash mismatch" >&2
  exit 1
}

EXEC_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.constructive-v3-gate.XXXXXX")"
for file in "${FILES[@]}"; do
  cp "$ROOT_DIR/ops/$file" "$EXEC_INPUT/$file"
done
mkdir -p "$EXEC_INPUT/protocol" "$EXEC_INPUT/inputs" "$EXEC_INPUT/testlib"
cp "$PROTOCOL" "$EXEC_INPUT/protocol/constructive_code_executable_slate_v3_20260729.md"
cp "$INDEX" "$EXEC_INPUT/inputs/candidate_source_index.json"
cp -a "$V1_ROOT" "$EXEC_INPUT/inputs/v1"
cp "$TESTLIB" "$EXEC_INPUT/testlib/testlib.h"
cp "$SLURM_SCRIPT" "$EXEC_INPUT/constructive_code_v3_gate.slurm"
EXEC_HASH="$(hash_tree "$EXEC_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v3_gate_${EXEC_HASH}"
if [[ ! -f "$OPS_ROOT/replay_constructive_code_v3.py" ]]; then
  mv "$EXEC_INPUT" "$OPS_ROOT"
else
  find "$EXEC_INPUT" -type f -delete
  find "$EXEC_INPUT" -depth -type d -empty -delete
fi
[[ "$(hash_tree "$OPS_ROOT")" == "$EXEC_HASH" ]] || {
  echo "ConstructiveCode v3 execution snapshot hash mismatch" >&2
  exit 1
}

V1_HASH="$(hash_tree "$OPS_ROOT/inputs/v1")"
mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT,OAT_ZERO_TESTLIB_ROOT=$OPS_ROOT/testlib,OAT_ZERO_CONSTRUCTIVE_V3_PROTOCOL=$OPS_ROOT/protocol/constructive_code_executable_slate_v3_20260729.md,OAT_ZERO_CONSTRUCTIVE_V3_INDEX=$OPS_ROOT/inputs/candidate_source_index.json,OAT_ZERO_CONSTRUCTIVE_V3_V1_ROOT=$OPS_ROOT/inputs/v1,OAT_ZERO_SANDBOX_SOURCE=$OPS_ROOT/constructive_code_sandbox.c,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
    "$OPS_ROOT/constructive_code_v3_gate.slurm"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid ConstructiveCode v3 job ID: $job_id" >&2
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
    echo "Held v3 job missing $required" >&2
    exit 1
  }
done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXEC_HASH" \
  "$OPS_ROOT/protocol/constructive_code_executable_slate_v3_20260729.md" \
  "$0" "$V1_HASH" "$OPS_ROOT/inputs/candidate_source_index.json" \
  "$IMAGE" "$OPS_ROOT/testlib/testlib.h" <<'PY'
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
    "schema_version": "constructive-code-v3-gate-identity-v1",
    "job_id": int(sys.argv[2]),
    "source_hash": sys.argv[3],
    "execution_hash": sys.argv[4],
    "protocol_sha256": digest(sys.argv[5]),
    "launcher_sha256": digest(sys.argv[6]),
    "v1_exclusion_tree_sha256": sys.argv[7],
    "candidate_index_sha256": digest(sys.argv[8]),
    "runtime_image_sha256": digest(sys.argv[9]),
    "testlib_h_sha256": digest(sys.argv[10]),
    "frozen_task_count": 12,
    "frozen_split_counts": {"train": 4, "development": 4, "evaluation": 4},
    "expected_submission_suite_replays": 4800,
    "suite_policy": "CodeContests-O if admissible, otherwise Plus-5x",
    "v1_job_id": 30184444,
    "v2_materialization_job_id": 30184636,
    "v2_replay_job_id": 30187501,
    "language_model_sampling": False,
    "evaluation_split_loaded": False,
    "network_allowed_only_for_frozen_source_materialization": True,
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
echo "[constructive-v3-gate] released job $job_id"
echo "[constructive-v3-gate] identity=$IDENTITY"
