#!/usr/bin/env bash
# Freeze and submit the ConstructiveCode v2 dual-suite executable replay gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/constructive_code_executable_slate_v2_20260729.md"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/replay_constructive_code_v2.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v2_replay_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/constructive_code_v2"
IMAGE="$ROOT_DIR/var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"
TESTLIB="$ROOT_DIR/third_party/testlib"
OUTPUTS=(
  "$ROOT_DIR/var/artifacts/constructive_code_v2_replays.jsonl"
  "$ROOT_DIR/var/artifacts/constructive_code_v2_replay_manifest.json"
  "$ROOT_DIR/var/artifacts/constructive_code_v2_checker_equivalence.json"
  "$ROOT_DIR/var/artifacts/constructive_code_v2_gate_audit.json"
  "$ROOT_DIR/var/artifacts/constructive_code_v2_run_audit.json"
)
FILES=(
  replay_constructive_code_v2.py
  audit_constructive_code_v2.py
  audit_constructive_code_checker_equivalence.py
  materialize_constructive_code_v2.py
  audit_constructive_code_sources.py
  materialize_constructive_code_review_slate.py
  materialize_constructive_code_plus_suites.py
  replay_constructive_code_review_slate.py
  constructive_code_sandbox.c
)
phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac
for required in "$PYTHON_BIN" "$PROTOCOL" "$SLURM_SCRIPT" "$IMAGE" "$TESTLIB/testlib.h"; do
  [[ -e "$required" ]] || { echo "Missing ConstructiveCode v2 replay prerequisite: $required" >&2; exit 1; }
done
for file in "${FILES[@]}"; do
  [[ -f "$ROOT_DIR/ops/$file" ]] || { echo "Missing ConstructiveCode v2 replay script: $file" >&2; exit 1; }
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
  echo "[constructive-v2-replay] configuration passed; no candidate executed"
  exit 0
fi
[[ -f "$DATA_ROOT/manifest.json" ]] || { echo "ConstructiveCode v2 materialization is incomplete" >&2; exit 1; }
"$PYTHON_BIN" - "$DATA_ROOT/manifest.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("schema_version") != "constructive-code-slate-v2" or payload.get("status") != "pending_executable_replay":
    raise SystemExit("ConstructiveCode v2 source manifest is not replay-ready")
PY
for fresh in "$IDENTITY" "${OUTPUTS[@]}"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh ConstructiveCode v2 replay target required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v2_replay_${SOURCE_HASH}"
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

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.constructive-v2-replay-ops.XXXXXX")"
for file in "${FILES[@]}"; do
  cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"
done
mkdir -p "$OPS_INPUT/testlib" "$OPS_INPUT/protocol" "$OPS_INPUT/inputs"
cp "$TESTLIB/testlib.h" "$OPS_INPUT/testlib/testlib.h"
cp "$PROTOCOL" "$OPS_INPUT/protocol/constructive_code_executable_slate_v2_20260729.md"
cp "$ROOT_DIR/var/artifacts/constructive_code_candidate_source_index.json" "$OPS_INPUT/inputs/candidate_source_index.json"
cp -a "$ROOT_DIR/var/data/constructive_code_review_slate_v1" "$OPS_INPUT/inputs/v1"
cp -a "$DATA_ROOT" "$OPS_INPUT/inputs/v2"
cp "$SLURM_SCRIPT" "$OPS_INPUT/replay_constructive_code_v2.slurm"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v2_replay_ops_${OPS_HASH}"
if [[ ! -f "$OPS_ROOT/replay_constructive_code_v2.py" ]]; then
  mv "$OPS_INPUT" "$OPS_ROOT"
else
  find "$OPS_INPUT" -type f -delete
  find "$OPS_INPUT" -depth -type d -empty -delete
fi
[[ "$(hash_tree "$OPS_ROOT")" == "$OPS_HASH" ]] || exit 1
DATA_HASH="$(hash_tree "$OPS_ROOT/inputs/v2")"
V1_HASH="$(hash_tree "$OPS_ROOT/inputs/v1")"

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_OPS_ROOT=$OPS_ROOT,OAT_ZERO_TESTLIB_ROOT=$OPS_ROOT/testlib,OAT_ZERO_CONSTRUCTIVE_V2_PROTOCOL=$OPS_ROOT/protocol/constructive_code_executable_slate_v2_20260729.md,OAT_ZERO_SANDBOX_SOURCE=$OPS_ROOT/constructive_code_sandbox.c,OAT_ZERO_CONSTRUCTIVE_V2_INDEX=$OPS_ROOT/inputs/candidate_source_index.json,OAT_ZERO_CONSTRUCTIVE_V2_SLATE=$OPS_ROOT/inputs/v2,OAT_ZERO_CONSTRUCTIVE_V2_V1_ROOT=$OPS_ROOT/inputs/v1,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
    "$OPS_ROOT/replay_constructive_code_v2.slurm"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid ConstructiveCode v2 replay job ID: $job_id" >&2; exit 1; }
cleanup() {
  local status="$?"
  trap - EXIT
  scancel "$job_id" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
job_record="$(scontrol show job "$job_id" -o)"
for required in \
  "JobState=PENDING" "Reason=JobHeldUser" \
  "OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT" "OAT_ZERO_OPS_ROOT=$OPS_ROOT" \
  "OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY"; do
  [[ "$job_record" == *"$required"* ]] || { echo "Held replay job missing $required" >&2; exit 1; }
done

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$OPS_HASH" \
  "$OPS_ROOT/protocol/constructive_code_executable_slate_v2_20260729.md" \
  "$0" "$DATA_HASH" "$V1_HASH" \
  "$OPS_ROOT/inputs/candidate_source_index.json" "$IMAGE" \
  "$OPS_ROOT/testlib/testlib.h" "$OPS_ROOT/inputs/v2/manifest.json" <<PY
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
    "schema_version": "constructive-code-v2-replay-identity-v1",
    "job_id": int(sys.argv[2]),
    "source_hash": sys.argv[3],
    "execution_hash": sys.argv[4],
    "protocol_sha256": digest(sys.argv[5]),
    "launcher_sha256": digest(sys.argv[6]),
    "data_tree_sha256": sys.argv[7],
    "v1_exclusion_tree_sha256": sys.argv[8],
    "candidate_index_sha256": digest(sys.argv[9]),
    "runtime_image_sha256": digest(sys.argv[10]),
    "testlib_h_sha256": digest(sys.argv[11]),
    "source_manifest_sha256": digest(sys.argv[12]),
    "materialization_job_id": 30184636,
    "expected_submission_suite_replays": 1600,
    "language_model_sampling": False,
    "evaluation_split_loaded": False,
    "network_allowed": False,
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
echo "[constructive-v2-replay] released gate job $job_id"
echo "[constructive-v2-replay] identity=$IDENTITY"
