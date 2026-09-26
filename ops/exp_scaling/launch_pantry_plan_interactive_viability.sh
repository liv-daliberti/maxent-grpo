#!/usr/bin/env bash
# Freeze and submit the prospective PantryPlan finite-action viability gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/pantry_plan_interactive_05b_viability_v1_20260729.md"
EVALUATOR="$ROOT_DIR/ops/evaluate_pantry_plan_interactive_viability.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/evaluate_pantry_plan_interactive_viability.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/pantry_plan_interactive_05b_viability_v1_identity.json"
RECEIPT="$ROOT_DIR/var/artifacts/pantry_plan_interactive_05b_viability_v1.json"
ADMISSION="$ROOT_DIR/var/artifacts/pantry_plan_modebench_v2_admission_audit.json"
MODEL="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
phase="${1:-}"

case "$phase" in
  config|run) ;;
  *)
    echo "Usage: $0 {config|run}" >&2
    exit 1
    ;;
esac

for required in \
  "$PYTHON_BIN" "$PROTOCOL" "$EVALUATOR" "$SLURM_SCRIPT" "$ADMISSION" \
  "$MODEL/config.json" "$ROOT_DIR/src/oat_drgrpo/pantry_plan_interactive.py"; do
  [[ -f "$required" ]] || {
    echo "Missing PantryPlan interactive prerequisite: $required" >&2
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
  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$EVALUATOR" --help >/dev/null
  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_pantry_plan_interactive.py" >/dev/null
  "$PYTHON_BIN" - "$ADMISSION" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
if receipt.get("status") != "pass":
    raise SystemExit("PantryPlan admission audit is not passing")
PY
  echo "[pantry-interactive] configuration passed; no model sampled"
  exit 0
fi

for fresh in "$IDENTITY" "$RECEIPT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh PantryPlan interactive target required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/pantry_interactive_${SOURCE_HASH}"
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

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.pantry-interactive-ops.XXXXXX")"
cp "$EVALUATOR" "$EXECUTION_INPUT/evaluate_pantry_plan_interactive_viability.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/evaluate_pantry_plan_interactive_viability.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/pantry_interactive_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/evaluate_pantry_plan_interactive_viability.py" ]]; then
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
    "$EXECUTION_ROOT/evaluate_pantry_plan_interactive_viability.slurm"
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid PantryPlan interactive job ID: $job_id" >&2
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
  "$PROTOCOL" "$ADMISSION" "$MODEL/config.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "pantry-plan-interactive-viability-identity-v1",
    "job_id": int(sys.argv[2]),
    "source_hash": sys.argv[3],
    "execution_hash": sys.argv[4],
    "protocol_sha256": hashlib.sha256(pathlib.Path(sys.argv[5]).read_bytes()).hexdigest(),
    "admission_audit_sha256": hashlib.sha256(pathlib.Path(sys.argv[6]).read_bytes()).hexdigest(),
    "model_config_sha256": hashlib.sha256(pathlib.Path(sys.argv[7]).read_bytes()).hexdigest(),
    "model": "Qwen2.5-0.5B-Instruct",
    "seed": 76101,
    "sample_count": 64,
    "prefix_count": 16,
    "policy_interface": "hierarchical_finite_action_one_token_v1",
    "development_only": True,
    "evaluation_prompts_loaded": False,
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
echo "[pantry-interactive] released viability job $job_id"
echo "[pantry-interactive] identity=$IDENTITY"
