#!/usr/bin/env bash
# Freeze and submit E49E's answer-bound singleton-gap repair.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

stage="${1:-}"
case "$stage" in
  toy|full) ;;
  *)
    echo "Usage: $0 {toy|full}" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e49e_trace_executed_bank_math_haarnoja_05b.md"
CLEAN_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_clean_relaunch_amendment_20260724.md"
REPAIR_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_singleton_repair_amendment_20260724.md"
NUMERIC_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_singleton_numeric_hardening_amendment_20260724.md"
CACHE_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_singleton_cache_replay_amendment_20260724.md"
KERNEL_V2_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_kernel_routewise_v2_amendment_20260724.md"
ORIGIN_REPLAY_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_repair_origin_replay_amendment_20260724.md"
ENDPOINT_SOURCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json"
SLURM="$ROOT_DIR/ops/slurm/e49e_repair_singletons_node302.slurm"
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots"

if [[ "$stage" == toy ]]; then
  SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
  INPUT_E49D_SOURCE="$ROOT_DIR/var/artifacts/e49d_math_strategy_menu_toy_v1"
  INPUT_E49E_SOURCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_v1"
  INPUT_AUGMENT_SOURCE="$ROOT_DIR/var/artifacts/e49e_kernel_augmentation_math_toy_v2"
  OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_toy_repaired"
  EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v1"
  INVALID_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_invalid_controls_toy.json"
  EQUIVALENT_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_equivalent_controls_toy.json"
  EXPECTED_ROWS=100
else
  SOURCE="$ROOT_DIR/var/data/math12k_384_math500"
  INPUT_E49D_SOURCE="$ROOT_DIR/var/artifacts/e49d_math_strategy_menu_full_v1"
  INPUT_E49E_SOURCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_full_v1"
  INPUT_AUGMENT_SOURCE="$ROOT_DIR/var/artifacts/e49e_kernel_augmentation_math_full_v2"
  OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_full_repaired"
  EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_full_repair_v1"
  INVALID_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_invalid_controls_full.json"
  EQUIVALENT_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_equivalent_controls_full.json"
  EXPECTED_ROWS=884
fi
RECORD="$EVIDENCE/repair_job.json"

if [[ ! -x "$PYTHON_BIN" ]] \
  || [[ ! -f "$PROTOCOL" ]] \
  || [[ ! -f "$CLEAN_AMENDMENT" ]] \
  || [[ ! -f "$REPAIR_AMENDMENT" ]] \
  || [[ ! -f "$NUMERIC_AMENDMENT" ]] \
  || [[ ! -f "$CACHE_AMENDMENT" ]] \
  || [[ ! -f "$KERNEL_V2_AMENDMENT" ]] \
  || [[ ! -f "$ORIGIN_REPLAY_AMENDMENT" ]] \
  || [[ ! -f "$ENDPOINT_SOURCE" ]] \
  || [[ ! -f "$SLURM" ]] \
  || [[ ! -f "$INPUT_E49D_SOURCE/menu_records.jsonl" ]] \
  || [[ ! -f "$INPUT_E49E_SOURCE/trace_bank_records.jsonl" ]] \
  || [[ ! -f "$INPUT_AUGMENT_SOURCE/augmentation_records.jsonl" ]] \
  || [[ ! -f "$INPUT_AUGMENT_SOURCE/generation_summary.json" ]] \
  || [[ ! -f "$INVALID_SOURCE" ]] \
  || [[ ! -f "$EQUIVALENT_SOURCE" ]] \
  || ! grep -q '^\*\*Status: FROZEN BEFORE ANY SINGLETON-REPAIR JUDGE REQUEST OR POLICY TRAINING' "$REPAIR_AMENDMENT"; then
  echo "E49E singleton-repair frozen prerequisite is missing" >&2
  exit 2
fi

augment_complete="$(
  "$PYTHON_BIN" - "$INPUT_AUGMENT_SOURCE/generation_summary.json" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    "yes"
    if record.get("schema") == "e49e_kernel_augmentation_summary_v1"
    and record.get("augmentation_version")
    == "e49e_finite_kernel_augmentation_v2"
    and record.get("complete") is True
    else "no"
)
PY
)"
if [[ "$augment_complete" != yes ]]; then
  echo "E49E finite-kernel augmentation is incomplete" >&2
  exit 2
fi

AUGMENT_JOB_RECORD="$INPUT_AUGMENT_SOURCE/augmentation_job.json"
if [[ ! -f "$AUGMENT_JOB_RECORD" ]]; then
  echo "E49E finite-kernel v2 job record is missing" >&2
  exit 2
fi
augment_job_id="$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' \
    "$AUGMENT_JOB_RECORD"
)"
augment_job_state="$(
  sacct -X -n -j "$augment_job_id" --format=State -P \
    | head -n 1 | cut -d'|' -f1
)"
case "$augment_job_state" in
  COMPLETED*) ;;
  *)
    echo "E49E finite-kernel v2 $augment_job_id is not complete: $augment_job_state" >&2
    exit 2
    ;;
esac

mkdir -p "$EVIDENCE" "$SNAPSHOT_PARENT"
if [[ -e "$RECORD" || -e "$OUTPUT" ]]; then
  echo "fresh E49E ${stage} singleton repair required" >&2
  exit 2
fi

TRACE_JOB_RECORD="$INPUT_E49E_SOURCE/materialization_job.json"
if [[ ! -f "$TRACE_JOB_RECORD" ]]; then
  echo "E49E trace writer job record is missing" >&2
  exit 2
fi
trace_job_id="$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' \
    "$TRACE_JOB_RECORD"
)"
trace_job_state="$(
  sacct -X -n -j "$trace_job_id" --format=State -P \
    | head -n 1 | cut -d'|' -f1
)"
case "$trace_job_state" in
  COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*) ;;
  *)
    echo "E49E trace writer $trace_job_id is not terminal: $trace_job_state" >&2
    exit 2
    ;;
esac

record_rows="$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(len({json.loads(x)["row_id"] for x in open(sys.argv[1]) if x.strip()}))' \
    "$INPUT_E49E_SOURCE/trace_bank_records.jsonl"
)"
if [[ "$record_rows" != "$EXPECTED_ROWS" ]]; then
  echo "E49E trace evidence is incomplete: $record_rows/$EXPECTED_ROWS" >&2
  exit 2
fi

hash_tree() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for item in sorted(path for path in root.rglob("*") if path.is_file()):
    digest.update(str(item.relative_to(root)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(item.read_bytes()).digest())
print(digest.hexdigest())
PY
}

snapshot_staging="$(mktemp -d "$SNAPSHOT_PARENT/.e49e-repair.XXXXXX")"
mkdir -p \
  "$snapshot_staging/src" \
  "$snapshot_staging/ops/math_strategy_calibration"
while IFS= read -r -d '' source_file; do
  relative="${source_file#"$ROOT_DIR/src/"}"
  mkdir -p "$snapshot_staging/src/$(dirname "$relative")"
  cp "$source_file" "$snapshot_staging/src/$relative"
done < <(find "$ROOT_DIR/src" -type f -name '*.py' -print0 | sort -z)
cp \
  "$ROOT_DIR/ops/math_strategy_calibration/materialize_e49d_strategy_menu_data.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/materialize_e49e_trace_bank_data.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps.py" \
  "$INVALID_SOURCE" \
  "$EQUIVALENT_SOURCE" \
  "$snapshot_staging/ops/math_strategy_calibration/"
SNAPSHOT_HASH="$(hash_tree "$snapshot_staging")"
SNAPSHOT_ROOT="$SNAPSHOT_PARENT/e49e_singleton_repair_${SNAPSHOT_HASH}"
if [[ ! -e "$SNAPSHOT_ROOT" ]]; then
  mv "$snapshot_staging" "$SNAPSHOT_ROOT"
else
  if [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]]; then
    echo "existing E49E repair snapshot hash mismatch" >&2
    exit 2
  fi
  find "$snapshot_staging" -type f -delete
  find "$snapshot_staging" -depth -type d -empty -delete
fi
REPAIR="$SNAPSHOT_ROOT/ops/math_strategy_calibration/repair_e49e_singleton_gaps.py"
INVALID="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$INVALID_SOURCE")"
EQUIVALENT="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$EQUIVALENT_SOURCE")"

FROZEN_E49D="$EVIDENCE/frozen_e49d_input"
FROZEN_E49E="$EVIDENCE/frozen_e49e_input"
FROZEN_AUGMENT="$EVIDENCE/frozen_kernel_augmentation"
mkdir -p "$FROZEN_E49D" "$FROZEN_E49E" "$FROZEN_AUGMENT"
cp "$INPUT_E49D_SOURCE/menu_records.jsonl" "$FROZEN_E49D/menu_records.jsonl"
cp \
  "$INPUT_E49E_SOURCE/trace_bank_records.jsonl" \
  "$INPUT_E49E_SOURCE/frozen_identity.json" \
  "$INPUT_E49E_SOURCE/materialization_job.json" \
  "$FROZEN_E49E/"
if [[ -f "$INPUT_E49E_SOURCE/generation_summary.json" ]]; then
  cp "$INPUT_E49E_SOURCE/generation_summary.json" "$FROZEN_E49E/"
fi
cp \
  "$INPUT_AUGMENT_SOURCE/augmentation_records.jsonl" \
  "$INPUT_AUGMENT_SOURCE/generation_summary.json" \
  "$INPUT_AUGMENT_SOURCE/frozen_identity.json" \
  "$INPUT_AUGMENT_SOURCE/augmentation_job.json" \
  "$FROZEN_AUGMENT/"
FROZEN_ENDPOINT="$EVIDENCE/qwen72_endpoint.json"
cp "$ENDPOINT_SOURCE" "$FROZEN_ENDPOINT"

PREFLIGHT_TMP="$(mktemp "$EVIDENCE/.preflight.XXXXXX")"
PYTHONPATH="$SNAPSHOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$REPAIR" \
  --source "$SOURCE" \
  --input-e49d "$FROZEN_E49D" \
  --input-e49e "$FROZEN_E49E" \
  --input-augmentation "$FROZEN_AUGMENT" \
  --output "$OUTPUT" \
  --evidence "$EVIDENCE" \
  --endpoint "$FROZEN_ENDPOINT" \
  --known-invalid-controls "$INVALID" \
  --known-equivalent-controls "$EQUIVALENT" \
  --preflight-only >"$PREFLIGHT_TMP"
PREFLIGHT="$EVIDENCE/preflight.json"
mv "$PREFLIGHT_TMP" "$PREFLIGHT"
"$PYTHON_BIN" - "$PREFLIGHT" "$EXPECTED_ROWS" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    record.get("schema") != "e49e_singleton_repair_preflight_v1"
    or record.get("row_count") != int(sys.argv[2])
    or record.get("all_controls_pass") is not True
):
    raise SystemExit("E49E singleton-repair preflight failed")
PY

IDENTITY="$EVIDENCE/frozen_identity.json"
"$PYTHON_BIN" - "$IDENTITY" "$stage" "$SNAPSHOT_ROOT" "$SNAPSHOT_HASH" \
  "$(sha256sum "$FROZEN_E49D/menu_records.jsonl" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_E49E/trace_bank_records.jsonl" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_AUGMENT/augmentation_records.jsonl" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_ENDPOINT" | cut -d' ' -f1)" \
  "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" \
  "$(sha256sum "$INVALID" | cut -d' ' -f1)" \
  "$(sha256sum "$EQUIVALENT" | cut -d' ' -f1)" \
  "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" \
  "$(sha256sum "$CLEAN_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$REPAIR_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$NUMERIC_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$CACHE_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$KERNEL_V2_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$ORIGIN_REPLAY_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)" \
  "$(sha256sum "$SLURM" | cut -d' ' -f1)" \
  "$(hash_tree "$SOURCE/train")" \
  "$(hash_tree "$SOURCE/eval")" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
keys = [
    "stage",
    "snapshot_root",
    "snapshot_tree_sha256",
    "e49d_input_sha256",
    "raw_trace_records_sha256",
    "kernel_augmentation_records_sha256",
    "endpoint_record_sha256",
    "preflight_sha256",
    "known_invalid_controls_sha256",
    "known_equivalent_controls_sha256",
    "protocol_sha256",
    "clean_relaunch_amendment_sha256",
    "singleton_repair_amendment_sha256",
    "singleton_numeric_hardening_amendment_sha256",
    "singleton_cache_replay_amendment_sha256",
    "kernel_routewise_v2_amendment_sha256",
    "repair_origin_replay_amendment_sha256",
    "launcher_sha256",
    "slurm_sha256",
    "source_train_tree_sha256",
    "source_eval_tree_sha256",
]
payload = {
    "schema": "e49e_singleton_repair_frozen_identity_v1",
    **dict(zip(keys, sys.argv[2:], strict=True)),
    "repair_judge_requests_at_freeze": 0,
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
IDENTITY_HASH="$(sha256sum "$IDENTITY" | cut -d' ' -f1)"

job_id="$(
  sbatch --parsable \
    --job-name="e49e-repair-${stage}" \
    --output="$EVIDENCE/repair-%j.out" \
    --error="$EVIDENCE/repair-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49E_REPAIR_STAGE=$stage,E49E_REPAIR_SOURCE_SNAPSHOT=$SNAPSHOT_ROOT,E49E_REPAIR_SOURCE_HASH=$SNAPSHOT_HASH,E49E_REPAIR_FROZEN_E49D=$FROZEN_E49D,E49E_REPAIR_FROZEN_E49E=$FROZEN_E49E,E49E_REPAIR_FROZEN_AUGMENT=$FROZEN_AUGMENT,E49E_REPAIR_ENDPOINT=$FROZEN_ENDPOINT,E49E_REPAIR_INVALID_CONTROLS=$INVALID,E49E_REPAIR_EQUIVALENT_CONTROLS=$EQUIVALENT,E49E_REPAIR_IDENTITY=$IDENTITY,E49E_REPAIR_IDENTITY_HASH=$IDENTITY_HASH" \
    "$SLURM"
)"
"$PYTHON_BIN" - "$RECORD" "$job_id" "$stage" "$IDENTITY_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e49e_singleton_repair_job_v1",
    "job_id": sys.argv[2],
    "stage": sys.argv[3],
    "frozen_identity_sha256": sys.argv[4],
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "submitted E49E ${stage} singleton-repair job $job_id"
