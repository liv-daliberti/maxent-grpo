#!/usr/bin/env bash
# Freeze and submit E49E finite-kernel diversity augmentation.
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
variant="${E49E_KERNEL_VARIANT:-v1}"
case "$variant" in
  v1|v2) ;;
  *)
    echo "E49E kernel variant must be v1 or v2" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e49e_trace_executed_bank_math_haarnoja_05b.md"
CLEAN_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_clean_relaunch_amendment_20260724.md"
KERNEL_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_finite_kernel_augmentation_amendment_20260724.md"
HARDENING_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_finite_kernel_hardening_amendment_20260724.md"
GAP_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_kernel_gap_eligibility_amendment_20260724.md"
CACHE_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_kernel_cache_replay_amendment_20260724.md"
V2_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_kernel_routewise_v2_amendment_20260724.md"
ENDPOINT_SOURCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json"
SLURM="$ROOT_DIR/ops/slurm/e49e_kernel_augmentation_node302.slurm"
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots"
BASE_AUGMENT_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/augment_e49e_kernel_routes.py"
if [[ "$variant" == v2 ]]; then
  AUGMENT_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/augment_e49e_kernel_routes_v2.py"
  VARIANT_AMENDMENT="$V2_AMENDMENT"
else
  AUGMENT_SOURCE="$BASE_AUGMENT_SOURCE"
  VARIANT_AMENDMENT="$KERNEL_AMENDMENT"
fi

if [[ "$stage" == toy ]]; then
  SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
  INPUT_E49D_SOURCE="$ROOT_DIR/var/artifacts/e49d_math_strategy_menu_toy_v1"
  INPUT_E49E_SOURCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_v1"
  EVIDENCE="$ROOT_DIR/var/artifacts/e49e_kernel_augmentation_math_toy_${variant}"
  INVALID_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_invalid_controls_toy.json"
  EQUIVALENT_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_equivalent_controls_toy.json"
  EXPECTED_ROWS=100
else
  SOURCE="$ROOT_DIR/var/data/math12k_384_math500"
  INPUT_E49D_SOURCE="$ROOT_DIR/var/artifacts/e49d_math_strategy_menu_full_v1"
  INPUT_E49E_SOURCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_full_v1"
  EVIDENCE="$ROOT_DIR/var/artifacts/e49e_kernel_augmentation_math_full_${variant}"
  INVALID_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_invalid_controls_full.json"
  EQUIVALENT_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_equivalent_controls_full.json"
  EXPECTED_ROWS=884
fi
RECORD="$EVIDENCE/augmentation_job.json"

if [[ ! -x "$PYTHON_BIN" ]] \
  || [[ ! -f "$PROTOCOL" ]] \
  || [[ ! -f "$CLEAN_AMENDMENT" ]] \
  || [[ ! -f "$KERNEL_AMENDMENT" ]] \
  || [[ ! -f "$HARDENING_AMENDMENT" ]] \
  || [[ ! -f "$GAP_AMENDMENT" ]] \
  || [[ ! -f "$CACHE_AMENDMENT" ]] \
  || [[ ! -f "$VARIANT_AMENDMENT" ]] \
  || [[ ! -f "$AUGMENT_SOURCE" ]] \
  || [[ ! -f "$ENDPOINT_SOURCE" ]] \
  || [[ ! -f "$SLURM" ]] \
  || [[ ! -f "$INPUT_E49D_SOURCE/menu_records.jsonl" ]] \
  || [[ ! -f "$INPUT_E49E_SOURCE/trace_bank_records.jsonl" ]] \
  || [[ ! -f "$INVALID_SOURCE" ]] \
  || [[ ! -f "$EQUIVALENT_SOURCE" ]] \
  || ! grep -q '^\*\*Status: FROZEN BEFORE ANY FINITE-KERNEL AUGMENTATION REQUEST OR POLICY TRAINING' "$KERNEL_AMENDMENT"; then
  echo "E49E kernel-augmentation frozen prerequisite is missing" >&2
  exit 2
fi

mkdir -p "$EVIDENCE" "$SNAPSHOT_PARENT"
if [[ -e "$RECORD" ]]; then
  echo "fresh E49E ${stage} kernel augmentation required" >&2
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

PRIOR_AUGMENT_SHA="none"
PRIOR_AUGMENT_SOURCE=""
FROZEN_PRIOR_RECORDS=""
if [[ "$variant" == v2 ]]; then
  PRIOR_EVIDENCE="$ROOT_DIR/var/artifacts/e49e_kernel_augmentation_math_${stage}_v1"
  PRIOR_RECORD="$PRIOR_EVIDENCE/augmentation_job.json"
  PRIOR_AUGMENT_SOURCE="$PRIOR_EVIDENCE/augmentation_records.jsonl"
  if [[ ! -f "$PRIOR_RECORD" || ! -f "$PRIOR_AUGMENT_SOURCE" ]]; then
    echo "E49E v1 augmentation evidence is incomplete" >&2
    exit 2
  fi
  prior_job_id="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' \
      "$PRIOR_RECORD"
  )"
  prior_job_state="$(
    sacct -X -n -j "$prior_job_id" --format=State -P \
      | head -n 1 | cut -d'|' -f1
  )"
  case "$prior_job_state" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*) ;;
    *)
      echo "E49E v1 augmentation $prior_job_id is not terminal: $prior_job_state" >&2
      exit 2
      ;;
  esac
  prior_rows="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(len({json.loads(x)["row_id"] for x in open(sys.argv[1]) if x.strip()}))' \
      "$PRIOR_AUGMENT_SOURCE"
  )"
  if [[ "$prior_rows" != "$EXPECTED_ROWS" ]]; then
    echo "E49E v1 augmentation is incomplete: $prior_rows/$EXPECTED_ROWS" >&2
    exit 2
  fi
  PRIOR_AUGMENT_SHA="$(sha256sum "$PRIOR_AUGMENT_SOURCE" | cut -d' ' -f1)"
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

snapshot_staging="$(mktemp -d "$SNAPSHOT_PARENT/.e49e-kernel.XXXXXX")"
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
  "$BASE_AUGMENT_SOURCE" \
  "$INVALID_SOURCE" \
  "$EQUIVALENT_SOURCE" \
  "$snapshot_staging/ops/math_strategy_calibration/"
if [[ "$AUGMENT_SOURCE" != "$BASE_AUGMENT_SOURCE" ]]; then
  cp \
    "$AUGMENT_SOURCE" \
    "$snapshot_staging/ops/math_strategy_calibration/"
fi
SNAPSHOT_HASH="$(hash_tree "$snapshot_staging")"
SNAPSHOT_ROOT="$SNAPSHOT_PARENT/e49e_kernel_augmentation_${SNAPSHOT_HASH}"
if [[ ! -e "$SNAPSHOT_ROOT" ]]; then
  mv "$snapshot_staging" "$SNAPSHOT_ROOT"
else
  if [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]]; then
    echo "existing E49E kernel snapshot hash mismatch" >&2
    exit 2
  fi
  find "$snapshot_staging" -type f -delete
  find "$snapshot_staging" -depth -type d -empty -delete
fi
AUGMENT="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$AUGMENT_SOURCE")"
INVALID="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$INVALID_SOURCE")"
EQUIVALENT="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$EQUIVALENT_SOURCE")"

FROZEN_E49D="$EVIDENCE/frozen_e49d_input"
FROZEN_E49E="$EVIDENCE/frozen_e49e_input"
mkdir -p "$FROZEN_E49D" "$FROZEN_E49E"
cp "$INPUT_E49D_SOURCE/menu_records.jsonl" "$FROZEN_E49D/menu_records.jsonl"
cp \
  "$INPUT_E49E_SOURCE/trace_bank_records.jsonl" \
  "$INPUT_E49E_SOURCE/frozen_identity.json" \
  "$INPUT_E49E_SOURCE/materialization_job.json" \
  "$FROZEN_E49E/"
if [[ -f "$INPUT_E49E_SOURCE/generation_summary.json" ]]; then
  cp "$INPUT_E49E_SOURCE/generation_summary.json" "$FROZEN_E49E/"
fi
if [[ "$variant" == v2 ]]; then
  FROZEN_PRIOR="$EVIDENCE/frozen_prior_augmentation"
  mkdir -p "$FROZEN_PRIOR"
  FROZEN_PRIOR_RECORDS="$FROZEN_PRIOR/augmentation_records.jsonl"
  cp "$PRIOR_AUGMENT_SOURCE" "$FROZEN_PRIOR_RECORDS"
fi
FROZEN_ENDPOINT="$EVIDENCE/qwen72_endpoint.json"
cp "$ENDPOINT_SOURCE" "$FROZEN_ENDPOINT"

PREFLIGHT_TMP="$(mktemp "$EVIDENCE/.preflight.XXXXXX")"
E49E_KERNEL_PRIOR_RECORDS="$FROZEN_PRIOR_RECORDS" \
PYTHONPATH="$SNAPSHOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$AUGMENT" \
  --source "$SOURCE" \
  --input-e49d "$FROZEN_E49D" \
  --input-e49e "$FROZEN_E49E" \
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
    record.get("schema") != "e49e_kernel_augmentation_preflight_v1"
    or record.get("row_count") != int(sys.argv[2])
    or record.get("all_controls_pass") is not True
):
    raise SystemExit("E49E kernel-augmentation preflight failed")
PY

IDENTITY="$EVIDENCE/frozen_identity.json"
"$PYTHON_BIN" - "$IDENTITY" "$stage" "$variant" "$SNAPSHOT_ROOT" "$SNAPSHOT_HASH" \
  "$(sha256sum "$FROZEN_E49D/menu_records.jsonl" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_E49E/trace_bank_records.jsonl" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_ENDPOINT" | cut -d' ' -f1)" \
  "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" \
  "$(sha256sum "$INVALID" | cut -d' ' -f1)" \
  "$(sha256sum "$EQUIVALENT" | cut -d' ' -f1)" \
  "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" \
  "$(sha256sum "$CLEAN_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$KERNEL_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$HARDENING_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$GAP_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$CACHE_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$VARIANT_AMENDMENT" | cut -d' ' -f1)" \
  "$PRIOR_AUGMENT_SHA" \
  "$(sha256sum "$AUGMENT" | cut -d' ' -f1)" \
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
    "variant",
    "snapshot_root",
    "snapshot_tree_sha256",
    "e49d_input_sha256",
    "raw_trace_records_sha256",
    "endpoint_record_sha256",
    "preflight_sha256",
    "known_invalid_controls_sha256",
    "known_equivalent_controls_sha256",
    "protocol_sha256",
    "clean_relaunch_amendment_sha256",
    "kernel_augmentation_amendment_sha256",
    "kernel_hardening_amendment_sha256",
    "kernel_gap_eligibility_amendment_sha256",
    "kernel_cache_replay_amendment_sha256",
    "variant_amendment_sha256",
    "prior_augmentation_records_sha256",
    "augmentation_script_sha256",
    "launcher_sha256",
    "slurm_sha256",
    "source_train_tree_sha256",
    "source_eval_tree_sha256",
]
payload = {
    "schema": "e49e_kernel_augmentation_frozen_identity_v1",
    **dict(zip(keys, sys.argv[2:], strict=True)),
    "augmentation_requests_at_freeze": 0,
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
    --job-name="e49e-kernel-${stage}-${variant}" \
    --output="$EVIDENCE/augment-%j.out" \
    --error="$EVIDENCE/augment-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49E_KERNEL_STAGE=$stage,E49E_KERNEL_VARIANT=$variant,E49E_KERNEL_EVIDENCE=$EVIDENCE,E49E_KERNEL_AUGMENT=$AUGMENT,E49E_KERNEL_PRIOR_RECORDS=$FROZEN_PRIOR_RECORDS,E49E_KERNEL_SOURCE_SNAPSHOT=$SNAPSHOT_ROOT,E49E_KERNEL_SOURCE_HASH=$SNAPSHOT_HASH,E49E_KERNEL_FROZEN_E49D=$FROZEN_E49D,E49E_KERNEL_FROZEN_E49E=$FROZEN_E49E,E49E_KERNEL_ENDPOINT=$FROZEN_ENDPOINT,E49E_KERNEL_INVALID_CONTROLS=$INVALID,E49E_KERNEL_EQUIVALENT_CONTROLS=$EQUIVALENT,E49E_KERNEL_IDENTITY=$IDENTITY,E49E_KERNEL_IDENTITY_HASH=$IDENTITY_HASH" \
    "$SLURM"
)"
"$PYTHON_BIN" - "$RECORD" "$job_id" "$stage" "$variant" "$IDENTITY_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e49e_kernel_augmentation_job_v1",
    "job_id": sys.argv[2],
    "stage": sys.argv[3],
    "variant": sys.argv[4],
    "frozen_identity_sha256": sys.argv[5],
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "submitted E49E ${stage} finite-kernel augmentation ${variant} job $job_id"
