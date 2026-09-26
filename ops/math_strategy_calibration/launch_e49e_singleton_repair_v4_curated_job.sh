#!/usr/bin/env bash
# Freeze and submit the two-row V4 curated-contract correction.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
if [[ "${1:-}" != toy ]]; then
  echo "Usage: $0 toy" >&2
  exit 2
fi

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
V3_EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v3"
V3_RECORDS="$V3_EVIDENCE/repair_records.jsonl"
V3_JOB="$V3_EVIDENCE/repair_job.json"
V3_IDENTITY="$V3_EVIDENCE/frozen_identity.json"
V3_CONTRACTS="$ROOT_DIR/ops/math_strategy_calibration/e49e_curated_singleton_contracts_toy.json"
V4_CONTRACTS="$ROOT_DIR/ops/math_strategy_calibration/e49e_curated_singleton_contracts_toy_v4.json"
V4_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_curated_v4_two_row_correction_amendment_20260724.md"
OBJECTIVE_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_curated_objective_dual_audit_amendment_20260724.md"
INVALID_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_invalid_controls_toy.json"
EQUIVALENT_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_equivalent_controls_toy.json"
OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_toy_repaired_v4"
EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v4"
SLURM="$ROOT_DIR/ops/slurm/e49e_repair_singletons_v4_curated_node915.slurm"
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots"
RECORD="$EVIDENCE/repair_job.json"

for required in \
  "$PYTHON_BIN" \
  "$V3_RECORDS" \
  "$V3_JOB" \
  "$V3_IDENTITY" \
  "$V3_EVIDENCE/frozen_e49d_input/menu_records.jsonl" \
  "$V3_EVIDENCE/frozen_e49e_input/trace_bank_records.jsonl" \
  "$V3_EVIDENCE/frozen_kernel_augmentation/augmentation_records.jsonl" \
  "$V3_EVIDENCE/qwen72_endpoint.json" \
  "$V3_CONTRACTS" \
  "$V4_CONTRACTS" \
  "$V4_AMENDMENT" \
  "$OBJECTIVE_AMENDMENT" \
  "$INVALID_SOURCE" \
  "$EQUIVALENT_SOURCE" \
  "$SLURM" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2b.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v3_curated.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v4_curated.py"; do
  if [[ ! -e "$required" ]]; then
    echo "missing V4 repair prerequisite: $required" >&2
    exit 2
  fi
done
if ! grep -q \
  '^\*\*Status: FROZEN BEFORE ANY V4 AUDIT REQUEST' \
  "$V4_AMENDMENT"; then
  echo "V4 correction amendment is not frozen" >&2
  exit 2
fi
if [[ -e "$RECORD" || -e "$OUTPUT" ]]; then
  echo "fresh V4 curated repair required" >&2
  exit 2
fi

v3_job_id="$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' \
    "$V3_JOB"
)"
v3_state="$(
  sacct -X -n -j "$v3_job_id" --format=State -P \
    | head -n 1 | cut -d'|' -f1
)"
case "$v3_state" in
  COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*) ;;
  *)
    echo "V3 repair $v3_job_id is not terminal: $v3_state" >&2
    exit 2
    ;;
esac
"$PYTHON_BIN" - "$V3_RECORDS" "$V4_CONTRACTS" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
contracts = json.load(open(sys.argv[2], encoding="utf-8"))
ids = [row.get("row_id") for row in rows]
failures = {row["row_id"] for row in rows if row.get("pass") is not True}
if (
    len(rows) != 11
    or len(set(ids)) != 11
    or sum(row.get("pass") is True for row in rows) != 9
    or failures != set(contracts)
):
    raise SystemExit("terminal V3 evidence or V4 correction set changed")
PY

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

mkdir -p "$EVIDENCE" "$SNAPSHOT_PARENT"
snapshot_staging="$(mktemp -d "$SNAPSHOT_PARENT/.e49e-repair-v4.XXXXXX")"
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
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2b.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v3_curated.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v4_curated.py" \
  "$V3_CONTRACTS" \
  "$V4_CONTRACTS" \
  "$INVALID_SOURCE" \
  "$EQUIVALENT_SOURCE" \
  "$snapshot_staging/ops/math_strategy_calibration/"
SNAPSHOT_HASH="$(hash_tree "$snapshot_staging")"
SNAPSHOT_ROOT="$SNAPSHOT_PARENT/e49e_singleton_repair_v4_${SNAPSHOT_HASH}"
if [[ ! -e "$SNAPSHOT_ROOT" ]]; then
  mv "$snapshot_staging" "$SNAPSHOT_ROOT"
else
  if [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]]; then
    echo "existing V4 repair snapshot hash mismatch" >&2
    exit 2
  fi
  find "$snapshot_staging" -type f -delete
  find "$snapshot_staging" -depth -type d -empty -delete
fi
REPAIR="$SNAPSHOT_ROOT/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v4_curated.py"
FROZEN_V3_CONTRACTS="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$V3_CONTRACTS")"
FROZEN_V4_CONTRACTS="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$V4_CONTRACTS")"
INVALID="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$INVALID_SOURCE")"
EQUIVALENT="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$EQUIVALENT_SOURCE")"

FROZEN_E49D="$EVIDENCE/frozen_e49d_input"
FROZEN_E49E="$EVIDENCE/frozen_e49e_input"
FROZEN_AUGMENT="$EVIDENCE/frozen_kernel_augmentation"
mkdir -p "$FROZEN_E49D" "$FROZEN_E49E" "$FROZEN_AUGMENT"
cp "$V3_EVIDENCE/frozen_e49d_input/"* "$FROZEN_E49D/"
cp "$V3_EVIDENCE/frozen_e49e_input/"* "$FROZEN_E49E/"
cp "$V3_EVIDENCE/frozen_kernel_augmentation/"* "$FROZEN_AUGMENT/"
FROZEN_ENDPOINT="$EVIDENCE/qwen72_endpoint.json"
FROZEN_PRIOR="$EVIDENCE/frozen_v3_repair_records.jsonl"
cp "$V3_EVIDENCE/qwen72_endpoint.json" "$FROZEN_ENDPOINT"
cp "$V3_RECORDS" "$FROZEN_PRIOR"

PREFLIGHT="$EVIDENCE/preflight.json"
E49E_PRIOR_REPAIR_V3_RECORDS="$FROZEN_PRIOR" \
E49E_V3_CURATED_CONTRACTS="$FROZEN_V3_CONTRACTS" \
E49E_V4_CURATED_CONTRACTS="$FROZEN_V4_CONTRACTS" \
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
  --preflight-only >"$PREFLIGHT"
"$PYTHON_BIN" - "$PREFLIGHT" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    record.get("row_count") != 100
    or record.get("singleton_repair_row_count") != 11
    or record.get("trace_certified_row_count") != 89
    or record.get("all_controls_pass") is not True
):
    raise SystemExit("V4 curated-repair preflight failed")
PY

IDENTITY="$EVIDENCE/frozen_identity.json"
"$PYTHON_BIN" - "$IDENTITY" \
  "$SNAPSHOT_ROOT" \
  "$SNAPSHOT_HASH" \
  "$(sha256sum "$V3_IDENTITY" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_PRIOR" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_V3_CONTRACTS" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_V4_CONTRACTS" | cut -d' ' -f1)" \
  "$(sha256sum "$V4_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$OBJECTIVE_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_ENDPOINT" | cut -d' ' -f1)" \
  "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" \
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
keys = (
    "snapshot_root",
    "snapshot_tree_sha256",
    "v3_frozen_identity_sha256",
    "prior_v3_repair_records_sha256",
    "v3_contracts_sha256",
    "v4_contracts_sha256",
    "v4_amendment_sha256",
    "objective_dual_audit_amendment_sha256",
    "endpoint_record_sha256",
    "preflight_sha256",
    "launcher_sha256",
    "slurm_sha256",
    "source_train_tree_sha256",
    "source_eval_tree_sha256",
)
payload = {
    "schema": "e49e_singleton_repair_v4_frozen_identity_v1",
    **dict(zip(keys, sys.argv[2:], strict=True)),
    "v4_audit_requests_at_freeze": 0,
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
    --job-name="e49e-repair-v4-toy" \
    --output="$EVIDENCE/repair-%j.out" \
    --error="$EVIDENCE/repair-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49E_REPAIR_SOURCE_SNAPSHOT=$SNAPSHOT_ROOT,E49E_REPAIR_SOURCE_HASH=$SNAPSHOT_HASH,E49E_REPAIR_FROZEN_E49D=$FROZEN_E49D,E49E_REPAIR_FROZEN_E49E=$FROZEN_E49E,E49E_REPAIR_FROZEN_AUGMENT=$FROZEN_AUGMENT,E49E_REPAIR_ENDPOINT=$FROZEN_ENDPOINT,E49E_REPAIR_INVALID_CONTROLS=$INVALID,E49E_REPAIR_EQUIVALENT_CONTROLS=$EQUIVALENT,E49E_REPAIR_PRIOR_V3=$FROZEN_PRIOR,E49E_REPAIR_V3_CONTRACTS=$FROZEN_V3_CONTRACTS,E49E_REPAIR_V4_CONTRACTS=$FROZEN_V4_CONTRACTS,E49E_REPAIR_IDENTITY=$IDENTITY,E49E_REPAIR_IDENTITY_HASH=$IDENTITY_HASH" \
    "$SLURM"
)"
"$PYTHON_BIN" - "$RECORD" "$job_id" "$IDENTITY_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e49e_singleton_repair_v4_job_v1",
    "job_id": sys.argv[2],
    "stage": "toy",
    "frozen_identity_sha256": sys.argv[3],
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "submitted E49E toy curated-repair V4 job $job_id"
