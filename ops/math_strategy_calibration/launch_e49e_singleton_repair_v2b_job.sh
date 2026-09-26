#!/usr/bin/env bash
# Freeze and submit the fresh-seed answer-blind V2b repair.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

if [[ "${1:-}" != toy ]]; then
  echo "Usage: $0 toy" >&2
  exit 2
fi

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
PRIOR_EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v1"
PRIOR_RECORDS="$PRIOR_EVIDENCE/repair_records.jsonl"
PRIOR_JOB_RECORD="$PRIOR_EVIDENCE/repair_job.json"
OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_toy_repaired_v2b"
EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v2b"
ABORTED_V2="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v2_aborted_node302_inflight_20260724"
FROZEN_E49D_SOURCE="$PRIOR_EVIDENCE/frozen_e49d_input"
FROZEN_E49E_SOURCE="$PRIOR_EVIDENCE/frozen_e49e_input"
FROZEN_AUGMENT_SOURCE="$PRIOR_EVIDENCE/frozen_kernel_augmentation"
ENDPOINT_SOURCE="$PRIOR_EVIDENCE/qwen72_endpoint.json"
INVALID_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_invalid_controls_toy.json"
EQUIVALENT_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_equivalent_controls_toy.json"
AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_answer_blind_symbolic_singleton_v2_amendment_20260724.md"
ABORT_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_v2_node302_inflight_abort_amendment_20260724.md"
RELAUNCH_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_answer_blind_v2b_node915_relaunch_amendment_20260724.md"
CURATED_AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_curated_singleton_contingency_amendment_20260724.md"
CURATED_CONTRACTS="$ROOT_DIR/ops/math_strategy_calibration/e49e_curated_singleton_contracts_toy.json"
SLURM="$ROOT_DIR/ops/slurm/e49e_repair_singletons_v2b_node915.slurm"
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots"
RECORD="$EVIDENCE/repair_job.json"

for required in \
  "$PYTHON_BIN" \
  "$PRIOR_RECORDS" \
  "$PRIOR_JOB_RECORD" \
  "$FROZEN_E49D_SOURCE/menu_records.jsonl" \
  "$FROZEN_E49E_SOURCE/trace_bank_records.jsonl" \
  "$FROZEN_AUGMENT_SOURCE/augmentation_records.jsonl" \
  "$ENDPOINT_SOURCE" \
  "$INVALID_SOURCE" \
  "$EQUIVALENT_SOURCE" \
  "$AMENDMENT" \
  "$ABORT_AMENDMENT" \
  "$RELAUNCH_AMENDMENT" \
  "$ABORTED_V2/frozen_identity.json" \
  "$ABORTED_V2/repair_job.json" \
  "$CURATED_AMENDMENT" \
  "$CURATED_CONTRACTS" \
  "$SLURM" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2b.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v3_curated.py"; do
  if [[ ! -e "$required" ]]; then
    echo "missing V2b repair prerequisite: $required" >&2
    exit 2
  fi
done
if ! grep -q \
  '^\*\*Status: FROZEN BEFORE ANY V2 SINGLETON PROPOSAL OR AUDIT REQUEST' \
  "$AMENDMENT"; then
  echo "V2 repair amendment is not frozen" >&2
  exit 2
fi
if ! grep -q \
  '^\*\*Status: FROZEN BEFORE ANY V2B PROPOSAL OR AUDIT REQUEST' \
  "$RELAUNCH_AMENDMENT"; then
  echo "V2b relaunch amendment is not frozen" >&2
  exit 2
fi
if ! grep -q \
  '^\*\*Status: FROZEN BEFORE ANY ANSWER-BLIND V2 PROPOSAL OR CURATED-CONTRACT AUDIT REQUEST' \
  "$CURATED_AMENDMENT"; then
  echo "curated singleton contingency is not frozen" >&2
  exit 2
fi
if [[ -e "$RECORD" || -e "$OUTPUT" ]]; then
  echo "fresh V2b singleton repair required" >&2
  exit 2
fi

prior_job_id="$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' \
    "$PRIOR_JOB_RECORD"
)"
prior_state="$(
  sacct -X -n -j "$prior_job_id" --format=State -P \
    | head -n 1 | cut -d'|' -f1
)"
case "$prior_state" in
  COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*) ;;
  *)
    echo "V1 repair $prior_job_id is not terminal: $prior_state" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" - "$PRIOR_RECORDS" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
ids = [row.get("row_id") for row in rows]
if len(rows) != 11 or len(set(ids)) != 11 or None in ids:
    raise SystemExit(f"terminal V1 repair evidence is incomplete: {len(set(ids))}/11")
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
snapshot_staging="$(mktemp -d "$SNAPSHOT_PARENT/.e49e-repair-v2b.XXXXXX")"
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
  "$CURATED_CONTRACTS" \
  "$INVALID_SOURCE" \
  "$EQUIVALENT_SOURCE" \
  "$snapshot_staging/ops/math_strategy_calibration/"
SNAPSHOT_HASH="$(hash_tree "$snapshot_staging")"
SNAPSHOT_ROOT="$SNAPSHOT_PARENT/e49e_singleton_repair_v2b_${SNAPSHOT_HASH}"
if [[ ! -e "$SNAPSHOT_ROOT" ]]; then
  mv "$snapshot_staging" "$SNAPSHOT_ROOT"
else
  if [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]]; then
    echo "existing V2b repair snapshot hash mismatch" >&2
    exit 2
  fi
  find "$snapshot_staging" -type f -delete
  find "$snapshot_staging" -depth -type d -empty -delete
fi

REPAIR="$SNAPSHOT_ROOT/ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2b.py"
INVALID="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$INVALID_SOURCE")"
EQUIVALENT="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$EQUIVALENT_SOURCE")"
FROZEN_E49D="$EVIDENCE/frozen_e49d_input"
FROZEN_E49E="$EVIDENCE/frozen_e49e_input"
FROZEN_AUGMENT="$EVIDENCE/frozen_kernel_augmentation"
mkdir -p "$FROZEN_E49D" "$FROZEN_E49E" "$FROZEN_AUGMENT"
cp "$FROZEN_E49D_SOURCE/menu_records.jsonl" "$FROZEN_E49D/"
cp "$FROZEN_E49E_SOURCE/"* "$FROZEN_E49E/"
cp "$FROZEN_AUGMENT_SOURCE/"* "$FROZEN_AUGMENT/"
FROZEN_ENDPOINT="$EVIDENCE/qwen72_endpoint.json"
FROZEN_PRIOR="$EVIDENCE/frozen_v1_repair_records.jsonl"
cp "$ENDPOINT_SOURCE" "$FROZEN_ENDPOINT"
cp "$PRIOR_RECORDS" "$FROZEN_PRIOR"

PREFLIGHT="$EVIDENCE/preflight.json"
E49E_PRIOR_REPAIR_RECORDS="$FROZEN_PRIOR" \
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
    raise SystemExit("V2b singleton-repair preflight failed")
PY

IDENTITY="$EVIDENCE/frozen_identity.json"
"$PYTHON_BIN" - "$IDENTITY" \
  "$SNAPSHOT_ROOT" \
  "$SNAPSHOT_HASH" \
  "$(sha256sum "$FROZEN_E49D/menu_records.jsonl" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_E49E/trace_bank_records.jsonl" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_AUGMENT/augmentation_records.jsonl" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_PRIOR" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_ENDPOINT" | cut -d' ' -f1)" \
  "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" \
  "$(sha256sum "$AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$ABORT_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$RELAUNCH_AMENDMENT" | cut -d' ' -f1)" \
  "$(hash_tree "$ABORTED_V2")" \
  "$(sha256sum "$CURATED_AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$CURATED_CONTRACTS" | cut -d' ' -f1)" \
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
    "e49d_input_sha256",
    "raw_trace_records_sha256",
    "kernel_augmentation_records_sha256",
    "prior_v1_repair_records_sha256",
    "endpoint_record_sha256",
    "preflight_sha256",
    "v2_amendment_sha256",
    "v2_abort_amendment_sha256",
    "v2b_relaunch_amendment_sha256",
    "aborted_v2_evidence_tree_sha256",
    "curated_contingency_amendment_sha256",
    "curated_contracts_sha256",
    "launcher_sha256",
    "slurm_sha256",
    "source_train_tree_sha256",
    "source_eval_tree_sha256",
)
payload = {
    "schema": "e49e_singleton_repair_v2b_frozen_identity_v1",
    **dict(zip(keys, sys.argv[2:], strict=True)),
    "proposal_input_scope": "problem_only",
    "v2b_judge_requests_at_freeze": 0,
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
    --job-name="e49e-repair-v2b-toy" \
    --output="$EVIDENCE/repair-%j.out" \
    --error="$EVIDENCE/repair-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49E_REPAIR_SOURCE_SNAPSHOT=$SNAPSHOT_ROOT,E49E_REPAIR_SOURCE_HASH=$SNAPSHOT_HASH,E49E_REPAIR_FROZEN_E49D=$FROZEN_E49D,E49E_REPAIR_FROZEN_E49E=$FROZEN_E49E,E49E_REPAIR_FROZEN_AUGMENT=$FROZEN_AUGMENT,E49E_REPAIR_ENDPOINT=$FROZEN_ENDPOINT,E49E_REPAIR_INVALID_CONTROLS=$INVALID,E49E_REPAIR_EQUIVALENT_CONTROLS=$EQUIVALENT,E49E_REPAIR_PRIOR_RECORDS=$FROZEN_PRIOR,E49E_REPAIR_IDENTITY=$IDENTITY,E49E_REPAIR_IDENTITY_HASH=$IDENTITY_HASH" \
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
    "schema": "e49e_singleton_repair_v2b_job_v1",
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
echo "submitted E49E toy answer-blind singleton-repair V2b job $job_id"
