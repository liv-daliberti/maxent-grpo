#!/usr/bin/env bash
# Freeze and submit E49H's curated, trace-executed distinct-route cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
SOURCE_SCRIPT="$ROOT_DIR/ops/math_strategy_calibration/certify_e49h_curated_routes.py"
EXPERIMENT="${E49H_EXPERIMENT:-e49h}"
EXPECTED_TRAIN="${E49H_EXPECTED_TRAIN:-12}"
EXPECTED_EVAL="${E49H_EXPECTED_EVAL:-10}"
STATUS_MARKER="${E49H_STATUS_MARKER:-E49H}"
PROTOCOL="${E49H_PROTOCOL:-$ROOT_DIR/paper/preregistration/e49h_curated_distinct_route_expansion_20260724.md}"
AMENDMENT="${E49H_AMENDMENT:-$ROOT_DIR/paper/preregistration/e49h_e49j_restricted_mathir_gate_amendment_20260724.md}"
CONTRACTS_SOURCE="${E49H_CONTRACTS_SOURCE:-$ROOT_DIR/ops/math_strategy_calibration/e49h_curated_distinct_routes_toy.json}"
SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
ENDPOINT_SOURCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v5/qwen72_endpoint.json"
E49J_REPORT_SOURCE="$ROOT_DIR/var/artifacts/e49j_mathir_signature_veto_v1/calibration_report.json"
SLURM="$ROOT_DIR/ops/slurm/e49h_curated_routes_node915.slurm"
OUTPUT="${E49H_OUTPUT_OVERRIDE:-$ROOT_DIR/var/artifacts/e49h_curated_distinct_routes_toy_v1}"
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots"
IDENTITY="$OUTPUT/frozen_identity.json"
PREFLIGHT="$OUTPUT/preflight.json"
JOB="$OUTPUT/certification_job.json"

for required in "$PYTHON_BIN" "$SOURCE_SCRIPT" "$PROTOCOL" \
  "$AMENDMENT" "$CONTRACTS_SOURCE" "$SOURCE/train" "$SOURCE/eval" \
  "$ENDPOINT_SOURCE" "$E49J_REPORT_SOURCE" "$SLURM"; do
  if [[ ! -e "$required" ]]; then
    echo "missing E49H prerequisite: $required" >&2
    exit 2
  fi
done
if ! grep -q "^\\*\\*Status: FROZEN BEFORE ANY $STATUS_MARKER 72B REQUEST" "$PROTOCOL"; then
  echo "E49H protocol is not frozen" >&2
  exit 2
fi
if ! grep -q "^\\*\\*Status: FROZEN BEFORE ANY $STATUS_MARKER 72B REQUEST" "$AMENDMENT"; then
  echo "E49H E49J-gate amendment is not frozen" >&2
  exit 2
fi
if [[ -e "$JOB" || -e "$OUTPUT/candidate_records.jsonl" ]]; then
  echo "fresh E49H evidence directory required" >&2
  exit 2
fi
"$PYTHON_BIN" - "$E49J_REPORT_SOURCE" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
checks = report.get("checks") or {}
if (
    report.get("schema") != "e49j_restricted_mathir_calibration_report_v1"
    or report.get("pass") is not True
    or checks.get("false_new_exactly_zero") is not True
    or checks.get("all_hidden_equivalent_controls_rejected") is not True
    or checks.get("at_least_three_of_four_true_distinct_recovered") is not True
):
    raise SystemExit("E49J calibration does not authorize E49H")
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

mkdir -p "$OUTPUT" "$SNAPSHOT_PARENT"
if [[ "$(find "$OUTPUT/request_cache" -type f 2>/dev/null | wc -l | tr -d ' ')" != 0 ]]; then
  echo "E49H request cache must be empty at freeze" >&2
  exit 2
fi

snapshot_staging="$(mktemp -d "$SNAPSHOT_PARENT/.e49h-curated.XXXXXX")"
mkdir -p "$snapshot_staging/src" "$snapshot_staging/ops/math_strategy_calibration"
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
  "$ROOT_DIR/ops/math_strategy_calibration/calibrate_e49g_hardened_pair_veto.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/calibrate_e49j_mathir_signature_veto.py" \
  "$SOURCE_SCRIPT" \
  "$snapshot_staging/ops/math_strategy_calibration/"
SNAPSHOT_HASH="$(hash_tree "$snapshot_staging")"
SNAPSHOT="$SNAPSHOT_PARENT/e49h_curated_routes_$SNAPSHOT_HASH"
if [[ ! -e "$SNAPSHOT" ]]; then
  mv "$snapshot_staging" "$SNAPSHOT"
else
  if [[ "$(hash_tree "$SNAPSHOT")" != "$SNAPSHOT_HASH" ]]; then
    echo "E49H snapshot hash collision" >&2
    exit 2
  fi
  find "$snapshot_staging" -type f -delete
  find "$snapshot_staging" -depth -type d -empty -delete
fi
FROZEN_SCRIPT="$SNAPSHOT/ops/math_strategy_calibration/certify_e49h_curated_routes.py"

FROZEN_CONTRACTS="$OUTPUT/curated_contracts.json"
FROZEN_ENDPOINT="$OUTPUT/qwen72_endpoint.json"
FROZEN_E49J_REPORT="$OUTPUT/e49j_calibration_report.json"
cp "$CONTRACTS_SOURCE" "$FROZEN_CONTRACTS"
cp "$ENDPOINT_SOURCE" "$FROZEN_ENDPOINT"
cp "$E49J_REPORT_SOURCE" "$FROZEN_E49J_REPORT"

"$PYTHON_BIN" "$FROZEN_SCRIPT" preflight \
  --source "$SOURCE" \
  --contracts "$FROZEN_CONTRACTS" \
  --experiment "$EXPERIMENT" \
  --expected-train "$EXPECTED_TRAIN" \
  --expected-eval "$EXPECTED_EVAL" >"$PREFLIGHT"
"$PYTHON_BIN" - "$PREFLIGHT" "$EXPERIMENT" \
  "$EXPECTED_TRAIN" "$EXPECTED_EVAL" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
experiment = sys.argv[2]
expected_train = int(sys.argv[3])
expected_eval = int(sys.argv[4])
if (
    record.get("schema") != f"{experiment}_curated_route_preflight_v1"
    or record.get("candidate_count") != expected_train + expected_eval
    or record.get("train_candidate_count") != expected_train
    or record.get("eval_candidate_count") != expected_eval
    or record.get("all_contracts_closed_and_nonleaking") is not True
):
    raise SystemExit("E49H preflight failed")
PY

SOURCE_TRAIN_HASH="$(hash_tree "$SOURCE/train")"
SOURCE_EVAL_HASH="$(hash_tree "$SOURCE/eval")"
"$PYTHON_BIN" - "$IDENTITY" \
  "$(sha256sum "$FROZEN_SCRIPT" | cut -d' ' -f1)" \
  "$SNAPSHOT_HASH" \
  "$SNAPSHOT" \
  "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" \
  "$(sha256sum "$AMENDMENT" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_CONTRACTS" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_ENDPOINT" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_E49J_REPORT" | cut -d' ' -f1)" \
  "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" \
  "$SOURCE_TRAIN_HASH" \
  "$SOURCE_EVAL_HASH" \
  "$(sha256sum "$SLURM" | cut -d' ' -f1)" \
  "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)" \
  "$EXPERIMENT" "$EXPECTED_TRAIN" "$EXPECTED_EVAL" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
keys = (
    "script_sha256",
    "snapshot_tree_sha256",
    "snapshot_root",
    "protocol_sha256",
    "e49j_gate_amendment_sha256",
    "contracts_sha256",
    "endpoint_sha256",
    "e49j_calibration_sha256",
    "preflight_sha256",
    "source_train_tree_sha256",
    "source_eval_tree_sha256",
    "slurm_sha256",
    "launcher_sha256",
)
payload = {
    "schema": f"{sys.argv[15]}_curated_routes_frozen_identity_v1",
    **dict(zip(keys, sys.argv[2:15], strict=True)),
    "requests_at_freeze": 0,
    "candidate_count": int(sys.argv[16]) + int(sys.argv[17]),
    "train_candidate_count": int(sys.argv[16]),
    "eval_candidate_count": int(sys.argv[17]),
    "soundness_assessment_count": 4 * (
        int(sys.argv[16]) + int(sys.argv[17])
    ),
    "maximum_pair_veto_assessment_count": 4 * (
        int(sys.argv[16]) + int(sys.argv[17])
    ),
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
    --output="$OUTPUT/certify-%j.out" \
    --error="$OUTPUT/certify-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49H_FROZEN_SCRIPT=$FROZEN_SCRIPT,E49H_SOURCE=$SOURCE,E49H_CONTRACTS=$FROZEN_CONTRACTS,E49H_ENDPOINT=$FROZEN_ENDPOINT,E49H_CALIBRATION=$FROZEN_E49J_REPORT,E49H_OUTPUT=$OUTPUT,E49H_IDENTITY=$IDENTITY,E49H_IDENTITY_HASH=$IDENTITY_HASH,E49H_EXPERIMENT=$EXPERIMENT,E49H_EXPECTED_TRAIN=$EXPECTED_TRAIN,E49H_EXPECTED_EVAL=$EXPECTED_EVAL" \
    "$SLURM"
)"
"$PYTHON_BIN" - "$JOB" "$job_id" "$IDENTITY_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "curated_routes_job_v1",
    "job_id": sys.argv[2],
    "frozen_identity_sha256": sys.argv[3],
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "submitted E49H curated route certification job $job_id"
