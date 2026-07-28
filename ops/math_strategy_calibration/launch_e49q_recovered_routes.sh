#!/usr/bin/env bash
# Freeze and submit E49Q's visible-trace recovery analysis.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
SOURCE_SCRIPT="$ROOT_DIR/ops/math_strategy_calibration/certify_e49q_recovered_routes.py"
PROTOCOL="$ROOT_DIR/paper/preregistration/e49q_visible_trace_recovery_20260724.md"
SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
ENDPOINT_SOURCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v5/qwen72_endpoint.json"
CALIBRATION_SOURCE="$ROOT_DIR/var/artifacts/e49j_mathir_signature_veto_v1/calibration_report.json"
SLURM="$ROOT_DIR/ops/slurm/e49q_recovered_routes_node915.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/e49q_recovered_routes_v1"
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots"
IDENTITY="$OUTPUT/frozen_identity.json"
PREFLIGHT="$OUTPUT/preflight.json"
JOB="$OUTPUT/recovery_job.json"
EVIDENCE_SOURCE=(
  "$ROOT_DIR/var/artifacts/e49h_curated_distinct_routes_toy_v1"
  "$ROOT_DIR/var/artifacts/e49k_curated_eval_buffer_v1"
  "$ROOT_DIR/var/artifacts/e49n_curated_eval_recovery_v1"
  "$ROOT_DIR/var/artifacts/e49p_curated_eval_reserve_v1"
  "$ROOT_DIR/var/artifacts/e49l_curated_train_buffer_v1"
)

for required in "$PYTHON_BIN" "$SOURCE_SCRIPT" "$PROTOCOL" \
  "$SOURCE/train" "$SOURCE/eval" "$ENDPOINT_SOURCE" \
  "$CALIBRATION_SOURCE" "$SLURM" "${EVIDENCE_SOURCE[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "missing E49Q prerequisite: $required" >&2
    exit 2
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE ANY E49Q 72B REQUEST' "$PROTOCOL"; then
  echo "E49Q protocol is not frozen" >&2
  exit 2
fi
if [[ -e "$JOB" || -e "$OUTPUT/recovered_records.jsonl" ]]; then
  echo "fresh E49Q evidence directory required" >&2
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

mkdir -p "$OUTPUT" "$SNAPSHOT_PARENT"
if [[ "$(find "$OUTPUT/request_cache" -type f 2>/dev/null | wc -l | tr -d ' ')" != 0 ]]; then
  echo "E49Q request cache must be empty at freeze" >&2
  exit 2
fi

snapshot_staging="$(mktemp -d "$SNAPSHOT_PARENT/.e49q-recover.XXXXXX")"
mkdir -p "$snapshot_staging/src" "$snapshot_staging/ops/math_strategy_calibration"
while IFS= read -r -d '' source_file; do
  relative="${source_file#"$ROOT_DIR/src/"}"
  mkdir -p "$snapshot_staging/src/$(dirname "$relative")"
  cp "$source_file" "$snapshot_staging/src/$relative"
done < <(find "$ROOT_DIR/src" -type f -name '*.py' -print0 | sort -z)
while IFS= read -r -d '' source_file; do
  cp "$source_file" "$snapshot_staging/ops/math_strategy_calibration/"
done < <(find "$ROOT_DIR/ops/math_strategy_calibration" -maxdepth 1 \
  -type f -name '*.py' -print0 | sort -z)
SNAPSHOT_HASH="$(hash_tree "$snapshot_staging")"
SNAPSHOT="$SNAPSHOT_PARENT/e49q_recovered_routes_$SNAPSHOT_HASH"
if [[ ! -e "$SNAPSHOT" ]]; then
  mv "$snapshot_staging" "$SNAPSHOT"
else
  if [[ "$(hash_tree "$SNAPSHOT")" != "$SNAPSHOT_HASH" ]]; then
    echo "E49Q snapshot hash collision" >&2
    exit 2
  fi
  find "$snapshot_staging" -type f -delete
  find "$snapshot_staging" -depth -type d -empty -delete
fi
FROZEN_SCRIPT="$SNAPSHOT/ops/math_strategy_calibration/certify_e49q_recovered_routes.py"

FROZEN_EVIDENCE="$OUTPUT/frozen_evidence"
mkdir -p "$FROZEN_EVIDENCE"
EVIDENCE_ARGS=()
EVIDENCE_DIRS=()
for source_evidence in "${EVIDENCE_SOURCE[@]}"; do
  name="$(basename "$source_evidence")"
  destination="$FROZEN_EVIDENCE/$name"
  mkdir -p "$destination"
  for file in frozen_identity.json run_summary.json \
    candidate_records.jsonl curated_contracts.json; do
    cp "$source_evidence/$file" "$destination/$file"
  done
  EVIDENCE_ARGS+=(--evidence "$destination")
  EVIDENCE_DIRS+=("$destination")
done
FROZEN_ENDPOINT="$OUTPUT/qwen72_endpoint.json"
FROZEN_CALIBRATION="$OUTPUT/e49j_calibration_report.json"
cp "$ENDPOINT_SOURCE" "$FROZEN_ENDPOINT"
cp "$CALIBRATION_SOURCE" "$FROZEN_CALIBRATION"

"$PYTHON_BIN" "$FROZEN_SCRIPT" preflight \
  --source "$SOURCE" \
  "${EVIDENCE_ARGS[@]}" \
  --e49j-calibration "$FROZEN_CALIBRATION" >"$PREFLIGHT"
"$PYTHON_BIN" - "$PREFLIGHT" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    record.get("schema") != "e49q_recovered_route_preflight_v1"
    or record.get("source_candidate_count") != 53
    or record.get("eligible_candidate_count") != 14
    or record.get("eligible_train_row_count") != 3
    or record.get("eligible_eval_row_count") != 11
    or record.get("request_count") != 56
    or record.get("all_four_execution_traces_present") is not True
    or record.get("at_least_one_exact_execution_per_route") is not True
):
    raise SystemExit("E49Q preflight failed")
PY

SOURCE_TRAIN_HASH="$(hash_tree "$SOURCE/train")"
SOURCE_EVAL_HASH="$(hash_tree "$SOURCE/eval")"
FROZEN_EVIDENCE_HASH="$(hash_tree "$FROZEN_EVIDENCE")"
"$PYTHON_BIN" - "$IDENTITY" \
  "$(sha256sum "$FROZEN_SCRIPT" | cut -d' ' -f1)" \
  "$SNAPSHOT_HASH" "$SNAPSHOT" \
  "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_ENDPOINT" | cut -d' ' -f1)" \
  "$(sha256sum "$FROZEN_CALIBRATION" | cut -d' ' -f1)" \
  "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" \
  "$FROZEN_EVIDENCE_HASH" "$SOURCE_TRAIN_HASH" "$SOURCE_EVAL_HASH" \
  "$(sha256sum "$SLURM" | cut -d' ' -f1)" \
  "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)" <<'PY'
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
    "endpoint_sha256",
    "e49j_calibration_sha256",
    "preflight_sha256",
    "frozen_evidence_tree_sha256",
    "source_train_tree_sha256",
    "source_eval_tree_sha256",
    "slurm_sha256",
    "launcher_sha256",
)
payload = {
    "schema": "e49q_recovered_routes_frozen_identity_v1",
    **dict(zip(keys, sys.argv[2:], strict=True)),
    "requests_at_freeze": 0,
    "source_candidate_count": 53,
    "eligible_candidate_count": 14,
    "request_count": 56,
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
IDENTITY_HASH="$(sha256sum "$IDENTITY" | cut -d' ' -f1)"
EVIDENCE_JOINED="$(IFS=:; echo "${EVIDENCE_DIRS[*]}")"

job_id="$(
  sbatch --parsable \
    --output="$OUTPUT/recover-%j.out" \
    --error="$OUTPUT/recover-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49Q_FROZEN_SCRIPT=$FROZEN_SCRIPT,E49Q_SOURCE=$SOURCE,E49Q_EVIDENCE_DIRS=$EVIDENCE_JOINED,E49Q_ENDPOINT=$FROZEN_ENDPOINT,E49Q_CALIBRATION=$FROZEN_CALIBRATION,E49Q_OUTPUT=$OUTPUT,E49Q_IDENTITY=$IDENTITY,E49Q_IDENTITY_HASH=$IDENTITY_HASH" \
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
    "schema": "e49q_recovered_routes_job_v1",
    "job_id": sys.argv[2],
    "frozen_identity_sha256": sys.argv[3],
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "submitted E49Q recovered-route job $job_id"
