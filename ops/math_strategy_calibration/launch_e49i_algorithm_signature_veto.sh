#!/usr/bin/env bash
# Freeze and submit E49I's executable algorithm-signature calibration.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
SOURCE_SCRIPT="$ROOT_DIR/ops/math_strategy_calibration/calibrate_e49i_algorithm_signature_veto.py"
BASE_SCRIPT="$ROOT_DIR/ops/math_strategy_calibration/calibrate_e49g_hardened_pair_veto.py"
PROTOCOL="$ROOT_DIR/paper/preregistration/e49i_algorithm_signature_veto_calibration_20260724.md"
V5_EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_toy_repair_v5"
PACKET="$V5_EVIDENCE/manual_audit_packet.jsonl"
LABELS="$V5_EVIDENCE/manual_audit_labels.json"
PRIVATE_KEY="$V5_EVIDENCE/private/manual_audit_key.jsonl"
ENDPOINT="$V5_EVIDENCE/qwen72_endpoint.json"
FAILED_E49G="$ROOT_DIR/var/artifacts/e49g_hardened_pair_veto_v1/calibration_report.json"
SLURM="$ROOT_DIR/ops/slurm/e49i_algorithm_signature_veto_node915.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/e49i_algorithm_signature_veto_v1"
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots"
IDENTITY="$OUTPUT/frozen_identity.json"
JOB="$OUTPUT/calibration_job.json"

for required in "$PYTHON_BIN" "$SOURCE_SCRIPT" "$BASE_SCRIPT" "$PROTOCOL" \
  "$PACKET" "$LABELS" "$PRIVATE_KEY" "$ENDPOINT" "$FAILED_E49G" \
  "$SLURM"; do
  if [[ ! -e "$required" ]]; then
    echo "missing E49I prerequisite: $required" >&2
    exit 2
  fi
done
if ! grep -q '^\*\*Status: FROZEN BEFORE ANY E49I 72B REQUEST' "$PROTOCOL"; then
  echo "E49I protocol is not frozen" >&2
  exit 2
fi
if [[ -e "$JOB" || -e "$OUTPUT/pair_decisions.jsonl" ]]; then
  echo "fresh E49I evidence directory required" >&2
  exit 2
fi
"$PYTHON_BIN" - "$PACKET" "$LABELS" "$PRIVATE_KEY" "$FAILED_E49G" <<'PY'
import json
import sys
packet = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
labels = json.load(open(sys.argv[2], encoding="utf-8"))["labels"]
private = [json.loads(x) for x in open(sys.argv[3], encoding="utf-8") if x.strip()]
failed = json.load(open(sys.argv[4], encoding="utf-8"))
ids = lambda rows: {row["pair_id"] for row in rows}
if (
    len(packet) != 29
    or len(labels) != 29
    or len(private) != 29
    or not ids(packet) == ids(labels) == ids(private)
    or failed.get("schema")
    != "e49g_hardened_pair_calibration_report_v1"
    or failed.get("pass") is not False
    or failed.get("checks", {}).get("false_new_exactly_zero") is not True
    or failed.get("checks", {}).get(
        "at_least_three_of_four_true_distinct_recovered"
    ) is not False
):
    raise SystemExit("E49I predecessor/cohort evidence changed")
PY

mkdir -p "$OUTPUT" "$SNAPSHOT_PARENT"
if [[ "$(find "$OUTPUT/request_cache" -type f 2>/dev/null | wc -l | tr -d ' ')" != 0 ]]; then
  echo "E49I request cache must be empty at freeze" >&2
  exit 2
fi
SCRIPT_HASH="$(sha256sum "$SOURCE_SCRIPT" | cut -d' ' -f1)"
BASE_HASH="$(sha256sum "$BASE_SCRIPT" | cut -d' ' -f1)"
SNAPSHOT="$SNAPSHOT_PARENT/e49i_algorithm_signature_${SCRIPT_HASH}_${BASE_HASH}"
if [[ ! -e "$SNAPSHOT" ]]; then
  mkdir "$SNAPSHOT"
  cp "$SOURCE_SCRIPT" "$SNAPSHOT/calibrate_e49i_algorithm_signature_veto.py"
  cp "$BASE_SCRIPT" "$SNAPSHOT/calibrate_e49g_hardened_pair_veto.py"
elif [[ "$(sha256sum "$SNAPSHOT/calibrate_e49i_algorithm_signature_veto.py" | cut -d' ' -f1)" != "$SCRIPT_HASH" ]] \
  || [[ "$(sha256sum "$SNAPSHOT/calibrate_e49g_hardened_pair_veto.py" | cut -d' ' -f1)" != "$BASE_HASH" ]]; then
  echo "E49I source snapshot collision" >&2
  exit 2
fi
FROZEN_SCRIPT="$SNAPSHOT/calibrate_e49i_algorithm_signature_veto.py"

"$PYTHON_BIN" - "$IDENTITY" \
  "$(sha256sum "$FROZEN_SCRIPT" | cut -d' ' -f1)" \
  "$BASE_HASH" \
  "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" \
  "$(sha256sum "$PACKET" | cut -d' ' -f1)" \
  "$(sha256sum "$LABELS" | cut -d' ' -f1)" \
  "$(sha256sum "$PRIVATE_KEY" | cut -d' ' -f1)" \
  "$(sha256sum "$ENDPOINT" | cut -d' ' -f1)" \
  "$(sha256sum "$FAILED_E49G" | cut -d' ' -f1)" \
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
    "transport_script_sha256",
    "protocol_sha256",
    "packet_sha256",
    "manual_labels_sha256",
    "private_key_sha256",
    "endpoint_sha256",
    "failed_e49g_report_sha256",
    "slurm_sha256",
    "launcher_sha256",
)
payload = {
    "schema": "e49i_algorithm_signature_identity_v1",
    **dict(zip(keys, sys.argv[2:], strict=True)),
    "requests_at_freeze": 0,
    "pair_count": 29,
    "assessment_count": 116,
    "decision_threshold": "at_least_3_of_4",
    "seeds": [493201, 493202, 493211, 493212],
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
    --output="$OUTPUT/calibrate-%j.out" \
    --error="$OUTPUT/calibrate-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49I_FROZEN_SCRIPT=$FROZEN_SCRIPT,E49I_PACKET=$PACKET,E49I_ENDPOINT=$ENDPOINT,E49I_OUTPUT=$OUTPUT,E49I_IDENTITY=$IDENTITY,E49I_IDENTITY_HASH=$IDENTITY_HASH" \
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
    "schema": "e49i_algorithm_signature_job_v1",
    "job_id": sys.argv[2],
    "frozen_identity_sha256": sys.argv[3],
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "submitted E49I algorithm-signature calibration job $job_id"
