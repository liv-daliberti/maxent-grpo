#!/usr/bin/env bash
# Submit one durable E49E trace-bank certification job.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONDONTWRITEBYTECODE=1

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
AMENDMENT="$ROOT_DIR/paper/preregistration/e49e_clean_relaunch_amendment_20260724.md"
ENDPOINT_SOURCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json"
EVIDENCE="$ROOT_DIR/var/artifacts/e49e_trace_bank_math_${stage}_v1"
RECORD="$EVIDENCE/materialization_job.json"
CACHE_ROOT="$EVIDENCE/request_cache"
SLURM="$ROOT_DIR/ops/slurm/e49e_materialize_trace_banks_node302.slurm"
SNAPSHOT_PARENT="$ROOT_DIR/var/artifacts/source_snapshots"

if [[ ! -x "$PYTHON_BIN" ]] \
  || [[ ! -f "$PROTOCOL" ]] \
  || [[ ! -f "$AMENDMENT" ]] \
  || [[ ! -f "$ENDPOINT_SOURCE" ]] \
  || [[ ! -f "$SLURM" ]] \
  || ! grep -q '^\*\*Status: FROZEN BEFORE ANY E49E JUDGE REQUEST OR TRAINING LAUNCH' "$PROTOCOL" \
  || ! grep -q '^\*\*Status: FROZEN BEFORE ANY CLEAN-RELAUNCH JUDGE REQUEST OR POLICY TRAINING' "$AMENDMENT"; then
  echo "E49E frozen prerequisite is missing" >&2
  exit 2
fi

if [[ "$stage" == toy ]]; then
  SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
  INPUT_SOURCE="$ROOT_DIR/var/artifacts/e49d_math_strategy_menu_toy_v1"
  OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_toy"
  CONTROLS_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_invalid_controls_toy.json"
  EQUIVALENT_CONTROLS_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_equivalent_controls_toy.json"
  EXPECTED_ROWS=100
else
  SOURCE="$ROOT_DIR/var/data/math12k_384_math500"
  INPUT_SOURCE="$ROOT_DIR/var/artifacts/e49d_math_strategy_menu_full_v1"
  OUTPUT="$ROOT_DIR/var/data/e49e_trace_bank_math_full"
  CONTROLS_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_invalid_controls_full.json"
  EQUIVALENT_CONTROLS_SOURCE="$ROOT_DIR/ops/math_strategy_calibration/e49e_known_equivalent_controls_full.json"
  EXPECTED_ROWS=884
fi

mkdir -p "$EVIDENCE" "$SNAPSHOT_PARENT"
if [[ -e "$RECORD" || -e "$OUTPUT" ]]; then
  echo "E49E ${stage} trace-bank job record already exists: $RECORD" >&2
  exit 2
fi

if [[ ! -f "$INPUT_SOURCE/menu_records.jsonl" ]] \
  || [[ ! -f "$CONTROLS_SOURCE" ]] \
  || [[ ! -f "$EQUIVALENT_CONTROLS_SOURCE" ]]; then
  echo "E49E ${stage} frozen proposal evidence or controls are missing" >&2
  exit 2
fi

# A proposal writer must be terminal before its append-only evidence is
# frozen.  A scientifically failed E49D job is allowed: E49E uses its
# proposals as candidates and re-certifies every retained route from scratch.
INPUT_JOB_RECORD="$INPUT_SOURCE/materialization_job.json"
if [[ -f "$INPUT_JOB_RECORD" ]]; then
  input_job_id="$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' \
      "$INPUT_JOB_RECORD"
  )"
  input_job_state="$(
    sacct -X -n -j "$input_job_id" --format=State -P \
      | head -n 1 | cut -d'|' -f1
  )"
  case "$input_job_state" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*) ;;
    *)
      echo "E49D proposal writer $input_job_id is not terminal: $input_job_state" >&2
      exit 2
      ;;
  esac
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

# A scientifically cancelled predecessor may contribute completed requests,
# but they are immutable and must not be resampled. Bind any inherited cache
# into the new identity before the first new request.
mkdir -p "$CACHE_ROOT"
INHERITED_CACHE_COUNT="$(
  find "$CACHE_ROOT" -type f -name '*.json' | wc -l | tr -d ' '
)"
INHERITED_CACHE_HASH="$(hash_tree "$CACHE_ROOT")"

# Freeze exactly the Python implementation used by preprocessing.  Running
# the snapshot itself makes later live worktree edits irrelevant.
snapshot_staging="$(mktemp -d "$SNAPSHOT_PARENT/.e49e-trace.XXXXXX")"
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
  "$CONTROLS_SOURCE" \
  "$EQUIVALENT_CONTROLS_SOURCE" \
  "$snapshot_staging/ops/math_strategy_calibration/"
SNAPSHOT_HASH="$(hash_tree "$snapshot_staging")"
SNAPSHOT_ROOT="$SNAPSHOT_PARENT/e49e_trace_bank_${SNAPSHOT_HASH}"
if [[ ! -e "$SNAPSHOT_ROOT" ]]; then
  mv "$snapshot_staging" "$SNAPSHOT_ROOT"
else
  if [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]]; then
    echo "existing E49E snapshot hash mismatch" >&2
    exit 2
  fi
  find "$snapshot_staging" -type f -delete
  find "$snapshot_staging" -depth -type d -empty -delete
fi
MATERIALIZER="$SNAPSHOT_ROOT/ops/math_strategy_calibration/materialize_e49e_trace_bank_data.py"
CONTROLS="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$CONTROLS_SOURCE")"
EQUIVALENT_CONTROLS="$SNAPSHOT_ROOT/ops/math_strategy_calibration/$(basename "$EQUIVALENT_CONTROLS_SOURCE")"

FROZEN_INPUT="$EVIDENCE/frozen_e49d_input"
mkdir -p "$FROZEN_INPUT"
cp "$INPUT_SOURCE/menu_records.jsonl" "$FROZEN_INPUT/menu_records.jsonl"
if [[ -f "$INPUT_SOURCE/generation_summary.json" ]]; then
  cp "$INPUT_SOURCE/generation_summary.json" "$FROZEN_INPUT/"
fi
FROZEN_ENDPOINT="$EVIDENCE/qwen72_endpoint.json"
cp "$ENDPOINT_SOURCE" "$FROZEN_ENDPOINT"

PREFLIGHT_TMP="$(mktemp "$EVIDENCE/.preflight.XXXXXX")"
"$PYTHON_BIN" "$MATERIALIZER" \
  --source "$SOURCE" \
  --input-evidence "$FROZEN_INPUT" \
  --output "$OUTPUT" \
  --evidence "$EVIDENCE" \
  --endpoint "$FROZEN_ENDPOINT" \
  --known-invalid-controls "$CONTROLS" \
  --known-equivalent-controls "$EQUIVALENT_CONTROLS" \
  --preflight-only >"$PREFLIGHT_TMP"
PREFLIGHT="$EVIDENCE/preflight.json"
mv "$PREFLIGHT_TMP" "$PREFLIGHT"
"$PYTHON_BIN" - "$PREFLIGHT" "$EXPECTED_ROWS" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    record.get("schema") != "e49e_trace_bank_preflight_v1"
    or record.get("row_count") != int(sys.argv[2])
    or record.get("rows_without_candidate") != []
):
    raise SystemExit("E49E candidate-bank preflight failed")
PY

INPUT_HASH="$(sha256sum "$FROZEN_INPUT/menu_records.jsonl" | cut -d' ' -f1)"
ENDPOINT_HASH="$(sha256sum "$FROZEN_ENDPOINT" | cut -d' ' -f1)"
PREFLIGHT_HASH="$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)"
CONTROLS_HASH="$(sha256sum "$CONTROLS" | cut -d' ' -f1)"
EQUIVALENT_CONTROLS_HASH="$(sha256sum "$EQUIVALENT_CONTROLS" | cut -d' ' -f1)"
PROTOCOL_HASH="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
AMENDMENT_HASH="$(sha256sum "$AMENDMENT" | cut -d' ' -f1)"
LAUNCHER_HASH="$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
SLURM_HASH="$(sha256sum "$SLURM" | cut -d' ' -f1)"
SOURCE_TRAIN_HASH="$(hash_tree "$SOURCE/train")"
SOURCE_EVAL_HASH="$(hash_tree "$SOURCE/eval")"
IDENTITY="$EVIDENCE/frozen_identity.json"
"$PYTHON_BIN" - "$IDENTITY" "$stage" "$SNAPSHOT_ROOT" "$SNAPSHOT_HASH" \
  "$INPUT_HASH" "$ENDPOINT_HASH" "$PREFLIGHT_HASH" "$PROTOCOL_HASH" \
  "$LAUNCHER_HASH" "$SLURM_HASH" "$SOURCE_TRAIN_HASH" "$SOURCE_EVAL_HASH" \
  "$CONTROLS_HASH" "$EQUIVALENT_CONTROLS_HASH" "$AMENDMENT_HASH" \
  "$INHERITED_CACHE_COUNT" "$INHERITED_CACHE_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e49e_trace_bank_frozen_identity_v1",
    "stage": sys.argv[2],
    "snapshot_root": sys.argv[3],
    "snapshot_tree_sha256": sys.argv[4],
    "input_evidence_sha256": sys.argv[5],
    "endpoint_record_sha256": sys.argv[6],
    "preflight_sha256": sys.argv[7],
    "protocol_sha256": sys.argv[8],
    "launcher_sha256": sys.argv[9],
    "slurm_sha256": sys.argv[10],
    "source_train_tree_sha256": sys.argv[11],
    "source_eval_tree_sha256": sys.argv[12],
    "known_invalid_controls_sha256": sys.argv[13],
    "known_equivalent_controls_sha256": sys.argv[14],
    "clean_relaunch_amendment_sha256": sys.argv[15],
    "judge_requests_at_freeze": int(sys.argv[16]),
    "inherited_completed_request_count": int(sys.argv[16]),
    "inherited_request_cache_tree_sha256": sys.argv[17],
    "new_judge_requests_at_freeze": 0,
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
    --job-name="e49e-trace-${stage}" \
    --output="$EVIDENCE/materialize-%j.out" \
    --error="$EVIDENCE/materialize-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49E_STAGE=$stage,E49E_SOURCE_SNAPSHOT=$SNAPSHOT_ROOT,E49E_SOURCE_HASH=$SNAPSHOT_HASH,E49E_FROZEN_INPUT=$FROZEN_INPUT,E49E_FROZEN_ENDPOINT=$FROZEN_ENDPOINT,E49E_FROZEN_IDENTITY=$IDENTITY,E49E_FROZEN_IDENTITY_HASH=$IDENTITY_HASH,E49E_KNOWN_INVALID_CONTROLS=$CONTROLS,E49E_KNOWN_EQUIVALENT_CONTROLS=$EQUIVALENT_CONTROLS,E49E_INHERITED_CACHE_COUNT=$INHERITED_CACHE_COUNT,E49E_INHERITED_CACHE_HASH=$INHERITED_CACHE_HASH" \
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
    "schema": "e49e_trace_bank_job_v1",
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
echo "submitted E49E ${stage} trace-bank job $job_id"
