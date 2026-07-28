#!/usr/bin/env bash
# Configure or atomically submit E69's six one-time MATH-500 evaluations.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|full) ;;
  *)
    echo "Usage: $0 {config|full}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e69_gate4_math500_one_time_transfer_20260728.md"
GATE2_AUDIT="$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_audit_latest.json"
GATE3_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate3_confirmatory_identity.json"
GATE3_AUDIT="$ROOT_DIR/var/artifacts/e69_gate3_confirmatory_audit_latest.json"
CHECKPOINT_AUDITOR="$ROOT_DIR/ops/route_successor/audit_e69_gate4_checkpoints.py"
CHECKPOINT_AUDIT="$ROOT_DIR/var/artifacts/e69_gate4_checkpoint_audit.json"
EVALUATOR="$ROOT_DIR/ops/route_successor/eval_e69_math500_checkpoint.py"
ANALYZER="$ROOT_DIR/ops/route_successor/analyze_e69_gate4_math500.py"
PLOTTER="$ROOT_DIR/ops/route_successor/plot_e69_five_area_panel.py"
SLURM_WRAPPER="$ROOT_DIR/ops/slurm/e69_math500_eval_node302.slurm"
DATA_ROOT="$ROOT_DIR/var/data/math12k_384_math500"
IDENTITY="$ROOT_DIR/var/artifacts/e69_gate4_math500_identity.json"
MANIFEST="$ROOT_DIR/var/artifacts/e69_gate4_math500_jobs.tsv"
OUTPUT_ROOT="$ROOT_DIR/var/artifacts/e69_gate4_math500"

for required in \
  "$PROTOCOL" "$CHECKPOINT_AUDITOR" "$EVALUATOR" "$ANALYZER" "$PLOTTER" \
  "$SLURM_WRAPPER" \
  "$DATA_ROOT/MATERIALIZATION_MANIFEST.json" \
  "$DATA_ROOT/eval/math/data-00000-of-00001.arrow"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E69 Gate 4 prerequisite: $required" >&2
    exit 1
  fi
done

if [[ "$phase" == config ]]; then
  echo "[e69-gate4] configuration passed; full remains gated on terminal Gate 3"
  exit 0
fi

for required in "$GATE2_AUDIT" "$GATE3_IDENTITY" "$GATE3_AUDIT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing terminal E69 Gate 4 prerequisite: $required" >&2
    exit 1
  fi
done

for fresh in "$IDENTITY" "$MANIFEST"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh E69 Gate 4 artifact required; already exists: $fresh" >&2
    exit 1
  fi
done
if [[ -d "$OUTPUT_ROOT" ]] && find "$OUTPUT_ROOT" -type f -print -quit | grep -q .; then
  echo "Fresh E69 Gate 4 output root required: $OUTPUT_ROOT" >&2
  exit 1
fi

"$PYTHON_BIN" - "$GATE2_AUDIT" "$GATE3_IDENTITY" "$GATE3_AUDIT" <<'PY'
import json
import pathlib
import sys

gate2 = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
identity = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
gate3 = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
if (
    gate2.get("status") != "pass"
    or gate2.get("summary", {}).get("terminal_physical_runs") != 18
    or gate2.get("summary", {}).get("integrity_violations") != 0
):
    raise SystemExit("E69 Gate 4 requires a clean passing Gate 2")
if (
    identity.get("schema") != "e69_gate3_confirmatory_v1"
    or identity.get("confirmatory_physical_jobs") != 30
    or identity.get("checkpoint_selection") != "terminal_pass_6_only"
):
    raise SystemExit("E69 Gate 4 requires the exact Gate 3 identity")
if (
    gate3.get("schema") != "e69_gate3_confirmatory_audit_v1"
    or gate3.get("status") != "complete"
    or gate3.get("summary", {}).get("terminal_physical_runs") != 30
    or gate3.get("summary", {}).get("integrity_violations") != 0
    or gate3.get("math500_sealed") is not True
):
    raise SystemExit("E69 Gate 4 requires clean, complete, sealed Gate 3")
PY

PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" "$PYTHON_BIN" "$CHECKPOINT_AUDITOR"
"$PYTHON_BIN" - "$CHECKPOINT_AUDIT" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    audit.get("schema") != "e69_gate4_checkpoint_audit_v1"
    or audit.get("status") != "pass"
    or audit.get("observed_checkpoints") != 6
    or audit.get("violations")
    or audit.get("math500_sealed") is not True
):
    raise SystemExit("E69 Gate 4 requires six clean terminal checkpoints")
PY

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | cut -d' ' -f1
  )
}

SOURCE_HASH="$("$PYTHON_BIN" - "$GATE3_IDENTITY" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["source_hash"])
PY
)"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_${SOURCE_HASH}/src"
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E69 Gate 4 source snapshot hash mismatch" >&2
  exit 1
fi

eval_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e69-g4-eval.XXXXXX")"
mkdir -p "$eval_input/ops/route_successor" "$eval_input/ops/slurm"
cp "$EVALUATOR" "$eval_input/ops/route_successor/"
cp "$SLURM_WRAPPER" "$eval_input/ops/slurm/"
EVAL_CODE_HASH="$(hash_tree "$eval_input")"
EVAL_SNAPSHOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate4_eval_${EVAL_CODE_HASH}"
if [[ ! -f "$EVAL_SNAPSHOT/ops/route_successor/eval_e69_math500_checkpoint.py" ]]; then
  mv "$eval_input" "$EVAL_SNAPSHOT"
else
  find "$eval_input" -type f -delete
  rmdir "$eval_input/ops/route_successor" "$eval_input/ops/slurm"
  rmdir "$eval_input/ops" "$eval_input"
fi
if [[ "$(hash_tree "$EVAL_SNAPSHOT")" != "$EVAL_CODE_HASH" ]]; then
  echo "E69 Gate 4 evaluator snapshot hash mismatch" >&2
  exit 1
fi
FROZEN_EVALUATOR="$EVAL_SNAPSHOT/ops/route_successor/eval_e69_math500_checkpoint.py"
FROZEN_SLURM="$EVAL_SNAPSHOT/ops/slurm/e69_math500_eval_node302.slurm"

mkdir -p "$OUTPUT_ROOT" "$ROOT_DIR/var/artifacts/logs"
mapfile -t checkpoint_rows < <(
  "$PYTHON_BIN" - "$CHECKPOINT_AUDIT" "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
output_root = pathlib.Path(sys.argv[2]).resolve()
for row in audit["checkpoints"]:
    print(
        "\t".join(
            [
                row["alias"],
                row["arm"],
                str(row["seed"]),
                row["checkpoint"],
                row["checkpoint_tree_sha256"],
                str(output_root / f"{row['alias']}.json"),
            ]
        )
    )
PY
)
if [[ "${#checkpoint_rows[@]}" -ne 6 ]]; then
  echo "E69 Gate 4 checkpoint audit did not yield six evaluations" >&2
  exit 1
fi

released=0
job_ids=()
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "[e69-gate4] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

rows_with_jobs=()
for checkpoint_row in "${checkpoint_rows[@]}"; do
  IFS=$'\t' read -r alias arm seed checkpoint checkpoint_hash output \
    <<< "$checkpoint_row"
  if [[ -e "$output" ]]; then
    echo "Immutable Gate 4 output already exists: $output" >&2
    exit 1
  fi
  export_args="ALL,ROOT_DIR=$ROOT_DIR"
  export_args+=",OAT_ZERO_PYTHON=$PYTHON_BIN"
  export_args+=",OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT"
  export_args+=",E69_GATE4_EVALUATOR=$FROZEN_EVALUATOR"
  export_args+=",E69_GATE4_IDENTITY=$IDENTITY"
  export_args+=",E69_GATE4_ALIAS=$alias"
  export_args+=",E69_GATE4_ARM=$arm"
  export_args+=",E69_GATE4_SEED=$seed"
  export_args+=",E69_GATE4_CHECKPOINT=$checkpoint"
  export_args+=",E69_GATE4_OUTPUT=$output"
  export_args+=",E69_GATE4_DATA_ROOT=$DATA_ROOT"
  job_id="$(sbatch --hold --parsable --export="$export_args" "$FROZEN_SLURM")"
  job_id="${job_id%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid Gate 4 job id: $job_id" >&2
    exit 1
  fi
  job_ids+=("$job_id")
  rows_with_jobs+=(
    "$alias"$'\t'"$arm"$'\t'"$seed"$'\t'"$job_id"$'\t'"$checkpoint"$'\t'"$checkpoint_hash"$'\t'"$output"
  )
done

{
  printf 'alias\tarm\tseed\tjob_id\tcheckpoint\tcheckpoint_tree_sha256\toutput\n'
  printf '%s\n' "${rows_with_jobs[@]}"
} > "$MANIFEST"

export SOURCE_HASH EVAL_CODE_HASH
"$PYTHON_BIN" - \
  "$IDENTITY" "$PROTOCOL" "$0" "$CHECKPOINT_AUDIT" "$GATE2_AUDIT" \
  "$GATE3_IDENTITY" "$GATE3_AUDIT" "$EVALUATOR" "$ANALYZER" "$PLOTTER" \
  "$MANIFEST" "$DATA_ROOT/MATERIALIZATION_MANIFEST.json" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[11])
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
if len(rows) != 6:
    raise SystemExit("Gate 4 manifest must contain six jobs")
payload = {
    "schema": "e69_gate4_math500_one_time_transfer_v1",
    "protocol_sha256": digest(sys.argv[2]),
    "launcher_sha256": digest(sys.argv[3]),
    "checkpoint_audit_sha256": digest(sys.argv[4]),
    "gate2_audit_sha256": digest(sys.argv[5]),
    "gate3_identity_sha256": digest(sys.argv[6]),
    "gate3_audit_sha256": digest(sys.argv[7]),
    "evaluator_sha256": digest(sys.argv[8]),
    "analyzer_sha256": digest(sys.argv[9]),
    "plotter_sha256": digest(sys.argv[10]),
    "manifest_sha256": digest(sys.argv[11]),
    "data_manifest_sha256": digest(sys.argv[12]),
    "source_hash": os.environ["SOURCE_HASH"],
    "evaluation_surface_hash": os.environ["EVAL_CODE_HASH"],
    "attempt_selection": "exact_six_held_manifest_job_ids",
    "checkpoint_selection": "gate3_terminal_pass_6_step_02305_only",
    "evaluations": [
        {
            "alias": row["alias"],
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "checkpoint": row["checkpoint"],
            "checkpoint_tree_sha256": row["checkpoint_tree_sha256"],
            "output": row["output"],
        }
        for row in rows
    ],
    "data": {
        "rows": 500,
        "ordered_row_sha256": (
            "1576fd11df21dc705a7c85000f232031212225cd9c00520faa26f6bdfc751166"
        ),
        "arrow_sha256": (
            "2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7"
        ),
    },
    "requests": {
        "template": "qwen_math",
        "max_tokens": 1024,
        "max_model_len": 2048,
        "greedy": {"n": 1, "temperature": 0, "top_p": 1, "seed": 0},
        "sampled": {"n": 8, "temperature": 1, "top_p": 1, "seed": 690401},
        "verifier": "math_verify",
    },
    "bootstrap": {"replicates": 10000, "seed": 690402},
    "math500_unsealed": True,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

for job_id in "${job_ids[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    "E69_GATE4_IDENTITY=$IDENTITY" \
    "E69_GATE4_DATA_ROOT=$DATA_ROOT" \
    "E69_GATE4_EVALUATOR=$FROZEN_EVALUATOR" \
    "OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT"; do
    if [[ "$record" != *"$required"* ]]; then
      echo "E69 Gate 4 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
done

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e69-gate4] unsealed and released six one-time evaluations: ${job_ids[*]}"
echo "[e69-gate4] identity=$IDENTITY"
