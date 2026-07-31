#!/usr/bin/env bash
# Conditionally snapshot and launch the frozen ConstructiveCode v5 Coder gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v5_coder_viability.py"
V4_EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v4_coder_viability.py"
BASE_EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v3_coder_viability.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/evaluate_constructive_code_v5_coder_viability.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/constructive_code_v5_coder_05b_viability_20260730.md"
GATE_AUDIT="$ROOT_DIR/var/artifacts/constructive_code_v5_gate_audit.json"
GATE_IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v5_gate_identity.json"
DATA_ROOT="$ROOT_DIR/var/data/constructive_code_v5"
MODEL="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
IMAGE="$ROOT_DIR/var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"
IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v5_coder_05b_viability_identity.json"
SUBMISSION="$ROOT_DIR/var/artifacts/constructive_code_v5_coder_05b_viability_submission.json"
RECEIPT="$ROOT_DIR/var/artifacts/constructive_code_v5_coder_05b_viability.json"

for required in "$PYTHON_BIN" "$EVALUATOR" "$V4_EVALUATOR" "$BASE_EVALUATOR" "$SLURM_SCRIPT" \
  "$PROTOCOL" "$GATE_IDENTITY" "$MODEL/config.json" "$MODEL/model.safetensors" \
  "$MODEL/tokenizer.json" "$IMAGE"; do
  [[ -e "$required" ]] || { echo "Missing ConstructiveCode v5 viability prerequisite: $required" >&2; exit 1; }
done
[[ -z "$(find -L "$MODEL" -type l -print -quit)" ]] || { echo "Coder snapshot contains a broken symlink" >&2; exit 1; }

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m py_compile "$EVALUATOR" "$V4_EVALUATOR" "$BASE_EVALUATOR"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_constructive_code_v5_coder_viability.py" \
    "$ROOT_DIR/tests/test_constructive_code_v4_coder_viability.py" \
    "$ROOT_DIR/tests/test_constructive_code_v3_coder_viability.py"
  bash -n "$SLURM_SCRIPT"
  echo "[constructive-v5-coder-viability] configuration passed; no model sampled"
  exit 0
fi

[[ -f "$GATE_AUDIT" ]] || { echo "ConstructiveCode v5 gate has not produced an audit; refusing to sample" >&2; exit 1; }
"$PYTHON_BIN" - "$GATE_AUDIT" <<'PYGATE'
import json,pathlib,sys
gate=json.loads(pathlib.Path(sys.argv[1]).read_text())
if gate.get("status") != "pass": raise SystemExit("v5 gate status is not pass")
if gate.get("expected_replay_count") != 2304 or gate.get("observed_replay_count") != 2304: raise SystemExit("v5 gate replay count differs from 2304/2304")
if gate.get("violations") not in ([],None) or gate.get("checker_equivalence_violations") not in ([],None): raise SystemExit("v5 gate contains a hard violation")
tasks=gate.get("task_results")
if not isinstance(tasks,list) or len(tasks)!=12 or any(row.get("status")!="pass" for row in tasks): raise SystemExit("v5 gate did not admit all 12 tasks")
PYGATE

for fresh in "$IDENTITY" "$SUBMISSION" "$RECEIPT"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh ConstructiveCode v5 viability target required: $fresh" >&2; exit 1; }
done
[[ -f "$DATA_ROOT/manifest.json" ]] || { echo "Missing admitted v5 data root" >&2; exit 1; }

SOURCE_HASH="$(jq -er '.source_hash' "$GATE_IDENTITY")"
GATE_EXEC_HASH="$(jq -er '.execution_hash' "$GATE_IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v5_source_${SOURCE_HASH}/src"
GATE_OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v5_gate_${GATE_EXEC_HASH}"
[[ -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" && -f "$GATE_OPS_ROOT/replay_constructive_code_v5.py" ]] || { echo "ConstructiveCode v5 gate snapshots are missing" >&2; exit 1; }
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "Gate source snapshot hash mismatch" >&2; exit 1; }
[[ "$(hash_tree "$GATE_OPS_ROOT")" == "$GATE_EXEC_HASH" ]] || { echo "Gate ops snapshot hash mismatch" >&2; exit 1; }

EXEC_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.constructive-v5-coder-viability.XXXXXX")"
cp "$EVALUATOR" "$EXEC_INPUT/evaluate_constructive_code_v5_coder_viability.py"
cp "$V4_EVALUATOR" "$EXEC_INPUT/evaluate_constructive_code_v4_coder_viability.py"
cp "$BASE_EVALUATOR" "$EXEC_INPUT/evaluate_constructive_code_v3_coder_viability.py"
cp "$SLURM_SCRIPT" "$EXEC_INPUT/evaluate_constructive_code_v5_coder_viability.slurm"
cp "$PROTOCOL" "$EXEC_INPUT/constructive_code_v5_coder_05b_viability_20260730.md"
EXECUTION_HASH="$(hash_tree "$EXEC_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v5_coder_viability_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/evaluate_constructive_code_v5_coder_viability.py" ]]; then
  mv "$EXEC_INPUT" "$EXECUTION_ROOT"
else
  find "$EXEC_INPUT" -type f -delete
  rmdir "$EXEC_INPUT"
fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || { echo "Viability execution snapshot mismatch" >&2; exit 1; }

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold --partition=all --account=allcs \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_GATE_OPS_ROOT=$GATE_OPS_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
  "$EXECUTION_ROOT/evaluate_constructive_code_v5_coder_viability.slurm")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid ConstructiveCode v5 viability job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$GATE_EXEC_HASH" "$EXECUTION_HASH" \
  "$PROTOCOL" "$0" "$GATE_AUDIT" "$GATE_IDENTITY" "$MODEL/config.json" "$MODEL/tokenizer.json" "$DATA_ROOT/manifest.json" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1])
payload={
 "schema_version":"constructive-code-v5-coder-05b-viability-identity-v1",
 "job_id":int(sys.argv[2]),"source_hash":sys.argv[3],"gate_execution_hash":sys.argv[4],"execution_hash":sys.argv[5],
 "protocol_sha256":digest(sys.argv[6]),"launcher_sha256":digest(sys.argv[7]),"gate_audit_sha256":digest(sys.argv[8]),
 "gate_identity_sha256":digest(sys.argv[9]),"model_config_sha256":digest(sys.argv[10]),"tokenizer_json_sha256":digest(sys.argv[11]),
 "v5_manifest_sha256":digest(sys.argv[12]),"model":"Qwen2.5-Coder-0.5B-Instruct",
 "model_snapshot":"ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
 "development_problem_ids":["359_B","988_A","1283_C","1399_D"],"evaluation_rows_loaded":False,
 "seed":77101,"sample_count_per_task":64,"prefix_count":16,"max_tokens":1024,
 "network_enabled":False,"language_model_sampling":True
}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID

record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'gres/gpu:a5000:1' 'NumCPUs=16' 'MinMemoryNode=64G' 'TimeLimit=04:00:00'; do
  [[ "$record" == *"$required"* ]] || { echo "Held ConstructiveCode v5 viability job missing $required" >&2; exit 1; }
done
[[ "$record" == *'Partition=all'* || "$record" == *'Partition=mltheory'* ]] || { echo "Held ConstructiveCode v5 viability job has unexpected partition" >&2; exit 1; }
"$PYTHON_BIN" - "$SUBMISSION" "$job_id" "$IDENTITY" "$record" <<'PYSUB'
import hashlib,json,os,pathlib,sys,tempfile
path=pathlib.Path(sys.argv[1]); identity=pathlib.Path(sys.argv[3])
payload={"schema_version":"constructive-code-v5-coder-05b-viability-submission-v1","job_id":int(sys.argv[2]),"identity_sha256":hashlib.sha256(identity.read_bytes()).hexdigest(),"held_job_audit":"pass","released":True,"scheduler_record":sys.argv[4]}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYSUB
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[constructive-v5-coder-viability] released job $job_id"
echo "[constructive-v5-coder-viability] identity=$IDENTITY"
