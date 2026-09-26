#!/usr/bin/env bash
# Snapshot and launch the preregistered ConstructiveCode 1.5B capacity retry.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v8_coder_15b_viability.py"
V6_EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v6_coder_viability.py"
V4_EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v4_coder_viability.py"
BASE_EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v3_coder_viability.py"
GATE_AUDITOR="$ROOT_DIR/ops/audit_constructive_code_v6_gate.py"
BATCH="$ROOT_DIR/ops/slurm/evaluate_constructive_code_v8_coder_15b_viability.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/constructive_code_v8_coder_15b_capacity_retry_20260730.md"
MODEL="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
GATE_AUDIT="$ROOT_DIR/var/artifacts/constructive_code_v6_gate_audit.json"
GATE_IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v6_gate_identity.json"
V6_MANIFEST="$ROOT_DIR/var/data/constructive_code_v6/manifest.json"
IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v8_coder_15b_viability_identity.json"
SUBMISSION="$ROOT_DIR/var/artifacts/constructive_code_v8_coder_15b_viability_submission.json"
RECEIPT="$ROOT_DIR/var/artifacts/constructive_code_v8_coder_15b_viability.json"

for required in "$PYTHON" "$EVALUATOR" "$V6_EVALUATOR" "$V4_EVALUATOR" \
  "$BASE_EVALUATOR" "$GATE_AUDITOR" "$BATCH" "$PROTOCOL" "$GATE_AUDIT" \
  "$GATE_IDENTITY" "$V6_MANIFEST" "$MODEL/config.json" "$MODEL/model.safetensors" \
  "$MODEL/tokenizer.json"; do
  [[ -e "$required" ]] || { echo "Missing ConstructiveCode v8 prerequisite: $required" >&2; exit 1; }
done
[[ -z "$(find -L "$MODEL" -type l -print -quit)" ]] || {
  echo "Qwen2.5-Coder-1.5B snapshot contains a broken symlink" >&2; exit 1;
}

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON" -m py_compile \
    "$EVALUATOR" "$V6_EVALUATOR" "$V4_EVALUATOR" "$BASE_EVALUATOR" "$GATE_AUDITOR"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON" -m pytest -q \
    "$ROOT_DIR/tests/test_constructive_code_v8_coder_15b_viability.py" \
    "$ROOT_DIR/tests/test_constructive_code_v6_coder_viability.py" \
    "$ROOT_DIR/tests/test_constructive_code_v6_gate.py"
  bash -n "$BATCH"
  echo "[constructive-v8-coder-15b] configuration passed; no model sampled"
  exit 0
fi

for fresh in "$IDENTITY" "$SUBMISSION" "$RECEIPT"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh ConstructiveCode v8 target required: $fresh" >&2; exit 1; }
done
"$PYTHON" - "$GATE_AUDIT" <<'PYGATE'
import json,pathlib,sys
gate=json.loads(pathlib.Path(sys.argv[1]).read_text())
if gate.get("status") != "pass": raise SystemExit("v6 gate status is not pass")
if gate.get("expected_replay_count") != 960 or gate.get("observed_replay_count") != 960: raise SystemExit("v6 gate is not 960/960")
if gate.get("violations") not in ([],None) or gate.get("checker_equivalence_violations") not in ([],None): raise SystemExit("v6 gate has violations")
if gate.get("evaluation_rows_loaded") is not False or gate.get("language_model_sampling") is not False: raise SystemExit("v6 gate crossed its boundary")
PYGATE

SOURCE_HASH="$(jq -er '.source_hash' "$GATE_IDENTITY")"
GATE_EXEC_HASH="$(jq -er '.execution_hash' "$GATE_IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v5_source_${SOURCE_HASH}/src"
GATE_OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v5_gate_${GATE_EXEC_HASH}"
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "source snapshot drift" >&2; exit 1; }
[[ "$(hash_tree "$GATE_OPS_ROOT")" == "$GATE_EXEC_HASH" ]] || { echo "gate snapshot drift" >&2; exit 1; }

EXEC_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.constructive-v8-coder-15b.XXXXXX")"
for source in "$EVALUATOR" "$V6_EVALUATOR" "$V4_EVALUATOR" "$BASE_EVALUATOR" "$GATE_AUDITOR" "$BATCH" "$PROTOCOL"; do
  cp "$source" "$EXEC_INPUT/$(basename "$source")"
done
EXECUTION_HASH="$(hash_tree "$EXEC_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v8_coder_15b_${EXECUTION_HASH}"
if [[ ! -d "$EXECUTION_ROOT" ]]; then mv "$EXEC_INPUT" "$EXECUTION_ROOT"; else find "$EXEC_INPUT" -type f -delete; rmdir "$EXEC_INPUT"; fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || { echo "v8 execution snapshot drift" >&2; exit 1; }

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold --partition=all --account=allcs \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_GATE_OPS_ROOT=$GATE_OPS_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
  "$EXECUTION_ROOT/$(basename "$BATCH")")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid v8 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$PYTHON" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$GATE_EXEC_HASH" "$EXECUTION_HASH" \
  "$PROTOCOL" "$0" "$GATE_AUDIT" "$GATE_IDENTITY" "$MODEL/config.json" \
  "$MODEL/tokenizer.json" "$V6_MANIFEST" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def d(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
p=pathlib.Path(sys.argv[1]); payload={
 "schema":"constructive-code-v8-coder-15b-viability-identity-v1",
 "job_id":int(sys.argv[2]),"source_hash":sys.argv[3],"gate_execution_hash":sys.argv[4],"execution_hash":sys.argv[5],
 "protocol_sha256":d(sys.argv[6]),"launcher_sha256":d(sys.argv[7]),"gate_audit_sha256":d(sys.argv[8]),
 "gate_identity_sha256":d(sys.argv[9]),"model_config_sha256":d(sys.argv[10]),"tokenizer_json_sha256":d(sys.argv[11]),
 "v6_manifest_sha256":d(sys.argv[12]),"model":"Qwen2.5-Coder-1.5B-Instruct",
 "model_snapshot":"2e1fd397ee46e1388853d2af2c993145b0f1098a",
 "development_problem_ids":["359_B","988_A","1399_D"],"evaluation_problem_ids":["361_B","1294_C","149_C"],
 "evaluation_rows_loaded":False,"seed":77101,"sample_count_per_task":64,"prefix_count":16,"max_tokens":1024,
 "network_enabled":False,"language_model_sampling":True,"capacity_retry":True,"separate_from_05b_rows":True
}
p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w") as h: json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PYID

record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'gres/gpu:a5000:1' 'NumCPUs=16' 'MinMemoryNode=64G' 'TimeLimit=04:00:00'; do
  [[ "$record" == *"$required"* ]] || { echo "Held v8 job missing $required" >&2; exit 1; }
done
"$PYTHON" - "$SUBMISSION" "$job_id" "$IDENTITY" "$record" <<'PYSUB'
import hashlib,json,os,pathlib,sys,tempfile
p=pathlib.Path(sys.argv[1]); i=pathlib.Path(sys.argv[3]); x={"schema":"constructive-code-v8-coder-15b-viability-submission-v1","job_id":int(sys.argv[2]),"identity_sha256":hashlib.sha256(i.read_bytes()).hexdigest(),"held_job_audit":"pass","released":True,"scheduler_record":sys.argv[4]}
fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w") as h: json.dump(x,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PYSUB
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[constructive-v8-coder-15b] released job $job_id"
