#!/usr/bin/env bash
# Snapshot and launch the frozen ConstructiveCode v7 train-only SFT + dev gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
TRAINER="$ROOT_DIR/ops/train_constructive_code_v7_sft.py"
EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v7_coder_viability.py"
V6_EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v6_coder_viability.py"
V4_EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v4_coder_viability.py"
BASE_EVALUATOR="$ROOT_DIR/ops/evaluate_constructive_code_v3_coder_viability.py"
GATE_AUDITOR="$ROOT_DIR/ops/audit_constructive_code_v6_gate.py"
MATERIALIZER="$ROOT_DIR/ops/materialize_constructive_code_v7_sft.py"
BATCH="$ROOT_DIR/ops/slurm/train_constructive_code_v7_sft_and_gate.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/constructive_code_v7_train_only_sft_20260730.md"
REPAIR_PROTOCOL="$ROOT_DIR/paper/preregistration/constructive_code_v7_token_boundary_repair_r1_20260730.md"
EXAMPLES="$ROOT_DIR/var/data/constructive_code_v7_sft/examples.jsonl"
SFT_MANIFEST="$ROOT_DIR/var/data/constructive_code_v7_sft/manifest.json"
V6_GATE="$ROOT_DIR/var/artifacts/constructive_code_v6_gate_audit.json"
V6_GATE_IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v6_gate_identity.json"
BASE_MODEL="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
IDENTITY="$ROOT_DIR/var/artifacts/constructive_code_v7_sft_gate_identity.json"
SUBMISSION="$ROOT_DIR/var/artifacts/constructive_code_v7_sft_gate_submission.json"
OUTPUT_MODEL="$ROOT_DIR/var/models/constructive_code_v7_sft"
SFT_RECEIPT="$ROOT_DIR/var/artifacts/constructive_code_v7_sft.json"
GATE_RECEIPT="$ROOT_DIR/var/artifacts/constructive_code_v7_post_sft_viability.json"

for required in "$PYTHON" "$TRAINER" "$EVALUATOR" "$V6_EVALUATOR" "$V4_EVALUATOR" \
  "$BASE_EVALUATOR" "$GATE_AUDITOR" "$MATERIALIZER" "$BATCH" "$PROTOCOL" \
  "$REPAIR_PROTOCOL" \
  "$EXAMPLES" "$SFT_MANIFEST" "$V6_GATE" "$V6_GATE_IDENTITY" \
  "$BASE_MODEL/config.json" "$BASE_MODEL/model.safetensors"; do
  [[ -e "$required" ]] || { echo "Missing ConstructiveCode v7 prerequisite: $required" >&2; exit 1; }
done

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON" -m py_compile \
    "$TRAINER" "$EVALUATOR" "$V6_EVALUATOR" "$V4_EVALUATOR" "$BASE_EVALUATOR" "$GATE_AUDITOR"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON" -m pytest -q \
    "$ROOT_DIR/tests/test_constructive_code_v7_sft.py" \
    "$ROOT_DIR/tests/test_constructive_code_v6_coder_viability.py" \
    "$ROOT_DIR/tests/test_constructive_code_v6_gate.py"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON" "$MATERIALIZER" --check
  bash -n "$BATCH"
  [[ "$(grep -F 'mkdir -p ' "$BATCH")" != *'"$RUN_ROOT/runtime"'* ]] || {
    echo "v7 batch pre-creates runtime extraction destination" >&2; exit 1;
  }
  echo "[constructive-v7-sft-gate] configuration passed; no model sampled"
  exit 0
fi

for fresh in "$IDENTITY" "$SUBMISSION" "$OUTPUT_MODEL" "$SFT_RECEIPT" "$GATE_RECEIPT"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh ConstructiveCode v7 target required: $fresh" >&2; exit 1; }
done
[[ "$(jq -er '.status' "$V6_GATE")" == pass ]] || { echo "v6 executable gate is not pass" >&2; exit 1; }
[[ "$(jq -er '.status' "$SFT_MANIFEST")" == pass ]] || { echo "v7 SFT manifest is not pass" >&2; exit 1; }

SOURCE_HASH="$(jq -er '.source_hash' "$V6_GATE_IDENTITY")"
GATE_EXEC_HASH="$(jq -er '.execution_hash' "$V6_GATE_IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v5_source_${SOURCE_HASH}/src"
GATE_OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v5_gate_${GATE_EXEC_HASH}"
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "v7 source antecedent hash mismatch" >&2; exit 1; }
[[ "$(hash_tree "$GATE_OPS_ROOT")" == "$GATE_EXEC_HASH" ]] || { echo "v7 gate antecedent hash mismatch" >&2; exit 1; }

EXEC_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.constructive-v7-sft-gate.XXXXXX")"
for source in "$TRAINER" "$EVALUATOR" "$V6_EVALUATOR" "$V4_EVALUATOR" "$BASE_EVALUATOR" "$GATE_AUDITOR" "$BATCH" "$PROTOCOL" "$REPAIR_PROTOCOL"; do
  cp "$source" "$EXEC_INPUT/$(basename "$source")"
done
EXECUTION_HASH="$(hash_tree "$EXEC_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/constructive_code_v7_sft_gate_${EXECUTION_HASH}"
if [[ ! -d "$EXECUTION_ROOT" ]]; then mv "$EXEC_INPUT" "$EXECUTION_ROOT"; else find "$EXEC_INPUT" -type f -delete; rmdir "$EXEC_INPUT"; fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || { echo "v7 execution snapshot mismatch" >&2; exit 1; }

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold --partition=all --account=allcs \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_GATE_OPS_ROOT=$GATE_OPS_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
  "$EXECUTION_ROOT/$(basename "$BATCH")")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid v7 SFT-gate job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$PYTHON" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$GATE_EXEC_HASH" "$EXECUTION_HASH" \
  "$PROTOCOL" "$0" "$EXAMPLES" "$SFT_MANIFEST" "$V6_GATE" "$V6_GATE_IDENTITY" \
  "$BASE_MODEL/config.json" "$BASE_MODEL/tokenizer.json" "$REPAIR_PROTOCOL" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def d(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
p=pathlib.Path(sys.argv[1]); payload={
 "schema":"constructive-code-v7-sft-gate-token-boundary-repair-r1","job_id":int(sys.argv[2]),
 "source_hash":sys.argv[3],"gate_execution_hash":sys.argv[4],"execution_hash":sys.argv[5],
 "protocol_sha256":d(sys.argv[6]),"launcher_sha256":d(sys.argv[7]),
 "examples_sha256":d(sys.argv[8]),"sft_manifest_sha256":d(sys.argv[9]),
 "v6_gate_sha256":d(sys.argv[10]),"v6_gate_identity_sha256":d(sys.argv[11]),
 "base_model_config_sha256":d(sys.argv[12]),"base_tokenizer_sha256":d(sys.argv[13]),
 "repair_protocol_sha256":d(sys.argv[14]),
 "base_model_snapshot":"ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
 "train_problem_ids":["327_B","659_C","1283_C","1102_B"],
 "development_problem_ids":["359_B","988_A","1399_D"],
 "evaluation_problem_ids":["361_B","1294_C","149_C"],
 "sft_seed":77201,"sft_examples":64,"sft_epochs":4,"sft_optimizer_updates":32,
 "gate_seed":77102,"gate_samples_per_task":64,"gate_prefix_count":16,
 "evaluation_rows_loaded":False,"shared_checkpoint_for_both_online_arms":True,
 "repair_scope":"offset-based assistant-only token masking; no corpus, schedule, model, or gate change",
 "network_enabled":False
}
p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w") as h: json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PYID

record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'gres/gpu:a5000:1' 'NumCPUs=16' 'MinMemoryNode=64G' 'TimeLimit=04:00:00'; do
  [[ "$record" == *"$required"* ]] || { echo "Held v7 SFT-gate job missing $required" >&2; exit 1; }
done
"$PYTHON" - "$SUBMISSION" "$job_id" "$IDENTITY" "$record" <<'PYSUB'
import hashlib,json,os,pathlib,sys,tempfile
p=pathlib.Path(sys.argv[1]); i=pathlib.Path(sys.argv[3]); x={"schema":"constructive-code-v7-sft-gate-submission-v1","job_id":int(sys.argv[2]),"identity_sha256":hashlib.sha256(i.read_bytes()).hexdigest(),"held_job_audit":"pass","released":True,"scheduler_record":sys.argv[4]}
fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w") as h: json.dump(x,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PYSUB
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[constructive-v7-sft-gate] released job $job_id"
