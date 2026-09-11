#!/usr/bin/env bash
# Materialize, snapshot, and submit the frozen PointMaze compact-state v2 gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/point_maze_modebench_v1"
WARMSTART_DATA="$ROOT_DIR/var/data/point_maze_interactive_warmstart_v2"
AUDIT="$ROOT_DIR/var/artifacts/point_maze_modebench_v1_admission_audit.json"
V1_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_warmstart_v1.json"
PROTOCOL="$ROOT_DIR/paper/preregistration/point_maze_interactive_warmstart_v2_20260730.md"
AMENDMENT="$ROOT_DIR/paper/preregistration/point_maze_warmstart_v2_prelaunch_placement_repair_20260730.md"
MATERIALIZER="$ROOT_DIR/ops/materialize_point_maze_train_warmstart_v1.py"
TRAINER="$ROOT_DIR/ops/train_point_maze_interactive_warmstart_v1.py"
EVALUATOR="$ROOT_DIR/ops/evaluate_point_maze_interactive_viability.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_point_maze_interactive_warmstart_v2.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_v2_identity.json"
OUTPUT_MODEL="$ROOT_DIR/var/models/point_maze_interactive_warmstart_v2"
SFT_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_sft_v2.json"
GATE_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_warmstart_v2.json"

for required in "$PYTHON_BIN" "$MODEL_ROOT/config.json" "$DATA_ROOT/identity.json" \
  "$DATA_ROOT/train/dataset_dict.json" "$DATA_ROOT/dev/dataset_dict.json" \
  "$AUDIT" "$V1_RECEIPT" "$PROTOCOL" "$AMENDMENT" "$MATERIALIZER" "$TRAINER" \
  "$EVALUATOR" "$SLURM_SCRIPT" "$ROOT_DIR/var/maze_runtime/venv/bin/python"; do
  [[ -e "$required" ]] || { echo "Missing PointMaze v2 prerequisite: $required" >&2; exit 1; }
done

"$PYTHON_BIN" - "$V1_RECEIPT" "$AUDIT" <<'PYV1'
import json,sys
receipt=json.load(open(sys.argv[1],encoding="utf-8"))
if receipt.get("status") != "fail" or receipt.get("summary",{}).get("verified_completions") != 0:
    raise SystemExit("PointMaze v2 requires the exact terminal zero-route v1 receipt")
counts={}
for attempt in receipt.get("attempts",[]):
    for action in attempt.get("actions",[]): counts[action]=counts.get(action,0)+1
total=sum(counts.values()); ns=counts.get("N",0)+counts.get("S",0)
if total != 24576 or ns != 24328:
    raise SystemExit("PointMaze v1 action-collapse antecedent changed")
audit=json.load(open(sys.argv[2],encoding="utf-8"))
if audit.get("status") != "pass": raise SystemExit("PointMaze admission audit is not passing")
PYV1

if [[ ! -e "$WARMSTART_DATA" ]]; then
  "$PYTHON_BIN" "$MATERIALIZER" \
    --data-root "$DATA_ROOT" \
    --train-split-root "$DATA_ROOT/train" \
    --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
    --protocol "$PROTOCOL" \
    --output-root "$WARMSTART_DATA" \
    --policy-interface compact_state_v2
fi

"$PYTHON_BIN" - "$WARMSTART_DATA/identity.json" "$WARMSTART_DATA/examples.jsonl" <<'PYDATA'
import hashlib,json,pathlib,sys
identity=json.load(open(sys.argv[1],encoding="utf-8"))
if identity.get("status") != "pass" or identity.get("policy_interface") != "compact_state_v2":
    raise SystemExit("PointMaze compact-state data identity is not passing")
if identity.get("example_count") != 644 or identity.get("episode_count") != 16:
    raise SystemExit("PointMaze v2 data counts changed")
if hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest() != identity["hashes"]["examples_sha256"]:
    raise SystemExit("PointMaze v2 examples hash changed")
PYDATA

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" -m py_compile "$MATERIALIZER" "$TRAINER" "$EVALUATOR"
  bash -n "$SLURM_SCRIPT"
  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_point_maze_interactive_policy.py" \
    "$ROOT_DIR/tests/test_point_maze_warmstart_v2.py"
  echo "[point-warmstart-v2] configuration and compact train-only data passed; no model updated or dev row sampled"
  exit 0
fi

for fresh in "$IDENTITY" "$OUTPUT_MODEL" "$SFT_RECEIPT" "$GATE_RECEIPT"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh PointMaze v2 artifact required: $fresh" >&2; exit 1; }
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v2_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$staging/src"; cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"; rmdir "$staging"
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || exit 1

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.point-warmstart-v2-ops.XXXXXX")"
cp "$TRAINER" "$EXECUTION_INPUT/train_point_maze_interactive_warmstart_v1.py"
cp "$EVALUATOR" "$EXECUTION_INPUT/evaluate_point_maze_interactive_viability.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/train_point_maze_interactive_warmstart_v2.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v2_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_point_maze_interactive_warmstart_v1.py" ]]; then mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"; else find "$EXECUTION_INPUT" -type f -delete; rmdir "$EXECUTION_INPUT"; fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || exit 1

mkdir -p "$ROOT_DIR/var/artifacts/logs" "$ROOT_DIR/var/models"
job_id="$(sbatch --parsable --hold --partition=all --account=mltheory \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH" \
  "$EXECUTION_ROOT/train_point_maze_interactive_warmstart_v2.slurm")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid PointMaze v2 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" "$PROTOCOL" \
  "$TRAINER" "$EVALUATOR" "$SLURM_SCRIPT" "$WARMSTART_DATA/identity.json" \
  "$WARMSTART_DATA/examples.jsonl" "$MODEL_ROOT/config.json" "$AUDIT" "$V1_RECEIPT" "$AMENDMENT" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema_version":"point-maze-interactive-warmstart-identity-v2","job_id":int(sys.argv[2]),
 "source_hash":sys.argv[3],"execution_hash":sys.argv[4],"protocol_sha256":digest(sys.argv[5]),
 "trainer_sha256":digest(sys.argv[6]),"evaluator_sha256":digest(sys.argv[7]),
 "slurm_sha256":digest(sys.argv[8]),"warmstart_data_identity_sha256":digest(sys.argv[9]),
 "warmstart_examples_sha256":digest(sys.argv[10]),"base_model_config_sha256":digest(sys.argv[11]),
 "admission_audit_sha256":digest(sys.argv[12]),"v1_failure_receipt_sha256":digest(sys.argv[13]),
 "placement_amendment_sha256":digest(sys.argv[14]),
 "prelaunch_canceled_job_ids":[30197923,30197998],
 "base_model":"Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
 "policy_interface":"compact_state_v2","sft_seed":75202,"sft_optimizer_steps":184,
 "development_sampling_seed":75104,"development_samples":256,"evaluation_split_loaded":False,
 "shared_checkpoint_for_both_online_arms":True}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID

record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'gres/gpu:a5000:1' 'MinMemoryNode=64G' 'TimeLimit=02:00:00'; do
  [[ "$record" == *"$required"* ]] || { echo "PointMaze v2 held job missing $required" >&2; exit 1; }
done
scontrol update "JobId=$job_id" Partition=all
record="$(scontrol show job "$job_id" -o)"
[[ "$record" == *'Partition=all'* ]] || { echo "PointMaze v2 placement repair did not reach partition all" >&2; exit 1; }
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[point-warmstart-v2] released job $job_id"
echo "[point-warmstart-v2] identity=$IDENTITY"
