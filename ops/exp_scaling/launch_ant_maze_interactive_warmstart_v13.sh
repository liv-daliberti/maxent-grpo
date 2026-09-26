#!/usr/bin/env bash
# Snapshot and submit the frozen AntMaze v13 SFT plus development gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA="$ROOT_DIR/var/data/ant_maze_modebench_v12"
WARM_DATA="$ROOT_DIR/var/data/ant_maze_interactive_warmstart_v13_r1"
V12_RECEIPT="$ROOT_DIR/var/artifacts/ant_maze_05b_viability_v12.json"
V12_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_05b_viability_v12_identity.json"
CROSS_NODE="$ROOT_DIR/var/artifacts/ant_maze_v12_cross_node_v1_r1.json"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_interactive_warmstart_v13_20260730.md"
TRAINER="$ROOT_DIR/ops/train_ant_maze_interactive_warmstart_v13.py"
EVALUATOR="$ROOT_DIR/ops/evaluate_ant_maze_interactive_viability_v13.py"
BASE_EVALUATOR="$ROOT_DIR/ops/evaluate_point_maze_interactive_viability.py"
BATCH="$ROOT_DIR/ops/slurm/train_ant_maze_interactive_warmstart_v13.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_interactive_warmstart_v13_identity.json"
OUTPUT_MODEL="$ROOT_DIR/var/models/ant_maze_interactive_warmstart_v13"
SFT_RECEIPT="$ROOT_DIR/var/artifacts/ant_maze_interactive_warmstart_sft_v13.json"
GATE_RECEIPT="$ROOT_DIR/var/artifacts/ant_maze_interactive_05b_viability_v13.json"

for required in "$PYTHON_BIN" "$MODEL/config.json" "$DATA/identity.json" \
  "$DATA/dev/dataset_dict.json" "$WARM_DATA/identity.json" \
  "$WARM_DATA/examples.jsonl" "$V12_RECEIPT" "$V12_IDENTITY" \
  "$CROSS_NODE" "$PROTOCOL" "$TRAINER" "$EVALUATOR" \
  "$BASE_EVALUATOR" "$BATCH" "$ROOT_DIR/var/maze_runtime/venv/bin/python"; do
  [[ -e "$required" ]] || { echo "Missing AntMaze v13 prerequisite: $required" >&2; exit 1; }
done

"$PYTHON_BIN" - "$V12_RECEIPT" "$CROSS_NODE" "$WARM_DATA/identity.json" "$WARM_DATA/examples.jsonl" <<'PYPRE'
import hashlib,json,pathlib,sys
v12=json.load(open(sys.argv[1])); cross=json.load(open(sys.argv[2])); warm=json.load(open(sys.argv[3]))
if v12.get("status")!="fail" or v12.get("summary")!={"multimode_prompts":0,"prefix_success_prompts":0,"prompt_count":4,"verified_completions":0}: raise SystemExit("v13 requires exact v12 free-form failure")
if cross.get("status")!="pass" or cross.get("decision")!="eligible_for_frozen_05b_viability_gate_v12" or cross.get("validated_count")!=216: raise SystemExit("v13 requires exact v12 cross-node pass")
if warm.get("status")!="pass" or warm.get("split")!="train_only" or warm.get("map_count")!=4 or warm.get("episode_count")!=8 or warm.get("example_count")!=32: raise SystemExit("v13 train-only materialization failed")
if warm.get("information_boundary")!={"certification_records_selected_by_split":"train","dev_dataset_loaded":False,"eval_dataset_loaded":False,"model_sampled":False,"online_reward_used":False,"train_dataset_rows_loaded":4}: raise SystemExit("v13 materialization firewall changed")
if hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest()!=warm["hashes"]["examples_sha256"]: raise SystemExit("v13 examples hash changed")
PYPRE

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m py_compile "$TRAINER" "$EVALUATOR" "$BASE_EVALUATOR"
  bash -n "$BATCH"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_ant_maze_interactive_v13.py" \
    "$ROOT_DIR/tests/test_ant_maze_warmstart_v13.py" \
    "$ROOT_DIR/tests/test_modebench_base_viability.py"
  echo "[ant-v13] configuration and 32 train-only examples passed; no model updated or development row sampled"
  exit 0
fi

for fresh in "$IDENTITY" "$OUTPUT_MODEL" "$SFT_RECEIPT" "$GATE_RECEIPT"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh AntMaze v13 artifact required: $fresh" >&2; exit 1; }
done
SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_interactive_v13_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/ant_maze_interactive_worker.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$staging/src"; cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"; rmdir "$staging"
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "AntMaze v13 source snapshot mismatch" >&2; exit 1; }

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-v13-ops.XXXXXX")"
cp "$TRAINER" "$OPS_INPUT/train_ant_maze_interactive_warmstart_v13.py"
cp "$EVALUATOR" "$OPS_INPUT/evaluate_ant_maze_interactive_viability_v13.py"
cp "$BASE_EVALUATOR" "$OPS_INPUT/evaluate_point_maze_interactive_viability.py"
cp "$BATCH" "$OPS_INPUT/train_ant_maze_interactive_warmstart_v13.slurm"
cp "$PROTOCOL" "$OPS_INPUT/ant_maze_interactive_warmstart_v13_20260730.md"
EXECUTION_HASH="$(hash_tree "$OPS_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_interactive_v13_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_ant_maze_interactive_warmstart_v13.py" ]]; then
  mv "$OPS_INPUT" "$EXECUTION_ROOT"
else
  find "$OPS_INPUT" -type f -delete; rmdir "$OPS_INPUT"
fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || { echo "AntMaze v13 ops snapshot mismatch" >&2; exit 1; }

mkdir -p "$ROOT_DIR/var/artifacts/logs" "$ROOT_DIR/var/models"
job_id="$(sbatch --parsable --hold --partition=all --account=mltheory \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
  "$EXECUTION_ROOT/train_ant_maze_interactive_warmstart_v13.slurm")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid AntMaze v13 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" "$SOURCE_ROOT" "$EXECUTION_ROOT" "$PROTOCOL" "$TRAINER" "$EVALUATOR" "$BASE_EVALUATOR" "$BATCH" "$WARM_DATA/identity.json" "$WARM_DATA/examples.jsonl" "$V12_RECEIPT" "$V12_IDENTITY" "$CROSS_NODE" "$MODEL/config.json" "$DATA/identity.json" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def d(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
p=pathlib.Path(sys.argv[1]); x={
 "schema_version":"ant-maze-interactive-warmstart-identity-v13","job_id":int(sys.argv[2]),
 "source_hash":sys.argv[3],"execution_hash":sys.argv[4],"source_root":sys.argv[5],"execution_root":sys.argv[6],
 "protocol_sha256":d(sys.argv[7]),"trainer_sha256":d(sys.argv[8]),"evaluator_sha256":d(sys.argv[9]),
 "base_evaluator_sha256":d(sys.argv[10]),"slurm_sha256":d(sys.argv[11]),
 "warmstart_data_identity_sha256":d(sys.argv[12]),"warmstart_examples_sha256":d(sys.argv[13]),
 "v12_failure_receipt_sha256":d(sys.argv[14]),"v12_failure_identity_sha256":d(sys.argv[15]),
 "v12_cross_node_audit_sha256":d(sys.argv[16]),"base_model_config_sha256":d(sys.argv[17]),
 "dataset_identity_sha256":d(sys.argv[18]),"policy_interface":"closed_loop_public_markov_compass_token_v13",
 "sft_seed":75313,"sft_examples":32,"sft_optimizer_updates":128,"development_sampling_seed":107313,
 "development_samples":256,"evaluation_split_loaded":False,"shared_checkpoint_for_both_online_arms":True,
 "controller_inference_threads":1,"freeform_v12_relabelled":False}
p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w") as h: json.dump(x,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PYID

record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'gres/gpu:a5000:1' 'MinMemoryNode=64G' 'TimeLimit=04:00:00' 'RunTime=00:00:00'; do
  [[ "$record" == *"$required"* ]] || { echo "Held AntMaze v13 job missing $required" >&2; exit 1; }
done
[[ "$record" == *'Partition=all'* || "$record" == *'Partition=mltheory'* ]] || { echo "AntMaze v13 partition mismatch" >&2; exit 1; }
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-v13] released job $job_id"
echo "[ant-v13] identity=$IDENTITY"
