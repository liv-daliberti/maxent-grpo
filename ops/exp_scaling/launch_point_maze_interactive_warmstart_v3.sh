#!/usr/bin/env bash
# Materialize, snapshot, and submit the frozen PointMaze public-Markov-state v3 gate.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"; case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/point_maze_modebench_v1"; WARMSTART_DATA="$ROOT_DIR/var/data/point_maze_interactive_warmstart_v3"
AUDIT="$ROOT_DIR/var/artifacts/point_maze_modebench_v1_admission_audit.json"
V2_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_warmstart_v2.json"
PROTOCOL="$ROOT_DIR/paper/preregistration/point_maze_interactive_warmstart_v3_20260730.md"
MATERIALIZER="$ROOT_DIR/ops/materialize_point_maze_train_warmstart_v1.py"; TRAINER="$ROOT_DIR/ops/train_point_maze_interactive_warmstart_v1.py"
EVALUATOR="$ROOT_DIR/ops/evaluate_point_maze_interactive_viability.py"; SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_point_maze_interactive_warmstart_v3.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_v3_identity.json"; OUTPUT_MODEL="$ROOT_DIR/var/models/point_maze_interactive_warmstart_v3"
SFT_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_sft_v3.json"; GATE_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_warmstart_v3.json"
for required in "$PYTHON_BIN" "$MODEL_ROOT/config.json" "$DATA_ROOT/identity.json" "$DATA_ROOT/train/dataset_dict.json" "$DATA_ROOT/dev/dataset_dict.json" "$AUDIT" "$V2_RECEIPT" "$PROTOCOL" "$MATERIALIZER" "$TRAINER" "$EVALUATOR" "$SLURM_SCRIPT" "$ROOT_DIR/var/maze_runtime/venv/bin/python"; do [[ -e "$required" ]] || { echo "Missing PointMaze v3 prerequisite: $required" >&2; exit 1; }; done
"$PYTHON_BIN" - "$V2_RECEIPT" "$AUDIT" <<'PYV2'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8")); a=json.load(open(sys.argv[2],encoding="utf-8"))
if r.get("status")!="fail" or r.get("decision")!="point_maze_compact_warmstart_v2_ineligible" or r.get("summary",{}).get("verified_completions")!=0: raise SystemExit("Point v3 requires exact terminal v2 failure")
counts={}
for row in r.get("attempts",[]):
    for action in row.get("actions",[]): counts[action]=counts.get(action,0)+1
if counts.get("N")!=20072 or counts.get("S")!=4445: raise SystemExit("Point v2 action-collapse antecedent changed")
if a.get("status")!="pass": raise SystemExit("Point admission audit is not passing")
PYV2
if [[ ! -e "$WARMSTART_DATA" ]]; then "$PYTHON_BIN" "$MATERIALIZER" --data-root "$DATA_ROOT" --train-split-root "$DATA_ROOT/train" --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" --protocol "$PROTOCOL" --output-root "$WARMSTART_DATA" --policy-interface velocity_state_v3; fi
"$PYTHON_BIN" - "$WARMSTART_DATA/identity.json" "$WARMSTART_DATA/examples.jsonl" <<'PYDATA'
import hashlib,json,pathlib,sys
i=json.load(open(sys.argv[1],encoding="utf-8"))
if i.get("status")!="pass" or i.get("policy_interface")!="velocity_state_v3" or i.get("example_count")!=644 or i.get("episode_count")!=16: raise SystemExit("Point v3 data identity failed")
if hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()!=i["hashes"]["examples_sha256"]: raise SystemExit("Point v3 examples hash changed")
PYDATA
hash_tree(){ local tree="$1"; (cd "$tree"; find . -type f -print0|sort -z|xargs -0 sha256sum|sha256sum|cut -d' ' -f1); }
if [[ "$phase" == config ]]; then "$PYTHON_BIN" -m py_compile "$MATERIALIZER" "$TRAINER" "$EVALUATOR"; bash -n "$SLURM_SCRIPT"; PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q "$ROOT_DIR/tests/test_point_maze_interactive_policy.py" "$ROOT_DIR/tests/test_point_maze_interactive_worker.py" "$ROOT_DIR/tests/test_point_maze_warmstart_v3.py"; echo "[point-warmstart-v3] configuration and public-velocity train data passed; no model updated or dev row sampled"; exit 0; fi
for fresh in "$IDENTITY" "$OUTPUT_MODEL" "$SFT_RECEIPT" "$GATE_RECEIPT"; do [[ ! -e "$fresh" ]] || { echo "Fresh Point v3 artifact required: $fresh" >&2; exit 1; }; done
SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"; SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v3_${SOURCE_HASH}"; SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then mkdir -p "$SOURCE_PARENT"; staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"; mkdir -p "$staging/src"; cp -a "$ROOT_DIR/src/." "$staging/src/"; mv "$staging/src" "$SOURCE_ROOT"; rmdir "$staging"; fi; [[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || exit 1
OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.point-warmstart-v3-ops.XXXXXX")"; cp "$TRAINER" "$OPS_INPUT/train_point_maze_interactive_warmstart_v1.py"; cp "$EVALUATOR" "$OPS_INPUT/evaluate_point_maze_interactive_viability.py"; cp "$SLURM_SCRIPT" "$OPS_INPUT/train_point_maze_interactive_warmstart_v3.slurm"
EXECUTION_HASH="$(hash_tree "$OPS_INPUT")"; EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v3_ops_${EXECUTION_HASH}"; if [[ ! -f "$EXECUTION_ROOT/train_point_maze_interactive_warmstart_v1.py" ]]; then mv "$OPS_INPUT" "$EXECUTION_ROOT"; else find "$OPS_INPUT" -type f -delete; rmdir "$OPS_INPUT"; fi; [[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || exit 1
mkdir -p "$ROOT_DIR/var/artifacts/logs" "$ROOT_DIR/var/models"; job_id="$(sbatch --parsable --hold --partition=all --account=mltheory --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH" "$EXECUTION_ROOT/train_point_maze_interactive_warmstart_v3.slurm")"; job_id="${job_id%%;*}"; [[ "$job_id" =~ ^[0-9]+$ ]] || exit 1
cleanup(){ local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null||true; exit "$status"; }; trap cleanup EXIT
"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" "$PROTOCOL" "$TRAINER" "$EVALUATOR" "$SLURM_SCRIPT" "$WARMSTART_DATA/identity.json" "$WARMSTART_DATA/examples.jsonl" "$MODEL_ROOT/config.json" "$AUDIT" "$V2_RECEIPT" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def d(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
p=pathlib.Path(sys.argv[1]); x={"schema_version":"point-maze-interactive-warmstart-identity-v3","job_id":int(sys.argv[2]),"source_hash":sys.argv[3],"execution_hash":sys.argv[4],"protocol_sha256":d(sys.argv[5]),"trainer_sha256":d(sys.argv[6]),"evaluator_sha256":d(sys.argv[7]),"slurm_sha256":d(sys.argv[8]),"warmstart_data_identity_sha256":d(sys.argv[9]),"warmstart_examples_sha256":d(sys.argv[10]),"base_model_config_sha256":d(sys.argv[11]),"admission_audit_sha256":d(sys.argv[12]),"v2_failure_receipt_sha256":d(sys.argv[13]),"policy_interface":"velocity_state_v3","sft_seed":75203,"sft_optimizer_steps":276,"development_sampling_seed":75105,"development_samples":256,"evaluation_split_loaded":False,"shared_checkpoint_for_both_online_arms":True}
p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(x,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PYID
record="$(scontrol show job "$job_id" -o)"; for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'gres/gpu:a5000:1' 'MinMemoryNode=64G' 'TimeLimit=02:00:00'; do [[ "$record" == *"$required"* ]] || { echo "Point v3 held job missing $required" >&2; exit 1; }; done; scontrol update "JobId=$job_id" Partition=all; record="$(scontrol show job "$job_id" -o)"; [[ "$record" == *'Partition=all'* ]] || exit 1; scontrol update "JobId=$job_id" Requeue=0; scontrol release "$job_id"; trap - EXIT; echo "[point-warmstart-v3] released job $job_id"; echo "[point-warmstart-v3] identity=$IDENTITY"
