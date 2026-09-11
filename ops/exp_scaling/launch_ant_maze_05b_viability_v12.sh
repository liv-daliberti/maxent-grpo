#!/usr/bin/env bash
# Submit the development-only AntMaze v12 base-model viability gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"; case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v12"
ROUTE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v12_r4_generation_identity.json"
CROSS_NODE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_v12_cross_node_v1_r1_identity.json"
CROSS_NODE_AUDIT="$ROOT_DIR/var/artifacts/ant_maze_v12_cross_node_v1_r1.json"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_maze_05b_viability_v12_20260730.md"
EVALUATOR="$ROOT_DIR/ops/evaluate_modebench_base_viability.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/evaluate_ant_maze_05b_viability_v12.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_05b_viability_v12_identity.json"
RECEIPT="$ROOT_DIR/var/artifacts/ant_maze_05b_viability_v12.json"
for required in "$PYTHON_BIN" "$MODEL_ROOT/config.json" "$PROTOCOL" "$EVALUATOR" "$SLURM_SCRIPT" "$ROOT_DIR/var/maze_runtime/venv/bin/python"; do [[ -e "$required" ]] || { echo "Missing AntMaze v12 viability prerequisite: $required" >&2; exit 1; }; done
hash_tree() { local tree="$1"; (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }
if [[ "$phase" == config ]]; then
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m py_compile "$EVALUATOR"; "$PYTHON_BIN" "$EVALUATOR" --help >/dev/null; bash -n "$SLURM_SCRIPT"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON_BIN" -m pytest -q "$ROOT_DIR/tests/test_modebench_base_viability.py" "$ROOT_DIR/tests/test_ant_maze_05b_viability_v10.py" "$ROOT_DIR/tests/test_ant_maze_05b_viability_v12.py"
  echo "[antmaze-v12-viability] configuration passed; no model sampled"; exit 0
fi
for required in "$DATA_ROOT/identity.json" "$DATA_ROOT/dev/dataset_dict.json" "$ROUTE_IDENTITY" "$CROSS_NODE_IDENTITY" "$CROSS_NODE_AUDIT"; do [[ -f "$required" ]] || { echo "Missing admitted AntMaze v12 antecedent: $required" >&2; exit 1; }; done
"$PYTHON_BIN" - "$ROUTE_IDENTITY" "$CROSS_NODE_IDENTITY" "$CROSS_NODE_AUDIT" <<'PYANT'
import json,sys
route=json.load(open(sys.argv[1])); identity=json.load(open(sys.argv[2])); audit=json.load(open(sys.argv[3]))
if route.get("schema_version")!="ant-maze-v12-route-generation-identity-v1" or route.get("job_id")!=30200585: raise SystemExit("wrong AntMaze v12 route identity")
if identity.get("schema_version")!="ant-maze-v12-cross-node-r1-identity-v1" or identity.get("language_model_sampling") is not False: raise SystemExit("wrong AntMaze v12 cross-node identity")
if audit.get("status")!="pass" or audit.get("decision")!="eligible_for_frozen_05b_viability_gate_v12" or audit.get("validated_count")!=216 or audit.get("node_count")!=3: raise SystemExit("AntMaze v12 cross-node gate is not admitted")
PYANT
for fresh in "$IDENTITY" "$RECEIPT"; do [[ ! -e "$fresh" ]] || { echo "Fresh AntMaze v12 viability artifact required: $fresh" >&2; exit 1; }; done
SOURCE_HASH="$(jq -er .source_hash "$ROUTE_IDENTITY")"; SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_route_${SOURCE_HASH}/src"
[[ -f "$SOURCE_ROOT/oat_drgrpo/ant_maze_worker_v12.py" && "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "Exact AntMaze v12 source snapshot mismatch" >&2; exit 1; }
EXEC_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-v12-viability.XXXXXX")"; cp "$EVALUATOR" "$EXEC_INPUT/evaluate_modebench_base_viability.py"; cp "$SLURM_SCRIPT" "$EXEC_INPUT/evaluate_ant_maze_05b_viability_v12.slurm"; cp "$PROTOCOL" "$EXEC_INPUT/ant_maze_05b_viability_v12_20260730.md"
EXECUTION_HASH="$(hash_tree "$EXEC_INPUT")"; EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_maze_v12_viability_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/evaluate_modebench_base_viability.py" ]]; then mv "$EXEC_INPUT" "$EXECUTION_ROOT"; else find "$EXEC_INPUT" -type f -delete; rmdir "$EXEC_INPUT"; fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || exit 1
mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold --export="ALL,PYTHONDONTWRITEBYTECODE=1,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH,OAT_ZERO_ROUTE_IDENTITY=$ROUTE_IDENTITY" "$EXECUTION_ROOT/evaluate_ant_maze_05b_viability_v12.slurm")"; job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid AntMaze v12 viability job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }; trap cleanup EXIT
record="$(scontrol show job "$job_id" -o)"; for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'TresPerNode=gres/gpu:a5000:1' "OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT" "OAT_ZERO_ROUTE_IDENTITY=$ROUTE_IDENTITY"; do [[ "$record" == *"$required"* ]] || { echo "Held AntMaze v12 viability job missing $required" >&2; exit 1; }; done
"$PYTHON_BIN" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" "$EXECUTION_ROOT/ant_maze_05b_viability_v12_20260730.md" "$EVALUATOR" "$SLURM_SCRIPT" "$DATA_ROOT/identity.json" "$ROUTE_IDENTITY" "$CROSS_NODE_IDENTITY" "$CROSS_NODE_AUDIT" "$MODEL_ROOT/config.json" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={"schema_version":"ant-maze-05b-viability-identity-v12","job_id":int(sys.argv[2]),"source_hash":sys.argv[3],"execution_hash":sys.argv[4],"protocol_sha256":digest(sys.argv[5]),"evaluator_sha256":digest(sys.argv[6]),"slurm_sha256":digest(sys.argv[7]),"dataset_identity_sha256":digest(sys.argv[8]),"route_identity_sha256":digest(sys.argv[9]),"cross_node_identity_sha256":digest(sys.argv[10]),"cross_node_audit_sha256":digest(sys.argv[11]),"model_config_sha256":digest(sys.argv[12]),"model":"Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775","split":"dev/multi_answer","evaluation_split_loaded":False,"sample_count":64,"prefix_count":16,"sampling_seed":107312,"minimum_prefix_success_prompts":2,"minimum_multimode_prompts":1,"assistant_prefix":"\\boxed{","prompt_repair":"ant_maze_v10","certified_route_programs_in_context":False}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID
scontrol update "JobId=$job_id" Requeue=0; scontrol release "$job_id"; trap - EXIT
echo "[antmaze-v12-viability] released job $job_id"; echo "[antmaze-v12-viability] identity=$IDENTITY"
