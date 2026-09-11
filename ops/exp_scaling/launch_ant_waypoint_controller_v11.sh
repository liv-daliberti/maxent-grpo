#!/usr/bin/env bash
# Snapshot and submit the frozen SE/NE-remediation Ant controller v11.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/maze_runtime/venv/bin/python"
IDENTITY_PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v11.py"
V10_TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v10.py"
V9_TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v9.py"
BASE="$ROOT_DIR/ops/train_ant_waypoint_controller_v7.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_ant_waypoint_controller_v11.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_waypoint_controller_v11_20260730.md"
AMENDMENT="$ROOT_DIR/paper/preregistration/ant_waypoint_controller_v11_prelaunch_placement_repair_20260730.md"
INITIAL_MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v10.zip"
V10_RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v10.evaluation.json"
V10_IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v10_identity.json"
V10_ROUTE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v10_generation_identity.json"
V10_ROUTE_DATA="$ROOT_DIR/var/data/ant_maze_modebench_v10"
V10_ROUTE_AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v10_admission_audit.json"
V10_VIABILITY="$ROOT_DIR/var/artifacts/ant_maze_05b_viability_v10.json"
IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v11_identity.json"
MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v11.zip"
RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v11.evaluation.json"
EXPECTED_INITIAL=fffb66696aa4435c83b8bed854834ce82ccb01a9a1ce9dfe22e813c7e94a7b94

for required in "$PYTHON_BIN" "$IDENTITY_PYTHON" "$TRAINER" "$V10_TRAINER" \
  "$V9_TRAINER" "$BASE" "$SLURM_SCRIPT" "$PROTOCOL" "$AMENDMENT" "$INITIAL_MODEL" \
  "$V10_RECEIPT" "$V10_IDENTITY"; do
  [[ -e "$required" ]] || { echo "Missing Ant waypoint v11 prerequisite: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$INITIAL_MODEL" | cut -d' ' -f1)" == "$EXPECTED_INITIAL" ]] || { echo "Ant v11 initialization hash mismatch" >&2; exit 1; }

"$IDENTITY_PYTHON" - "$V10_RECEIPT" "$V10_IDENTITY" "$EXPECTED_INITIAL" \
  "$V10_ROUTE_IDENTITY" "$V10_ROUTE_DATA" "$V10_ROUTE_AUDIT" "$V10_VIABILITY" <<'PYV10'
import json,pathlib,sys
receipt=json.load(open(sys.argv[1],encoding="utf-8")); identity=json.load(open(sys.argv[2],encoding="utf-8"))
if receipt.get("status") != "fail" or receipt.get("decision") != "ant_waypoint_v10_ineligible": raise SystemExit("Ant v11 requires the exact failed-v10 receipt")
if receipt.get("hashes",{}).get("model_sha256") != sys.argv[3]: raise SystemExit("Ant v10 receipt does not bind v11 initialization")
summary=receipt.get("evaluation",{}).get("summary",{})
if summary.get("success_rate") != 89/96 or summary.get("heading_success_rates",{}).get("3") != 2/3 or summary.get("heading_success_rates",{}).get("1") != 10/12: raise SystemExit("Ant v10 antecedent strata changed")
if identity.get("job_id") != 30193111 or identity.get("seed") != 73010: raise SystemExit("Ant v10 identity differs from the consumed gate")
for raw in sys.argv[4:]:
    if pathlib.Path(raw).exists(): raise SystemExit(f"Ant v11 requires no v10 route or language-model execution: {raw}")
PYV10

hash_tree() { local tree="$1"; (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" "$TRAINER" --help >/dev/null
  "$PYTHON_BIN" -c "import sys;from collections import Counter;sys.path.insert(0,'$ROOT_DIR/ops');import train_ant_waypoint_controller_v11 as v11;from stable_baselines3 import PPO;assert Counter(v11.TRAINING_HEADING_SCHEDULE)==Counter({3:6,1:4,0:1,2:1,4:1,5:1,6:1,7:1});assert len(v11.DEVELOPMENT_MAPS)==4 and all(len(m)==12 and all(len(r)==12 for r in m) for m in v11.DEVELOPMENT_MAPS);assert all([v11._stratified_edge_index(e,r) for r in range(3)]==[0,(len(e)-1)//2,len(e)-1] for h in v11.DEVELOPMENT_EDGES for e in h);env=v11.v10.v9.AntMazeLocalWaypointEnv(rank=0,episode_steps=400);obs,_=env.reset(seed=73011);model=PPO.load('$INITIAL_MODEL',device='cpu');assert obs.shape==model.observation_space.shape;env.close()"
  bash -n "$SLURM_SCRIPT"
  "$IDENTITY_PYTHON" -m pytest -q "$ROOT_DIR/tests/test_ant_waypoint_controller_v11.py"
  echo "[ant-waypoint-v11] configuration passed; no controller trained"
  exit 0
fi
for fresh in "$IDENTITY" "$MODEL" "$RECEIPT"; do [[ ! -e "$fresh" ]] || { echo "Fresh Ant waypoint v11 artifact required: $fresh" >&2; exit 1; }; done

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-waypoint-v11-ops.XXXXXX")"
cp "$TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v11.py"
cp "$V10_TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v10.py"
cp "$V9_TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v9.py"
cp "$BASE" "$EXECUTION_INPUT/train_ant_waypoint_controller_v7.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/train_ant_waypoint_controller_v11.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_waypoint_v11_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_ant_waypoint_controller_v11.py" ]]; then mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"; else find "$EXECUTION_INPUT" -type f -delete; rmdir "$EXECUTION_INPUT"; fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || exit 1

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold --partition=all --account=allcs \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
  "$EXECUTION_ROOT/train_ant_waypoint_controller_v11.slurm")"
job_id="${job_id%%;*}"; [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Ant waypoint v11 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }; trap cleanup EXIT

"$IDENTITY_PYTHON" - "$IDENTITY" "$job_id" "$EXECUTION_HASH" "$PROTOCOL" "$AMENDMENT" "$TRAINER" "$V10_TRAINER" "$V9_TRAINER" "$BASE" "$SLURM_SCRIPT" "$INITIAL_MODEL" "$V10_RECEIPT" "$V10_IDENTITY" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema_version":"ant-waypoint-controller-v11-identity-v1","job_id":int(sys.argv[2]),"execution_hash":sys.argv[3],
 "protocol_sha256":digest(sys.argv[4]),"placement_amendment_sha256":digest(sys.argv[5]),
 "trainer_sha256":digest(sys.argv[6]),"v10_trainer_sha256":digest(sys.argv[7]),
 "v9_trainer_sha256":digest(sys.argv[8]),"base_trainer_sha256":digest(sys.argv[9]),"slurm_sha256":digest(sys.argv[10]),
 "initial_model_sha256":digest(sys.argv[11]),"v10_receipt_sha256":digest(sys.argv[12]),"v10_identity_sha256":digest(sys.argv[13]),
 "prelaunch_canceled_job_id":30198258,
 "language_model_sampled":False,"route_map_executed":False,"seed":73011,"timesteps":2000000,"learning_rate":1e-6,
 "training_map_count":4,"development_map_count":4,"evaluation_map_size":12,
 "training_heading_schedule":[3,1,3,0,3,1,3,2,3,1,3,4,1,5,6,7],"evaluation_seed_base":6073011,
 "evaluation_episodes":96,"evaluation_edge_indices":"0,floor((n-1)/2),n-1",
 "thresholds":{"overall":0.90,"per_heading":0.75,"per_map":0.75,"unhealthy":0.10,"median_steps":300}}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID

record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'NumCPUs=12' 'MinMemoryNode=32G' 'TimeLimit=02:00:00'; do [[ "$record" == *"$required"* ]] || { echo "Ant v11 held job missing $required" >&2; exit 1; }; done
scontrol update "JobId=$job_id" Partition=all
record="$(scontrol show job "$job_id" -o)"
[[ "$record" == *'Partition=all'* ]] || { echo "Ant v11 placement repair did not reach partition all" >&2; exit 1; }
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-waypoint-v11] released job $job_id"
echo "[ant-waypoint-v11] identity=$IDENTITY"
