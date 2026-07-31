#!/usr/bin/env bash
# Snapshot and submit the frozen northeast-remediation Ant controller v10.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/maze_runtime/venv/bin/python"
IDENTITY_PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v10.py"
V9_TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v9.py"
BASE="$ROOT_DIR/ops/train_ant_waypoint_controller_v7.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_ant_waypoint_controller_v10.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_waypoint_controller_v10_20260729.md"
INITIAL_MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v9.zip"
V9_RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v9.evaluation.json"
V9_IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v9_identity.json"
V9_ROUTE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v9_generation_identity.json"
V9_ROUTE_DATA="$ROOT_DIR/var/data/ant_maze_modebench_v9"
V9_ROUTE_AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v9_admission_audit.json"
V9_VIABILITY="$ROOT_DIR/var/artifacts/ant_maze_05b_viability_v9.json"
IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v10_identity.json"
MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v10.zip"
RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v10.evaluation.json"
EXPECTED_INITIAL=c49cdb1b8c1bf0d89027766d062492e84538c827d6aa88a2079067a6867b19b7

for required in "$PYTHON_BIN" "$IDENTITY_PYTHON" "$TRAINER" "$V9_TRAINER" \
  "$BASE" "$SLURM_SCRIPT" "$PROTOCOL" "$INITIAL_MODEL" "$V9_RECEIPT" \
  "$V9_IDENTITY"; do
  [[ -e "$required" ]] || {
    echo "Missing Ant waypoint v10 prerequisite: $required" >&2
    exit 1
  }
done
[[ "$(sha256sum "$INITIAL_MODEL" | cut -d' ' -f1)" == "$EXPECTED_INITIAL" ]] || {
  echo "Ant v10 initialization hash mismatch" >&2
  exit 1
}
"$IDENTITY_PYTHON" - "$V9_RECEIPT" "$V9_IDENTITY" "$EXPECTED_INITIAL" \
  "$V9_ROUTE_IDENTITY" "$V9_ROUTE_DATA" "$V9_ROUTE_AUDIT" "$V9_VIABILITY" <<'PYV9'
import json, pathlib, sys
receipt=json.load(open(sys.argv[1],encoding="utf-8"))
identity=json.load(open(sys.argv[2],encoding="utf-8"))
if receipt.get("status") != "fail" or receipt.get("decision") != "ant_waypoint_v9_ineligible":
    raise SystemExit("Ant v10 requires the exact terminal failed-v9 receipt")
if receipt.get("hashes",{}).get("model_sha256") != sys.argv[3]:
    raise SystemExit("Ant v9 receipt does not bind the v10 initialization")
summary=receipt.get("evaluation",{}).get("summary",{})
if summary.get("success_rate") != 0.90625 or summary.get("heading_success_rates",{}).get("1") != 1/3:
    raise SystemExit("Ant v9 failure stratum differs from the registered antecedent")
if identity.get("job_id") != 30192731 or identity.get("seed") != 73009:
    raise SystemExit("Ant v9 identity differs from the consumed gate")
for raw in sys.argv[4:]:
    if pathlib.Path(raw).exists():
        raise SystemExit(f"Ant v10 requires no v9 route or language-model execution: {raw}")
PYV9

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" "$TRAINER" --help >/dev/null
  "$PYTHON_BIN" -c "import sys;from collections import Counter;sys.path.insert(0,'$ROOT_DIR/ops');import train_ant_waypoint_controller_v10 as v10;from stable_baselines3 import PPO;assert Counter(v10.TRAINING_HEADING_SCHEDULE)==Counter({1:7,0:1,2:1,3:1,4:1,5:1,6:1,7:1});assert len(v10.DEVELOPMENT_MAPS)==4 and all(len(m)==10 and all(len(r)==10 for r in m) for m in v10.DEVELOPMENT_MAPS);assert all([v10._stratified_edge_index(e,r) for r in range(3)]==[0,(len(e)-1)//2,len(e)-1] for h in v10.DEVELOPMENT_EDGES for e in h);env=v10.v9.AntMazeLocalWaypointEnv(rank=0,episode_steps=400);obs,_=env.reset(seed=73010);model=PPO.load('$INITIAL_MODEL',device='cpu');assert obs.shape==model.observation_space.shape;env.close()"
  bash -n "$SLURM_SCRIPT"
  "$IDENTITY_PYTHON" -m pytest -q \
    "$ROOT_DIR/tests/test_ant_waypoint_controller_v10.py"
  echo "[ant-waypoint-v10] configuration passed; no controller trained"
  exit 0
fi
for fresh in "$IDENTITY" "$MODEL" "$RECEIPT"; do
  [[ ! -e "$fresh" ]] || {
    echo "Fresh Ant waypoint v10 artifact required: $fresh" >&2
    exit 1
  }
done

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-waypoint-v10-ops.XXXXXX")"
cp "$TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v10.py"
cp "$V9_TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v9.py"
cp "$BASE" "$EXECUTION_INPUT/train_ant_waypoint_controller_v7.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/train_ant_waypoint_controller_v10.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_waypoint_v10_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_ant_waypoint_controller_v10.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || {
  echo "Ant v10 execution snapshot mismatch" >&2
  exit 1
}

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
  "$EXECUTION_ROOT/train_ant_waypoint_controller_v10.slurm")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Ant waypoint v10 job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$IDENTITY_PYTHON" - "$IDENTITY" "$job_id" "$EXECUTION_HASH" "$PROTOCOL" \
  "$TRAINER" "$V9_TRAINER" "$BASE" "$SLURM_SCRIPT" "$INITIAL_MODEL" \
  "$V9_RECEIPT" "$V9_IDENTITY" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema_version":"ant-waypoint-controller-v10-identity-v1","job_id":int(sys.argv[2]),
 "execution_hash":sys.argv[3],"protocol_sha256":digest(sys.argv[4]),
 "trainer_sha256":digest(sys.argv[5]),"v9_trainer_sha256":digest(sys.argv[6]),
 "base_trainer_sha256":digest(sys.argv[7]),"slurm_sha256":digest(sys.argv[8]),
 "initial_model_sha256":digest(sys.argv[9]),"v9_receipt_sha256":digest(sys.argv[10]),
 "v9_identity_sha256":digest(sys.argv[11]),"language_model_sampled":False,
 "route_map_executed":False,"seed":73010,"timesteps":2000000,"learning_rate":2e-6,
 "training_map_count":4,"development_map_count":4,"evaluation_map_size":10,
 "training_heading_schedule":[1,0,1,2,1,3,1,4,1,5,1,6,1,7],
 "evaluation_seed_base":5073010,"evaluation_episodes":96,
 "evaluation_edge_indices":"0,floor((n-1)/2),n-1",
 "thresholds":{"overall":0.90,"per_heading":0.75,"per_map":0.75,"unhealthy":0.10,"median_steps":300}}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID

job_record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'NumCPUs=12' \
  'MinMemoryNode=32G' 'TimeLimit=02:00:00' \
  "OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY"; do
  [[ "$job_record" == *"$required"* ]] || {
    echo "Ant v10 held-job audit missing: $required" >&2
    exit 1
  }
done
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-waypoint-v10] released job $job_id"
echo "[ant-waypoint-v10] identity=$IDENTITY"
