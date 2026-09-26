#!/usr/bin/env bash
# Snapshot and submit the frozen maze-local Ant waypoint controller v9.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/maze_runtime/venv/bin/python"
IDENTITY_PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v9.py"
BASE="$ROOT_DIR/ops/train_ant_waypoint_controller_v7.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_ant_waypoint_controller_v9.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_waypoint_controller_v9_20260729.md"
INITIAL_MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v8.zip"
V8_RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v8.evaluation.json"
V8_IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v8_identity.json"
V8_ROUTE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v8_generation_identity.json"
V8_ROUTE_LOG="$ROOT_DIR/var/artifacts/logs/antmaze-v8-route-gate-30188228.err"
IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v9_identity.json"
MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v9.zip"
RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v9.evaluation.json"
EXPECTED_INITIAL=77e780dfff1147bc2f542c1761f6efeaa2305fd5ddb625244ebf8e82b3fa871d

for required in "$PYTHON_BIN" "$IDENTITY_PYTHON" "$TRAINER" "$BASE" \
  "$SLURM_SCRIPT" "$PROTOCOL" "$INITIAL_MODEL" "$V8_RECEIPT" \
  "$V8_IDENTITY" "$V8_ROUTE_IDENTITY" "$V8_ROUTE_LOG"; do
  [[ -e "$required" ]] || {
    echo "Missing Ant waypoint v9 prerequisite: $required" >&2
    exit 1
  }
done
[[ "$(sha256sum "$INITIAL_MODEL" | cut -d' ' -f1)" == "$EXPECTED_INITIAL" ]] || {
  echo "Ant v9 initialization hash mismatch" >&2
  exit 1
}
"$IDENTITY_PYTHON" - "$V8_RECEIPT" "$V8_IDENTITY" "$V8_ROUTE_IDENTITY" \
  "$V8_ROUTE_LOG" "$EXPECTED_INITIAL" <<'PYV8'
import json, pathlib, sys
receipt=json.load(open(sys.argv[1],encoding="utf-8"))
identity=json.load(open(sys.argv[2],encoding="utf-8"))
route_identity=json.load(open(sys.argv[3],encoding="utf-8"))
route_log=pathlib.Path(sys.argv[4]).read_text(encoding="utf-8")
if receipt.get("status") != "pass" or receipt.get("decision") != "admitted_to_fresh_maze_route_gate_v8":
    raise SystemExit("Ant v9 requires the exact admitted open-plane v8 gate")
if receipt.get("hashes",{}).get("model_sha256") != sys.argv[5]:
    raise SystemExit("Ant v8 receipt does not bind the v9 initialization")
if identity.get("job_id") != 30187810 or identity.get("seed") != 73008:
    raise SystemExit("Ant v8 identity differs from the consumed gate")
if route_identity.get("job_id") != 30188228 or route_identity.get("map_substitution") is not False:
    raise SystemExit("Ant v8 route identity differs from the stopped slate")
if "ant_v8_admission_train_00 upper fixture failed" not in route_log:
    raise SystemExit("Ant v8 route failure boundary is missing")
PYV8

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" "$TRAINER" --help >/dev/null
  "$PYTHON_BIN" -c "import sys;sys.path.insert(0,'$ROOT_DIR/ops');import train_ant_waypoint_controller_v9 as v9;from stable_baselines3 import PPO;env=v9.AntMazeLocalWaypointEnv(rank=0,episode_steps=400);obs,_=env.reset(seed=73009);model=PPO.load('$INITIAL_MODEL',device='cpu');assert obs.shape==model.observation_space.shape;env.close()"
  bash -n "$SLURM_SCRIPT"
  echo "[ant-waypoint-v9] configuration passed; no controller trained"
  exit 0
fi
for fresh in "$IDENTITY" "$MODEL" "$RECEIPT"; do
  [[ ! -e "$fresh" ]] || {
    echo "Fresh Ant waypoint v9 artifact required: $fresh" >&2
    exit 1
  }
done

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-waypoint-v9-ops.XXXXXX")"
cp "$TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v9.py"
cp "$BASE" "$EXECUTION_INPUT/train_ant_waypoint_controller_v7.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/train_ant_waypoint_controller_v9.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_waypoint_v9_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_ant_waypoint_controller_v9.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || {
  echo "Ant v9 execution snapshot mismatch" >&2
  exit 1
}

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY" \
  "$EXECUTION_ROOT/train_ant_waypoint_controller_v9.slurm")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid Ant waypoint v9 job ID: $job_id" >&2
  exit 1
}
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$IDENTITY_PYTHON" - "$IDENTITY" "$job_id" "$EXECUTION_HASH" "$PROTOCOL" \
  "$TRAINER" "$BASE" "$SLURM_SCRIPT" "$INITIAL_MODEL" "$V8_RECEIPT" \
  "$V8_IDENTITY" "$V8_ROUTE_IDENTITY" "$V8_ROUTE_LOG" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema_version":"ant-waypoint-controller-v9-identity-v1","job_id":int(sys.argv[2]),
 "execution_hash":sys.argv[3],"protocol_sha256":digest(sys.argv[4]),
 "trainer_sha256":digest(sys.argv[5]),"base_trainer_sha256":digest(sys.argv[6]),
 "slurm_sha256":digest(sys.argv[7]),"initial_model_sha256":digest(sys.argv[8]),
 "v8_receipt_sha256":digest(sys.argv[9]),"v8_identity_sha256":digest(sys.argv[10]),
 "v8_route_identity_sha256":digest(sys.argv[11]),"v8_route_failure_log_sha256":digest(sys.argv[12]),
 "environment":"four frozen 7x7 training maps; one-cell free-edge episodes only",
 "training_map_count":4,"development_map_count":4,"development_maps_loaded_during_training":False,
 "language_model_sampled":False,"seed":73009,"timesteps":5000000,"learning_rate":5e-6,
 "training_waypoint_distances":[4.0],"heading_schedule":"cyclic balanced eight headings",
 "evaluation_seed_base":4073009,"evaluation_episodes":96,
 "thresholds":{"overall":0.90,"per_heading":0.75,"per_map":0.75,"unhealthy":0.10,"median_steps":300}}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
os.replace(tmp,path)
PYID

job_record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'NumCPUs=12' \
  'MinMemoryNode=32G' 'TimeLimit=03:00:00' \
  "OAT_ZERO_PROTOCOL_IDENTITY=$IDENTITY"; do
  [[ "$job_record" == *"$required"* ]] || {
    echo "Ant v9 held-job audit missing: $required" >&2
    exit 1
  }
done
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-waypoint-v9] released job $job_id"
echo "[ant-waypoint-v9] identity=$IDENTITY"
