#!/usr/bin/env bash
# Snapshot and submit the frozen maze-blind Ant waypoint controller v8.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON_BIN="$ROOT_DIR/var/maze_runtime/venv/bin/python"
IDENTITY_PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v8.py"
BASE="$ROOT_DIR/ops/train_ant_waypoint_controller_v7.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_ant_waypoint_controller_v8.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_waypoint_controller_v8_20260729.md"
INITIAL_MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v7.zip"
V7_RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v7.evaluation.json"
V7_IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v7_identity.json"
IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v8_identity.json"
MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v8.zip"
RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v8.evaluation.json"
EXPECTED_INITIAL=2bb5d192698639a294ac8266ae9be68484c9d493f4f7ba4317ee06d2d4b04442

for required in "$PYTHON_BIN" "$IDENTITY_PYTHON" "$TRAINER" "$BASE" \
  "$SLURM_SCRIPT" "$PROTOCOL" "$INITIAL_MODEL" "$V7_RECEIPT" "$V7_IDENTITY"; do
  [[ -e "$required" ]] || { echo "Missing Ant waypoint v8 prerequisite: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$INITIAL_MODEL" | cut -d' ' -f1)" == "$EXPECTED_INITIAL" ]] || {
  echo "Ant v8 initialization hash mismatch" >&2; exit 1;
}
"$IDENTITY_PYTHON" - "$V7_RECEIPT" "$V7_IDENTITY" "$EXPECTED_INITIAL" <<'PYV7'
import json, sys
receipt=json.load(open(sys.argv[1],encoding="utf-8"))
identity=json.load(open(sys.argv[2],encoding="utf-8"))
summary=receipt.get("evaluation",{}).get("summary",{})
if receipt.get("status") != "fail" or receipt.get("decision") != "ant_waypoint_v7_ineligible":
    raise SystemExit("Ant v8 requires the exact frozen failed v7 gate")
if receipt.get("hashes",{}).get("model_sha256") != sys.argv[3]:
    raise SystemExit("Ant v7 receipt does not bind the v8 initialization")
if identity.get("job_id") != 30187375 or identity.get("seed") != 73007:
    raise SystemExit("Ant v7 identity differs from the consumed development gate")
if summary.get("episodes") != 96 or summary.get("success_rate") != 86/96:
    raise SystemExit("Ant v7 aggregate antecedent differs from the frozen result")
PYV7

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" "$TRAINER" --help >/dev/null
  echo "[ant-waypoint-v8] configuration passed; no controller trained"
  exit 0
fi
for fresh in "$IDENTITY" "$MODEL" "$RECEIPT"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh Ant waypoint v8 artifact required: $fresh" >&2; exit 1; }
done

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-waypoint-v8-ops.XXXXXX")"
cp "$TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v8.py"
cp "$BASE" "$EXECUTION_INPUT/train_ant_waypoint_controller_v7.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/train_ant_waypoint_controller_v8.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_waypoint_v8_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_ant_waypoint_controller_v8.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || { echo "Ant v8 execution snapshot mismatch" >&2; exit 1; }

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT" \
  "$EXECUTION_ROOT/train_ant_waypoint_controller_v8.slurm")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Ant waypoint v8 job ID: $job_id" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$IDENTITY_PYTHON" - "$IDENTITY" "$job_id" "$EXECUTION_HASH" "$PROTOCOL" \
  "$TRAINER" "$BASE" "$SLURM_SCRIPT" "$INITIAL_MODEL" "$V7_RECEIPT" "$V7_IDENTITY" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema_version":"ant-waypoint-controller-v8-identity-v1","job_id":int(sys.argv[2]),
 "execution_hash":sys.argv[3],"protocol_sha256":digest(sys.argv[4]),
 "trainer_sha256":digest(sys.argv[5]),"base_trainer_sha256":digest(sys.argv[6]),
 "slurm_sha256":digest(sys.argv[7]),"initial_model_sha256":digest(sys.argv[8]),
 "v7_receipt_sha256":digest(sys.argv[9]),"v7_identity_sha256":digest(sys.argv[10]),
 "environment":"open-plane Ant-v5 only","maze_loaded":False,"language_model_sampled":False,
 "seed":73008,"timesteps":3000000,"learning_rate":5e-6,
 "training_waypoint_distances":[4.0],
 "heading_schedule":"unchanged cyclic worker-rank offsets across eight headings",
 "evaluation_seed_base":3073008,"evaluation_episodes":96,
 "thresholds":{"overall":0.90,"per_heading":0.75,"unhealthy":0.10,"median_steps":300}}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
os.replace(tmp,path)
PYID

job_record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'NumCPUs=12' \
  'MinMemoryNode=24G' 'TimeLimit=03:00:00'; do
  [[ "$job_record" == *"$required"* ]] || { echo "Ant v8 held-job audit missing: $required" >&2; exit 1; }
done
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-waypoint-v8] released job $job_id"
echo "[ant-waypoint-v8] identity=$IDENTITY"
