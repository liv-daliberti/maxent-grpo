#!/usr/bin/env bash
# Snapshot and submit the frozen maze-blind Ant waypoint controller v6.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac

PYTHON_BIN="$ROOT_DIR/var/maze_runtime/venv/bin/python"
TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v6.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_ant_waypoint_controller_v6.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_waypoint_controller_v6_20260729.md"
INITIAL_MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_heading_v1.zip"
V5_RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_heading_v5.evaluation.json"
FAILED_ROUTE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_modebench_v1_generation_r1_identity.json"
FAILED_ROUTE_LOG="$ROOT_DIR/var/artifacts/logs/antmaze-route-gate-30184769.err"
IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v6_identity.json"
MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v6.zip"
RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v6.evaluation.json"

for required in "$PYTHON_BIN" "$TRAINER" "$SLURM_SCRIPT" "$PROTOCOL" \
  "$INITIAL_MODEL" "$V5_RECEIPT" "$FAILED_ROUTE_IDENTITY" "$FAILED_ROUTE_LOG"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing Ant waypoint v6 prerequisite: $required" >&2
    exit 1
  fi
done

observed_initial="$(sha256sum "$INITIAL_MODEL" | cut -d' ' -f1)"
if [[ "$observed_initial" != "526b669bb14cf8a07b44a1c725f94411632a3ac8a8e84966db09336cc96e4b0f" ]]; then
  echo "Ant waypoint initialization hash mismatch" >&2
  exit 1
fi
"$PYTHON_BIN" - "$V5_RECEIPT" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "pass":
    raise SystemExit("Ant v5 open-plane receipt is not passing")
PY

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
  )
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" "$TRAINER" --help >/dev/null
  echo "[ant-waypoint-v6] configuration passed; no full controller trained"
  exit 0
fi

for fresh in "$IDENTITY" "$MODEL" "$RECEIPT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh Ant waypoint v6 artifact required: $fresh" >&2
    exit 1
  fi
done

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-waypoint-v6-ops.XXXXXX")"
cp "$TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v6.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/train_ant_waypoint_controller_v6.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_waypoint_v6_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_ant_waypoint_controller_v6.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
if [[ "$(hash_tree "$EXECUTION_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "Ant waypoint v6 execution snapshot mismatch" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT" \
    "$EXECUTION_ROOT/train_ant_waypoint_controller_v6.slurm"
)"
job_id="${job_id%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Invalid Ant waypoint v6 job ID: $job_id" >&2
  exit 1
fi

cleanup() {
  local status="$?"
  trap - EXIT
  scancel "$job_id" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - \
  "$IDENTITY" "$job_id" "$EXECUTION_HASH" "$PROTOCOL" "$TRAINER" \
  "$SLURM_SCRIPT" "$INITIAL_MODEL" "$V5_RECEIPT" \
  "$FAILED_ROUTE_IDENTITY" "$FAILED_ROUTE_LOG" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1])
payload={
    "schema_version":"ant-waypoint-controller-v6-identity-v1",
    "job_id":int(sys.argv[2]),
    "execution_hash":sys.argv[3],
    "protocol_sha256":digest(sys.argv[4]),
    "trainer_sha256":digest(sys.argv[5]),
    "slurm_sha256":digest(sys.argv[6]),
    "initial_model_sha256":digest(sys.argv[7]),
    "v5_receipt_sha256":digest(sys.argv[8]),
    "failed_route_identity_sha256":digest(sys.argv[9]),
    "failed_route_log_sha256":digest(sys.argv[10]),
    "environment":"open-plane Ant-v5 only",
    "maze_loaded":False,
    "language_model_sampled":False,
    "seed":73006,
    "timesteps":3000000,
    "evaluation_episodes":96,
}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY

scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-waypoint-v6] released job $job_id"
echo "[ant-waypoint-v6] identity=$IDENTITY"
