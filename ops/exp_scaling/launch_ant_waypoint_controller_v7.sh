#!/usr/bin/env bash
# Snapshot and submit the frozen maze-blind Ant waypoint controller v7.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac

PYTHON_BIN="$ROOT_DIR/var/maze_runtime/venv/bin/python"
TRAINER="$ROOT_DIR/ops/train_ant_waypoint_controller_v7.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_ant_waypoint_controller_v7.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/ant_waypoint_controller_v7_20260729.md"
INITIAL_MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v6.zip"
V6_RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v6.evaluation.json"
V6_IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v6_identity.json"
IDENTITY="$ROOT_DIR/var/artifacts/ant_waypoint_controller_v7_identity.json"
MODEL="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v7.zip"
RECEIPT="$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v7.evaluation.json"
EXPECTED_INITIAL=ac21f650cb25141670d2316b8f9dd691956e6c23d97791152a826835d61b7d17

for required in "$PYTHON_BIN" "$TRAINER" "$SLURM_SCRIPT" "$PROTOCOL" \
  "$INITIAL_MODEL" "$V6_RECEIPT" "$V6_IDENTITY"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing Ant waypoint v7 prerequisite: $required" >&2
    exit 1
  fi
done
observed_initial="$(sha256sum "$INITIAL_MODEL" | cut -d' ' -f1)"
if [[ "$observed_initial" != "$EXPECTED_INITIAL" ]]; then
  echo "Ant v7 initialization hash mismatch" >&2
  exit 1
fi
"$PYTHON_BIN" - "$V6_RECEIPT" "$V6_IDENTITY" "$EXPECTED_INITIAL" <<'PYV6'
import json, sys
receipt=json.load(open(sys.argv[1], encoding="utf-8"))
identity=json.load(open(sys.argv[2], encoding="utf-8"))
if receipt.get("status") != "fail" or receipt.get("decision") != "ant_waypoint_v6_ineligible":
    raise SystemExit("Ant v7 requires the exact frozen failed v6 gate")
if receipt.get("hashes", {}).get("model_sha256") != sys.argv[3]:
    raise SystemExit("Ant v6 receipt does not bind the v7 initialization")
if identity.get("job_id") != 30185409 or identity.get("seed") != 73006:
    raise SystemExit("Ant v6 identity differs from the consumed development gate")
PYV6

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  "$PYTHON_BIN" "$TRAINER" --help >/dev/null
  echo "[ant-waypoint-v7] configuration passed; no controller trained"
  exit 0
fi

for fresh in "$IDENTITY" "$MODEL" "$RECEIPT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh Ant waypoint v7 artifact required: $fresh" >&2
    exit 1
  fi
done

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.ant-waypoint-v7-ops.XXXXXX")"
cp "$TRAINER" "$EXECUTION_INPUT/train_ant_waypoint_controller_v7.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/train_ant_waypoint_controller_v7.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/ant_waypoint_v7_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_ant_waypoint_controller_v7.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
if [[ "$(hash_tree "$EXECUTION_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "Ant waypoint v7 execution snapshot mismatch" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT" \
    "$EXECUTION_ROOT/train_ant_waypoint_controller_v7.slurm"
)"
job_id="${job_id%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Invalid Ant waypoint v7 job ID: $job_id" >&2
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
  "$SLURM_SCRIPT" "$INITIAL_MODEL" "$V6_RECEIPT" "$V6_IDENTITY" <<'PYID'
import hashlib, json, os, pathlib, sys, tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1])
payload={
    "schema_version":"ant-waypoint-controller-v7-identity-v1",
    "job_id":int(sys.argv[2]),
    "execution_hash":sys.argv[3],
    "protocol_sha256":digest(sys.argv[4]),
    "trainer_sha256":digest(sys.argv[5]),
    "slurm_sha256":digest(sys.argv[6]),
    "initial_model_sha256":digest(sys.argv[7]),
    "v6_receipt_sha256":digest(sys.argv[8]),
    "v6_identity_sha256":digest(sys.argv[9]),
    "environment":"open-plane Ant-v5 only",
    "maze_loaded":False,
    "language_model_sampled":False,
    "seed":73007,
    "timesteps":5000000,
    "learning_rate":1e-5,
    "training_waypoint_distances":[4.0],
    "heading_schedule":"cyclic worker-rank offsets across eight headings",
    "evaluation_seed_base":2073007,
    "evaluation_episodes":96,
}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PYID

scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[ant-waypoint-v7] released job $job_id"
echo "[ant-waypoint-v7] identity=$IDENTITY"
