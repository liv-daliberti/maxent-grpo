#!/usr/bin/env bash
# Snapshot and launch the prospective PointMaze-GeometryShift dev gate.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac

PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
EVALUATOR="$ROOT_DIR/ops/evaluate_point_maze_interactive_viability.py"
BATCH="$ROOT_DIR/ops/slurm/evaluate_point_maze_geometry_shift_viability_v1.slurm"
PROTOCOL="$ROOT_DIR/paper/preregistration/point_maze_geometry_shift_replacement_v1_20260730.md"
DATA="$ROOT_DIR/var/data/point_maze_geometry_shift_v1"
ADMISSION="$ROOT_DIR/var/artifacts/point_maze_geometry_shift_v1_admission_audit.json"
MODEL="$ROOT_DIR/var/models/point_maze_interactive_warmstart_v3"
MODEL_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_sft_v3.json"
IDENTITY="$ROOT_DIR/var/artifacts/point_maze_geometry_shift_viability_v1_identity.json"
SUBMISSION="$ROOT_DIR/var/artifacts/point_maze_geometry_shift_viability_v1_submission.json"
RECEIPT="$ROOT_DIR/var/artifacts/point_maze_geometry_shift_05b_viability_v1.json"

for required in "$PYTHON" "$EVALUATOR" "$BATCH" "$PROTOCOL" "$DATA/identity.json" \
  "$DATA/dev/dataset_dict.json" "$ADMISSION" "$MODEL/config.json" "$MODEL/model.safetensors" \
  "$MODEL_RECEIPT" "$ROOT_DIR/var/maze_runtime/venv/bin/python"; do
  [[ -e "$required" ]] || { echo "Missing geometry-shift prerequisite: $required" >&2; exit 1; }
done

hash_tree() {
  local tree="$1"
  (cd "$tree"; find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

if [[ "$phase" == config ]]; then
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON" -m py_compile "$EVALUATOR"
  PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src" "$PYTHON" -m pytest -q \
    "$ROOT_DIR/tests/test_point_maze_geometry_shift_v1.py" \
    "$ROOT_DIR/tests/test_point_maze_interactive_policy.py" \
    "$ROOT_DIR/tests/test_point_maze_interactive_worker.py"
  bash -n "$BATCH"
  echo "[point-geometry-shift-v1] configuration passed; no new-map model sample"
  exit 0
fi

for fresh in "$IDENTITY" "$SUBMISSION" "$RECEIPT"; do
  [[ ! -e "$fresh" ]] || { echo "Fresh geometry-shift target required: $fresh" >&2; exit 1; }
done
[[ "$(jq -er '.status' "$ADMISSION")" == pass ]] || { echo "geometry-shift admission is not pass" >&2; exit 1; }

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/point_geometry_shift_v1_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$staging/src"; cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"; rmdir "$staging"
fi
[[ "$(hash_tree "$SOURCE_ROOT")" == "$SOURCE_HASH" ]] || { echo "geometry-shift source snapshot drift" >&2; exit 1; }

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.point-geometry-shift-v1.XXXXXX")"
cp "$EVALUATOR" "$OPS_INPUT/$(basename "$EVALUATOR")"
cp "$BATCH" "$OPS_INPUT/$(basename "$BATCH")"
cp "$PROTOCOL" "$OPS_INPUT/$(basename "$PROTOCOL")"
EXECUTION_HASH="$(hash_tree "$OPS_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/point_geometry_shift_v1_ops_${EXECUTION_HASH}"
if [[ ! -d "$EXECUTION_ROOT" ]]; then mv "$OPS_INPUT" "$EXECUTION_ROOT"; else find "$OPS_INPUT" -type f -delete; rmdir "$OPS_INPUT"; fi
[[ "$(hash_tree "$EXECUTION_ROOT")" == "$EXECUTION_HASH" ]] || { echo "geometry-shift execution snapshot drift" >&2; exit 1; }

mkdir -p "$ROOT_DIR/var/artifacts/logs"
job_id="$(sbatch --parsable --hold --partition=all --account=allcs \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH" \
  "$EXECUTION_ROOT/$(basename "$BATCH")")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid geometry-shift job ID" >&2; exit 1; }
cleanup() { local status="$?"; trap - EXIT; scancel "$job_id" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT

"$PYTHON" - "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" "$PROTOCOL" \
  "$0" "$DATA/identity.json" "$ADMISSION" "$MODEL_RECEIPT" "$MODEL/config.json" <<'PYID'
import hashlib,json,os,pathlib,sys,tempfile
def d(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
p=pathlib.Path(sys.argv[1]); x={
 "schema":"point-maze-geometry-shift-viability-identity-v1","job_id":int(sys.argv[2]),
 "source_hash":sys.argv[3],"execution_hash":sys.argv[4],"protocol_sha256":d(sys.argv[5]),
 "launcher_sha256":d(sys.argv[6]),"dataset_identity_sha256":d(sys.argv[7]),
 "admission_audit_sha256":d(sys.argv[8]),"model_receipt_sha256":d(sys.argv[9]),
 "model_config_sha256":d(sys.argv[10]),"model":"point_maze_interactive_warmstart_v3",
 "policy_interface":"velocity_state_v3","seed":75121,"samples_per_map":64,"prefix_count":16,
 "development_maps":4,"evaluation_rows_loaded":False,"shared_checkpoint_with_point_maze":True,
 "replacement_configuration":True,"language_model_parameters":"0.5B"
}
p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w") as h: json.dump(x,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PYID
record="$(scontrol show job "$job_id" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'gres/gpu:a5000:1' 'NumCPUs=8' 'MinMemoryNode=64G' 'TimeLimit=02:00:00'; do
  [[ "$record" == *"$required"* ]] || { echo "Held geometry-shift job missing $required" >&2; exit 1; }
done
"$PYTHON" - "$SUBMISSION" "$job_id" "$IDENTITY" "$record" <<'PYSUB'
import hashlib,json,os,pathlib,sys,tempfile
p=pathlib.Path(sys.argv[1]); i=pathlib.Path(sys.argv[3]); x={"schema":"point-maze-geometry-shift-submission-v1","job_id":int(sys.argv[2]),"identity_sha256":hashlib.sha256(i.read_bytes()).hexdigest(),"held_job_audit":"pass","released":True,"scheduler_record":sys.argv[4]}
fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w") as h: json.dump(x,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PYSUB
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
trap - EXIT
echo "[point-geometry-shift-v1] released job $job_id"
