#!/usr/bin/env bash
# Snapshot and submit the frozen PointMaze train-only shared warm start.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/point_maze_modebench_v1"
WARMSTART_DATA="$ROOT_DIR/var/data/point_maze_interactive_warmstart_v1"
AUDIT="$ROOT_DIR/var/artifacts/point_maze_modebench_v1_admission_audit.json"
PROTOCOL="$ROOT_DIR/paper/preregistration/point_maze_interactive_warmstart_sft_v1_20260729.md"
TRAINER="$ROOT_DIR/ops/train_point_maze_interactive_warmstart_v1.py"
EVALUATOR="$ROOT_DIR/ops/evaluate_point_maze_interactive_viability.py"
SLURM_SCRIPT="$ROOT_DIR/ops/slurm/train_point_maze_interactive_warmstart_v1.slurm"
IDENTITY="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_v1_identity.json"
OUTPUT_MODEL="$ROOT_DIR/var/models/point_maze_interactive_warmstart_v1"
SFT_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_sft_v1.json"
GATE_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_warmstart_v1.json"

for required in \
  "$PYTHON_BIN" "$MODEL_ROOT/config.json" "$DATA_ROOT/identity.json" \
  "$DATA_ROOT/dev/dataset_dict.json" "$WARMSTART_DATA/identity.json" \
  "$WARMSTART_DATA/examples.jsonl" "$AUDIT" "$PROTOCOL" "$TRAINER" \
  "$EVALUATOR" "$SLURM_SCRIPT" \
  "$ROOT_DIR/var/maze_runtime/venv/bin/python"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing PointMaze warm-start prerequisite: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$WARMSTART_DATA/identity.json" "$WARMSTART_DATA/examples.jsonl" "$AUDIT" <<'PY'
import hashlib, json, pathlib, sys
identity=json.load(open(sys.argv[1],encoding="utf-8"))
if identity.get("status") != "pass" or identity.get("split") != "train_only":
    raise SystemExit("PointMaze warm-start data identity is not passing")
boundary=identity.get("information_boundary",{})
if boundary.get("dev_dataset_loaded") or boundary.get("eval_dataset_loaded"):
    raise SystemExit("PointMaze warm-start data firewall failed")
digest=hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
if digest != identity["hashes"]["examples_sha256"]:
    raise SystemExit("PointMaze warm-start examples hash changed")
audit=json.load(open(sys.argv[3],encoding="utf-8"))
if audit.get("status") != "pass":
    raise SystemExit("PointMaze admission audit is not passing")
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
  "$PYTHON_BIN" "$EVALUATOR" --help >/dev/null
  echo "[point-warmstart-v1] configuration passed; no model updated or sampled"
  exit 0
fi

for fresh in "$IDENTITY" "$OUTPUT_MODEL" "$SFT_RECEIPT" "$GATE_RECEIPT"; do
  if [[ -e "$fresh" ]]; then
    echo "Fresh PointMaze warm-start artifact required: $fresh" >&2
    exit 1
  fi
done

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_PARENT="$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v1_${SOURCE_HASH}"
SOURCE_ROOT="$SOURCE_PARENT/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$SOURCE_PARENT"
  staging="$(mktemp -d "$SOURCE_PARENT/.source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"
  rmdir "$staging"
fi
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "PointMaze warm-start source snapshot mismatch" >&2
  exit 1
fi

EXECUTION_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.point-warmstart-v1-ops.XXXXXX")"
cp "$TRAINER" "$EXECUTION_INPUT/train_point_maze_interactive_warmstart_v1.py"
cp "$EVALUATOR" "$EXECUTION_INPUT/evaluate_point_maze_interactive_viability.py"
cp "$SLURM_SCRIPT" "$EXECUTION_INPUT/train_point_maze_interactive_warmstart_v1.slurm"
EXECUTION_HASH="$(hash_tree "$EXECUTION_INPUT")"
EXECUTION_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v1_ops_${EXECUTION_HASH}"
if [[ ! -f "$EXECUTION_ROOT/train_point_maze_interactive_warmstart_v1.py" ]]; then
  mv "$EXECUTION_INPUT" "$EXECUTION_ROOT"
else
  find "$EXECUTION_INPUT" -type f -delete
  rmdir "$EXECUTION_INPUT"
fi
if [[ "$(hash_tree "$EXECUTION_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "PointMaze warm-start execution snapshot mismatch" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/var/artifacts/logs" "$ROOT_DIR/var/models"
job_id="$(
  sbatch --parsable --hold \
    --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_SOURCE_ROOT=$SOURCE_ROOT,OAT_ZERO_EXECUTION_ROOT=$EXECUTION_ROOT,OAT_ZERO_SOURCE_HASH=$SOURCE_HASH,OAT_ZERO_EXECUTION_HASH=$EXECUTION_HASH" \
    "$EXECUTION_ROOT/train_point_maze_interactive_warmstart_v1.slurm"
)"
job_id="${job_id%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Invalid PointMaze warm-start job ID: $job_id" >&2
  exit 1
fi

cleanup() {
  local status="$?"
  trap - EXIT
  scancel "$job_id" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

"$PYTHON_BIN" - \
  "$IDENTITY" "$job_id" "$SOURCE_HASH" "$EXECUTION_HASH" "$PROTOCOL" \
  "$TRAINER" "$EVALUATOR" "$SLURM_SCRIPT" "$WARMSTART_DATA/identity.json" \
  "$WARMSTART_DATA/examples.jsonl" "$MODEL_ROOT/config.json" "$AUDIT" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1])
payload={
    "schema_version":"point-maze-interactive-warmstart-identity-v1",
    "job_id":int(sys.argv[2]),
    "source_hash":sys.argv[3],
    "execution_hash":sys.argv[4],
    "protocol_sha256":digest(sys.argv[5]),
    "trainer_sha256":digest(sys.argv[6]),
    "evaluator_sha256":digest(sys.argv[7]),
    "slurm_sha256":digest(sys.argv[8]),
    "warmstart_data_identity_sha256":digest(sys.argv[9]),
    "warmstart_examples_sha256":digest(sys.argv[10]),
    "base_model_config_sha256":digest(sys.argv[11]),
    "admission_audit_sha256":digest(sys.argv[12]),
    "base_model":"Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
    "sft_seed":75201,
    "sft_optimizer_steps":69,
    "development_sampling_seed":75103,
    "development_samples":256,
    "evaluation_split_loaded":False,
    "shared_checkpoint_for_both_online_arms":True,
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
echo "[point-warmstart-v1] released job $job_id"
echo "[point-warmstart-v1] identity=$IDENTITY"
