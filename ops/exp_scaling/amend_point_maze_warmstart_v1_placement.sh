#!/usr/bin/env bash
# Broaden only PointMaze warm-start v1 placement to the all A5000 pool.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
IDENTITY="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_v1_identity.json"
PROTOCOL="$ROOT_DIR/paper/preregistration/point_maze_warmstart_v1_a5000_placement_amendment_20260729.md"
OUTPUT="$ROOT_DIR/var/artifacts/point_maze_warmstart_v1_placement_amendment.json"
MODEL_OUT="$ROOT_DIR/var/models/point_maze_interactive_warmstart_v1"
SFT_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_sft_v1.json"
GATE_RECEIPT="$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_warmstart_v1.json"
JOB_ID=30185642

for required in "$PYTHON_BIN" "$IDENTITY" "$PROTOCOL"; do
  [[ -f "$required" ]] || { echo "Missing placement-amendment input: $required" >&2; exit 1; }
done
for fresh in "$OUTPUT" "$MODEL_OUT" "$SFT_RECEIPT" "$GATE_RECEIPT"; do
  [[ ! -e "$fresh" ]] || { echo "PointMaze placement amendment requires absent target: $fresh" >&2; exit 1; }
done

before="$(scontrol show job "$JOB_ID" -o)"
for required in \
  "JobId=$JOB_ID" "JobState=PENDING" "RunTime=00:00:00" \
  "Partition=mltheory" "TresPerNode=gres/gpu:a5000:1" \
  "OAT_ZERO_SOURCE_ROOT=$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v1_6e8f932018888b340a3988d3f19b6a3816ec40f20fdc3ff4ca72a341e2eb9d89/src" \
  "OAT_ZERO_EXECUTION_ROOT=$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v1_ops_909570a0e373f8a0108cd32ef7fa4c32a03b283f942222f516f38ef28da7a69b"; do
  [[ "$before" == *"$required"* ]] || { echo "PointMaze pre-amendment job missing: $required" >&2; exit 1; }
done

"$PYTHON_BIN" - "$IDENTITY" <<'PY'
import json, sys
identity=json.load(open(sys.argv[1],encoding="utf-8"))
expected={
 "job_id":30185642,
 "source_hash":"6e8f932018888b340a3988d3f19b6a3816ec40f20fdc3ff4ca72a341e2eb9d89",
 "execution_hash":"909570a0e373f8a0108cd32ef7fa4c32a03b283f942222f516f38ef28da7a69b",
 "sft_seed":75201,
 "sft_optimizer_steps":69,
 "development_sampling_seed":75103,
}
for key,value in expected.items():
    if identity.get(key) != value:
        raise SystemExit(f"PointMaze identity mismatch for {key}")
if identity.get("shared_checkpoint_for_both_online_arms") is not True:
    raise SystemExit("PointMaze identity does not preserve the shared checkpoint")
PY

scontrol update "JobId=$JOB_ID" Partition=all
after="$(scontrol show job "$JOB_ID" -o)"
for required in \
  "JobId=$JOB_ID" "Partition=all" "TresPerNode=gres/gpu:a5000:1" \
  "OAT_ZERO_SOURCE_ROOT=$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v1_6e8f932018888b340a3988d3f19b6a3816ec40f20fdc3ff4ca72a341e2eb9d89/src" \
  "OAT_ZERO_EXECUTION_ROOT=$ROOT_DIR/var/artifacts/source_snapshots/point_warmstart_v1_ops_909570a0e373f8a0108cd32ef7fa4c32a03b283f942222f516f38ef28da7a69b"; do
  [[ "$after" == *"$required"* ]] || { echo "PointMaze post-amendment job missing: $required" >&2; exit 1; }
done
if [[ "$after" != *"JobState=PENDING"* && "$after" != *"JobState=RUNNING"* ]]; then
  echo "PointMaze job entered an unexpected state after placement amendment" >&2
  exit 1
fi

"$PYTHON_BIN" - "$OUTPUT" "$PROTOCOL" "$0" "$IDENTITY" "$before" "$after" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema_version":"point-maze-warmstart-v1-placement-amendment-v1",
 "job_id":30185642,"scientific_change":False,"placement_only":True,
 "accelerator_type_before":"a5000","accelerator_type_after":"a5000",
 "partition_before":"mltheory","partition_after":"all",
 "protocol_sha256":digest(sys.argv[2]),"amendment_sha256":digest(sys.argv[3]),
 "identity_sha256":digest(sys.argv[4]),"slurm_before":sys.argv[5],
 "slurm_after":sys.argv[6],"model_or_receipt_existed_before":False}
path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w",encoding="utf-8") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY

echo "[point-warmstart-placement] job $JOB_ID broadened to all; A5000 unchanged"
