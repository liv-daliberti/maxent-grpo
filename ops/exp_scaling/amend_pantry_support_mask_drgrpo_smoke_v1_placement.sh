#!/usr/bin/env bash
# Apply the frozen placement-only amendment to unstarted Pantry job 30187473.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_ID=30187473
IDENTITY="$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_identity.json"
AMENDMENT="$ROOT_DIR/paper/preregistration/pantry_support_mask_drgrpo_smoke_v1_a5000_placement_amendment_20260730.md"
RECEIPT="$ROOT_DIR/var/artifacts/pantry_support_mask_drgrpo_smoke_v1_placement_amendment.json"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
[[ -f "$IDENTITY" && -f "$AMENDMENT" && ! -e "$RECEIPT" ]] || { echo "Pantry placement amendment prerequisites failed" >&2; exit 1; }
before="$(scontrol show job "$JOB_ID" -o)"
for required in 'JobState=PENDING' 'RunTime=00:00:00' 'Requeue=0' 'ReqNodeList=node302' 'gres/gpu:a100:1' 'NumCPUs=8' 'MinMemoryNode=64G' 'TimeLimit=04:00:00' 'OAT_ZERO_SEED=76201' 'OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES=32' 'OAT_ZERO_VARIANT=grpo'; do [[ "$before" == *"$required"* ]] || { echo "Pantry pre-amendment record missing $required" >&2; exit 1; }; done
scontrol hold "$JOB_ID"
cleanup(){ local status="$?"; trap - EXIT; scontrol release "$JOB_ID" 2>/dev/null || true; exit "$status"; }
trap cleanup EXIT
scontrol update "JobId=$JOB_ID" Partition=all Gres=gpu:a5000:1 NodeList=
after="$(scontrol show job "$JOB_ID" -o)"
for required in 'JobState=PENDING' 'Reason=JobHeldUser' 'Partition=all' 'ReqNodeList=(null)' 'gres/gpu:a5000:1' 'NumCPUs=8' 'MinMemoryNode=64G' 'TimeLimit=04:00:00' 'OAT_ZERO_SEED=76201' 'OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES=32' 'OAT_ZERO_VARIANT=grpo'; do [[ "$after" == *"$required"* ]] || { echo "Pantry post-amendment record missing $required" >&2; exit 1; }; done
"$PYTHON_BIN" - "$RECEIPT" "$IDENTITY" "$AMENDMENT" "$before" "$after" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def d(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
p=pathlib.Path(sys.argv[1]); x={"schema_version":"pantry-support-mask-drgrpo-smoke-placement-v1","job_id":30187473,"identity_sha256":d(sys.argv[2]),"amendment_sha256":d(sys.argv[3]),"before":sys.argv[4],"after":sys.argv[5],"placement_only":True,"scientific_change":False,"runtime_before":"00:00:00","gpu_before":"a100:1","gpu_after":"a5000:1","node_before":"node302","node_after":None}
p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(x,h,indent=2,sort_keys=True); h.write("\n")
os.replace(t,p)
PY
scontrol release "$JOB_ID"
trap - EXIT
echo "[pantry-placement] amended and released job $JOB_ID"
