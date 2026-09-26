#!/usr/bin/env bash
# Submit the fail-closed post-smoke PointMaze Stage-B launcher dependency.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_point_maze_stage_b_05b_12pass.py"
BATCH="$ROOT_DIR/ops/slurm/launch_point_maze_stage_b_after_smoke.slurm"
SMOKE_IDENTITY="$ROOT_DIR/var/artifacts/point_maze_interactive_paired_smoke_v3_identity.json"
OUTPUT="$ROOT_DIR/var/artifacts/point_maze_stage_b_v3_dependency_identity.json"
phase="${1:-}"
case "$phase" in config|run) ;; *) echo "Usage: $0 {config|run}" >&2; exit 1 ;; esac
for required in "$PYTHON_BIN" "$LAUNCHER" "$BATCH" "$SMOKE_IDENTITY"; do
  [[ -e "$required" ]] || { echo "Missing PointMaze Stage-B dependency prerequisite: $required" >&2; exit 1; }
done
bash -n "$BATCH"
audit_job_id="$(jq -er '.audit_job_id' "$SMOKE_IDENTITY")"
"$PYTHON_BIN" - "$SMOKE_IDENTITY" <<'PY'
import json,pathlib,sys
payload=json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("schema")!="point-maze-interactive-paired-smoke-identity-v3": raise SystemExit("PointMaze v3 identity schema drift")
if payload.get("seed")!=75303 or payload.get("row_indices")!=[1,3,5,7]: raise SystemExit("PointMaze v3 slate drift")
if payload.get("policy_microbatch_size")!=16: raise SystemExit("PointMaze v3 microbatch drift")
if set(payload.get("jobs",{}))!={"grpo","verified_first_global_replay_canonical"}: raise SystemExit("PointMaze v3 jobs drift")
PY
launcher_sha256="$(sha256sum "$LAUNCHER" | cut -d' ' -f1)"
if [[ "$phase" == config ]]; then
  echo "[point-stage-b-dependency] configuration passed; no job submitted"
  exit 0
fi
[[ ! -e "$OUTPUT" ]] || { echo "Fresh PointMaze Stage-B dependency identity required" >&2; exit 1; }
job_id="$(sbatch --parsable --dependency="afterok:$audit_job_id" \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXPECTED_LAUNCHER_SHA256=$launcher_sha256,OAT_ZERO_SMOKE_AUDIT_JOB_ID=$audit_job_id" \
  "$BATCH")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid PointMaze Stage-B dependency job ID" >&2; exit 1; }
"$PYTHON_BIN" - "$OUTPUT" "$job_id" "$audit_job_id" "$launcher_sha256" "$BATCH" "$SMOKE_IDENTITY" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema":"point-maze-stage-b-dependency-identity-v3","job_id":int(sys.argv[2]),
 "dependency":f"afterok:{sys.argv[3]}","smoke_audit_job_id":int(sys.argv[3]),
 "expected_stage_b_launcher_sha256":sys.argv[4],"dependency_batch_sha256":sha(sys.argv[5]),
 "smoke_identity_sha256":sha(sys.argv[6]),"fail_closed":True,"paper_jobs_before_smoke_pass":False
}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY
echo "[point-stage-b-dependency] submitted job $job_id afterok:$audit_job_id"
