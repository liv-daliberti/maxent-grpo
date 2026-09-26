#!/usr/bin/env bash
# Submit the fail-closed post-smoke AntMaze Stage-B launcher dependency.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_ant_maze_stage_b_05b_12pass.py"
BATCH="$ROOT_DIR/ops/slurm/launch_ant_maze_stage_b_after_smoke.slurm"
SMOKE_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_v13r2_identity.json"
OUTPUT="$ROOT_DIR/var/artifacts/ant_maze_stage_b_r2_dependency_identity.json"
phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac
for required in "$PYTHON_BIN" "$LAUNCHER" "$BATCH"; do
  [[ -e "$required" ]] || {
    echo "Missing AntMaze Stage-B dependency prerequisite: $required" >&2
    exit 1
  }
done
bash -n "$BATCH"
"$PYTHON_BIN" "$LAUNCHER" config
if [[ "$phase" == config ]]; then
  echo "[ant-stage-b-dependency] configuration passed; no job submitted"
  exit 0
fi
[[ -e "$SMOKE_IDENTITY" ]] || {
  echo "AntMaze paired-smoke identity is absent" >&2
  exit 1
}
audit_job_id="$(jq -er '.audit_job_id' "$SMOKE_IDENTITY")"
[[ "$audit_job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid AntMaze smoke audit job ID" >&2
  exit 1
}
"$PYTHON_BIN" - "$SMOKE_IDENTITY" <<'PY'
import json,pathlib,sys
payload=json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("schema")!="ant-maze-interactive-paired-smoke-identity-v13": raise SystemExit("Ant smoke identity schema drift")
if payload.get("artifact_cohort")!="v13r2" or payload.get("policy_microbatch_size")!=16: raise SystemExit("Ant smoke repair cohort drift")
if payload.get("arms")!=["grpo","verified_first_global_replay_canonical"]: raise SystemExit("Ant smoke arms drift")
if payload.get("seed")!=76313 or payload.get("development_only") is not True or payload.get("final_seed") is not False: raise SystemExit("Ant smoke cohort drift")
if set(payload.get("jobs",{}))!={"grpo","verified_first_global_replay_canonical"}: raise SystemExit("Ant smoke jobs drift")
PY
launcher_sha256="$(sha256sum "$LAUNCHER" | cut -d' ' -f1)"
[[ ! -e "$OUTPUT" ]] || {
  echo "Fresh AntMaze Stage-B dependency identity required" >&2
  exit 1
}
job_id="$(sbatch --parsable --dependency="afterok:$audit_job_id" \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXPECTED_LAUNCHER_SHA256=$launcher_sha256,OAT_ZERO_SMOKE_AUDIT_JOB_ID=$audit_job_id" \
  "$BATCH")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid AntMaze Stage-B dependency job ID" >&2
  exit 1
}
"$PYTHON_BIN" - \
  "$OUTPUT" "$job_id" "$audit_job_id" "$launcher_sha256" "$BATCH" "$SMOKE_IDENTITY" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema":"ant-maze-stage-b-dependency-identity-r2","job_id":int(sys.argv[2]),
 "dependency":f"afterok:{sys.argv[3]}","smoke_audit_job_id":int(sys.argv[3]),
 "expected_stage_b_launcher_sha256":sys.argv[4],"dependency_batch_sha256":sha(sys.argv[5]),
 "smoke_identity_sha256":sha(sys.argv[6]),"fail_closed":True,
 "paper_jobs_before_smoke_pass":False
}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY
echo "[ant-stage-b-dependency] submitted job $job_id afterok:$audit_job_id"
