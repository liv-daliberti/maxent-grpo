#!/usr/bin/env bash
# Submit the job that installs the final Ant Stage-B dependency after smoke launch.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
DEPENDENCY_LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_ant_maze_stage_b_dependency.sh"
BATCH="$ROOT_DIR/ops/slurm/prepare_ant_maze_stage_b_dependency_after_smoke_launch.slurm"
SMOKE_DEPENDENCY_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_paired_smoke_v13_dependency_identity.json"
OUTPUT="$ROOT_DIR/var/artifacts/ant_maze_stage_b_dependency_preparer_identity.json"
phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac
for required in "$PYTHON_BIN" "$DEPENDENCY_LAUNCHER" "$BATCH" "$SMOKE_DEPENDENCY_IDENTITY"; do
  [[ -e "$required" ]] || { echo "Missing Ant Stage-B preparer prerequisite: $required" >&2; exit 1; }
done
bash -n "$BATCH" "$DEPENDENCY_LAUNCHER"
bash "$DEPENDENCY_LAUNCHER" config
smoke_launch_job_id="$(jq -er '.job_id' "$SMOKE_DEPENDENCY_IDENTITY")"
[[ "$smoke_launch_job_id" == 30202318 ]] || {
  echo "AntMaze smoke-launch dependency identity drift" >&2
  exit 1
}
dependency_launcher_sha256="$(sha256sum "$DEPENDENCY_LAUNCHER" | cut -d' ' -f1)"
if [[ "$phase" == config ]]; then
  echo "[ant-stage-b-preparer] configuration passed; no job submitted"
  exit 0
fi
[[ ! -e "$OUTPUT" ]] || { echo "Fresh Ant Stage-B preparer identity required" >&2; exit 1; }
job_id="$(sbatch --parsable --dependency="afterok:$smoke_launch_job_id" \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXPECTED_DEPENDENCY_LAUNCHER_SHA256=$dependency_launcher_sha256,OAT_ZERO_SMOKE_LAUNCH_JOB_ID=$smoke_launch_job_id" \
  "$BATCH")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Ant Stage-B preparer job ID" >&2; exit 1; }
"$PYTHON_BIN" - \
  "$OUTPUT" "$job_id" "$smoke_launch_job_id" "$dependency_launcher_sha256" "$BATCH" "$SMOKE_DEPENDENCY_IDENTITY" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]); payload={
 "schema":"ant-maze-stage-b-dependency-preparer-identity-v1","job_id":int(sys.argv[2]),
 "dependency":f"afterok:{sys.argv[3]}","smoke_launch_job_id":int(sys.argv[3]),
 "expected_dependency_launcher_sha256":sys.argv[4],"preparer_batch_sha256":sha(sys.argv[5]),
 "smoke_dependency_identity_sha256":sha(sys.argv[6]),"fail_closed":True,
 "paper_jobs_before_smoke_pass":False
}
path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(tmp,path)
PY
echo "[ant-stage-b-preparer] submitted job $job_id afterok:$smoke_launch_job_id"
