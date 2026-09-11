#!/bin/bash
#SBATCH --job-name=prepare-ant-stage-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cs
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_ant_maze_stage_b_dependency.sh"
EXPECTED_SHA256="${OAT_ZERO_EXPECTED_DEPENDENCY_LAUNCHER_SHA256:?dependency launcher hash is required}"
SMOKE_LAUNCH_JOB_ID="${OAT_ZERO_SMOKE_LAUNCH_JOB_ID:?smoke-launch job ID is required}"
SMOKE_DEPENDENCY_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_paired_smoke_v13_dependency_identity.json"

[[ "$(sha256sum "$LAUNCHER" | cut -d' ' -f1)" == "$EXPECTED_SHA256" ]] || {
  echo "AntMaze Stage-B dependency launcher changed after preparer submission" >&2
  exit 1
}
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - \
  "$SMOKE_DEPENDENCY_IDENTITY" "$SMOKE_LAUNCH_JOB_ID" <<'PY'
import json,pathlib,sys
payload=json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("schema")!="ant-maze-paired-smoke-v13-dependency-identity-v1": raise SystemExit("Ant smoke dependency identity schema drift")
if payload.get("job_id")!=int(sys.argv[2]) or payload.get("v13_job_id")!=30202183: raise SystemExit("Ant smoke dependency job drift")
if payload.get("fail_closed") is not True or payload.get("paper_jobs_before_paired_smoke_pass") is not False: raise SystemExit("Ant smoke fail-closed contract drift")
PY
exec bash "$LAUNCHER" run
