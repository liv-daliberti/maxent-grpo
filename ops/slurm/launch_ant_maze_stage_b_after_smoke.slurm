#!/bin/bash
#SBATCH --job-name=launch-ant-stage-b
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
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_ant_maze_stage_b_05b_12pass.py"
EXPECTED_SHA256="${OAT_ZERO_EXPECTED_LAUNCHER_SHA256:?launcher hash is required}"
AUDIT_JOB_ID="${OAT_ZERO_SMOKE_AUDIT_JOB_ID:?smoke audit job ID is required}"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_v13r2_identity.json"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_v13r2_audit.json"

[[ "$(sha256sum "$LAUNCHER" | cut -d' ' -f1)" == "$EXPECTED_SHA256" ]] || {
  echo "AntMaze Stage-B launcher changed after dependency submission" >&2
  exit 1
}
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - \
  "$IDENTITY" "$AUDIT" "$AUDIT_JOB_ID" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
audit = json.loads(pathlib.Path(sys.argv[2]).read_text())
audit_job_id = int(sys.argv[3])
if identity.get("audit_job_id") != audit_job_id:
    raise SystemExit("AntMaze smoke audit job identity mismatch")
if identity.get("schema") != "ant-maze-interactive-paired-smoke-identity-v13":
    raise SystemExit("AntMaze smoke identity schema mismatch")
if identity.get("artifact_cohort") != "v13r2" or identity.get("policy_microbatch_size") != 16:
    raise SystemExit("AntMaze smoke repair cohort mismatch")
if audit.get("status") != "pass":
    raise SystemExit("AntMaze smoke audit is not pass")
if audit.get("decision") != "eligible_for_ten_ant_maze_stage_b_jobs":
    raise SystemExit("AntMaze smoke did not authorize ten cells")
if audit.get("errors") not in ([], None):
    raise SystemExit("AntMaze smoke audit has errors")
PY
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" "$LAUNCHER" run
