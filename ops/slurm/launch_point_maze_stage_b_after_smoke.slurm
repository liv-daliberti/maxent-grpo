#!/bin/bash
#SBATCH --job-name=launch-point-stage-b
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
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_point_maze_stage_b_05b_12pass.py"
EXPECTED_SHA256="${OAT_ZERO_EXPECTED_LAUNCHER_SHA256:?launcher hash is required}"
AUDIT_JOB_ID="${OAT_ZERO_SMOKE_AUDIT_JOB_ID:?smoke audit job ID is required}"
AUDIT="$ROOT_DIR/var/artifacts/point_maze_interactive_paired_smoke_v3_audit.json"
[[ "$(sha256sum "$LAUNCHER" | cut -d' ' -f1)" == "$EXPECTED_SHA256" ]] || {
  echo "PointMaze Stage-B launcher changed after dependency submission" >&2
  exit 1
}
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - "$AUDIT" <<'PY'
import json,pathlib,sys
payload=json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("schema") != "point-maze-interactive-paired-smoke-audit-v3": raise SystemExit("PointMaze smoke audit schema is not v3")
if payload.get("status") != "pass": raise SystemExit("PointMaze smoke audit is not pass")
if payload.get("decision") != "eligible_for_ten_point_maze_stage_b_jobs": raise SystemExit("PointMaze smoke did not authorize ten cells")
if payload.get("errors") not in ([],None): raise SystemExit("PointMaze smoke audit has errors")
PY
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" "$LAUNCHER" run
