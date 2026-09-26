#!/bin/bash
#SBATCH --job-name=launch-ant-paired-v13
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
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_ant_maze_interactive_paired_smoke_v13.py"
EXPECTED_SHA256="${OAT_ZERO_EXPECTED_LAUNCHER_SHA256:?launcher hash is required}"
V13_JOB_ID="${OAT_ZERO_V13_JOB_ID:?v13 job ID is required}"
IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_interactive_warmstart_v13_identity.json"
VIABILITY="$ROOT_DIR/var/artifacts/ant_maze_interactive_05b_viability_v13.json"

[[ "$(sha256sum "$LAUNCHER" | cut -d' ' -f1)" == "$EXPECTED_SHA256" ]] || {
  echo "AntMaze v13 paired-smoke launcher changed after dependency submission" >&2
  exit 1
}
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - \
  "$IDENTITY" "$VIABILITY" "$V13_JOB_ID" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
viability = json.loads(pathlib.Path(sys.argv[2]).read_text())
job_id = int(sys.argv[3])
summary = viability.get("summary", {})
if identity.get("job_id") != job_id:
    raise SystemExit("AntMaze v13 identity job mismatch")
if viability.get("job_id") != job_id:
    raise SystemExit("AntMaze v13 receipt job mismatch")
if viability.get("status") != "pass":
    raise SystemExit("AntMaze v13 viability is not pass")
if viability.get("decision") != "eligible_for_ant_v13_paired_online_smoke":
    raise SystemExit("AntMaze v13 did not authorize paired smoke")
if summary.get("prompt_count") != 4:
    raise SystemExit("AntMaze v13 prompt count changed")
if int(summary.get("prefix_success_prompts", 0)) < 2:
    raise SystemExit("AntMaze v13 prefix-success threshold failed")
if int(summary.get("multimode_prompts", 0)) < 1:
    raise SystemExit("AntMaze v13 multimode threshold failed")
PY
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" "$LAUNCHER" run
