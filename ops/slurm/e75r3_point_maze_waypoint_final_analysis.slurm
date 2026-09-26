#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=00:20:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
PLAN="${OAT_ZERO_E75R3_FINAL_PLAN:?frozen plan is required}"
PLAN_SHA256="${OAT_ZERO_E75R3_FINAL_PLAN_SHA256:?frozen plan hash is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/analyze_e75r3_point_maze_waypoint_final_eval.py" analyze \
  --plan "$PLAN" --plan-sha256 "$PLAN_SHA256" \
  --output-json "$ROOT_DIR/var/artifacts/e75r3_point_maze_waypoint_final_analysis.json" \
  --output-markdown "$ROOT_DIR/var/artifacts/e75r3_point_maze_waypoint_final_analysis.md"
