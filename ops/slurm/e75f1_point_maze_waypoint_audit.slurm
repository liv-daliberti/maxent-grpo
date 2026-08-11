#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_EXECUTION_ROOT="$EXECUTION_ROOT"

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_e75f1_point_maze_waypoint_05b_12pass.py" \
  --identity "$ROOT_DIR/var/artifacts/e75f1_point_maze_waypoint_05b_identity.json" \
  --manifest "$ROOT_DIR/var/artifacts/e75f1_point_maze_waypoint_05b_jobs.tsv" \
  --data-identity "$ROOT_DIR/var/data/point_maze_waypoint_e75f1/identity.json" \
  --qualification "$ROOT_DIR/var/artifacts/e75f1_point_maze_waypoint_dev_qualification.json" \
  --protocol "$EXECUTION_ROOT/e75f1_point_maze_waypoint_05b_12pass_20260804.md" \
  --output "$ROOT_DIR/var/artifacts/e75f1_point_maze_waypoint_05b_12pass_audit.json"
