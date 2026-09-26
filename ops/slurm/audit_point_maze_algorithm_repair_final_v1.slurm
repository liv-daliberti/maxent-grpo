#!/bin/bash
#SBATCH --job-name=point-repair-final-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cs
#SBATCH --time=12:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?root is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_point_maze_algorithm_repair_final_v1.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_final_v1_identity.json" \
  --submission "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_final_v1_submission.json" \
  --manifest "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_final_v1_jobs.tsv" \
  --output "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_final_v1_audit.json"
