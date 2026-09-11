#!/bin/bash
#SBATCH --job-name=point-repair-v2r2-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?root is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_point_maze_algorithm_repair_pair_v2.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_pair_v2r2_identity.json" \
  --submission "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_pair_v2r2_submission.json" \
  --manifest "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_pair_v2r2_jobs.tsv" \
  --output "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_pair_v2r2_audit.json"
