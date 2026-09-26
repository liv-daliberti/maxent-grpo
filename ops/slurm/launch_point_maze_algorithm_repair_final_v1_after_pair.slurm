#!/bin/bash
#SBATCH --job-name=launch-point-repair-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?root is required}"
EXPECTED="${EXPECTED_LAUNCHER_SHA256:?launcher hash is required}"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_point_maze_algorithm_repair_final_v1.py"
ACTUAL="$(sha256sum "$LAUNCHER" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || {
  echo "PointMaze repair-final launcher drift: $ACTUAL != $EXPECTED" >&2
  exit 1
}
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONDONTWRITEBYTECODE=1
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$LAUNCHER" run
