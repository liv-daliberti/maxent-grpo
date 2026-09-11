#!/bin/bash
#SBATCH --job-name=audit-point-paired-smoke-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
COHORT="${OAT_ZERO_POINT_SMOKE_COHORT:-v1}"
[[ "$COHORT" == "v1" || "$COHORT" == "v2" || "$COHORT" == "v3" || "$COHORT" == "v4" ]] || { echo "invalid PointMaze smoke cohort" >&2; exit 1; }
STEM="point_maze_interactive_paired_smoke_${COHORT}"
DATA_ROOT="$ROOT_DIR/var/data/point_maze_modebench_v1"
if [[ "$COHORT" == "v4" ]]; then
  STEM="point_maze_geometry_shift_paired_smoke_v1"
  DATA_ROOT="$ROOT_DIR/var/data/point_maze_geometry_shift_v1"
fi
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR" OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/audit_point_maze_interactive_paired_smoke_v1.py" \
  --repo-root "$ROOT_DIR" \
  --identity "$ROOT_DIR/var/artifacts/${STEM}_identity.json" \
  --manifest "$ROOT_DIR/var/artifacts/${STEM}_jobs.tsv" \
  --model "$ROOT_DIR/var/models/point_maze_interactive_warmstart_v3" \
  --data-split-root "$DATA_ROOT/train" \
  --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  --output "$ROOT_DIR/var/artifacts/${STEM}_audit.json"
