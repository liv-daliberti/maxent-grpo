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
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/make_point_maze_waypoint_pilot_v1_data.py" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --output-root "$ROOT_DIR/var/data/point_maze_waypoint_pilot_e75r3" \
  --seed 88103 \
  --exclude-identity "$ROOT_DIR/var/data/point_maze_waypoint_pilot_v1/identity.json" \
  --exclude-identity "$ROOT_DIR/var/data/point_maze_waypoint_pilot_e75r1/identity.json" \
  --exclude-identity "$ROOT_DIR/var/data/point_maze_waypoint_pilot_e75r2/identity.json"

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/materialize_point_maze_waypoint_warmstart_v1.py" \
  --data-root "$ROOT_DIR/var/data/point_maze_waypoint_pilot_e75r3" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --output-root "$ROOT_DIR/var/data/point_maze_waypoint_warmstart_e75r3"
