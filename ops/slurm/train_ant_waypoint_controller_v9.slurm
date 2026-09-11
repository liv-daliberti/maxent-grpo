#!/usr/bin/env bash
#SBATCH --job-name=ant-waypoint-controller-v9
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=var/artifacts/logs/ant-waypoint-controller-v9-%j.out
#SBATCH --error=var/artifacts/logs/ant-waypoint-controller-v9-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?OAT_ZERO_EXECUTION_ROOT is required}"
IDENTITY="${OAT_ZERO_PROTOCOL_IDENTITY:?OAT_ZERO_PROTOCOL_IDENTITY is required}"
cd "$ROOT_DIR"
mkdir -p var/artifacts/logs var/maze_runtime/controllers
[[ -f "$IDENTITY" ]] || { echo "Missing Ant v9 identity" >&2; exit 1; }
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=egl

exec "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  "$EXECUTION_ROOT/train_ant_waypoint_controller_v9.py" \
  --output "$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v9" \
  --timesteps 5000000 \
  --workers 8 \
  --seed 73009 \
  --episode-steps 400 \
  --evaluation-episodes-per-heading 12 \
  --learning-rate 0.000005 \
  --device cpu
