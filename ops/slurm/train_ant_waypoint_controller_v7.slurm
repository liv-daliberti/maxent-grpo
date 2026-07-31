#!/usr/bin/env bash
#SBATCH --job-name=ant-waypoint-controller-v7
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/ant-waypoint-controller-v7-%j.out
#SBATCH --error=var/artifacts/logs/ant-waypoint-controller-v7-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?OAT_ZERO_EXECUTION_ROOT is required}"
cd "$ROOT_DIR"
mkdir -p var/artifacts/logs var/maze_runtime/controllers
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=egl

exec "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  "$EXECUTION_ROOT/train_ant_waypoint_controller_v7.py" \
  --output "$ROOT_DIR/var/maze_runtime/controllers/ant_waypoint_v7" \
  --timesteps 5000000 \
  --workers 8 \
  --seed 73007 \
  --episode-steps 400 \
  --evaluation-episodes-per-heading 12 \
  --learning-rate 0.00001 \
  --device cpu
