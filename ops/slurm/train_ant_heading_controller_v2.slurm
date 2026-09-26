#!/usr/bin/env bash
# Prospective v2 auxiliary controller for the 0.5B AntMaze language policy.
#SBATCH --job-name=ant-heading-controller-v2
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/ant-heading-controller-v2-%j.out
#SBATCH --error=var/artifacts/logs/ant-heading-controller-v2-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/n/fs/similarity/maxent-grpo}"
cd "$ROOT_DIR"
mkdir -p var/artifacts/logs var/maze_runtime/controllers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=egl
exec "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  "$ROOT_DIR/ops/train_ant_heading_controller_v2.py" \
  --output "$ROOT_DIR/var/maze_runtime/controllers/ant_heading_v2" \
  --timesteps 3000000 \
  --workers 8 \
  --seed 73002 \
  --episode-steps 300 \
  --device cpu
