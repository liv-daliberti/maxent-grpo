#!/usr/bin/env bash
# Frozen auxiliary controller for the 0.5B AntMaze language-policy interface.
#SBATCH --job-name=ant-heading-controller-v1
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/ant-heading-controller-v1-%j.out
#SBATCH --error=var/artifacts/logs/ant-heading-controller-v1-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/n/fs/similarity/maxent-grpo}"
cd "$ROOT_DIR"
mkdir -p var/artifacts/logs var/maze_runtime/controllers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=egl
exec "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  "$ROOT_DIR/ops/train_ant_heading_controller.py" \
  --output "$ROOT_DIR/var/maze_runtime/controllers/ant_heading_v1" \
  --timesteps 2000000 \
  --workers 8 \
  --seed 73001 \
  --episode-steps 300 \
  --device cpu
