#!/usr/bin/env bash
# Prospective new-seed evaluation of the frozen v5 heading macro wrapper.
#SBATCH --job-name=ant-heading-controller-v5
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/ant-heading-controller-v5-%j.out
#SBATCH --error=var/artifacts/logs/ant-heading-controller-v5-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/n/fs/similarity/maxent-grpo}"
cd "$ROOT_DIR"
mkdir -p var/artifacts/logs var/maze_runtime/controllers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=egl
exec "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  "$ROOT_DIR/ops/evaluate_ant_heading_controller_v5.py"
