#!/bin/bash
#SBATCH --job-name=ant-sequential-v16r1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
IDENTITY="${OAT_ZERO_PROTOCOL_IDENTITY:?protocol identity is required}"
[[ -f "$IDENTITY" ]] || { echo "Missing Ant sequential v16r1 identity" >&2; exit 1; }
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  "$EXECUTION_ROOT/train_ant_sequential_waypoint_controller_v16_r1.py" \
  --output "$ROOT_DIR/var/maze_runtime/controllers/ant_sequential_waypoint_v16" \
  --timesteps 2000000 \
  --workers 8 \
  --seed 73016 \
  --episode-steps 1600 \
  --evaluation-episodes-per-heading 4 \
  --learning-rate 0.000001 \
  --device cpu
