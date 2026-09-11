#!/bin/bash
#SBATCH --job-name=launch-ant-v15-v17-r2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=00:20:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
MANIFEST="${OAT_ZERO_FROZEN_MANIFEST:?frozen manifest is required}"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/exp_scaling/schedule_ant_maze_v15_controller_v17_r2.py" \
  verify-run \
  --manifest "$MANIFEST"
