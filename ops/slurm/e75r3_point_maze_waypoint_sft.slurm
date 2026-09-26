#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=12:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
BASE_MODEL="${OAT_ZERO_BASE_MODEL:?base model is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_EXECUTION_ROOT="$EXECUTION_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/train_point_maze_waypoint_warmstart_v1.py" \
  --model "$BASE_MODEL" \
  --data-root "$ROOT_DIR/var/data/point_maze_waypoint_warmstart_e75r3" \
  --protocol "$EXECUTION_ROOT/e75r3_point_maze_waypoint_weak_sft_05b_20260804.md" \
  --output-model "$ROOT_DIR/var/models/point_maze_waypoint_warmstart_e75r3" \
  --output-receipt "$ROOT_DIR/var/artifacts/e75r3_point_maze_waypoint_warmstart.json" \
  --seed 88404 --epochs 1 --max-updates 72 --batch-size 4 --gradient-accumulation 8 \
  --learning-rate 0.00002 --max-length 1536
