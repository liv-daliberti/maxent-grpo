#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
ARM="${OAT_ZERO_ARM:?arm is required}"
SEED="${OAT_ZERO_SEED:?seed is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_EXECUTION_ROOT="$EXECUTION_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

STEM="$ROOT_DIR/var/artifacts/e78pm_point_maze_${ARM}_s${SEED}"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/train_point_maze_verified_replay_only.py" \
  --arm "$ARM" --seed "$SEED" \
  --model "$ROOT_DIR/var/models/point_maze_waypoint_warmstart_e75r3" \
  --data-root "$ROOT_DIR/var/data/point_maze_waypoint_e78pm" \
  --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  --passes 8 --evaluation-interval 192 --evaluation-prompts 128 \
  --output-model "$ROOT_DIR/var/models/e78pm_point_maze_${ARM}_s${SEED}" \
  --checkpoint-dir "$ROOT_DIR/var/checkpoints/e78pm_point_maze_${ARM}_s${SEED}" \
  --output-receipt "${STEM}.json" \
  --output-metrics "${STEM}.metrics.jsonl" \
  --output-replay "${STEM}.replay.jsonl" \
  --learning-rate 0.0000002 --replay-weight 0.10 \
  --microbatch-size 16 --max-length 1536 --auto-resume
