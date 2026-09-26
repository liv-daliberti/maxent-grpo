#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
PLAN="${OAT_ZERO_E75R3_FINAL_PLAN:?frozen plan is required}"
PLAN_SHA256="${OAT_ZERO_E75R3_FINAL_PLAN_SHA256:?frozen plan hash is required}"
ARM="${OAT_ZERO_ARM:?arm is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_EXECUTION_ROOT="$EXECUTION_ROOT"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

PYTHON="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
ANALYZER="$EXECUTION_ROOT/analyze_e75r3_point_maze_waypoint_final_eval.py"
"$PYTHON" "$ANALYZER" verify-arm \
  --plan "$PLAN" --plan-sha256 "$PLAN_SHA256" --arm "$ARM"

SHORT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_short"][sys.argv[2]])' "$PLAN" "$ARM")"
MODEL="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"][sys.argv[2]]["path"])' "$PLAN" "$ARM")"
STEM="$ROOT_DIR/var/artifacts/e75r3_point_maze_waypoint_final_eval_${SHORT}"
exec "$PYTHON" "$EXECUTION_ROOT/train_point_maze_waypoint_pilot_v1.py" \
  --arm "$ARM" --seed 88504 \
  --model "$MODEL" \
  --data-root "$ROOT_DIR/var/data/point_maze_waypoint_pilot_e75r3" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --evaluation-only --evaluation-prompts 64 --evaluation-split eval \
  --output-receipt "${STEM}.json" \
  --output-metrics "${STEM}.metrics.jsonl" \
  --output-replay "${STEM}.replay.jsonl" \
  --microbatch-size 16 --max-length 1536
