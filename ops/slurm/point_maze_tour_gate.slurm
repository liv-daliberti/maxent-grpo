#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

# Stage 1 and stage 2 of the PointMaze Tour admission ladder.  Both run on
# development maps only; the evaluation split stays sealed until stage 4.

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
ARM="${OAT_ZERO_ARM:?arm is required}"
SEED="${OAT_ZERO_SEED:?seed is required}"
STAGE="${OAT_ZERO_TOUR_STAGE:?stage is required}"
PASSES="${OAT_ZERO_TOUR_PASSES:-0}"
MODEL="${OAT_ZERO_TOUR_MODEL:?model is required}"
EVAL_ONLY="${OAT_ZERO_TOUR_EVAL_ONLY:-0}"

source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$ROOT_DIR/src"
export OAT_ZERO_EXECUTION_ROOT="$ROOT_DIR/ops"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

STEM="$ROOT_DIR/var/artifacts/tour_${STAGE}_${ARM}_s${SEED}"
EXTRA=()
if [ "$EVAL_ONLY" = "1" ]; then
  EXTRA+=(--evaluation-only)
fi

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/train_point_maze_tour.py" \
  --arm "$ARM" --seed "$SEED" \
  --model "$MODEL" \
  --data-root "${OAT_ZERO_TOUR_DATA:-$ROOT_DIR/var/data/point_maze_tour_v1r1}" \
  --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  --passes "$PASSES" \
  --evaluation-interval "${OAT_ZERO_TOUR_EVAL_INTERVAL:-192}" \
  --evaluation-prompts "${OAT_ZERO_TOUR_EVAL_PROMPTS:-64}" \
  --evaluation-split dev \
  --output-receipt "${STEM}.json" \
  --output-metrics "${STEM}.metrics.jsonl" \
  --learning-rate 0.0000002 --replay-weight 0.10 \
  --microbatch-size 16 --max-length 1536 \
  --prompt-format "${OAT_ZERO_TOUR_PROMPT_FORMAT:-qwen_chatml}" \
  --experiment-id "tour-${STAGE}" \
  "${EXTRA[@]}"
