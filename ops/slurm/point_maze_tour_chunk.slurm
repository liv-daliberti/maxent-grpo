#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

# One resumable chunk of a PointMaze Tour run.
#
# This cluster routes by requested walltime: --time of one hour or less lands in
# the large `all` partition and starts immediately, anything longer is rerouted
# into the contested cs/mltheory queues. So a long run is executed as a chain of
# one-hour chunks, each resuming from the previous chunk's checkpoint.

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
ARM="${OAT_ZERO_ARM:?arm is required}"
SEED="${OAT_ZERO_SEED:?seed is required}"
STAGE="${OAT_ZERO_TOUR_STAGE:?stage is required}"
PASSES="${OAT_ZERO_TOUR_PASSES:?passes is required}"
MODEL="${OAT_ZERO_TOUR_MODEL:?model is required}"
CHUNK="${OAT_ZERO_TOUR_CHUNK_UPDATES:-384}"
EVAL_INTERVAL="${OAT_ZERO_TOUR_EVAL_INTERVAL:-384}"
EVAL_PROMPTS="${OAT_ZERO_TOUR_EVAL_PROMPTS:-64}"
CKPT_EVERY="${OAT_ZERO_TOUR_CKPT_EVERY:-192}"
DATA="${OAT_ZERO_TOUR_DATA:-$ROOT_DIR/var/data/point_maze_tour_v1r1}"

source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$ROOT_DIR/src"
export OAT_ZERO_EXECUTION_ROOT="$ROOT_DIR/ops"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

STEM="$ROOT_DIR/var/artifacts/tour_${STAGE}_${ARM}_s${SEED}"
CKPT="$ROOT_DIR/var/checkpoints/tour_${STAGE}_${ARM}_s${SEED}"

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/train_point_maze_tour.py" \
  --arm "$ARM" --seed "$SEED" \
  --model "$MODEL" \
  --data-root "$DATA" \
  --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  --passes "$PASSES" \
  --evaluation-interval "$EVAL_INTERVAL" \
  --evaluation-prompts "$EVAL_PROMPTS" \
  --evaluation-split dev \
  --output-receipt "${STEM}.json" \
  --output-metrics "${STEM}.metrics.jsonl" \
  --checkpoint-dir "$CKPT" \
  --checkpoint-every "$CKPT_EVERY" \
  --max-updates-per-run "$CHUNK" \
  --auto-resume \
  --learning-rate 0.0000002 --replay-weight 0.10 \
  --microbatch-size 16 --max-length 1536 \
  --experiment-id "tour-${STAGE}"
