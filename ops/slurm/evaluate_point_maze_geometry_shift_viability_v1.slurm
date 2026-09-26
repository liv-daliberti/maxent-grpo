#!/bin/bash
#SBATCH --job-name=point-geometry-shift-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
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
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?source hash is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?execution hash is required}"

source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT" OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/evaluate_point_maze_interactive_viability.py" \
  --data-root "$ROOT_DIR/var/data/point_maze_geometry_shift_v1" \
  --data-split-root "$ROOT_DIR/var/data/point_maze_geometry_shift_v1/dev" \
  --split multi_answer \
  --model "$ROOT_DIR/var/models/point_maze_interactive_warmstart_v3" \
  --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  --output "$ROOT_DIR/var/artifacts/point_maze_geometry_shift_05b_viability_v1.json" \
  --protocol "$EXECUTION_ROOT/point_maze_geometry_shift_replacement_v1_20260730.md" \
  --admission-audit "$ROOT_DIR/var/artifacts/point_maze_geometry_shift_v1_admission_audit.json" \
  --policy-interface velocity_state_v3 --sample-count 64 --prefix-count 16 \
  --minimum-prefix-success-prompts 2 --minimum-multimode-prompts 1 \
  --temperature 1.0 --top-p 1.0 --max-model-len 1536 --batch-size 256 \
  --seed 75121 --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH" \
  --job-id "${SLURM_JOB_ID:?job ID is required}"
