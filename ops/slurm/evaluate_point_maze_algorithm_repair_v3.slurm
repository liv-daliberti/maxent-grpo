#!/bin/bash
#SBATCH --job-name=point-repair-v3-viability
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
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
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?source hash is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?execution hash is required}"
IDENTITY="${OAT_ZERO_VIABILITY_IDENTITY:?viability identity is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR" OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
OUTPUT="$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_v3_viability.json"
QUALIFICATION="$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_v3_qualification.json"
"$PYTHON_BIN" "$EXECUTION_ROOT/evaluate_point_maze_interactive_viability.py" \
  --data-root "$ROOT_DIR/var/data/point_maze_algorithm_repair_v3" \
  --data-split-root "$ROOT_DIR/var/data/point_maze_algorithm_repair_v3/dev" \
  --split multi_answer \
  --model "$ROOT_DIR/var/models/point_maze_interactive_warmstart_v3" \
  --worker-python "$ROOT_DIR/var/maze_runtime/venv/bin/python" \
  --output "$OUTPUT" \
  --protocol "$EXECUTION_ROOT/point_maze_algorithm_repair_v3_viability_20260730.md" \
  --admission-audit "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_v3_admission_audit.json" \
  --sample-count 64 --prefix-count 16 \
  --minimum-prefix-success-prompts 2 --minimum-multimode-prompts 2 \
  --temperature 1 --top-p 1 --max-model-len 1536 --batch-size 256 \
  --seed 76530 --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH" \
  --job-id "${SLURM_JOB_ID:?job ID is required}" \
  --policy-interface velocity_state_v3
exec "$PYTHON_BIN" "$EXECUTION_ROOT/qualify_point_maze_algorithm_repair_v3.py" \
  --viability "$OUTPUT" \
  --viability-identity "$IDENTITY" \
  --admission "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_v3_admission_audit.json" \
  --data-identity "$ROOT_DIR/var/data/point_maze_algorithm_repair_v3/identity.json" \
  --protocol "$EXECUTION_ROOT/point_maze_algorithm_repair_v3_viability_20260730.md" \
  --output "$QUALIFICATION"
