#!/bin/bash
#SBATCH --job-name=point-repair-v4-admit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
"$PYTHON_BIN" "$EXECUTION_ROOT/make_point_maze_algorithm_repair_data_v4.py" \
  --output-root "$ROOT_DIR/var/data/point_maze_algorithm_repair_v4" \
  --worker-python "$WORKER_PYTHON"
exec "$PYTHON_BIN" "$EXECUTION_ROOT/audit_point_maze_algorithm_repair_data_v4.py" \
  --data-root "$ROOT_DIR/var/data/point_maze_algorithm_repair_v4" \
  --worker-python "$WORKER_PYTHON" \
  --output "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_v4_admission_audit.json"
