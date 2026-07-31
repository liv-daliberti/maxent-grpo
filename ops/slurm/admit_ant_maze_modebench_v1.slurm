#!/bin/bash
#SBATCH --job-name=antmaze-route-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?OAT_ZERO_SOURCE_ROOT is required}"
OPS_ROOT="${OAT_ZERO_OPS_ROOT:?OAT_ZERO_OPS_ROOT is required}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_OPS_ROOT="$OPS_ROOT"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v1"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v1_admission_audit.json"

if [[ ! -e "$DATA_ROOT" ]]; then
  "$PYTHON_BIN" "$OPS_ROOT/make_ant_maze_mode_data.py" \
    --output-root "$DATA_ROOT" \
    --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON"
fi

exec "$PYTHON_BIN" "$OPS_ROOT/audit_ant_maze_mode_data.py" \
  --data-root "$DATA_ROOT" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --output "$AUDIT"
