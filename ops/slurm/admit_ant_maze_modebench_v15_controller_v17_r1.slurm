#!/bin/bash
#SBATCH --job-name=antmaze-v15-v17-admit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
OPS_ROOT="${OAT_ZERO_OPS_ROOT:?ops root is required}"
IDENTITY="${OAT_ZERO_PROTOCOL_IDENTITY:?protocol identity is required}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_OPS_ROOT="$OPS_ROOT"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export PYTHONPATH="$OPS_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
DATA_ROOT="$ROOT_DIR/var/data/ant_maze_modebench_v15_controller_v17_r1"
AUDIT="$ROOT_DIR/var/artifacts/ant_maze_modebench_v15_controller_v17_r1_admission_audit.json"
[[ -f "$IDENTITY" ]] || { echo "Missing AntMaze v15/v17 identity" >&2; exit 1; }
"$PYTHON_BIN" "$OPS_ROOT/make_ant_maze_mode_data_v15_controller_v17_r1.py" \
  --output-root "$DATA_ROOT" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON"
exec "$PYTHON_BIN" "$OPS_ROOT/audit_ant_maze_mode_data_v15_controller_v17_r1.py" \
  --data-root "$DATA_ROOT" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --output "$AUDIT"
