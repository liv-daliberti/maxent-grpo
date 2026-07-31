#!/bin/bash
#SBATCH --job-name=ant-interactive-v17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?source hash is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?execution hash is required}"
ROUTE_IDENTITY="${OAT_ZERO_ROUTE_IDENTITY:?route identity is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$ROUTE_IDENTITY"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 MUJOCO_GL=egl
MODEL="$ROOT_DIR/var/models/ant_maze_interactive_warmstart_v13"
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/evaluate_ant_maze_interactive_viability_v17.py" \
  --data-root "$ROOT_DIR/var/data/ant_maze_modebench_v15_controller_v17_r1" \
  --data-split-root "$ROOT_DIR/var/data/ant_maze_modebench_v15_controller_v17_r1/dev" \
  --split multi_answer \
  --model "$MODEL" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --output "$ROOT_DIR/var/artifacts/ant_maze_interactive_05b_viability_v17.json" \
  --protocol "$EXECUTION_ROOT/ant_maze_interactive_viability_v17_20260730.md" \
  --admission-audit "$ROOT_DIR/var/artifacts/ant_maze_modebench_v15_controller_v17_r1_admission_audit.json" \
  --sample-count 64 --prefix-count 16 \
  --minimum-prefix-success-prompts 2 --minimum-multimode-prompts 1 \
  --temperature 1.0 --top-p 1.0 --max-model-len 1536 \
  --batch-size 256 --seed 108317 \
  --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH" \
  --job-id "${SLURM_JOB_ID:?job ID required}" \
  --policy-interface velocity_state_v3
