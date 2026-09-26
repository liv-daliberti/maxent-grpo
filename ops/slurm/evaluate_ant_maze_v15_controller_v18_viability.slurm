#!/bin/bash
#SBATCH --job-name=ant-v15-v18-viability
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
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT" OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export OAT_ZERO_PROTOCOL_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_v15_controller_v18_admission_identity.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$EXECUTION_ROOT/ant_maze_v15_controller_v18_05b_viability_20260730.md"
ADMISSION="$ROOT_DIR/var/artifacts/ant_maze_modebench_v15_controller_v18_admission_audit.json"

"$PYTHON_BIN" "$EXECUTION_ROOT/evaluate_ant_maze_interactive_viability_v18.py" \
  --data-root "$ROOT_DIR/var/data/ant_maze_modebench_v15_controller_v18" \
  --data-split-root "$ROOT_DIR/var/data/ant_maze_modebench_v15_controller_v18/dev" \
  --split multi_answer \
  --model "$ROOT_DIR/var/models/ant_maze_interactive_warmstart_v13" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --output "$ROOT_DIR/var/artifacts/ant_maze_v15_controller_v18_05b_viability.json" \
  --protocol "$PROTOCOL" --admission-audit "$ADMISSION" \
  --policy-interface velocity_state_v3 --sample-count 64 --prefix-count 16 \
  --minimum-prefix-success-prompts 2 --minimum-multimode-prompts 2 \
  --temperature 1.0 --top-p 1.0 --max-model-len 1536 --batch-size 256 \
  --seed 76701 --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH" \
  --job-id "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

"$PYTHON_BIN" "$EXECUTION_ROOT/qualify_ant_maze_v15_controller_v18_viability.py" \
  --viability "$ROOT_DIR/var/artifacts/ant_maze_v15_controller_v18_05b_viability.json" \
  --identity "$IDENTITY" --admission "$ADMISSION" --protocol "$PROTOCOL" \
  --output "$ROOT_DIR/var/artifacts/ant_maze_v15_controller_v18_qualification.json"
