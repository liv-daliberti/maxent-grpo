#!/bin/bash
#SBATCH --job-name=point-warmstart-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?OAT_ZERO_SOURCE_ROOT is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?OAT_ZERO_EXECUTION_ROOT is required}"
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?OAT_ZERO_SOURCE_HASH is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?OAT_ZERO_EXECUTION_HASH is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT" OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_MAZE_WORKER_PYTHON="$ROOT_DIR/var/maze_runtime/venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
MODEL="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
OUTPUT_MODEL="$ROOT_DIR/var/models/point_maze_interactive_warmstart_v3"
PROTOCOL="$ROOT_DIR/paper/preregistration/point_maze_interactive_warmstart_v3_20260730.md"

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" "$EXECUTION_ROOT/train_point_maze_interactive_warmstart_v1.py" \
  --model "$MODEL" --data-root "$ROOT_DIR/var/data/point_maze_interactive_warmstart_v3" \
  --output-model "$OUTPUT_MODEL" --output-receipt "$ROOT_DIR/var/artifacts/point_maze_interactive_warmstart_sft_v3.json" \
  --protocol "$PROTOCOL" --policy-interface velocity_state_v3 --seed 75203 --epochs 12 \
  --batch-size 4 --gradient-accumulation 7 --learning-rate 0.00002 --weight-decay 0.01 \
  --warmup-steps 28 --max-grad-norm 1.0 --max-length 1536

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" "$EXECUTION_ROOT/evaluate_point_maze_interactive_viability.py" \
  --data-root "$ROOT_DIR/var/data/point_maze_modebench_v1" --data-split-root "$ROOT_DIR/var/data/point_maze_modebench_v1/dev" \
  --split multi_answer --model "$OUTPUT_MODEL" --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --output "$ROOT_DIR/var/artifacts/point_maze_interactive_05b_viability_warmstart_v3.json" \
  --protocol "$PROTOCOL" --admission-audit "$ROOT_DIR/var/artifacts/point_maze_modebench_v1_admission_audit.json" \
  --policy-interface velocity_state_v3 --sample-count 64 --prefix-count 16 \
  --minimum-prefix-success-prompts 2 --minimum-multimode-prompts 1 --temperature 1.0 --top-p 1.0 \
  --max-model-len 1536 --batch-size 256 --seed 75105 --source-hash "$SOURCE_HASH" \
  --execution-hash "$EXECUTION_HASH" --job-id "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
