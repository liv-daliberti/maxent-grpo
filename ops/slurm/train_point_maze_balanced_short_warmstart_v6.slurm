#!/bin/bash
#SBATCH --job-name=point-balanced-short-v6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --time=03:00:00
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
BASE_MODEL="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
OUTPUT_MODEL="$ROOT_DIR/var/models/point_maze_interactive_warmstart_v6_balanced_short"
PROTOCOL="$EXECUTION_ROOT/point_maze_balanced_short_warmstart_v6_20260730.md"

"$PYTHON_BIN" "$EXECUTION_ROOT/train_point_maze_interactive_warmstart_v1.py" \
  --model "$BASE_MODEL" \
  --data-root "$ROOT_DIR/var/data/point_maze_interactive_warmstart_v5_balanced" \
  --output-model "$OUTPUT_MODEL" \
  --output-receipt "$ROOT_DIR/var/artifacts/point_maze_balanced_short_warmstart_sft_v6.json" \
  --protocol "$PROTOCOL" --policy-interface velocity_state_v3 \
  --frozen-short-balanced-v6 --seed 76621 --epochs 3 \
  --batch-size 4 --gradient-accumulation 7 --learning-rate 0.00002 \
  --weight-decay 0.01 --warmup-steps 7 --max-grad-norm 1.0 --max-length 1536

"$PYTHON_BIN" "$EXECUTION_ROOT/evaluate_point_maze_interactive_viability.py" \
  --data-root "$ROOT_DIR/var/data/point_maze_algorithm_repair_v3" \
  --data-split-root "$ROOT_DIR/var/data/point_maze_algorithm_repair_v3/dev" \
  --split multi_answer --model "$OUTPUT_MODEL" \
  --worker-python "$OAT_ZERO_MAZE_WORKER_PYTHON" \
  --output "$ROOT_DIR/var/artifacts/point_maze_balanced_short_warmstart_v6_viability.json" \
  --protocol "$PROTOCOL" \
  --admission-audit "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_v3_admission_audit.json" \
  --policy-interface velocity_state_v3 --sample-count 64 --prefix-count 16 \
  --minimum-prefix-success-prompts 2 --minimum-multimode-prompts 2 \
  --temperature 1.0 --top-p 1.0 --max-model-len 1536 --batch-size 256 \
  --seed 76622 --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH" \
  --job-id "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

"$PYTHON_BIN" "$EXECUTION_ROOT/qualify_point_maze_balanced_short_warmstart_v6.py" \
  --source-identity "$ROOT_DIR/var/data/point_maze_balanced_warmstart_source_v5/identity.json" \
  --warmstart-identity "$ROOT_DIR/var/data/point_maze_interactive_warmstart_v5_balanced/identity.json" \
  --sft-receipt "$ROOT_DIR/var/artifacts/point_maze_balanced_short_warmstart_sft_v6.json" \
  --viability "$ROOT_DIR/var/artifacts/point_maze_balanced_short_warmstart_v6_viability.json" \
  --viability-identity "$IDENTITY" \
  --admission "$ROOT_DIR/var/artifacts/point_maze_algorithm_repair_v3_admission_audit.json" \
  --v5-qualification "$ROOT_DIR/var/artifacts/point_maze_balanced_warmstart_v5_qualification.json" \
  --protocol "$PROTOCOL" \
  --output "$ROOT_DIR/var/artifacts/point_maze_balanced_short_warmstart_v6_qualification.json"
