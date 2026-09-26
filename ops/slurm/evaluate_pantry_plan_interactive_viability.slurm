#!/bin/bash
#SBATCH --job-name=pantry-interactive-05b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --time=04:00:00
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
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/evaluate_pantry_plan_interactive_viability.py" \
  --data-root "$ROOT_DIR/var/data/pantry_plan_modebench_v2" \
  --data-split-root "$ROOT_DIR/var/data/pantry_plan_modebench_v2/dev" \
  --split multi_answer \
  --model "$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775" \
  --output "$ROOT_DIR/var/artifacts/pantry_plan_interactive_05b_viability_v1.json" \
  --protocol "$ROOT_DIR/paper/preregistration/pantry_plan_interactive_05b_viability_v1_20260729.md" \
  --admission-audit "$ROOT_DIR/var/artifacts/pantry_plan_modebench_v2_admission_audit.json" \
  --sample-count 64 \
  --prefix-count 16 \
  --minimum-prefix-success-prompts 32 \
  --minimum-multimode-prompts 16 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-model-len 2048 \
  --batch-size 256 \
  --seed 76101 \
  --source-hash "$SOURCE_HASH" \
  --execution-hash "$EXECUTION_HASH" \
  --job-id "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
