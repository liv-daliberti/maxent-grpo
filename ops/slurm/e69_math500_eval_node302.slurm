#!/bin/bash
# Evaluate one frozen E69 terminal checkpoint on held-out MATH-500.

#SBATCH --job-name=e69_math500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=2-00:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${ROOT_DIR}/ops/repo_env.sh"

export PYTHONPATH="${OAT_ZERO_SOURCE_ROOT:?}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_V1=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "${OAT_ZERO_PYTHON:?}" "${E69_GATE4_EVALUATOR:?}" \
  --identity "${E69_GATE4_IDENTITY:?}" \
  --alias "${E69_GATE4_ALIAS:?}" \
  --arm "${E69_GATE4_ARM:?}" \
  --seed "${E69_GATE4_SEED:?}" \
  --checkpoint "${E69_GATE4_CHECKPOINT:?}" \
  --output "${E69_GATE4_OUTPUT:?}" \
  --data-root "${E69_GATE4_DATA_ROOT:?}"
