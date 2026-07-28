#!/bin/bash
#SBATCH --job-name=ff05b_eval4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --time=08:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$ROOT_DIR/ops/repo_env.sh"
CODE_ROOT="${OAT_ZERO_EVAL_CODE_ROOT:-$ROOT_DIR}"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS=1

echo "[fixed-eval-slurm] host=$(hostname) job=${SLURM_JOB_ID:-} task=${OAT_ZERO_COMPARATIVE_TASK:?}"
exec "$CODE_ROOT/ops/exp_scaling/run_freeform_05b_repeated_eval.sh"
