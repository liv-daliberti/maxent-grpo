#!/bin/bash
# Run the comparative evaluation (per-eval-seed sampled passes + greedy) over
# every arm's final checkpoint via ops/run_countdown_comparative_eval.sh.

#SBATCH --job-name=cdcomp_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=06:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${ROOT_DIR}/ops/repo_env.sh"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS=1

echo "[cdcomp-eval-slurm] host=$(hostname)"
echo "[cdcomp-eval-slurm] job_id=${SLURM_JOB_ID:-}"
echo "[cdcomp-eval-slurm] stamp_prefix=${RUN_STAMP_PREFIX:?RUN_STAMP_PREFIX is required}"

exec "${ROOT_DIR}/ops/run_countdown_comparative_eval.sh"
