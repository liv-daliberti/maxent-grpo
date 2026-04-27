#!/bin/bash
# Run the full dr.GRPO / Dr.X-GRPO pass@k sweep in one 8-GPU node302 job.

#SBATCH --job-name=eval_pk_grid302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=48:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${ROOT_DIR}/ops/repo_env.sh"

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] root_dir=${ROOT_DIR}"
echo "[slurm] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[slurm] output_root=${PASSK_SWEEP_OUTPUT_ROOT:-unset}"
echo "[slurm] model_families=${PASSK_SWEEP_MODEL_FAMILIES:-drgrpo,drx_grpo}"
echo "[slurm] model_seeds=${PASSK_SWEEP_MODEL_SEEDS:-42,43,44}"
echo "[slurm] templates=${PASSK_SWEEP_TEMPLATES:-no,qwen_math,r1}"
echo "[slurm] inference_seeds=${PASSK_SWEEP_INFERENCE_SEEDS:-1,2,3,4,5}"

exec "${ROOT_DIR}/ops/run_eval_model_grid_node302.sh"
