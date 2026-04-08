#!/bin/bash
# Single-node 8xA100 pure MaxEnt tokenwise launcher using the shared OAT backend.

#SBATCH --job-name=maxent_tokenwise_pure_1p5b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --time=48:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

mkdir -p var/artifacts/logs var/tmp var/cache

source "$ROOT_DIR/ops/repo_env.sh"

export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/maxent_tokenwise_1p5b_pure_oat_r1_${RUN_STAMP}}"
export PURE_DRGRPO_ROOT="${PURE_DRGRPO_ROOT:-$ROOT_DIR/var/external/understand-r1-zero}"
export PURE_DRGRPO_PYTHON="${PURE_DRGRPO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export N_GPU="${N_GPU:-8}"

echo "[slurm] host=$(hostname)"
echo "[slurm] job_id=${SLURM_JOB_ID:-}"
echo "[slurm] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[slurm] pure_checkout=${PURE_DRGRPO_ROOT}"
echo "[slurm] pure_python=${PURE_DRGRPO_PYTHON}"
echo "[slurm] save_path=${SAVE_PATH}"

exec "$ROOT_DIR/archive/trl/ops/run_maxent_tokenwise_1p5b_pure_single_node.sh" "$@"
