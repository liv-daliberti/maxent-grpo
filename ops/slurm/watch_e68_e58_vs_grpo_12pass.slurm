#!/bin/bash
# Keep the user-facing five-domain E68/E69 decision figure fresh once per minute.
#SBATCH --job-name=watch_e68_figure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --time=7-00:00:00
#SBATCH --output=var/artifacts/logs/watch_e68_figure-%j.out
#SBATCH --error=var/artifacts/logs/watch_e68_figure-%j.err
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_E68_MONITOR_INTERVAL_SECONDS=60
exec "$ROOT_DIR/ops/exp_scaling/watch_e68_e58_vs_grpo_12pass.sh"
