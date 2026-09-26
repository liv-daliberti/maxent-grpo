#!/bin/bash
#SBATCH --job-name=watch_e70_full80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=7-00:00:00
#SBATCH --output=var/artifacts/logs/watch_e70_full80-%j.out
#SBATCH --error=var/artifacts/logs/watch_e70_full80-%j.err
#SBATCH --requeue

set -euo pipefail
ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_E70_MONITOR_INTERVAL_SECONDS=60
exec "$ROOT_DIR/ops/exp_scaling/watch_e70_clean_full80.sh"
