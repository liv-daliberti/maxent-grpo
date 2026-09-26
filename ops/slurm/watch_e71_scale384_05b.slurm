#!/bin/bash
# Refresh the E71 audit and both 384/128 scaling curves each minute.
#SBATCH --job-name=watch_e71_scale384
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --time=7-00:00:00
#SBATCH --output=var/artifacts/logs/watch_e71_scale384-%j.out
#SBATCH --error=var/artifacts/logs/watch_e71_scale384-%j.err
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_E71_MONITOR_INTERVAL_SECONDS=60
exec "$ROOT_DIR/ops/exp_scaling/watch_e71_scale384_05b.sh"
