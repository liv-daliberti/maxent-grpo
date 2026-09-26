#!/bin/bash
#SBATCH --job-name=e58_sentinel_monitor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node916
#SBATCH --time=4-00:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
exec "$ROOT_DIR/ops/exp_scaling/watch_e55_sentinel.sh"
