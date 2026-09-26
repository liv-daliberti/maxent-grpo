#!/usr/bin/env bash
#SBATCH --job-name=e69g2_route_obs
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
cd "$ROOT_DIR"
exec bash ops/route_successor/monitor_e69_gate2_route_snapshots.sh
