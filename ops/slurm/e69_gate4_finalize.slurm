#!/usr/bin/env bash
#SBATCH --job-name=e69g4_final
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
cd "$ROOT_DIR"
exec bash ops/route_successor/finalize_e69_gate4.sh
