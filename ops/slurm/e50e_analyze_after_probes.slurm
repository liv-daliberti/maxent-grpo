#!/usr/bin/env bash
#SBATCH --job-name=e50e-analyze
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/e50e-analyze-%j.out
#SBATCH --error=var/artifacts/e50e-analyze-%j.err

set -euo pipefail

: "${E50E_ROUTE_VARIANT:?missing E50E_ROUTE_VARIANT}"

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

exec bash "$ROOT_DIR/ops/math_strategy_calibration/run_e50e_analysis.sh"
