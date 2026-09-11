#!/usr/bin/env bash
#SBATCH --job-name=e50-analyze-matched
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/e50-analyze-matched-%j.out
#SBATCH --error=var/artifacts/e50-analyze-matched-%j.err

set -euo pipefail

: "${E50_ANALYSIS_VARIANT:?missing E50_ANALYSIS_VARIANT}"

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

exec bash \
  "$ROOT_DIR/ops/math_strategy_calibration/run_e50_analysis_and_maybe_full.sh"
