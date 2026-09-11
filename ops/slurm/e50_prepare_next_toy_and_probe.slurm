#!/usr/bin/env bash
#SBATCH --job-name=e50-prep-toy
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/e50-prep-toy-%j.out
#SBATCH --error=var/artifacts/e50-prep-toy-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

exec bash \
  "$ROOT_DIR/ops/math_strategy_calibration/prepare_e50_next_toy_and_probe.sh"
