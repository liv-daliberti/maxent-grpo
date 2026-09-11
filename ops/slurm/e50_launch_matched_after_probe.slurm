#!/usr/bin/env bash
#SBATCH --job-name=e50-launch-matched
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/e50-launch-matched-%j.out
#SBATCH --error=var/artifacts/e50-launch-matched-%j.err

set -euo pipefail

: "${E50_MATCHED_VARIANT:?missing E50_MATCHED_VARIANT}"

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

exec bash "$ROOT_DIR/ops/exp_scaling/launch_e50_matched_math_toy_05b.sh" \
  "$E50_MATCHED_VARIANT" run
