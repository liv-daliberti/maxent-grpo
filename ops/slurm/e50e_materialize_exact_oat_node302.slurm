#!/usr/bin/env bash
#SBATCH --job-name=e50e-materialize
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=var/artifacts/e50e-materialize-%j.out
#SBATCH --error=var/artifacts/e50e-materialize-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

exec bash \
  "$ROOT_DIR/ops/math_strategy_calibration/run_e50e_materialize_and_launch.sh"
