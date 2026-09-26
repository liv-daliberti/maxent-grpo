#!/usr/bin/env bash
#SBATCH --job-name=e49t-route-cov
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/e49t-route-cov-%j.out
#SBATCH --error=var/artifacts/e49t-route-cov-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
exec "$PYTHON_BIN" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e49t_toy_route_coverage.py" \
  --steps 0,150 \
  --out "$ROOT_DIR/var/artifacts/e49t_toy_route_coverage_v1/result.json"
