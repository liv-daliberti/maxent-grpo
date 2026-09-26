#!/usr/bin/env bash
#SBATCH --job-name=e50f3-family
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/e50f3-family-%j.out
#SBATCH --error=var/artifacts/e50f3-family-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/ops/math_strategy_calibration/run_e50f3_finite_family_canonicalization.py" \
  --output-root "$ROOT_DIR/var/artifacts/e50f3_finite_family_canonicalization_v1" \
  --timeout 1200
