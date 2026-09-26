#!/usr/bin/env bash
#SBATCH --job-name=e50f2-pair-cal
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=var/artifacts/e50f2-pair-cal-%j.out
#SBATCH --error=var/artifacts/e50f2-pair-cal-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/ops/math_strategy_calibration/run_e50f2_frozen_pairwise_relation_calibration.py" \
  --output-root "$ROOT_DIR/var/artifacts/e50f2_frozen_pairwise_relation_calibration_v1" \
  --workers 4 \
  --timeout 1200
