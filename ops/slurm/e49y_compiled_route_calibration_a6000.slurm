#!/usr/bin/env bash
#SBATCH --job-name=e49y-route-cal
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/e49y-route-cal-%j.out
#SBATCH --error=var/artifacts/e49y-route-cal-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/run_e49y_compiled_route_calibration.py" \
  --output-root \
    "$ROOT_DIR/var/artifacts/e49y_compiled_bottom_up_route_calibration_v1" \
  --workers 4 \
  --timeout 600
