#!/usr/bin/env bash
#SBATCH --job-name=e50g-safe-route
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=var/artifacts/e50g-safe-route-%j.out
#SBATCH --error=var/artifacts/e50g-safe-route-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/run_e50g_safe_signature_teacher_route_calibration.py" \
  --output-root "$ROOT_DIR/var/artifacts/e50g_safe_signature_teacher_route_calibration_v1" \
  --workers 2 \
  --timeout 1200
