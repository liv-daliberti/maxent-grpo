#!/usr/bin/env bash
#SBATCH --job-name=e49ac-singleton
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=var/artifacts/e49ac-singleton-%j.out
#SBATCH --error=var/artifacts/e49ac-singleton-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
OUTPUT_ROOT="$ROOT_DIR/var/artifacts/e49ac_confirmed_singleton_observed_route_discovery_v1"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/ops/math_strategy_calibration/run_e49ac_confirmed_singleton_observed_route_discovery.py" \
  --output-root "$OUTPUT_ROOT" \
  --workers 4 \
  --timeout 600
