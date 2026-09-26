#!/usr/bin/env bash
#SBATCH --job-name=e50a-route-cal
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=var/artifacts/e50a-route-cal-%j.out
#SBATCH --error=var/artifacts/e50a-route-cal-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/ops/math_strategy_calibration/run_e49y_observed_route_discovery.py" \
  --output-root "$ROOT_DIR/var/artifacts/e50a_consensus_observed_route_calibration_v1" \
  --reuse-natural-root "$ROOT_DIR/var/artifacts/e49y_observed_route_discovery_v1" \
  --expected-natural-sha256 ab6c5c081919caaf358c65ff63e394b0fc6587dc98fb69318a4c77b5aa76401f \
  --partition-mode consensus_intersection \
  --protocol "$ROOT_DIR/paper/preregistration/e50a_consensus_observed_route_calibration_20260726.md" \
  --result-schema e50a_consensus_observed_route_calibration_v1 \
  --workers 2 \
  --timeout 1200
