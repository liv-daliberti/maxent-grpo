#!/usr/bin/env bash
#SBATCH --job-name=e49v-materialize
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=var/artifacts/e49v-materialize-%j.out
#SBATCH --error=var/artifacts/e49v-materialize-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
exec "$PYTHON_BIN" \
  "$ROOT_DIR/ops/math_strategy_calibration/materialize_e49v_exact_oat_natural_menu.py" \
  --output-root "$ROOT_DIR/var/data/e49v_exact_oat_natural_menu_full" \
  --evidence-root "$ROOT_DIR/var/artifacts/e49v_exact_oat_natural_menu_full_v1" \
  --endpoint-record \
    "$ROOT_DIR/var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json" \
  --workers 4 \
  --timeout 600
