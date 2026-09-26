#!/usr/bin/env bash
# CPU-only E49C menu generation/auditing colocated with the private fabric.
#SBATCH --job-name=e49c-menus
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
STAGE="${E49C_STAGE:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$STAGE" != toy && "$STAGE" != full ]] \
  || [[ "$(hostname -s)" != node302 ]]; then
  echo "E49C menu materialization requires stage toy/full on node302" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENDPOINT="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json"
MATERIALIZER="$ROOT_DIR/ops/math_strategy_calibration/materialize_e49c_strategy_menu_data.py"

if [[ "$STAGE" == toy ]]; then
  SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
  OUTPUT="$ROOT_DIR/var/data/e49c_math_strategy_menu_toy"
  EVIDENCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_menu_toy_v1"
else
  SOURCE="$ROOT_DIR/var/data/math12k_384_math500"
  OUTPUT="$ROOT_DIR/var/data/e49c_math_strategy_menu_full"
  EVIDENCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_menu_full_v1"
fi

if [[ ! -f "$ENDPOINT" ]]; then
  echo "E49C Qwen72 endpoint is not ready" >&2
  exit 3
fi

exec "$PYTHON_BIN" "$MATERIALIZER" \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --evidence "$EVIDENCE" \
  --endpoint "$ENDPOINT" \
  --workers 4 --timeout 900
