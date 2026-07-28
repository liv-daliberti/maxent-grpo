#!/usr/bin/env bash
#SBATCH --job-name=e49t-term-gate
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/e49t-terminal-gate-%j.out
#SBATCH --error=var/artifacts/e49t-terminal-gate-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PREFIX=e49t_natural_menu_math_toy_05b_3ep_v1
IDENTITY="$ROOT_DIR/var/artifacts/${PREFIX}_identity.json"
ENDPOINT="$ROOT_DIR/var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
TERMINAL_AUDIT="$ROOT_DIR/var/artifacts/e49t_terminal_natural_menu_eval_v1/result.json"
ROUTE_COVERAGE="$ROOT_DIR/var/artifacts/e49t_toy_route_coverage_v1/result.json"
ADVANCEMENT="$ROOT_DIR/var/artifacts/e49t_natural_menu_math_toy_advancement_v1.json"

SOURCE_HASH="$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["source_hash"])' \
  "$IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49t_natural_menu_${SOURCE_HASH}/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/math_strategy_canonicalizer.py" ]]; then
  echo "E49T frozen source snapshot is missing" >&2
  exit 2
fi

export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib"

"$PYTHON_BIN" "$ROOT_DIR/ops/exp_scaling/audit_e49t_eval_contract.py" \
  --prefix "$PREFIX" \
  --endpoint "$ENDPOINT" \
  --output "$TERMINAL_AUDIT"

"$PYTHON_BIN" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e49t_toy_route_coverage.py" \
  --stamp-prefix "$PREFIX" \
  --endpoint-record "$ENDPOINT" \
  --steps all \
  --out "$ROUTE_COVERAGE"

"$PYTHON_BIN" "$ROOT_DIR/ops/exp_scaling/analyze_e49t_natural_menu_math.py" \
  --stamp-prefix "$PREFIX" \
  --route-coverage "$ROUTE_COVERAGE" \
  --out "$ADVANCEMENT" \
  --require-complete
