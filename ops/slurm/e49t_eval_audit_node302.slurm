#!/usr/bin/env bash
# CPU-only terminal E49T audit using the live node302 Qwen2.5-72B service.
#SBATCH --job-name=e49t-audit
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
PREFIX="${E49T_PREFIX:-}"
OUTPUT="${E49T_AUDIT_OUTPUT:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ -z "$PREFIX" ]] \
  || [[ -z "$OUTPUT" ]] \
  || [[ "$(hostname -s)" != node302 ]]; then
  echo "E49T eval audit requires root, prefix, output, and node302" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
ENDPOINT="$ROOT_DIR/var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
IDENTITY="$ROOT_DIR/var/artifacts/${PREFIX}_identity.json"
if [[ ! -f "$ENDPOINT" ]] || [[ ! -f "$IDENTITY" ]]; then
  echo "E49T Qwen72 endpoint or experiment identity is missing" >&2
  exit 3
fi

SOURCE_HASH="$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["source_hash"])' \
  "$IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49t_natural_menu_${SOURCE_HASH}/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/math_strategy_canonicalizer.py" ]]; then
  echo "E49T frozen training source snapshot is missing" >&2
  exit 4
fi
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PYTHON_BIN" "$ROOT_DIR/ops/exp_scaling/audit_e49t_eval_contract.py" \
  --prefix "$PREFIX" \
  --endpoint "$ENDPOINT" \
  --output "$OUTPUT"
