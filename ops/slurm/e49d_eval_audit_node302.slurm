#!/usr/bin/env bash
# CPU-only terminal E49D execution audit on the node105-reachable fabric.
#SBATCH --job-name=e49d-audit
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
STAGE="${E49D_STAGE:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$STAGE" != toy && "$STAGE" != full ]] \
  || [[ "$(hostname -s)" != node302 ]]; then
  echo "E49D eval audit requires stage toy/full on node302" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
SOURCE_HASH="27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49d_maximal_support_math_${SOURCE_HASH}/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/math_strategy_canonicalizer.py" ]]; then
  echo "E49D frozen source snapshot is missing" >&2
  exit 4
fi
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENDPOINT="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json"
OUTPUT="$ROOT_DIR/var/artifacts/e49d_${STAGE}_terminal_execution_audit.json"

if [[ ! -f "$ENDPOINT" ]]; then
  echo "E49D Qwen72 endpoint is not ready" >&2
  exit 3
fi
exec "$PYTHON_BIN" "$ROOT_DIR/ops/exp_scaling/audit_e49d_eval_contract.py" \
  --stage "$STAGE" --endpoint "$ENDPOINT" --output "$OUTPUT"
