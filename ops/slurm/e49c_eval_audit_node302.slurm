#!/usr/bin/env bash
# CPU-only terminal E49C execution audit on the node105-reachable fabric.
#SBATCH --job-name=e49c-audit
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
STAGE="${E49C_STAGE:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$STAGE" != toy && "$STAGE" != full ]] \
  || [[ "$(hostname -s)" != node302 ]]; then
  echo "E49C eval audit requires stage toy/full on node302" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENDPOINT="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json"
OUTPUT="$ROOT_DIR/var/artifacts/e49c_${STAGE}_terminal_execution_audit.json"

if [[ ! -f "$ENDPOINT" ]]; then
  echo "E49C Qwen72 endpoint is not ready" >&2
  exit 3
fi
exec "$PYTHON_BIN" "$ROOT_DIR/ops/exp_scaling/audit_e49c_eval_contract.py" \
  --stage "$STAGE" --endpoint "$ENDPOINT" --output "$OUTPUT"
