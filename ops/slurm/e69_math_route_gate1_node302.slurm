#!/usr/bin/env bash
#SBATCH --job-name=e69r2-route-g1
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/e69-route-g1-%j.out
#SBATCH --error=var/artifacts/e69-route-g1-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/ops/route_successor/sample_e69_math_route_gate1.py" \
  --output-root "$ROOT_DIR/var/artifacts/e69_math_route_gate1_base_rpn_v2" \
  --workers 16 \
  --expected-input-identity "$OAT_E69_GATE1_INPUT_IDENTITY"
