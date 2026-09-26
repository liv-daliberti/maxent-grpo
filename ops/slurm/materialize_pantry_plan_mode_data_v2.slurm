#!/bin/bash
#SBATCH --job-name=pantry-plan-v2-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=08:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?OAT_ZERO_SOURCE_ROOT is required}"
OPS_ROOT="${OAT_ZERO_OPS_ROOT:?OAT_ZERO_OPS_ROOT is required}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_OPS_ROOT="$OPS_ROOT"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
DATA_ROOT="$ROOT_DIR/var/data/pantry_plan_modebench_v2"
AUDIT="$ROOT_DIR/var/artifacts/pantry_plan_modebench_v2_admission_audit.json"

if [[ ! -e "$DATA_ROOT" ]]; then
  "$PYTHON_BIN" "$OPS_ROOT/make_pantry_plan_mode_data_v2.py" \
    --output-root "$DATA_ROOT" \
    --train-per-family 96 \
    --dev-per-family 16 \
    --eval-per-family 32 \
    --seed 74002
fi

exec "$PYTHON_BIN" "$OPS_ROOT/audit_pantry_plan_mode_data_v2.py" \
  --data-root "$DATA_ROOT" \
  --output "$AUDIT"
