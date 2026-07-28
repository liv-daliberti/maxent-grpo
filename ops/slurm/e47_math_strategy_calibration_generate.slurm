#!/usr/bin/env bash
#SBATCH --job-name=e47-cal-05b
#SBATCH --output=var/artifacts/e47_math_strategy_calibration_v1/generate-%j.out
#SBATCH --error=var/artifacts/e47_math_strategy_calibration_v1/generate-%j.err
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT_DIR" || ! -f "$ROOT_DIR/ops/math_strategy_calibration/e47_calibration.py" ]]; then
  echo "Cannot resolve E47 repository root from OAT_ZERO_REPO_ROOT/SLURM_SUBMIT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT_DIR/var/cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT_DIR/var/cache/huggingface/transformers"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PIPELINE="$ROOT_DIR/ops/math_strategy_calibration/e47_calibration.py"

"$PYTHON_BIN" "$PIPELINE" generate --batch-size 2 --gpu-memory 0.80
"$PYTHON_BIN" "$PIPELINE" grade
