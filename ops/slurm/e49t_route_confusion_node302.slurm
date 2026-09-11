#!/bin/bash

#SBATCH --job-name=e49t-route-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/e49t-route-cal-%j.out
#SBATCH --error=var/artifacts/e49t-route-cal-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
COHORT_DIR="$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1"
ENDPOINT_RECORD="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json"
OUTPUT="$COHORT_DIR/result.json"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e49t_route_confusion_calibration.py" \
  --cohort-dir "$COHORT_DIR" \
  --endpoint-record "$ENDPOINT_RECORD" \
  --output "$OUTPUT"
