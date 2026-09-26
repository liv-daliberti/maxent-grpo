#!/bin/bash
#SBATCH --job-name=e49t-decl-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=02:00:00
#SBATCH --output=var/artifacts/e49t-decl-cal-%j.out
#SBATCH --error=var/artifacts/e49t-decl-cal-%j.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
ENDPOINT="$ROOT_DIR/var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
PRIMARY="$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1/result.json"
ready=0
for _ in $(seq 1 180); do
  if [[ -f "$ENDPOINT" ]] && [[ -f "$PRIMARY" ]] \
    && "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - \
      "$PRIMARY" <<'PY' >/dev/null 2>&1
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("pass") is True else 1)
PY
  then
    ready=1
    break
  fi
  sleep 10
done
if [[ "$ready" != 1 ]]; then
  echo "E49T primary calibration/endpoint did not become ready" >&2
  exit 1
fi
exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e49t_declaration_mismatch_calibration.py" \
  --artifact-dir "$ROOT_DIR/var/artifacts/e49t_declaration_mismatch_calibration_v1" \
  --endpoint-record "$ENDPOINT" \
  --output "$ROOT_DIR/var/artifacts/e49t_declaration_mismatch_calibration_v1/result.json"
