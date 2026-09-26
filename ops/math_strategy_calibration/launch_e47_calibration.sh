#!/usr/bin/env bash
# Freeze, prepare, and launch E47-CAL policy sampling without changing E46.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PIPELINE="$ROOT_DIR/ops/math_strategy_calibration/e47_calibration.py"
PROTOCOL="$ROOT_DIR/paper/preregistration/e47_math_strategy_canonicalizer_calibration.md"
OUTPUT="$ROOT_DIR/var/artifacts/e47_math_strategy_calibration_v1"

if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E47-CAL protocol is not frozen" >&2
  exit 1
fi
if [[ -e "$OUTPUT/generation_job.json" ]]; then
  echo "E47-CAL generation was already submitted" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$PIPELINE" prepare
"$PYTHON_BIN" "$PIPELINE" audit-packet

job_id="$(
  sbatch --parsable \
    --nodes=1 \
    --nodelist=node103,node104,node208 \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR" \
    "$ROOT_DIR/ops/slurm/e47_math_strategy_calibration_generate.slurm"
)"
"$PYTHON_BIN" - "$OUTPUT/generation_job.json" "$job_id" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "schema": "e47_generation_job_v1",
            "job_id": sys.argv[2],
            "stage": "generate_then_full_math_validator",
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "submitted E47-CAL generation job $job_id"
