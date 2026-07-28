#!/usr/bin/env bash
# Submit the exact frozen E47 72B judge on four node105 A5000s.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROTOCOL="$ROOT_DIR/paper/preregistration/e47_math_strategy_canonicalizer_calibration.md"
OUTPUT="$ROOT_DIR/var/artifacts/e47_math_strategy_calibration_v1"
JOB_RECORD="$OUTPUT/qwen72_server_job.json"

if ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E47-CAL protocol is not frozen" >&2
  exit 1
fi
if [[ -e "$JOB_RECORD" ]]; then
  echo "E47 Qwen72 server was already submitted: $JOB_RECORD" >&2
  exit 1
fi
if [[ ! -f "$OUTPUT/validation_summary.json" ]]; then
  echo "E47 policy validation is incomplete" >&2
  exit 1
fi

job_id="$(
  sbatch --parsable \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR" \
    "$ROOT_DIR/ops/slurm/e47_qwen72_node105.slurm"
)"

python - "$JOB_RECORD" "$job_id" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema": "e47_qwen72_node105_job_v1",
            "job_id": sys.argv[2],
            "node": "node105",
            "gpu": "a5000",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.88,
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "submitted E47 Qwen72 node105 server job $job_id"
