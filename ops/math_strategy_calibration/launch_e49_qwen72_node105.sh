#!/usr/bin/env bash
# Start E49's long-lived copy of the frozen E47G Qwen72 judge.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER="$ROOT_DIR/ops/slurm/e47_qwen72_node105.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/e49_math_strategy_qwen72_v1"
RECORD="$OUTPUT/server_job.json"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT"
if [[ -f "$RECORD" ]]; then
  echo "E49 Qwen72 job record already exists: $RECORD" >&2
  exit 2
fi

job_id="$(
  sbatch --parsable \
    --job-name=e49-qwen72 \
    --time=1-00:00:00 \
    --output="$OUTPUT/server-%j.out" \
    --error="$OUTPUT/server-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,OAT_ZERO_QWEN72_OUTPUT=$OUTPUT,OAT_ZERO_QWEN72_PORT=8767" \
    "$SERVER"
)"
printf '{"job_id":"%s","port":8767,"time_limit":"1-00:00:00"}\n' \
  "$job_id" >"$RECORD"
echo "submitted E49 Qwen72 judge job $job_id"
