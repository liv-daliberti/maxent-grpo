#!/usr/bin/env bash
# Start a fresh, identical Qwen72 judge allocation for E49B's full stage.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER="$ROOT_DIR/ops/slurm/e47_qwen72_node105.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/e49_math_strategy_qwen72_continuity_v1"
RECORD="$OUTPUT/server_job.json"
PORT=8768
TIME_LIMIT=1-00:00:00

cd "$ROOT_DIR"
mkdir -p "$OUTPUT"
if [[ -f "$RECORD" ]]; then
  echo "E49 Qwen72 continuity job record already exists: $RECORD" >&2
  exit 2
fi

job_id="$(
  sbatch --parsable \
    --job-name=e49-qwen72b \
    --time="$TIME_LIMIT" \
    --output="$OUTPUT/server-%j.out" \
    --error="$OUTPUT/server-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,OAT_ZERO_QWEN72_OUTPUT=$OUTPUT,OAT_ZERO_QWEN72_PORT=$PORT" \
    "$SERVER"
)"

python - "$RECORD" "$job_id" "$PORT" "$TIME_LIMIT" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "job_id": sys.argv[2],
    "port": int(sys.argv[3]),
    "time_limit": sys.argv[4],
    "purpose": "E49B full-stage judge continuity",
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

echo "submitted E49 Qwen72 continuity judge job $job_id on port $PORT"
