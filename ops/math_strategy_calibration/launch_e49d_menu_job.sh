#!/usr/bin/env bash
# Submit one durable CPU-only E49D menu materialization job.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
stage="${1:-}"
case "$stage" in
  toy|full) ;;
  *)
    echo "Usage: $0 {toy|full}" >&2
    exit 2
    ;;
esac
OUTPUT="$ROOT_DIR/var/artifacts/e49d_math_strategy_menu_${stage}_v1"
RECORD="$OUTPUT/materialization_job.json"
SLURM="$ROOT_DIR/ops/slurm/e49d_materialize_menus_node302.slurm"

mkdir -p "$OUTPUT"
if [[ -f "$RECORD" ]]; then
  echo "E49D ${stage} menu job record already exists: $RECORD" >&2
  exit 2
fi
job_id="$(
  sbatch --parsable \
    --job-name="e49d-menu-${stage}" \
    --output="$OUTPUT/materialize-%j.out" \
    --error="$OUTPUT/materialize-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49D_STAGE=$stage" \
    "$SLURM"
)"
python - "$RECORD" "$job_id" "$stage" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e49d_menu_job_v1",
    "job_id": sys.argv[2],
    "stage": sys.argv[3],
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "submitted E49D ${stage} menu job $job_id"
