#!/usr/bin/env bash
# Submit the terminal execution-gated evaluation audit for one E49C stage.
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
OUTPUT="$ROOT_DIR/var/artifacts"
RECORD="$OUTPUT/e49c_${stage}_terminal_execution_audit_job.json"
SLURM="$ROOT_DIR/ops/slurm/e49c_eval_audit_node302.slurm"
if [[ -f "$RECORD" ]]; then
  echo "E49C ${stage} eval-audit job record already exists: $RECORD" >&2
  exit 2
fi
job_id="$(
  sbatch --parsable \
    --job-name="e49c-audit-${stage}" \
    --output="$OUTPUT/e49c-${stage}-audit-%j.out" \
    --error="$OUTPUT/e49c-${stage}-audit-%j.out" \
    --export="ALL,OAT_ZERO_REPO_ROOT=$ROOT_DIR,E49C_STAGE=$stage" \
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
    "schema": "e49c_eval_audit_job_v1",
    "job_id": sys.argv[2],
    "stage": sys.argv[3],
}
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "submitted E49C ${stage} eval-audit job $job_id"
