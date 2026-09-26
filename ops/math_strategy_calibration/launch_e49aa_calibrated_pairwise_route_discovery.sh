#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
PROTOCOL="$ROOT_DIR/paper/preregistration/e49aa_calibrated_pairwise_observed_route_discovery_20260726.md"
E49Y_RESULT="$ROOT_DIR/var/artifacts/e49y_observed_route_discovery_v1/result.json"
OUTPUT="$ROOT_DIR/var/artifacts/e49aa_calibrated_pairwise_observed_route_discovery_v1"
JOB_RECORD="$ROOT_DIR/var/artifacts/e49aa_calibrated_pairwise_route_discovery_job.txt"

for required in "$PYTHON_BIN" "$PROTOCOL" "$E49Y_RESULT"; do
  if [[ ! -e "$required" ]]; then
    echo "E49AA prerequisite is missing: $required" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT/result.json" ]]; then
  echo "Fresh E49AA output required: $OUTPUT/result.json" >&2
  exit 2
fi
"$PYTHON_BIN" - "$E49Y_RESULT" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    result.get("schema") != "e49y_observed_route_discovery_v1"
    or result.get("pass") is True
):
    raise SystemExit("E49AA activates only after final E49Y failure")
PY

job_id="$(
  sbatch --parsable \
    "$ROOT_DIR/ops/slurm/e49aa_calibrated_pairwise_route_discovery_node302.slurm"
)"
printf '%s\n' "$job_id" >"$JOB_RECORD"
echo "submitted E49AA calibrated pairwise discovery job $job_id"
