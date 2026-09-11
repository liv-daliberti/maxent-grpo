#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULT="$ROOT_DIR/var/artifacts/e49y_compiled_bottom_up_route_calibration_v1/result.json"
DIAGNOSTIC="$ROOT_DIR/var/artifacts/e49w_local_contract_diagnostic_v1.json"
if [[ -e "$RESULT" ]]; then
  echo "E49Y result already exists: $RESULT" >&2
  exit 1
fi
if [[ ! -f "$DIAGNOSTIC" ]]; then
  echo "E49W parser diagnostic is missing: $DIAGNOSTIC" >&2
  exit 1
fi
"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - \
  "$DIAGNOSTIC" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    result.get("schema") != "e49w_local_contract_diagnostic_v1"
    or result.get("local_contract_pass") is not False
    or result.get("local_contract_failures")
    != ["menu_parse:ValueError:strategy action combos must be distinct"]
):
    raise SystemExit("E49W parser diagnosis does not authorize E49Y")
PY

job_id="$(sbatch --parsable \
  "$ROOT_DIR/ops/slurm/e49y_compiled_route_calibration_a6000.slurm")"
printf '%s\n' "$job_id" > \
  "$ROOT_DIR/var/artifacts/e49y_compiled_route_calibration_job.txt"
printf 'submitted E49Y compiled route calibration job %s\n' "$job_id"
