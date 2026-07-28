#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
ADVANCEMENT="$ROOT_DIR/var/artifacts/e49t_natural_menu_math_toy_advancement_v1.json"
OUTPUT="$ROOT_DIR/var/data/e49v_exact_oat_natural_menu_full"
MANIFEST="$OUTPUT/MATERIALIZATION_MANIFEST.json"

if [[ ! -f "$ADVANCEMENT" ]]; then
  echo "E49T toy advancement decision is missing: $ADVANCEMENT" >&2
  exit 1
fi
"$PYTHON_BIN" - "$ADVANCEMENT" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    result.get("schema") != "e49t_natural_menu_math_toy_advancement_v1"
    or result.get("complete_evidence") is not True
    or result.get("advance_to_exact_oat_full") is not True
    or not all(result.get("checks", {}).values())
):
    raise SystemExit("E49T toy has not passed every frozen advancement check")
PY

if [[ -e "$MANIFEST" ]]; then
  echo "Fresh E49V materialization required: $MANIFEST" >&2
  exit 1
fi

job_id="$(sbatch --parsable \
  "$ROOT_DIR/ops/slurm/e49v_materialize_exact_oat_node302.slurm")"
printf '%s\n' "$job_id" > \
  "$ROOT_DIR/var/artifacts/e49v_exact_oat_materialization_job.txt"
printf 'submitted E49V exact-OAT materialization job %s\n' "$job_id"
