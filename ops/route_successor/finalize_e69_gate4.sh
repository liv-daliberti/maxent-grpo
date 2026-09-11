#!/usr/bin/env bash
# Analyze the immutable E69 Gate 4 outputs and render the five-area panel.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
ANALYSIS="$ROOT_DIR/var/artifacts/e69_gate4_math500_analysis.json"
IDENTITY="$ROOT_DIR/var/artifacts/e69_gate4_math500_identity.json"
ANALYZER="$ROOT_DIR/ops/route_successor/analyze_e69_gate4_math500.py"
PLOTTER="$ROOT_DIR/ops/route_successor/plot_e69_five_area_panel.py"
FINALIZER_SLURM="$ROOT_DIR/ops/slurm/e69_gate4_finalize.slurm"

cd "$ROOT_DIR"
"$PYTHON_BIN" - \
  "$IDENTITY" "$ANALYZER" "$PLOTTER" "$0" "$FINALIZER_SLURM" <<'PY'
import hashlib
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if identity.get("schema") != "e69_gate4_math500_one_time_transfer_v1":
    raise SystemExit("E69 finalizer requires the exact Gate 4 identity schema")
for field, raw_path in (
    ("analyzer_sha256", sys.argv[2]),
    ("plotter_sha256", sys.argv[3]),
    ("finalizer_sha256", sys.argv[4]),
    ("finalizer_slurm_sha256", sys.argv[5]),
):
    observed = hashlib.sha256(pathlib.Path(raw_path).read_bytes()).hexdigest()
    if identity.get(field) != observed:
        raise SystemExit(f"E69 Gate 4 analysis surface drift: {field}")
PY
PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
  "$PYTHON_BIN" "$ANALYZER"

"$PYTHON_BIN" - "$ANALYSIS" <<'PY'
import json
import pathlib
import sys

analysis = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = analysis.get("summary", {})
if analysis.get("schema") != "e69_gate4_math500_analysis_v1":
    raise SystemExit("E69 finalizer requires the exact Gate 4 analysis schema")
if analysis.get("status") != "complete":
    raise SystemExit(
        "E69 finalizer stopped: Gate 4 analysis status is "
        f"{analysis.get('status')!r}"
    )
if summary.get("observed_results") != 6:
    raise SystemExit("E69 finalizer requires all six immutable results")
if summary.get("integrity_violations") != 0:
    raise SystemExit("E69 finalizer requires zero integrity violations")
if analysis.get("final_classification") is None:
    raise SystemExit("E69 finalizer requires a frozen final classification")
print("[e69-finalize] Gate 4 is complete and clean; rendering paper panel")
PY

PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
  "$PYTHON_BIN" "$PLOTTER"
