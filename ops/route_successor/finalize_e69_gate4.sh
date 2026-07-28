#!/usr/bin/env bash
# Analyze the immutable E69 Gate 4 outputs and render the five-area panel.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
ANALYSIS="$ROOT_DIR/var/artifacts/e69_gate4_math500_analysis.json"

cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
  "$PYTHON_BIN" ops/route_successor/analyze_e69_gate4_math500.py

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
  "$PYTHON_BIN" ops/route_successor/plot_e69_five_area_panel.py
