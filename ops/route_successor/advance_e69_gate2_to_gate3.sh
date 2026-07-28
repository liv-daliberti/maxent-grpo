#!/usr/bin/env bash
# Advance E69 only when the complete frozen Gate 2 audit passes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
AUDIT="$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_audit_latest.json"

cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
  "$PYTHON_BIN" ops/route_successor/audit_e69_gate2_screen.py
PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
  "$PYTHON_BIN" ops/route_successor/snapshot_e69_gate2_route_replay.py
PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
  "$PYTHON_BIN" ops/route_successor/audit_e69_gate2_screen.py

"$PYTHON_BIN" - "$AUDIT" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = audit.get("summary", {})
if audit.get("status") != "pass":
    raise SystemExit(
        "E69 automatic advance stopped: frozen Gate 2 status is "
        f"{audit.get('status')!r}"
    )
if summary.get("terminal_physical_runs") != 18:
    raise SystemExit("E69 automatic advance requires 18 terminal Gate 2 runs")
if summary.get("integrity_violations") != 0:
    raise SystemExit("E69 automatic advance requires zero integrity violations")
if audit.get("math500_sealed") is not True:
    raise SystemExit("E69 automatic advance requires the MATH-500 seal")
print("[e69-advance] Gate 2 passed cleanly; launching Gate 3")
PY

bash ops/route_successor/launch_e69_gate3_confirmatory.sh full
