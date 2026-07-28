#!/usr/bin/env bash
# Unseal E69 MATH-500 only after the complete frozen Gate 3 audit is clean.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
AUDIT="$ROOT_DIR/var/artifacts/e69_gate3_confirmatory_audit_latest.json"

cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
  "$PYTHON_BIN" ops/route_successor/audit_e69_gate3_confirmatory.py

"$PYTHON_BIN" - "$AUDIT" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = audit.get("summary", {})
if audit.get("schema") != "e69_gate3_confirmatory_audit_v1":
    raise SystemExit("E69 Gate 4 advance requires the exact Gate 3 audit schema")
if audit.get("status") != "complete":
    raise SystemExit(
        "E69 automatic advance stopped: frozen Gate 3 status is "
        f"{audit.get('status')!r}"
    )
if summary.get("terminal_physical_runs") != 30:
    raise SystemExit("E69 automatic advance requires 30 terminal Gate 3 runs")
if summary.get("integrity_violations") != 0:
    raise SystemExit("E69 automatic advance requires zero integrity violations")
if audit.get("math500_sealed") is not True:
    raise SystemExit("E69 automatic advance requires the MATH-500 seal")
print("[e69-advance] Gate 3 is complete and clean; unsealing Gate 4")
PY

bash ops/route_successor/launch_e69_gate4_math500.sh full
