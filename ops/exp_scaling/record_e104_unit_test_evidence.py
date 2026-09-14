#!/usr/bin/env python3
"""Record production tests against E104's exact immutable source snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e102_full_open_bank_maxent_replay as shared  # noqa: E402
import launch_e104_group_centered_semantic_repair_three_scale as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "var/artifacts/e104_group_centered_semantic_repair_unit_tests.json"
PRODUCTION_PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
PRODUCTION_LIB = ROOT / "var/seed_paper_eval/paper310/lib"
TESTS = (
    "tests/test_args.py",
    "tests/test_semantic_shannon.py",
    "tests/test_semantic_shannon_group_centered.py",
    "tests/test_semantic_shannon_group_centered_theory.py",
)
EXPECTED_SUMMARY = "155 passed"
SNAPSHOT_SOURCES = (
    "src/oat_drgrpo/args.py",
    "src/oat_drgrpo/semantic_shannon.py",
    "src/oat_drgrpo/learner/init.py",
    "src/oat_drgrpo/learner/grpo.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_evidence(
    payload: dict[str, Any],
    *,
    snapshot: Path,
) -> list[str]:
    violations: list[str] = []
    if payload.get("schema") != "e104_snapshot_unit_test_evidence_v1":
        violations.append("unit-test evidence schema mismatch")
    if payload.get("snapshot_root") != str(snapshot):
        violations.append("unit-test evidence names a different snapshot")
    if payload.get("passed") is not True or payload.get("returncode") != 0:
        violations.append("snapshot unit tests did not pass")
    if EXPECTED_SUMMARY not in str(payload.get("stdout", "")):
        violations.append(f"snapshot unit tests lack {EXPECTED_SUMMARY!r}")
    expected_tests = {
        relative: digest(ROOT / relative) for relative in TESTS
    }
    if payload.get("test_sha256") != expected_tests:
        violations.append("snapshot unit-test source digest mismatch")
    expected_sources = {
        relative: digest(snapshot / relative) for relative in SNAPSHOT_SOURCES
    }
    if payload.get("snapshot_source_sha256") != expected_sources:
        violations.append("tested snapshot source digest mismatch")
    return violations


def main() -> int:
    ledger_path = ROOT / launch.LEDGER
    if not ledger_path.is_file():
        raise SystemExit(f"E104 ledger is absent: {ledger_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    snapshot = Path(str(ledger["snapshot_root"]))
    launch.verify_snapshot(snapshot)
    command = [
        str(PRODUCTION_PYTHON),
        "-m",
        "pytest",
        "-q",
        *TESTS,
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(snapshot / "src")
    existing_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = str(PRODUCTION_LIB) + (
        f":{existing_library_path}" if existing_library_path else ""
    )
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "schema": "e104_snapshot_unit_test_evidence_v1",
        "snapshot_root": str(snapshot),
        "command": command,
        "tests": list(TESTS),
        "test_sha256": {
            relative: digest(ROOT / relative) for relative in TESTS
        },
        "snapshot_source_sha256": {
            relative: digest(snapshot / relative)
            for relative in SNAPSHOT_SOURCES
        },
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0 and EXPECTED_SUMMARY in result.stdout,
    }
    shared.atomic_json(OUT, payload)
    violations = validate_evidence(payload, snapshot=snapshot)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    print(f"[e104-unit] passed={not violations} evidence={OUT}")
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
