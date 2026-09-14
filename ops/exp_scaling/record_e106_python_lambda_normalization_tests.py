#!/usr/bin/env python3
"""Record production-runtime tests against E106's immutable snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e102_full_open_bank_maxent_replay as shared  # noqa: E402
import launch_e106_python_lambda_normalization_three_scale as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / launch.UNIT_EVIDENCE
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
LIB = ROOT / "var/seed_paper_eval/paper310/lib"
TESTS = (
    "tests/test_python_modebench.py",
    "tests/test_verified_transformations.py",
    "tests/test_semantic_shannon.py",
    "tests/test_semantic_shannon_group_centered.py",
    "tests/test_semantic_shannon_group_centered_contract.py",
    "tests/test_semantic_shannon_group_centered_theory.py",
)
EXPECTED = "53 passed"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    snapshot = (ROOT / launch.SNAPSHOT).resolve()
    launch.verify_snapshot(ROOT, snapshot)
    command = [str(PYTHON), "-m", "pytest", "-q", *[str(ROOT / p) for p in TESTS]]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{snapshot / 'src'}:{snapshot}"
    existing = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = str(LIB) + (f":{existing}" if existing else "")
    result = subprocess.run(
        command,
        cwd=snapshot,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "schema": "e106_python_lambda_normalization_unit_tests_v1",
        "snapshot_root": str(snapshot),
        "snapshot_sha256": launch.SNAPSHOT_SHA256,
        "command": command,
        "tests": list(TESTS),
        "test_sha256": {path: digest(ROOT / path) for path in TESTS},
        "snapshot_source_sha256": {
            "src/oat_drgrpo/math_grader.py": digest(snapshot / "src/oat_drgrpo/math_grader.py"),
            "src/oat_drgrpo/semantic_shannon.py": digest(snapshot / "src/oat_drgrpo/semantic_shannon.py"),
        },
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0 and EXPECTED in result.stdout,
    }
    shared.atomic_json(OUT, payload)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    passed = payload["passed"]
    print(f"[e106-unit] passed={passed} evidence={OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
