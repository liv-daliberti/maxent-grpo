#!/usr/bin/env python3
"""Record group-16 sparse-success theory tests against the frozen snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / (
    "var/artifacts/source_snapshots/e106_python_lambda_b853595e3b158046"
)
SNAPSHOT_SHA256 = (
    "b853595e3b158046f73dee899a2c2a0558d4167e5b6971a36aa50565d9337d02"
)
TEST = ROOT / "tests/test_e105_sparse_success_theory.py"
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
LIBRARY_PATH = ROOT / "var/seed_paper_eval/paper310/lib"
OUT = ROOT / "var/artifacts/e105_sparse_success_theory_tests.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    command = [str(PYTHON), "-m", "pytest", "-q", str(TEST)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SNAPSHOT / "src")
    environment["LD_LIBRARY_PATH"] = str(LIBRARY_PATH)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "schema": "e105_sparse_success_theory_tests_v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "passed": result.returncode == 0 and "2 passed" in result.stdout,
        "returncode": result.returncode,
        "command": command,
        "snapshot_root": str(SNAPSHOT),
        "snapshot_sha256": SNAPSHOT_SHA256,
        "test": str(TEST.relative_to(ROOT)),
        "test_sha256": digest(TEST),
        "implementation_sha256": {
            "src/oat_drgrpo/semantic_shannon.py": digest(
                SNAPSHOT / "src/oat_drgrpo/semantic_shannon.py"
            )
        },
        "group_size": 16,
        "sparse_success_scenarios": [
            {"failure_mass": 0.90, "successful_mode_masses": [0.08, 0.02]},
            {
                "failure_mass": 0.88,
                "successful_mode_masses": [0.07, 0.035, 0.015],
            },
        ],
        "post_e104_or_e106_update_outcomes_inspected": False,
        "pointmaze": "excluded",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

