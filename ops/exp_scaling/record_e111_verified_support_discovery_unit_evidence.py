#!/usr/bin/env python3
"""Record production tests against E111's exact immutable source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e102_full_open_bank_maxent_replay as shared  # noqa: E402
import launch_e111_verified_support_discovery_mechanism_gate_three_scale as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "var/artifacts/e111_verified_support_discovery_unit_tests.json"
PRODUCTION_PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
PRODUCTION_LIB = ROOT / "var/seed_paper_eval/paper310/lib"
TESTS = (
    "tests/test_semantic_shannon.py",
    "tests/test_semantic_shannon_quality_gated.py",
    "tests/test_semantic_shannon_separate_advantage.py",
    "tests/test_semantic_shannon_success_conditioned_signed.py",
    "tests/test_semantic_shannon_group_centered.py",
    "tests/test_semantic_shannon_group_centered_contract.py",
    "tests/test_semantic_shannon_group_centered_theory.py",
    "tests/test_semantic_shannon_verified_support.py",
    "tests/test_semantic_shannon_verified_support_contract.py",
    "tests/test_online_canonical_bank.py",
    "tests/test_e105_policy_gradient_direction.py",
    "tests/test_args.py",
    "tests/test_e103_starvation_fallback.py",
    "tests/test_e111_verified_support_discovery_mechanism_gate.py",
    "tests/test_e38_semantic_shannon_05b_contract.py",
    "tests/test_e41_semantic_shannon_advantage_05b_contract.py",
    "tests/test_e43_success_conditioned_signed_semantic_shannon_05b_contract.py",
)
EXPECTED_SUMMARY = "260 passed"
SNAPSHOT_SOURCES = (
    "src/oat_drgrpo/args.py",
    "src/oat_drgrpo/semantic_shannon.py",
    "src/oat_drgrpo/online_canonical_bank.py",
    "src/oat_drgrpo/learner/init.py",
    "src/oat_drgrpo/learner/grpo.py",
    "ops/train.sh",
    "ops/run_experiment.sh",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_evidence(
    payload: dict[str, Any],
    *,
    snapshot: Path,
) -> list[str]:
    violations: list[str] = []
    if payload.get("schema") != "e111_verified_support_discovery_unit_tests_v1":
        violations.append("unit-test evidence schema mismatch")
    if payload.get("snapshot_root") != str(snapshot):
        violations.append("unit-test evidence names a different snapshot")
    if payload.get("passed") is not True or payload.get("returncode") != 0:
        violations.append("snapshot unit tests did not pass")
    if payload.get("outcomes_used") is not False:
        violations.append("unit-test evidence used outcome results")
    if payload.get("pointmaze") != "excluded":
        violations.append("unit-test evidence included PointMaze")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot_root.resolve()
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
        "schema": "e111_verified_support_discovery_unit_tests_v1",
        "snapshot_root": str(snapshot),
        "outcomes_used": False,
        "pointmaze": "excluded",
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
    print(f"[e111-unit] passed={not violations} evidence={OUT}")
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
