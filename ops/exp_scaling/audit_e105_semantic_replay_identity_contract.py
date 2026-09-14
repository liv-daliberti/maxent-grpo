#!/usr/bin/env python3
"""Freeze the E105 semantic/ReplayDr verified-identity contract."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / (
    "var/artifacts/source_snapshots/"
    "e106_python_lambda_b853595e3b158046"
)
SNAPSHOT_SHA256 = (
    "b853595e3b158046f73dee899a2c2a0558d4167e5b6971a36aa50565d9337d02"
)
TEST = ROOT / "tests/test_e105_semantic_replay_identity_contract.py"
OUT = ROOT / "var/artifacts/e105_semantic_replay_identity_contract_tests.json"

sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import audit_e102_full_open_bank_maxent_replay as shared  # noqa: E402
import launch_e106_python_lambda_normalization_three_scale as e106  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    e106.verify_snapshot(ROOT, SNAPSHOT)
    if e106.SNAPSHOT_SHA256 != SNAPSHOT_SHA256:
        raise SystemExit("E106 snapshot authority drifted")
    if not TEST.is_file():
        raise SystemExit(f"semantic/replay identity test is absent: {TEST}")

    command = [sys.executable, "-m", "pytest", "-q", str(TEST)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SNAPSHOT / "src")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "schema": "e105_semantic_replay_identity_contract_tests_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": result.returncode == 0,
        "returncode": int(result.returncode),
        "command": command,
        "python_executable": sys.executable,
        "snapshot_root": str(SNAPSHOT),
        "snapshot_sha256": SNAPSHOT_SHA256,
        "test": str(TEST.relative_to(ROOT)),
        "test_sha256": sha256(TEST),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "implementation_sha256": {
            relative: sha256(SNAPSHOT / relative)
            for relative in (
                "src/oat_drgrpo/learner/grpo.py",
                "src/oat_drgrpo/learner/run.py",
                "src/oat_drgrpo/math_grader.py",
                "src/oat_drgrpo/online_canonical_bank.py",
                "src/oat_drgrpo/semantic_shannon.py",
            )
        },
        "domains": [
            "graph_coloring",
            "countdown",
            "python_factors",
            "mathir",
            "pantry_plan",
        ],
        "group_size": 16,
        "post_e104_or_e106_update_outcomes_inspected": False,
        "pointmaze": "excluded",
        "assertions": {
            "actor_reward_matches_replay_validator_admission": True,
            "semantic_and_replay_verified_keys_match": True,
            "semantic_and_replay_persistent_counts_match": True,
            "parseable_task_failures_are_excluded": True,
            "inactive_verified_rows_are_excluded": True,
            "rare_verified_mode_has_positive_semantic_pressure": True,
            "common_verified_mode_has_negative_semantic_pressure": True,
            "verified_replay_retains_both_modes": True,
            "joint_auto_resume_is_exact": True,
        },
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    shared.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(result.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
