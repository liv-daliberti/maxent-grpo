from __future__ import annotations

import copy
import hashlib

from audit_constructive_code_checker_equivalence import (
    MANIFEST_SCHEMA,
    REPLAY_SCHEMA,
    build_checker_equivalence_audit,
)
from audit_constructive_code_v2 import (
    EXPECTED_TASKS,
    REQUIRED_SUITE_IDS,
    build_v2_gate_audit,
)


import pytest as _pytest

from conftest import reload_constructive_code_v2_bases as _reload_v2_bases


@_pytest.fixture(autouse=True)
def _pristine_constructive_code_v2_bases():
    """Keep this module's v2 assertions independent of test execution order."""

    _reload_v2_bases()
    yield


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fixture():
    checker = "a" * 64
    tasks = []
    replays = []
    for problem_id, (family, adapter) in EXPECTED_TASKS.items():
        problem_key = f"Codeforces:{problem_id}:title"
        tasks.append(
            {
                "source_problem_id": problem_id,
                "problem_key": problem_key,
                "witness_family": family,
                "task_adapter": adapter,
                "suites": [
                    {
                        "suite_id": suite_id,
                        "suite_sha256": _sha(f"{problem_id}:{suite_id}"),
                        "checker_sha256": checker,
                        "test_count": 1,
                        "required_correct_replays": 100,
                        "required_incorrect_replays": 100,
                        "verified_threshold": 0.9,
                    }
                    for suite_id in sorted(REQUIRED_SUITE_IDS)
                ],
            }
        )
        for suite_id in REQUIRED_SUITE_IDS:
            suite_sha = _sha(f"{problem_id}:{suite_id}")
            for label in ("correct", "incorrect"):
                for index in range(100):
                    submission_sha = _sha(f"{problem_id}:{label}:{index}")
                    accepted = label == "correct"
                    replays.append(
                        {
                            "schema_version": REPLAY_SCHEMA,
                            "problem_key": problem_key,
                            "source_problem_id": problem_id,
                            "task_adapter": adapter,
                            "witness_family": family,
                            "suite_id": suite_id,
                            "suite_sha256": suite_sha,
                            "checker_sha256": checker,
                            "submission_sha256": submission_sha,
                            "known_label": label,
                            "released_checker_accepted": accepted,
                            "wrapper_accepted": accepted,
                            "behavior_key": (
                                f"constructive_behavior:v1:{index % 2:064x}"
                                if accepted
                                else None
                            ),
                            "execution": {
                                "candidate_invocation_wall_seconds": [0.1],
                                "candidate_wall_seconds": 0.1,
                                "checker_wall_seconds": 0.01,
                                "executed_tests": 1,
                                "suite_tests": 1,
                                "first_failure": (
                                    None
                                    if accepted
                                    else {
                                        "stage": "released_checker",
                                        "timed_out": False,
                                    }
                                ),
                            },
                        }
                    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "selection": {
            "languages": ["py3", "pypy3", "python3"],
            "order": "exact code SHA-256 ascending within known label",
            "correct_per_task": 100,
            "incorrect_per_task": 100,
        },
        "preflight": {"identity_and_hash_checks": "pass"},
        "tasks": tasks,
    }
    return manifest, replays


def _audit(manifest, replays):
    equivalence = build_checker_equivalence_audit(manifest, replays)
    return build_v2_gate_audit(manifest, replays, equivalence)


def test_v2_gate_passes_complete_dual_suite_fixture():
    manifest, replays = _fixture()
    audit = _audit(manifest, replays)
    assert audit["status"] == "pass"
    assert audit["observed_replay_count"] == 1600
    assert all(row["status"] == "pass" for row in audit["task_results"])


def test_v2_gate_rejects_cross_suite_submission_drift():
    manifest, replays = _fixture()
    changed = copy.deepcopy(replays)
    target = next(
        row
        for row in changed
        if row["suite_id"] == "codecontests_plus_5x_v2"
        and row["source_problem_id"] == "327_B"
        and row["known_label"] == "correct"
    )
    target["submission_sha256"] = _sha("substituted")
    audit = _audit(manifest, changed)
    assert audit["status"] == "fail"
    assert "cross_suite_submission_set_mismatch" in {
        row["code"] for row in audit["violations"]
    }


def test_v2_gate_rejects_replay_suite_hash_drift():
    manifest, replays = _fixture()
    changed = copy.deepcopy(replays)
    changed[0]["suite_sha256"] = _sha("wrong-suite")
    audit = _audit(manifest, changed)
    assert audit["status"] == "fail"
    assert "replay_identity_or_hash_mismatch" in {
        row["code"] for row in audit["violations"]
    }


def test_v2_gate_rejects_timeout_and_latency_violation():
    manifest, replays = _fixture()
    changed = copy.deepcopy(replays)
    target = changed[0]
    same_suite = [
        row
        for row in changed
        if row["problem_key"] == target["problem_key"]
        and row["suite_id"] == target["suite_id"]
    ]
    for row in same_suite[:20]:
        row["execution"]["candidate_invocation_wall_seconds"] = [1.1]
        row["execution"]["candidate_wall_seconds"] = 1.1
    target["execution"]["first_failure"] = {
        "stage": "candidate",
        "timed_out": True,
        "sandbox_violation": False,
        "output_limited": False,
    }
    audit = _audit(manifest, changed)
    assert audit["status"] == "fail"
    failed = [
        row
        for row in audit["suite_results"]
        if row["problem_key"] == target["problem_key"]
        and row["suite_id"] == target["suite_id"]
    ][0]
    assert not failed["checks"]["zero_timeout_violations"]
    assert not failed["checks"]["p95_candidate_execution"]
