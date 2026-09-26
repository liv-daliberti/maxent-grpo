from __future__ import annotations

import copy
import hashlib

from audit_constructive_code_checker_equivalence import (
    MANIFEST_SCHEMA,
    REPLAY_SCHEMA,
    build_checker_equivalence_audit,
)
from audit_constructive_code_v2 import REQUIRED_SUITE_IDS, build_v2_gate_audit
from materialize_constructive_code_v3 import V3_SPLIT_ASSIGNMENT, V3_TASKS


import pytest as _pytest

from conftest import reload_constructive_code_v2_bases as _reload_v2_bases


@_pytest.fixture(autouse=True)
def _pristine_constructive_code_v2_bases():
    """Keep this module's v2 assertions independent of test execution order."""

    _reload_v2_bases()
    yield


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _single_task_fixture():
    problem_id = "327_B"
    family, adapter = V3_TASKS[problem_id]
    problem_key = f"Codeforces:{problem_id}:title"
    checker = "a" * 64
    suites = [
        {
            "suite_id": suite_id,
            "suite_sha256": _sha(suite_id),
            "checker_sha256": checker,
            "test_count": 1,
            "required_correct_replays": 100,
            "required_incorrect_replays": 100,
            "verified_threshold": 0.9,
        }
        for suite_id in sorted(REQUIRED_SUITE_IDS)
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "selection": {
            "languages": ["py3", "pypy3", "python3"],
            "order": "exact code SHA-256 ascending within known label",
            "correct_per_task": 100,
            "incorrect_per_task": 100,
        },
        "preflight": {"identity_and_hash_checks": "pass"},
        "tasks": [
            {
                "source_problem_id": problem_id,
                "problem_key": problem_key,
                "witness_family": family,
                "task_adapter": adapter,
                "suites": suites,
            }
        ],
    }
    replays = []
    for suite in suites:
        suite_id = suite["suite_id"]
        for label in ("correct", "incorrect"):
            for index in range(100):
                accepted = label == "correct"
                replays.append(
                    {
                        "schema_version": REPLAY_SCHEMA,
                        "problem_key": problem_key,
                        "source_problem_id": problem_id,
                        "task_adapter": adapter,
                        "witness_family": family,
                        "suite_id": suite_id,
                        "suite_sha256": suite["suite_sha256"],
                        "checker_sha256": checker,
                        "submission_sha256": _sha(f"{label}:{index}"),
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
    return manifest, replays


def _v3_audit(manifest, replays):
    equivalence = build_checker_equivalence_audit(manifest, replays)
    return build_v2_gate_audit(
        manifest,
        replays,
        equivalence,
        expected_tasks={"327_B": V3_TASKS["327_B"]},
        required_suite_ids=REQUIRED_SUITE_IDS,
        suite_policy="at_least_one",
        schema_version="constructive-code-v3-gate-audit-v1",
    )


def test_v3_split_has_one_of_each_family_per_disjoint_split():
    assert set(V3_SPLIT_ASSIGNMENT) == set(V3_TASKS)
    assert len(V3_TASKS) == 12
    for split in ("train", "development", "evaluation"):
        ids = [
            problem_id
            for problem_id, assigned in V3_SPLIT_ASSIGNMENT.items()
            if assigned == split
        ]
        assert len(ids) == 4
        assert {V3_TASKS[problem_id][0] for problem_id in ids} == {
            "ordered_sequence",
            "unordered_set",
            "assignment",
            "unordered_partition",
        }


def test_v3_falls_back_to_plus_when_overlay_only_misses_tnr():
    manifest, replays = _single_task_fixture()
    changed = copy.deepcopy(replays)
    overlay = "codecontests_o_corner_cases_v2"
    incorrect = [
        row
        for row in changed
        if row["suite_id"] == overlay and row["known_label"] == "incorrect"
    ]
    for row in incorrect[:20]:
        row["released_checker_accepted"] = True
        row["wrapper_accepted"] = True
        row["behavior_key"] = "constructive_behavior:v1:" + "f" * 64
        row["execution"]["first_failure"] = None
    audit = _v3_audit(manifest, changed)
    assert audit["status"] == "pass"
    task = audit["task_results"][0]
    assert task["selected_suite_id"] == "codecontests_plus_5x_v2"
    assert task["passing_suite_ids"] == ["codecontests_plus_5x_v2"]


def test_v3_fallback_cannot_waive_wrapper_checker_disagreement():
    manifest, replays = _single_task_fixture()
    changed = copy.deepcopy(replays)
    target = next(
        row
        for row in changed
        if row["suite_id"] == "codecontests_o_corner_cases_v2"
    )
    target["wrapper_accepted"] = not target["released_checker_accepted"]
    audit = _v3_audit(manifest, changed)
    assert audit["status"] == "fail"
    assert "suite_executable_gate_failed" in {
        row["code"] for row in audit["violations"]
    }
