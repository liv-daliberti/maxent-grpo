from __future__ import annotations

import hashlib

import pytest

from audit_constructive_code_checker_equivalence import (
    MANIFEST_SCHEMA,
    REPLAY_SCHEMA,
)
from audit_constructive_code_dual_suite_admission import (
    build_dual_suite_admission,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(suite_id: str):
    return {
        "schema_version": MANIFEST_SCHEMA,
        "selection": {
            "language": "py3",
            "order": "exact SHA-256 ascending",
            "correct_per_task": 20,
            "incorrect_per_task": 20,
        },
        "execution_limits": {"cpu": 3},
        "runtime": {"image": _sha("image")},
        "testlib": {"revision": _sha("testlib")},
        "tasks": [
            {
                "problem_key": "Codeforces:1_A:Example",
                "source_problem_id": "1_A",
                "task_adapter": "fixed_integer_sequence_v1",
                "witness_family": "ordered_sequence",
                "suites": [
                    {
                        "suite_id": suite_id,
                        "checker_sha256": _sha("checker"),
                        "required_correct_replays": 20,
                        "required_incorrect_replays": 20,
                        "verified_threshold": 0.9,
                    }
                ],
            }
        ],
    }


def _replays(suite_id: str, incorrect_accepts: int = 0):
    records = []
    for index in range(20):
        records.append(
            {
                "schema_version": REPLAY_SCHEMA,
                "problem_key": "Codeforces:1_A:Example",
                "suite_id": suite_id,
                "submission_sha256": _sha(f"correct-{index}"),
                "checker_sha256": _sha("checker"),
                "known_label": "correct",
                "released_checker_accepted": True,
                "wrapper_accepted": True,
                "behavior_key": f"constructive_behavior:v1:{_sha(f'key-{index % 2}')}",
            }
        )
    for index in range(20):
        accepted = index < incorrect_accepts
        records.append(
            {
                "schema_version": REPLAY_SCHEMA,
                "problem_key": "Codeforces:1_A:Example",
                "suite_id": suite_id,
                "submission_sha256": _sha(f"incorrect-{index}"),
                "checker_sha256": _sha("checker"),
                "known_label": "incorrect",
                "released_checker_accepted": accepted,
                "wrapper_accepted": accepted,
                "behavior_key": (
                    f"constructive_behavior:v1:{_sha(f'bad-key-{index}') }"
                    if accepted
                    else None
                ),
            }
        )
    return records


def test_dual_suite_admits_task_only_when_both_suites_pass():
    overlay_id = "codecontests_o_corner_cases_v1"
    plus_id = "codecontests_plus_5x_v1"

    report = build_dual_suite_admission(
        _manifest(overlay_id),
        _replays(overlay_id),
        _manifest(plus_id),
        _replays(plus_id),
    )

    assert report["status"] == "complete"
    assert report["eligible_task_count"] == 1
    assert report["tasks"][0]["status"] == "eligible"


def test_dual_suite_excludes_task_on_one_suite_threshold_failure():
    overlay_id = "codecontests_o_corner_cases_v1"
    plus_id = "codecontests_plus_5x_v1"

    report = build_dual_suite_admission(
        _manifest(overlay_id),
        _replays(overlay_id),
        _manifest(plus_id),
        _replays(plus_id, incorrect_accepts=3),
    )

    assert report["eligible_task_count"] == 0
    assert report["tasks"][0]["status"] == "ineligible"
    assert report["tasks"][0]["failures"] == [
        {
            "suite_id": plus_id,
            "failed_checks": ["verified_true_negative_rate"],
        }
    ]


def test_dual_suite_rejects_cross_suite_selection_drift():
    overlay_id = "codecontests_o_corner_cases_v1"
    plus_id = "codecontests_plus_5x_v1"
    plus_replays = _replays(plus_id)
    plus_replays[0]["submission_sha256"] = _sha("replacement")

    with pytest.raises(ValueError, match="selection drift"):
        build_dual_suite_admission(
            _manifest(overlay_id),
            _replays(overlay_id),
            _manifest(plus_id),
            plus_replays,
        )
