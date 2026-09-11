from __future__ import annotations

import hashlib

from audit_constructive_code_checker_equivalence import (
    MANIFEST_SCHEMA,
    REPLAY_SCHEMA,
    build_checker_equivalence_audit,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest():
    return {
        "schema_version": MANIFEST_SCHEMA,
        "tasks": [
            {
                "problem_key": "Codeforces:1_A",
                "witness_family": "ordered_sequence",
                "suites": [
                    {
                        "suite_id": "codecontests_plus_5x",
                        "checker_sha256": _sha("checker"),
                        "required_correct_replays": 2,
                        "required_incorrect_replays": 2,
                        "verified_threshold": 0.5,
                    }
                ],
            }
        ],
    }


def _replay(index, label, accepted, key_index=None):
    behavior_key = None
    if accepted:
        behavior_key = f"constructive_behavior:v1:{_sha(f'key-{key_index}')}"
    return {
        "schema_version": REPLAY_SCHEMA,
        "problem_key": "Codeforces:1_A",
        "suite_id": "codecontests_plus_5x",
        "submission_sha256": _sha(f"submission-{index}"),
        "checker_sha256": _sha("checker"),
        "known_label": label,
        "released_checker_accepted": accepted,
        "wrapper_accepted": accepted,
        "behavior_key": behavior_key,
    }


def test_equivalence_gate_passes_complete_matching_multi_witness_replay():
    replays = [
        _replay(0, "correct", True, 0),
        _replay(1, "correct", True, 1),
        _replay(2, "incorrect", False),
        _replay(3, "incorrect", False),
    ]

    audit = build_checker_equivalence_audit(_manifest(), replays)

    assert audit["status"] == "pass"
    assert audit["violations"] == []
    assert audit["suite_results"][0]["counts"] == {
        "correct": 2,
        "incorrect": 2,
        "true_positive": 2,
        "true_negative": 2,
        "distinct_correct_behavior_keys": 2,
    }


def test_equivalence_gate_fails_closed_on_wrapper_mismatch_and_missing_key():
    replays = [
        _replay(0, "correct", True, 0),
        {
            **_replay(1, "correct", True, 1),
            "wrapper_accepted": False,
            "behavior_key": None,
        },
        _replay(2, "incorrect", False),
        _replay(3, "incorrect", False),
    ]

    audit = build_checker_equivalence_audit(_manifest(), replays)
    codes = {violation["code"] for violation in audit["violations"]}

    assert audit["status"] == "fail"
    assert "wrapper_released_checker_mismatch" in codes


def test_equivalence_gate_rejects_duplicate_replay_and_low_coverage():
    first = _replay(0, "correct", True, 0)
    replays = [
        first,
        first,
        _replay(2, "incorrect", False),
    ]

    audit = build_checker_equivalence_audit(_manifest(), replays)
    codes = {violation["code"] for violation in audit["violations"]}

    assert audit["status"] == "fail"
    assert "duplicate_submission_replay" in codes
    assert "suite_correct_replay_count_failed" in codes
    assert "suite_incorrect_replay_count_failed" in codes
    assert "suite_multiple_behavior_keys_failed" in codes
