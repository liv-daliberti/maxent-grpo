from __future__ import annotations

import hashlib

from materialize_constructive_code_review_slate import (
    SLATE,
    build_overlay_suite,
    select_python_replays,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_review_slate_freezes_four_tasks_in_each_family():
    family_counts = {}
    for family, _adapter in SLATE.values():
        family_counts[family] = family_counts.get(family, 0) + 1

    assert family_counts == {
        "assignment": 4,
        "ordered_sequence": 4,
        "unordered_partition": 4,
        "unordered_set": 4,
    }


def test_python_replay_selection_is_unique_hash_sorted_and_capped():
    submissions = [
        {"language": "cpp", "code": "ignored"},
        {"language": "Py3", "code": "print(3)"},
        {"language": "python3", "code": "print(1)"},
        {"language": "pypy3", "code": "print(2)"},
        {"language": "python", "code": "print(1)"},
        {"language": "python", "code": ""},
    ]

    selected, available = select_python_replays(submissions, limit=2)

    expected_hashes = sorted({_sha("print(1)"), _sha("print(2)"), _sha("print(3)")})
    assert available == 3
    assert [record["submission_sha256"] for record in selected] == expected_hashes[:2]


def test_overlay_suite_identity_binds_order_hash_and_byte_count():
    records, digest = build_overlay_suite(["1 2\n", "é\n"])
    repeated, repeated_digest = build_overlay_suite(["1 2\n", "é\n"])
    reversed_records, reversed_digest = build_overlay_suite(["é\n", "1 2\n"])

    assert records == repeated
    assert digest == repeated_digest
    assert digest != reversed_digest
    assert records[0]["input_sha256"] == _sha("1 2\n")
    assert records[1]["input_bytes"] == 3
    assert reversed_records[0]["test_index"] == 0
