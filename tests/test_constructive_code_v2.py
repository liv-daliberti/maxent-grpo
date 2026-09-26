from __future__ import annotations

import hashlib

import pytest

from materialize_constructive_code_v2 import (
    _execution_limits,
    select_heldout_python3,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_v2_selection_is_explicit_py3_disjoint_unique_and_hash_sorted():
    submissions = [
        {"language": "py2", "code": "old"},
        {"language": "python", "code": "ambiguous"},
        {"language": "Py3", "code": "three"},
        {"language": "python3", "code": "one"},
        {"language": "pypy3", "code": "two"},
        {"language": "py3", "code": "three"},
        {"language": "py3", "code": "excluded"},
    ]

    selected, counts = select_heldout_python3(
        submissions,
        {_sha("excluded")},
        limit=2,
    )

    expected = sorted({_sha("one"), _sha("two"), _sha("three")})[:2]
    assert [row["submission_sha256"] for row in selected] == expected
    assert counts == {
        "explicit_python3_records": 5,
        "excluded_v1_records": 1,
        "available_unique_heldout_python3": 3,
        "selected": 2,
    }


def test_v2_selection_fails_closed_on_too_few_heldout_programs():
    with pytest.raises(ValueError, match="held-out Python-3 programs"):
        select_heldout_python3(
            [{"language": "py3", "code": "only"}],
            set(),
            limit=2,
        )


def test_v2_execution_limits_use_sandbox_schema():
    assert _execution_limits({"time_limit": 1000, "memory_limit": 256}) == {
        "time_milliseconds": 1000,
        "memory_megabytes": 256,
    }


@pytest.mark.parametrize(
    "row",
    [
        {"time_limit": None, "memory_limit": 256},
        {"time_limit": True, "memory_limit": 256},
        {"time_limit": 1000.0, "memory_limit": 256},
        {"time_limit": 0, "memory_limit": 256},
        {"time_limit": 1000, "memory_limit": -1},
    ],
)
def test_v2_execution_limits_fail_closed(row):
    with pytest.raises(ValueError, match="invalid positive integer"):
        _execution_limits(row)
