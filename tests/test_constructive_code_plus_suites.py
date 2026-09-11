from __future__ import annotations

import pytest

from materialize_constructive_code_plus_suites import (
    _plus_inputs,
    _selected_row_groups,
)


def test_plus_inputs_extracts_only_ordered_input_text():
    row = {
        "test_cases": [
            {"input": "1\n", "output": "secret-a"},
            {"input": "2\n", "output": "secret-b"},
        ]
    }

    assert _plus_inputs(row) == ["1\n", "2\n"]


@pytest.mark.parametrize(
    "cases",
    [None, [], [{}], [{"input": 1}]],
)
def test_plus_inputs_fails_closed_on_malformed_rows(cases):
    with pytest.raises(ValueError):
        _plus_inputs({"test_cases": cases})


def test_selected_row_groups_maps_global_rows_to_local_offsets():
    assert _selected_row_groups([3, 2, 4], [8, 0, 3, 2]) == (
        (0, 0, (0, 2)),
        (1, 3, (0,)),
        (2, 5, (3,)),
    )


@pytest.mark.parametrize("indices", [[], [-1], [9]])
def test_selected_row_groups_fails_closed_on_invalid_indices(indices):
    with pytest.raises(ValueError):
        _selected_row_groups([3, 2, 4], indices)
