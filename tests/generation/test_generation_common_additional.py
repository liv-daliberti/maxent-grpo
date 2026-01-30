"""Additional coverage for generation.common helpers."""

from __future__ import annotations

from maxent_grpo.generation import common


def test_append_completion_group_initializes_meta_and_truncates():
    grouped = [[] for _ in range(2)]
    grouped_meta = None
    # meta_group longer than completions should be truncated.
    grouped_meta = common.append_completion_group(
        grouped_comps=grouped,
        grouped_meta=grouped_meta,
        prompt_idx=0,
        completions=["a", "b"],
        meta_group=["m1", "m2", "extra"],
    )
    assert grouped[0] == ["a", "b"]
    assert grouped_meta is not None
    assert grouped_meta[0] == ["m1", "m2"]
    # Subsequent append with no meta should pad existing meta list.
    grouped_meta = common.append_completion_group(
        grouped_comps=grouped,
        grouped_meta=grouped_meta,
        prompt_idx=0,
        completions=["c"],
        meta_group=None,
    )
    assert grouped_meta[0][-1] is None
