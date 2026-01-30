"""Coverage for generation.common helpers."""

from __future__ import annotations

from types import SimpleNamespace


import maxent_grpo.generation.common as common


def test_pending_and_retry_limit_branches():
    comps = [["a"], []]
    assert common.pending_generation_indices(comps, expected_generations=1) == [1]
    assert common.pending_generation_indices(comps, expected_generations=0) == []
    assert common.determine_retry_limit(2, max_retry_rounds=None) == 2
    assert common.determine_retry_limit(0, max_retry_rounds=5) == 5
    assert common.determine_retry_limit(0, max_retry_rounds=None) == 3


def test_retry_incomplete_prompts_handles_meta_and_empty():
    prompts = ["p1", "p2"]
    aggregated = common.AggregatedGenerationState([[], []], metadata=[[None], [None]])

    def _gen(subprompts, expected, counts):
        assert subprompts == prompts
        return [["x"], ["y"]], [[SimpleNamespace()], [None]]

    updated = common.retry_incomplete_prompts(prompts, _gen, 1, aggregated, None)
    assert updated.completions == [["x"], ["y"]]
    assert updated.metadata and updated.metadata[0][0] is not None


def test_drop_empty_prompt_groups_and_truncate():
    prompts = ["p1", "p2"]
    answers = ["a1", "a2"]
    comps = [["c1"], []]
    meta = [[None], []]
    stats = {"dropped_prompts": 0}
    p, a, c, m = common.drop_empty_prompt_groups(prompts, answers, comps, meta, stats)
    assert p == ["p1"] and a == ["a1"]
    assert stats["dropped_prompts"] == 1

    comps2 = [["c1", "c2"]]
    meta2 = [[None, None]]
    trimmed, trimmed_meta, partial = common.truncate_to_expected_counts(
        comps2, meta2, expected_generations=1
    )
    assert trimmed == [["c1"]]
    assert partial == 0
    assert trimmed_meta and trimmed_meta[0] == [None]
