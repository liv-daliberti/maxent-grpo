"""Branch coverage for advantage grouping in training.rewards."""

from __future__ import annotations

import maxent_grpo.training.rewards as rewards_mod


def test_group_advantages_handles_empty_groups():
    grouped, samples = rewards_mod.group_advantages([[], [1.0, 2.0]], [0.0, 1.0, 2.0])
    assert grouped[0] == []
    assert len(samples) == len([val for group in grouped for val in group])
