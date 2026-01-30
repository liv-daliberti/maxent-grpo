"""Ensure maxent reward shim re-exports expected helpers."""

from __future__ import annotations

import maxent_grpo.rewards.maxent as maxent_mod
import maxent_grpo.training.rewards as training_rewards


def test_maxent_module_reexports_training_rewards():
    assert (
        maxent_mod.compute_reward_statistics
        is training_rewards.compute_reward_statistics
    )
    for name in (
        "AggregatedGenerationState",
        "compute_reward_totals",
        "group_advantages",
        "prepare_generation_batch",
        "reward_moments",
    ):
        assert hasattr(maxent_mod, name)
