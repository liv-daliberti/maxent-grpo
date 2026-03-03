"""Ensure maxent reward shim re-exports expected helpers."""

from __future__ import annotations

import maxent_grpo.rewards.maxent as maxent_mod
import maxent_grpo.training.rewards as training_rewards


def test_maxent_module_reexports_training_rewards():
    # the thin shim should expose the same reward helpers as the core
    # ``training.rewards`` package.
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


def test_maxent_module_reexports_weighting_helpers():
    # MaxEnt utilities should also expose the public interface of the
    # ``training.weighting`` package.  We just check a few representative
    # names; the weighting package has its own tests so we don't repeat them
    # here.
    for name in (
        "compute_weight_stats",
        "weight_vector_from_q",
        "WeightingSettings",
        "build_uniform_weight_stats",
    ):
        assert hasattr(maxent_mod, name), f"{name} missing from maxent exports"
