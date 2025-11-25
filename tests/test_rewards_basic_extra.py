"""Additional coverage for rewards.basic helpers."""

from __future__ import annotations

import pytest

from maxent_grpo.rewards import basic as basic_rewards


def test_get_reward_funcs_unknown_name_raises():
    with pytest.raises(KeyError):
        basic_rewards.get_reward_funcs(
            type("Args", (), {"reward_funcs": ["does-not-exist"]})()
        )


def test_canon_math_strips_wrappers_and_trailing_zeros():
    assert basic_rewards._canon_math("{123.0}") == "123"
