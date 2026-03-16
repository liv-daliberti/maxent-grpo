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
    assert basic_rewards._canon_math(r"Thus the answer is \boxed{42}.") == "42"


def test_pure_accuracy_reward_math_accepts_numeric_gold_labels():
    rewards = basic_rewards.pure_accuracy_reward_math(
        ["<think>steps</think><answer>42</answer>"],
        [42.0],
    )
    assert rewards == [1.0]


def test_accuracy_reward_accepts_boxed_answers_and_list_gold_labels():
    rewards = basic_rewards.accuracy_reward(
        [r"After working it out, the result is \boxed{42}."],
        [["42"]],
    )
    assert rewards == [1.0]


def test_boxed_accuracy_reward_requires_boxed_or_answer_payload():
    rewards = basic_rewards.boxed_accuracy_reward_math(
        [
            r"After working it out, the result is \boxed{42}.",
            "The answer is 42.",
        ],
        ["42", "42"],
    )
    assert rewards == [1.0, 0.0]


def test_truncate_after_first_boxed_answer_stops_before_tail() -> None:
    text = r"Reasoning... \boxed{42} Extra chatter \boxed{0}"
    assert basic_rewards.truncate_after_first_boxed_answer(text) == r"Reasoning... \boxed{42}"
