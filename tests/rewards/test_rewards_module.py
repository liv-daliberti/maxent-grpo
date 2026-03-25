"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Unit tests for rewards module helpers.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

from maxent_grpo.rewards import (
    _canon_math,
    _extract_content,
    get_reward_funcs,
    get_missing_boxed_answer_penalty_reward,
    pure_accuracy_math_correctness,
    pure_accuracy_reward_math,
    seed_paper_boxed_accuracy_reward_math,
    uses_pure_accuracy_math_reward,
)


def test_extract_content_handles_shapes():
    assert _extract_content("text") == "text"
    assert _extract_content([{"content": "hi"}]) == "hi"
    assert _extract_content({"content": "raw"}) == "{'content': 'raw'}"
    assert _extract_content(None) == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("{42}", "42"),
        ("(  42.0 )", "42"),
        ("-0", "0"),
        ("+0", "0"),
        (" 3 .0 ", "3"),
        ("(\\sqrt{4})", "(\\sqrt{4})"),
        ("2 1", "21"),
    ],
)
def test_canon_math_normalizes(raw, expected):
    assert _canon_math(raw) == expected


def test_pure_accuracy_reward_math_checks_tags_and_canon():
    comps = [
        "<think>...</think><answer> 42 </answer>",
        "<think>no</think><answer>-0</answer>",
        "missing tags",
        "<think>no answer tag</think>",
    ]
    rewards = pure_accuracy_reward_math(comps, ["42", "0", "0", "x"])
    assert rewards == [1.0, 1.0, 0.0, 0.0]


def test_pure_accuracy_reward_math_handles_multiple_rewards():
    completions = [
        "<think>t</think><answer>1</answer>",
        "<think>t</think><answer>2</answer>",
    ]
    rewards = pure_accuracy_reward_math(completions, ["1", "3"])
    assert rewards == [1.0, 0.05]


def test_pure_accuracy_reward_math_allows_outer_whitespace():
    completions = [
        "\n  <think>t</think><answer>1</answer>  ",
        "\n<think>t</think><answer>2</answer>\n",
    ]
    rewards = pure_accuracy_reward_math(completions, ["1", "3"])
    assert rewards == [1.0, 0.05]


def test_pure_accuracy_reward_math_handles_missing_answer(monkeypatch):
    # Force the answer extractor to fail even when the format regex matches.
    monkeypatch.setattr(
        "rewards._answer_pat", type("Pat", (), {"search": lambda *_a, **_k: None})()
    )
    comps = ["<think>t</think><answer>missing</answer>"]
    rewards = pure_accuracy_reward_math(comps, ["42"])
    assert rewards == [0.05]


def test_get_reward_funcs_resolves_known_names():
    args = SimpleNamespace(reward_funcs=["pure_accuracy_math"])
    funcs = get_reward_funcs(args)
    assert funcs[0] is pure_accuracy_reward_math
    with pytest.raises(KeyError):
        get_reward_funcs(SimpleNamespace(reward_funcs=["unknown"]))


def test_get_reward_funcs_resolves_seed_paper_reward() -> None:
    args = SimpleNamespace(reward_funcs=["seed_paper_boxed_accuracy_math"])
    funcs = get_reward_funcs(args)
    assert funcs[0] is seed_paper_boxed_accuracy_reward_math


def test_get_reward_funcs_resolves_missing_boxed_answer_penalty() -> None:
    args = SimpleNamespace(
        reward_funcs=["missing_boxed_answer_penalty_math"],
        missing_boxed_answer_penalty=-0.125,
    )
    funcs = get_reward_funcs(args)
    rewards = funcs[0]([r"Work \boxed{1}", "No final answer"])
    assert rewards == [0.0, -0.125]


def test_missing_boxed_answer_penalty_reward_clamps_positive_values() -> None:
    reward_fn = get_missing_boxed_answer_penalty_reward(0.3)
    assert reward_fn(["still missing"]) == [0.0]


def test_pure_accuracy_reward_math_missing_answer_via_basic(monkeypatch):
    import maxent_grpo.rewards.basic as basic

    monkeypatch.setattr(
        basic,
        "_answer_pat",
        type("Pat", (), {"search": lambda *_a, **_k: None})(),
    )
    comps = ["<think>trace</think><answer>?</answer>"]
    rewards = basic.pure_accuracy_reward_math(comps, ["42"])
    assert rewards == [0.05]


def test_pure_accuracy_reward_math_relaxed_eval_allows_missing_think():
    import maxent_grpo.rewards.basic as basic

    comps = ["<answer>7</answer>"]
    rewards_train = basic.pure_accuracy_reward_math(comps, ["7"])
    rewards_eval = basic.pure_accuracy_reward_math(comps, ["7"], is_eval=True)
    assert rewards_train == [0.25]
    assert rewards_eval == [0.25]


def test_pure_accuracy_reward_math_applies_tag_multipliers():
    import maxent_grpo.rewards.basic as basic

    comps = [
        "42",  # 0 tags
        "<think>42",  # 1 tag
        "<answer>42</answer>",  # 2 tags
        "<think><think><answer>42</answer>",  # missing </think> (3 unique tags)
        "<think>...</think><answer>42</answer>",  # 4 tags (override)
        "<think>...</think><answer>42</answer><answer>extra</answer>",  # >4 tags
    ]
    rewards = basic.pure_accuracy_reward_math(comps, ["42"] * len(comps))
    assert rewards == [0.05, 0.125, 0.25, 0.375, 1.0, 0.25]


def test_pure_accuracy_math_correctness_ignores_format_bonus():
    comps = [
        "<answer>7</answer>",
        "<think>t</think><answer>8</answer>",
        "scratch\n7",
    ]
    flags = pure_accuracy_math_correctness(comps, ["7", "7", "7"])
    assert flags == [True, False, False]


def test_pure_accuracy_math_correctness_optional_last_line_fallback():
    comps = ["scratch\n7"]
    flags = pure_accuracy_math_correctness(
        comps,
        ["7"],
        allow_last_line_fallback=True,
    )
    assert flags == [True]


def test_uses_pure_accuracy_math_reward_unwraps_partial() -> None:
    wrapped = functools.partial(pure_accuracy_reward_math)
    assert uses_pure_accuracy_math_reward([wrapped]) is True


def test_uses_pure_accuracy_math_reward_unwraps_decorated_function() -> None:
    @functools.wraps(pure_accuracy_reward_math)
    def wrapped(*args, **kwargs):
        return pure_accuracy_reward_math(*args, **kwargs)

    assert uses_pure_accuracy_math_reward([wrapped]) is True
