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

from types import SimpleNamespace

import pytest

from maxent_grpo.rewards import (
    _canon_math,
    _extract_content,
    get_reward_funcs,
    pure_accuracy_reward_math,
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
    assert rewards == [1.0, 0.0]


def test_pure_accuracy_reward_math_handles_missing_answer(monkeypatch):
    # Force the answer extractor to fail even when the format regex matches.
    monkeypatch.setattr(
        "rewards._answer_pat", type("Pat", (), {"search": lambda *_a, **_k: None})()
    )
    comps = ["<think>t</think><answer>missing</answer>"]
    rewards = pure_accuracy_reward_math(comps, ["42"])
    assert rewards == [0.0]


def test_get_reward_funcs_resolves_known_names():
    args = SimpleNamespace(reward_funcs=["pure_accuracy_math"])
    funcs = get_reward_funcs(args)
    assert funcs[0] is pure_accuracy_reward_math
    with pytest.raises(KeyError):
        get_reward_funcs(SimpleNamespace(reward_funcs=["unknown"]))


def test_pure_accuracy_reward_math_missing_answer_via_basic(monkeypatch):
    import maxent_grpo.rewards.basic as basic

    monkeypatch.setattr(
        basic,
        "_answer_pat",
        type("Pat", (), {"search": lambda *_a, **_k: None})(),
    )
    comps = ["<think>trace</think><answer>?</answer>"]
    rewards = basic.pure_accuracy_reward_math(comps, ["42"])
    assert rewards == [0.0]


def test_pure_accuracy_reward_math_relaxed_eval_allows_missing_think():
    import maxent_grpo.rewards.basic as basic

    comps = ["<answer>7</answer>"]
    rewards_train = basic.pure_accuracy_reward_math(comps, ["7"])
    rewards_eval = basic.pure_accuracy_reward_math(comps, ["7"], is_eval=True)
    assert rewards_train == [0.0]
    assert rewards_eval == [1.0]
