"""Unit tests for rewards module helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rewards import (
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


def test_get_reward_funcs_resolves_known_names():
    args = SimpleNamespace(reward_funcs=["pure_accuracy_math"])
    funcs = get_reward_funcs(args)
    assert funcs[0] is pure_accuracy_reward_math
    with pytest.raises(KeyError):
        get_reward_funcs(SimpleNamespace(reward_funcs=["unknown"]))
