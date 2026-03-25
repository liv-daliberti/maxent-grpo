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

Additional tests for rewards module coverage gaps.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import maxent_grpo.rewards as rewards


def test_canon_math_none_returns_empty():
    assert rewards._canon_math(None) == ""


def test_get_reward_funcs_requires_known_name():
    cfg = SimpleNamespace(reward_funcs=["pure_accuracy_math"])
    funcs = rewards.get_reward_funcs(cfg)
    assert len(funcs) == 1
    assert funcs[0] is rewards.pure_accuracy_reward_math
    with pytest.raises(KeyError):
        rewards.get_reward_funcs(SimpleNamespace(reward_funcs=["does_not_exist"]))


def test_open_r1_reward_aliases_resolve():
    cfg = SimpleNamespace(
        reward_funcs=[
            "accuracy",
            "boxed_accuracy_math",
            "seed_paper_boxed_accuracy_math",
            "format",
            "tag_count",
        ],
        cosine_min_value_wrong=-1.0,
        cosine_max_value_wrong=-0.5,
        cosine_min_value_correct=0.5,
        cosine_max_value_correct=1.0,
        cosine_max_len=1000,
        repetition_n_grams=3,
        repetition_max_penalty=-1.0,
        code_language="python",
    )
    funcs = rewards.get_reward_funcs(cfg)
    assert len(funcs) == 5
    assert callable(funcs[0])
    assert callable(funcs[1])
    assert callable(funcs[2])
    assert callable(funcs[3])
    assert callable(funcs[4])


def test_open_r1_math_alias_behavior():
    completions = [
        [{"role": "assistant", "content": "<think>\n...\n</think>\n<answer>\n42\n</answer>"}],
        [{"role": "assistant", "content": "<think>\n...\n</think>\n<answer>\n41\n</answer>"}],
    ]
    answers = ["42", "42"]
    fmt = rewards.format_reward(completions)
    tag = rewards.tag_count_reward(completions)
    acc = rewards.accuracy_reward(completions, answers)
    assert fmt == [1.0, 1.0]
    assert tag == [1.0, 1.0]
    assert acc == [1.0, 0.0]
