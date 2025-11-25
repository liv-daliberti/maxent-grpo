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
