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

Reward utilities shared across MaxEnt-GRPO components.

Two layers of rewards are exposed:

``maxent_grpo.rewards.basic``
    Lightweight registry used by the baseline GRPO trainer and inference helpers.
``maxent_grpo.rewards.maxent``
    Re-exports the richer reward/statistics helpers used inside the MaxEnt runner.
"""

from __future__ import annotations

import sys

from .basic import (
    RewardConfig,
    RewardFunction,
    _canon_math,
    _extract_content,
    _answer_pat,
    get_reward_funcs,
    pure_accuracy_reward_math,
)

__all__ = [
    "RewardConfig",
    "RewardFunction",
    "_canon_math",
    "_extract_content",
    "_answer_pat",
    "get_reward_funcs",
    "pure_accuracy_reward_math",
]

# Ensure a top-level "rewards" import resolves to this module.
sys.modules["rewards"] = sys.modules[__name__]
