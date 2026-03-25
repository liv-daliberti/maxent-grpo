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
    Lightweight registry used by the baseline GRPO trainer.
``maxent_grpo.rewards.maxent``
    Re-exports the richer reward/statistics helpers used inside the MaxEnt runner.
"""

from __future__ import annotations

import sys

from .basic import (
    RewardConfig,
    RewardFunction,
    accuracy_reward,
    binary_code_reward,
    boxed_accuracy_reward_math,
    get_missing_boxed_answer_penalty_reward,
    seed_paper_boxed_accuracy_reward_math,
    _canon_math,
    _extract_content,
    _answer_pat,
    format_reward,
    get_reward_funcs,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    pure_accuracy_math_correctness,
    pure_accuracy_reward_math,
    python_unit_test_reward,
    reasoning_steps_reward,
    tag_count_reward,
    uses_pure_accuracy_math_reward,
)

__all__ = [
    "RewardConfig",
    "RewardFunction",
    "accuracy_reward",
    "binary_code_reward",
    "boxed_accuracy_reward_math",
    "get_missing_boxed_answer_penalty_reward",
    "seed_paper_boxed_accuracy_reward_math",
    "_canon_math",
    "_extract_content",
    "_answer_pat",
    "format_reward",
    "get_reward_funcs",
    "get_code_format_reward",
    "get_cosine_scaled_reward",
    "get_repetition_penalty_reward",
    "len_reward",
    "pure_accuracy_math_correctness",
    "pure_accuracy_reward_math",
    "python_unit_test_reward",
    "reasoning_steps_reward",
    "tag_count_reward",
    "uses_pure_accuracy_math_reward",
]

# Ensure a top-level "rewards" import resolves to this module.
sys.modules["rewards"] = sys.modules[__name__]
