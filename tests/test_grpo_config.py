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

Unit tests for GRPO config dataclasses.
"""

from __future__ import annotations

import pytest

from maxent_grpo.config.grpo import GRPOConfig, GRPOScriptArguments


def test_grpo_config_eval_alias_sets_eval_strategy(monkeypatch):
    """evaluation_strategy alias should populate eval_strategy when available."""
    cfg = GRPOConfig(evaluation_strategy="steps")
    assert getattr(cfg, "eval_strategy", None) == "steps"

    # When IntervalStrategy is available, ensure conversion uses the enum type
    class _Interval:
        def __init__(self, val):
            self.val = val

        def __str__(self):
            return self.val

    monkeypatch.setattr(
        "maxent_grpo.config.grpo.IntervalStrategy", _Interval, raising=False
    )
    cfg2 = GRPOConfig(evaluation_strategy=_Interval("epoch"))
    assert str(cfg2.eval_strategy) == "epoch"


def test_grpo_script_arguments_defaults():
    args = GRPOScriptArguments(dataset_name="dummy")
    assert args.reward_funcs == ["pure_accuracy_math"]
    assert args.dataset_prompt_column == "problem"


def test_grpo_config_validates_tau_bounds():
    with pytest.raises(ValueError):
        GRPOConfig(maxent_tau_min=-0.1)