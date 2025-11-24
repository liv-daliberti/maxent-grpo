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

Validation tests for GRPOConfig bounds and aliases.
"""

from __future__ import annotations

import sys

import pytest

from maxent_grpo.config.grpo import GRPOConfig


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("kl_target", -1),
        ("kl_horizon", -1),
        ("kl_ctl_step_size", -0.1),
    ],
)
def test_grpo_config_rejects_negative_kl_params(field_name: str, value: float):
    kwargs = {field_name: value}
    with pytest.raises(ValueError):
        GRPOConfig(**kwargs)


def test_grpo_config_eval_strategy_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers.training_args", None)
    cfg = GRPOConfig(evaluation_strategy="steps")
    assert getattr(cfg, "eval_strategy") == "steps"
