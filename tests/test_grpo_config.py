from __future__ import annotations

import sys
from types import SimpleNamespace

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

    monkeypatch.setitem(
        sys.modules,
        "transformers.training_args",
        SimpleNamespace(IntervalStrategy=_Interval),
    )
    cfg2 = GRPOConfig(evaluation_strategy=_Interval("epoch"))
    assert isinstance(cfg2.eval_strategy, _Interval)
    assert str(cfg2.eval_strategy) == "epoch"


def test_grpo_script_arguments_defaults():
    args = GRPOScriptArguments(dataset_name="dummy")
    assert args.reward_funcs == ["pure_accuracy_math"]
    assert args.dataset_prompt_column == "problem"


def test_grpo_config_validates_tau_bounds():
    with pytest.raises(ValueError):
        GRPOConfig(maxent_tau_min=-0.1)
