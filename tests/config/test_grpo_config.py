from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from maxent_grpo.config.grpo import GRPOConfig, GRPOScriptArguments
from maxent_grpo.config.dataset import trl


def test_grpo_config_eval_alias_sets_eval_strategy(monkeypatch):
    """eval_strategy alias should populate eval_strategy when available."""
    cfg = GRPOConfig(eval_strategy="steps")
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
    cfg2 = GRPOConfig(eval_strategy=_Interval("epoch"))
    assert isinstance(cfg2.eval_strategy, _Interval)
    assert str(cfg2.eval_strategy) == "epoch"


def test_grpo_config_eval_strategy_field_is_parser_friendly():
    field_type = GRPOConfig.__dataclass_fields__["eval_strategy"].type
    assert field_type is not Any


def test_grpo_script_arguments_defaults():
    args = GRPOScriptArguments(dataset_name="dummy")
    assert args.dataset_prompt_column == "problem"


def test_grpo_config_syncs_loss_type_from_grpo_loss_type():
    cfg = GRPOConfig(grpo_loss_type="dr_grpo")
    assert cfg.grpo_loss_type == "dr_grpo"
    assert getattr(cfg, "loss_type", None) == "dr_grpo"


def test_grpo_config_normalizes_dr_grpo_denominator_mode():
    cfg = GRPOConfig(dr_grpo_denominator_mode="tokens")
    assert cfg.dr_grpo_denominator_mode == "active_tokens"


def test_grpo_config_validates_tau_bounds():
    with pytest.raises(ValueError):
        GRPOConfig(maxent_tau_min=-0.1)


def test_grpo_config_rejects_positive_missing_boxed_answer_penalty():
    with pytest.raises(ValueError):
        GRPOConfig(missing_boxed_answer_penalty=0.1)


def test_grpo_config_final_model_save_enabled_defaults_true():
    cfg = GRPOConfig()
    assert cfg.final_model_save_enabled is True


def test_grpo_config_preserves_num_generations_on_divisibility_warning(
    monkeypatch, caplog
):
    """Resume divisibility errors from the base class should not overwrite num_generations."""

    def _failing_post_init(self):
        raise ValueError("num_generations must be divisible by tensor parallelism")

    monkeypatch.setattr(
        trl.GRPOConfig, "__post_init__", _failing_post_init, raising=False
    )
    caplog.set_level("WARNING")

    cfg = GRPOConfig(num_generations=16)

    assert cfg.num_generations == 16
    assert any(
        "Ignoring num_generations divisibility constraint" in record.message
        for record in caplog.records
    )


def test_grpo_config_appends_generate_suffix_for_plain_hosts():
    cfg = GRPOConfig(vllm_url="http://localhost:29525")
    assert cfg.vllm_url == "http://localhost:29525/generate"


def test_grpo_config_preserves_custom_vllm_paths():
    cfg = GRPOConfig(vllm_url="http://localhost:29525/custom/path")
    assert cfg.vllm_url == "http://localhost:29525/custom/path"


def test_grpo_config_preserves_vllm_include_stop_str_setting():
    cfg = GRPOConfig(vllm_include_stop_str_in_output=True)
    assert cfg.vllm_include_stop_str_in_output is True
