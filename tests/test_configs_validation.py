"""
Validation tests for configuration dataclasses in ``src/maxent_grpo/config``.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from maxent_grpo.config import (
    DatasetConfig,
    DatasetMixtureConfig,
    GRPOConfig,
    ScriptArguments,
)


def test_dataset_mixture_rejects_invalid_test_split():
    with pytest.raises(ValueError):
        DatasetMixtureConfig(
            datasets=[DatasetConfig(id="a")],
            test_split_size=1.0,
        )


def test_script_arguments_require_dict_like_mixture():
    with pytest.raises(ValueError):
        ScriptArguments(dataset_name=None, dataset_mixture="not-a-dict")


def test_script_arguments_require_dataset_source():
    with pytest.raises(ValueError):
        ScriptArguments(dataset_name=None, dataset_mixture=None)


def test_script_arguments_validate_dataset_payload_shape():
    with pytest.raises(ValueError):
        ScriptArguments(dataset_name=None, dataset_mixture={"seed": 1})


def test_script_arguments_require_dataset_list_entries():
    with pytest.raises(ValueError):
        ScriptArguments(dataset_name=None, dataset_mixture={"datasets": {"id": "a"}})


def test_script_arguments_reject_column_mismatches():
    mixture = {
        "datasets": [
            {"id": "a", "columns": ["prompt", "answer"]},
            {"id": "b", "columns": ["prompt", "solution"]},
        ]
    }
    with pytest.raises(ValueError):
        ScriptArguments(dataset_name=None, dataset_mixture=mixture)


def test_script_arguments_convert_mixture_to_dataclass():
    mixture = {
        "datasets": [
            {"id": "a", "columns": ["prompt", "answer"], "weight": 0.5},
            {"id": "b", "columns": ["prompt", "answer"], "weight": 0.5},
        ],
        "seed": 123,
        "test_split_size": 0.25,
    }
    args = ScriptArguments(dataset_name=None, dataset_mixture=mixture)
    assert isinstance(args.dataset_mixture, DatasetMixtureConfig)
    assert args.dataset_mixture.seed == 123
    assert args.dataset_mixture.test_split_size == 0.25
    assert len(args.dataset_mixture.datasets) == 2


def test_dataset_module_prefers_trl_script_arguments_when_present(monkeypatch):
    trl_stub = SimpleNamespace(
        ScriptArguments=type("ScriptArguments", (), {}),
        GRPOConfig=type("GRPOConfig", (), {}),
    )
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    mod = importlib.reload(importlib.import_module("maxent_grpo.config.dataset"))
    assert mod._BaseScriptArgs is trl_stub.ScriptArguments


def test_grpo_config_sets_eval_strategy_alias(monkeypatch):
    # Ensure the IntervalStrategy branch is exercised even when transformers is absent.
    training_args_mod = ModuleType("transformers.training_args")

    class _IntervalStrategy(str):
        def __new__(cls, value):
            return str.__new__(cls, value)

    training_args_mod.IntervalStrategy = _IntervalStrategy
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.training_args", training_args_mod)

    cfg = GRPOConfig(evaluation_strategy="steps")
    assert isinstance(getattr(cfg, "eval_strategy"), _IntervalStrategy)
    assert str(getattr(cfg, "eval_strategy")) == "steps"


def test_grpo_config_accepts_maxent_knobs():
    cfg = GRPOConfig(
        maxent_tau=0.3,
        maxent_q_temperature=0.7,
        maxent_clip_range=0.1,
        maxent_target_weight_entropy=1.2,
    )
    assert cfg.maxent_tau == 0.3
    assert cfg.maxent_q_temperature == 0.7
    assert cfg.maxent_clip_range == 0.1
    assert cfg.maxent_target_weight_entropy == 1.2


def test_grpo_config_validates_tau_bounds():
    with pytest.raises(ValueError):
        GRPOConfig(maxent_tau_min=0.5, maxent_tau_max=0.1)


def test_grpo_config_rejects_non_positive_q_epsilon():
    with pytest.raises(ValueError):
        GRPOConfig(maxent_q_epsilon=0.0)


def test_grpo_config_rejects_negative_kl_and_tau_lr():
    with pytest.raises(ValueError):
        GRPOConfig(kl_target=-0.1)
    with pytest.raises(ValueError):
        GRPOConfig(maxent_tau_lr=-0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maxent_tau_warmup_steps": -2},
        {"maxent_q_temperature": 0.0},
        {"maxent_logprob_chunk_size": -1},
        {"maxent_clip_objective_coef": -0.5},
        {"maxent_clip_range": -0.1},
        {"kl_horizon": -1},
        {"kl_ctl_step_size": -0.2},
    ],
)
def test_grpo_config_rejects_other_negative_maxent_and_kl_knobs(kwargs):
    with pytest.raises(ValueError):
        GRPOConfig(**kwargs)
