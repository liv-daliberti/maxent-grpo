"""
Validation tests for configuration dataclasses in ``src/maxent_grpo/config``.
"""

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


def test_grpo_config_sets_eval_strategy_alias():
    cfg = GRPOConfig(evaluation_strategy="steps")
    assert getattr(cfg, "eval_strategy") == "steps"


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
