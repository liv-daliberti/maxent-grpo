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

Validation tests for configuration dataclasses in ``src/maxent_grpo/config``.
"""

from dataclasses import MISSING
from pathlib import Path

import pytest
import yaml

from maxent_grpo.config import (
    DatasetConfig,
    DatasetMixtureConfig,
    GRPOConfig,
    GRPOScriptArguments,
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


def test_grpo_config_accepts_maxent_knobs():
    cfg = GRPOConfig(
        objective="maxent_listwise",
        maxent_tau=0.3,
        maxent_q_temperature=0.7,
        maxent_clip_range=0.1,
        maxent_target_weight_entropy=1.2,
    )
    assert cfg.objective == "maxent_listwise"
    assert cfg.maxent_objective_variant == "listwise"
    assert cfg.maxent_tau == 0.3
    assert cfg.maxent_q_temperature == 0.7
    assert cfg.maxent_clip_range == 0.1
    assert cfg.maxent_target_weight_entropy == 1.2


@pytest.mark.parametrize(
    ("raw_objective", "expected_objective", "expected_variant"),
    [
        ("baseline", "grpo", "entropy"),
        ("entropy", "maxent_entropy", "entropy"),
        ("listwise", "maxent_listwise", "listwise"),
    ],
)
def test_grpo_config_normalizes_objective_aliases(
    raw_objective: str,
    expected_objective: str,
    expected_variant: str,
) -> None:
    kwargs = {"objective": raw_objective}
    if raw_objective == "listwise":
        kwargs["maxent_tau"] = 0.3
    cfg = GRPOConfig(**kwargs)
    assert cfg.objective == expected_objective
    assert cfg.maxent_objective_variant == expected_variant


def test_grpo_config_validates_tau_bounds():
    with pytest.raises(ValueError):
        GRPOConfig(maxent_tau_min=0.5, maxent_tau_max=0.1)


def test_grpo_config_requires_positive_tau_for_listwise_objective() -> None:
    with pytest.raises(ValueError, match="maxent_tau > 0"):
        GRPOConfig(
            objective="maxent_listwise",
            maxent_tau=0.0,
        )


def test_grpo_config_enables_policy_entropy_when_bonus_is_active() -> None:
    cfg = GRPOConfig(
        objective="grpo_entropy_bonus",
        policy_entropy_bonus_coef=0.2,
        maxent_policy_entropy=False,
    )
    assert cfg.maxent_policy_entropy is True


def test_grpo_config_requires_exact_entropy_for_entropy_regularized_maxent() -> None:
    with pytest.raises(
        ValueError, match="requires maxent_policy_entropy_mode='exact'"
    ):
        GRPOConfig(
            objective="maxent_entropy",
            maxent_alpha=0.2,
            maxent_policy_entropy_mode="sample",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maxent_reward_signal_gate": True},
        {"maxent_bonus_positive_only": True},
        {"maxent_cusp_gate": True},
    ],
)
def test_grpo_config_rejects_unsupported_reward_shaping_knobs(kwargs) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        GRPOConfig(**kwargs)


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
        {"objective": "unknown"},
        {"maxent_logprob_chunk_size": -1},
        {"maxent_clip_objective_coef": -0.5},
        {"maxent_clip_range": -0.1},
        {"maxent_reference_ema_beta": -0.1},
        {"maxent_reference_ema_beta": 1.1},
        {"maxent_reference_ema_update_interval": 0},
        {"maxent_reference_ema_warmup_steps": -1},
        {"kl_horizon": -1},
        {"kl_ctl_step_size": -0.2},
    ],
)
def test_grpo_config_rejects_other_negative_maxent_and_kl_knobs(kwargs):
    with pytest.raises(ValueError):
        GRPOConfig(**kwargs)


def _dataclass_default(owner: type[object], field_name: str):
    field_obj = owner.__dataclass_fields__[field_name]
    if field_obj.default is not MISSING:
        return field_obj.default
    if field_obj.default_factory is not MISSING:
        return field_obj.default_factory()
    raise KeyError(field_name)


def _recipe_value(payload: dict[str, object], field_name: str, owner: type[object]):
    if field_name in payload:
        return payload[field_name]
    return _dataclass_default(owner, field_name)


@pytest.mark.parametrize(
    ("grpo_rel", "maxent_rel"),
    [
        (
            "configs/recipes/Qwen2.5-0.5B-Instruct/grpo/config_math.yaml",
            "configs/recipes/Qwen2.5-0.5B-Instruct/maxent-grpo/config_math.yaml",
        ),
        (
            "configs/recipes/Qwen2.5-0.5B-Instruct/grpo/config_code_mbpp.yaml",
            "configs/recipes/Qwen2.5-0.5B-Instruct/maxent-grpo/config_code_mbpp.yaml",
        ),
        (
            "configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml",
            "configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml",
        ),
    ],
)
def test_paired_recipes_keep_sampling_reference_and_kl_settings_aligned(
    grpo_rel: str,
    maxent_rel: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_name in (
        "GRPO_RECIPE_USED",
        "MAXENT_VLLM_URL",
        "VLLM_URL",
        "MAXENT_VLLM_MODE",
        "VLLM_MODE",
        "MAXENT_VLLM_SYNC_WEIGHTS",
        "MAXENT_RUN_NAME",
        "WANDB_RUN_NAME",
        "MAXENT_WANDB_RUN_GROUP",
        "WANDB_RUN_GROUP",
        "MAXENT_LOG_LEVEL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    repo = Path(__file__).resolve().parents[2]
    with (repo / grpo_rel).open("r", encoding="utf-8") as handle:
        grpo_payload = yaml.safe_load(handle)
    with (repo / maxent_rel).open("r", encoding="utf-8") as handle:
        maxent_payload = yaml.safe_load(handle)
    assert isinstance(grpo_payload, dict)
    assert isinstance(maxent_payload, dict)

    for field in (
        "dataset_name",
        "dataset_prompt_column",
        "eval_dataset_name",
        "eval_dataset_split",
        "eval_dataset_prompt_column",
        "eval_dataset_solution_column",
    ):
        assert _recipe_value(maxent_payload, field, GRPOScriptArguments) == _recipe_value(
            grpo_payload, field, GRPOScriptArguments
        ), field

    for field in (
        "prompt_template",
        "system_prompt",
        "use_vllm",
        "vllm_mode",
        "vllm_return_logprobs",
        "vllm_request_logprobs",
        "vllm_sync_weights",
        "vllm_sync_interval_steps",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "gen_temperature",
        "gen_top_p",
        "gen_top_k",
        "gen_best_of",
        "max_prompt_length",
        "max_completion_length",
        "max_steps",
        "num_generations",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "learning_rate",
        "optim",
        "adam_beta2",
        "adam_epsilon",
        "lr_scheduler_type",
        "reward_funcs",
        "reward_weights",
        "beta",
        "kl_target",
        "kl_horizon",
        "kl_ctl_step_size",
        "clip_range",
        "max_grad_norm",
        "warmup_ratio",
        "maxent_reference_logprobs_source",
        "maxent_trl_reference_scoring",
        "behavior_logprobs_source",
        "maxent_length_normalize_ref",
        "maxent_policy_entropy_mode",
        "policy_entropy_bonus_coef",
    ):
        assert _recipe_value(maxent_payload, field, GRPOConfig) == _recipe_value(
            grpo_payload, field, GRPOConfig
        ), field

    assert _recipe_value(maxent_payload, "objective", GRPOConfig) == "maxent_entropy"
    assert _recipe_value(maxent_payload, "maxent_policy_entropy_mode", GRPOConfig) == "exact"
    assert _recipe_value(maxent_payload, "maxent_reference_ema_enabled", GRPOConfig) is False
