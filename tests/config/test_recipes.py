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

Unit tests for recipe loading helpers.
"""

from __future__ import annotations

import pytest

from maxent_grpo.config.recipes import (
    _MaxentRecipeSchema,
    _dataclass_field_names,
    _split_recipe_payload,
    load_grpo_recipe,
)
from maxent_grpo.config.grpo import GRPOScriptArguments, GRPOConfig


def test_dataclass_field_names():
    fields = _dataclass_field_names(GRPOScriptArguments)
    assert "reward_funcs" not in fields
    assert "cosine_max_len" in fields


def test_split_recipe_payload_routes_fields():
    class _ModelCfg:
        __dataclass_fields__ = {"model_name_or_path": None, "trust_remote_code": None}

    payload = {
        "reward_funcs": ["foo"],
        "beta": 0.2,
        "model_name_or_path": "m",
        "trust_remote_code": True,
        "extra": "x",
    }
    script, training, model, other = _split_recipe_payload(payload, _ModelCfg)
    assert script == {}
    assert training == {"reward_funcs": ["foo"], "beta": 0.2}
    assert model == {"model_name_or_path": "m", "trust_remote_code": True}
    assert other == {"extra": "x"}


def test_load_grpo_recipe_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "recipe.yaml"
    path.write_text(
        "\n".join(
            [
                "reward_funcs: ['r1']",
                "maxent_tau: 0.3",
                "dataset_name: ds",
                "model_name_or_path: repo/model",
                "objective: maxent_listwise",
                "output_dir: /tmp/out",
                "logging_steps: 10",
                "save_steps: 10",
                "beta: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    class _ModelCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    args, cfg, model_cfg = load_grpo_recipe(str(path), model_config_cls=_ModelCfg)
    assert isinstance(args, GRPOScriptArguments)
    assert isinstance(cfg, GRPOConfig)
    assert isinstance(model_cfg, _ModelCfg)
    assert cfg.reward_funcs == ["r1"]
    assert cfg.maxent_tau == 0.3
    assert model_cfg.kwargs["model_name_or_path"] == "repo/model"


def test_load_grpo_recipe_accepts_maxent_alpha_without_tau(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(
        "\n".join(
            [
                "reward_funcs: ['r1']",
                "maxent_alpha: 0.01",
                "dataset_name: ds",
                "model_name_or_path: repo/model",
                "objective: maxent_entropy",
                "output_dir: /tmp/out",
                "logging_steps: 10",
                "save_steps: 10",
                "beta: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    class _ModelCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    _, cfg, _ = load_grpo_recipe(str(path), model_config_cls=_ModelCfg)
    assert cfg.maxent_alpha == 0.01


def test_load_grpo_recipe_accepts_listwise_variant_without_alpha(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(
        "\n".join(
            [
                "reward_funcs: ['r1']",
                "objective: maxent_listwise",
                "maxent_tau: 0.2",
                "dataset_name: ds",
                "model_name_or_path: repo/model",
                "output_dir: /tmp/out",
                "logging_steps: 10",
                "save_steps: 10",
                "beta: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    class _ModelCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    _, cfg, _ = load_grpo_recipe(str(path), model_config_cls=_ModelCfg)
    assert cfg.maxent_objective_variant == "listwise"
    assert cfg.maxent_tau == 0.2


def test_maxent_recipe_schema_requires_alpha_or_tau_for_maxent():
    with pytest.raises(
        ValueError,
        match=(
            "maxent_alpha \\(or legacy maxent_tau\\) is required when "
            "objective=maxent_entropy"
        ),
    ):
        _MaxentRecipeSchema(
            reward_funcs=["r1"],
            dataset_name="ds",
            model_name_or_path="repo/model",
            objective="maxent_entropy",
            output_dir="/tmp/out",
            logging_steps=10,
            save_steps=10,
            beta=0.5,
        )


def test_maxent_recipe_schema_allows_listwise_variant_without_alpha() -> None:
    schema = _MaxentRecipeSchema(
        reward_funcs=["r1"],
        dataset_name="ds",
        model_name_or_path="repo/model",
        objective="maxent_listwise",
        maxent_tau=0.2,
        output_dir="/tmp/out",
        logging_steps=10,
        save_steps=10,
        beta=0.5,
    )
    assert schema.objective == "maxent_listwise"


def test_maxent_recipe_schema_requires_positive_tau_for_listwise() -> None:
    with pytest.raises(ValueError, match="maxent_tau > 0"):
        _MaxentRecipeSchema(
            reward_funcs=["r1"],
            dataset_name="ds",
            model_name_or_path="repo/model",
            objective="maxent_listwise",
            maxent_tau=0.0,
            output_dir="/tmp/out",
            logging_steps=10,
            save_steps=10,
            beta=0.5,
        )


def test_maxent_recipe_schema_allows_grpo_entropy_bonus_without_listwise_knobs() -> None:
    schema = _MaxentRecipeSchema(
        reward_funcs=["r1"],
        dataset_name="ds",
        model_name_or_path="repo/model",
        objective="grpo_entropy_bonus",
        policy_entropy_bonus_coef=0.1,
        output_dir="/tmp/out",
        logging_steps=10,
        save_steps=10,
        beta=0.5,
    )
    assert schema.policy_entropy_bonus_coef == pytest.approx(0.1)


def test_maxent_recipe_schema_rejects_listwise_knobs_for_grpo_entropy_bonus() -> None:
    with pytest.raises(ValueError, match="native GRPO loss"):
        _MaxentRecipeSchema(
            reward_funcs=["r1"],
            dataset_name="ds",
            model_name_or_path="repo/model",
            objective="grpo_entropy_bonus",
            policy_entropy_bonus_coef=0.1,
            maxent_tau=0.2,
            output_dir="/tmp/out",
            logging_steps=10,
            save_steps=10,
            beta=0.5,
        )


def test_maxent_recipe_schema_rejects_zero_bonus_for_grpo_entropy_bonus() -> None:
    with pytest.raises(ValueError, match="requires policy_entropy_bonus_coef > 0"):
        _MaxentRecipeSchema(
            reward_funcs=["r1"],
            dataset_name="ds",
            model_name_or_path="repo/model",
            objective="grpo_entropy_bonus",
            output_dir="/tmp/out",
            logging_steps=10,
            save_steps=10,
            beta=0.5,
        )


def test_maxent_recipe_schema_rejects_listwise_alpha() -> None:
    with pytest.raises(ValueError, match="does not use maxent_alpha"):
        _MaxentRecipeSchema(
            reward_funcs=["r1"],
            dataset_name="ds",
            model_name_or_path="repo/model",
            objective="maxent_listwise",
            maxent_tau=0.2,
            maxent_alpha=0.1,
            output_dir="/tmp/out",
            logging_steps=10,
            save_steps=10,
            beta=0.5,
        )
