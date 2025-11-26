"""
Unit tests for recipe loading helpers.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from maxent_grpo.config import recipes
from maxent_grpo.config.grpo import GRPOConfig, GRPOScriptArguments


def test_split_recipe_payload_routes_fields():
    payload = {
        "dataset_name": "ds",  # script arg
        "maxent_tau_min": 0.2,  # training arg
        "model_name": "m",  # model arg
        "unknown": 123,  # defaults to training
    }
    model_cls = SimpleNamespace(__dataclass_fields__={"model_name": None})
    script_kwargs, training_kwargs, model_kwargs, other_kwargs = (
        recipes._split_recipe_payload(payload, model_cls)
    )
    assert script_kwargs["dataset_name"] == "ds"
    assert training_kwargs["maxent_tau_min"] == 0.2
    assert model_kwargs["model_name"] == "m"
    assert other_kwargs["unknown"] == 123


def test_load_grpo_recipe_with_yaml(monkeypatch, tmp_path):
    recipe_path = tmp_path / "recipe.yml"
    recipe_path.write_text(
        "dataset_name: ds\nmaxent_tau_min: 0.5\nmaxent_tau_max: 1.0\nmodel_name: modelx\n"
    )
    monkeypatch.setenv("GRPO_RECIPE_USED", "preset")
    # force yaml loader path
    monkeypatch.setattr(recipes, "OmegaConf", None)

    # simple model config class
    class _ModelCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    script_args, training_args, model_args = recipes.load_grpo_recipe(
        str(recipe_path), model_config_cls=_ModelCfg
    )
    assert isinstance(script_args, GRPOScriptArguments)
    assert isinstance(training_args, GRPOConfig)
    assert isinstance(model_args, _ModelCfg)
    assert getattr(script_args, "recipe_path") == str(recipe_path)
    assert getattr(training_args, "recipe_path") == str(recipe_path)
    assert getattr(model_args, "recipe_path") == str(recipe_path)
    assert os.environ.get("GRPO_RECIPE_USED") == "preset"


def test_load_grpo_recipe_requires_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(recipes, "OmegaConf", None)
    monkeypatch.setattr(recipes, "yaml", None)
    with pytest.raises(ImportError):
        recipes.load_grpo_recipe(
            str(tmp_path / "missing.yml"), model_config_cls=type("M", (), {})
        )


def test_load_grpo_recipe_rejects_non_mapping(monkeypatch, tmp_path):
    recipe_path = tmp_path / "recipe.yml"
    recipe_path.write_text("list:\n  - 1\n  - 2\n")
    monkeypatch.setattr(recipes, "OmegaConf", None)

    class _YamlStub:
        @staticmethod
        def safe_load(_handle):
            return ["not-a-mapping"]

    monkeypatch.setattr(recipes, "yaml", _YamlStub)
    with pytest.raises(ValueError):
        recipes.load_grpo_recipe(str(recipe_path), model_config_cls=type("M", (), {}))


def test_load_grpo_recipe_sets_recipe_path_best_effort(monkeypatch, tmp_path):
    recipe_path = tmp_path / "recipe.yml"
    recipe_path.write_text("dataset_name: ds\nmodel_name: m\n")
    monkeypatch.setattr(recipes, "OmegaConf", None)

    class _YamlStub:
        @staticmethod
        def safe_load(_handle):
            return {"dataset_name": "ds"}

    class _ModelCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __setattr__(self, name, value):
            if name == "recipe_path":
                raise AttributeError("no recipe_path")
            return super().__setattr__(name, value)

    monkeypatch.setattr(recipes, "yaml", _YamlStub)
    script_args, training_args, model_args = recipes.load_grpo_recipe(
        str(recipe_path), model_config_cls=_ModelCfg
    )
    assert getattr(script_args, "recipe_path") == str(recipe_path)
    assert getattr(training_args, "recipe_path") == str(recipe_path)
    assert not hasattr(model_args, "recipe_path")
