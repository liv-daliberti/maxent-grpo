"""Unit tests for recipe helpers under maxent_grpo.config.recipes."""

from __future__ import annotations

from dataclasses import dataclass

from maxent_grpo.config.dataset import ScriptArguments
from maxent_grpo.config.grpo import GRPOConfig
from maxent_grpo.config.recipes import (
    _dataclass_field_names,
    _split_recipe_payload,
    load_grpo_recipe,
)


@dataclass
class _DummyModelConfig:
    foo: int = 0
    bar: str = "bar"


def test_dataclass_field_names_extracts_all_fields():
    names = _dataclass_field_names(_DummyModelConfig)
    assert {"foo", "bar"} <= names


def test_split_recipe_payload_routes_sections():
    payload = {
        "dataset_name": "demo/ds",
        "benchmarks": ["math"],
        "foo": 3,
        "maxent_tau": 0.7,
    }
    script_kwargs, training_kwargs, model_kwargs = _split_recipe_payload(
        payload, _DummyModelConfig
    )
    assert script_kwargs["dataset_name"] == "demo/ds"
    assert training_kwargs["benchmarks"] == ["math"]
    assert training_kwargs["maxent_tau"] == 0.7
    assert model_kwargs["foo"] == 3


def test_load_grpo_recipe_builds_dataclasses(tmp_path):
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        "dataset_name: demo/ds\nbenchmarks: [math]\nfoo: 5\n", encoding="utf-8"
    )
    script_args, training_args, model_cfg = load_grpo_recipe(
        str(recipe_path), model_config_cls=_DummyModelConfig
    )
    assert isinstance(script_args, ScriptArguments)
    assert isinstance(training_args, GRPOConfig)
    assert isinstance(model_cfg, _DummyModelConfig)
    assert script_args.dataset_name == "demo/ds"
    assert training_args.benchmarks == ["math"]
    assert model_cfg.foo == 5
