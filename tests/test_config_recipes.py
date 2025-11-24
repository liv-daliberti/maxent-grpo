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

Unit tests for recipe helpers under maxent_grpo.config.recipes.
"""

from __future__ import annotations

from types import SimpleNamespace
from dataclasses import dataclass

import pytest

import maxent_grpo.config.recipes as recipes_mod
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


def test_load_grpo_recipe_falls_back_to_yaml(tmp_path, monkeypatch):
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("foo: 7\ndataset_name: demo\nbenchmarks: [math]\n", encoding="utf-8")
    monkeypatch.setattr(recipes_mod, "OmegaConf", None)
    called = {}

    def _safe_load(handle):
        called["loaded"] = handle.name
        return {"foo": 7, "dataset_name": "demo", "benchmarks": ["math"]}

    monkeypatch.setattr(recipes_mod, "yaml", SimpleNamespace(safe_load=_safe_load))
    script_args, training_args, model_cfg = load_grpo_recipe(
        str(recipe_path), model_config_cls=_DummyModelConfig
    )
    assert called["loaded"] == str(recipe_path)
    assert script_args.dataset_name == "demo"
    assert training_args.benchmarks == ["math"]
    assert model_cfg.foo == 7


def test_load_grpo_recipe_validates_mapping(tmp_path, monkeypatch):
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("ignored: true\n", encoding="utf-8")

    class _OmegaStub:
        @staticmethod
        def load(path):
            return ["not", "a", "dict"]

        @staticmethod
        def to_container(cfg, resolve=True):
            return cfg

    monkeypatch.setattr(recipes_mod, "OmegaConf", _OmegaStub)
    with pytest.raises(ValueError):
        load_grpo_recipe(str(recipe_path), model_config_cls=_DummyModelConfig)


def test_load_grpo_recipe_requires_loader(tmp_path, monkeypatch):
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("foo: 1\n", encoding="utf-8")
    monkeypatch.setattr(recipes_mod, "OmegaConf", None)
    monkeypatch.setattr(recipes_mod, "yaml", None)

    with pytest.raises(ImportError):
        load_grpo_recipe(str(recipe_path), model_config_cls=_DummyModelConfig)