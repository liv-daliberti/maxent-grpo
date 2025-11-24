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

Tests for the TRL CLI parsing helper.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_parse_grpo_args_uses_env_recipe(monkeypatch):
    from maxent_grpo.training.cli import trl as cli_trl

    dummy_result = ("s", "t", "m")
    monkeypatch.setenv("GRPO_RECIPE", "/path/to/recipe.yaml")
    # Provide a minimal trl stub so import succeeds even when the real package is missing.
    trl_stub = SimpleNamespace(ModelConfig=object, TrlParser=None)
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    monkeypatch.setitem(
        cli_trl.__dict__,
        "load_grpo_recipe",
        lambda path, model_config_cls: dummy_result,
    )
    # Ensure ModelConfig/TrlParser are not required when recipe path is set.
    result = cli_trl.parse_grpo_args()
    assert result == dummy_result


def test_parse_grpo_args_requires_trl(monkeypatch):
    from maxent_grpo.training.cli import trl as cli_trl

    monkeypatch.delenv("GRPO_RECIPE", raising=False)
    # Remove any cached trl module to force the import error path.
    monkeypatch.setitem(sys.modules, "trl", None)
    with pytest.raises(ImportError):
        cli_trl.parse_grpo_args()