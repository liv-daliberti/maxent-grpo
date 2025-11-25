"""Unit tests for training.cli.trl helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_parse_grpo_args_uses_recipe(monkeypatch):
    called = {}
    fake_trl = SimpleNamespace(ModelConfig=type("ModelConfig", (), {}), TrlParser=None)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setattr(
        "maxent_grpo.training.cli.trl.load_grpo_recipe",
        lambda path, model_config_cls: called.setdefault(
            "args", (path, model_config_cls)
        ),
    )
    monkeypatch.setenv("GRPO_RECIPE", "path/to/recipe.yaml")
    import maxent_grpo.training.cli.trl as cli_trl

    cli_trl.parse_grpo_args()
    assert called["args"][0].endswith("recipe.yaml")
    assert called["args"][1] is fake_trl.ModelConfig


def test_parse_grpo_args_uses_trl_parser(monkeypatch):
    parsed = ("s", "t", "m")

    class _Parser:
        def __init__(self, _cfgs):
            self.cfgs = _cfgs

        def parse_args_and_config(self):
            return parsed

    fake_trl = SimpleNamespace(
        ModelConfig=type("ModelConfig", (), {}), TrlParser=_Parser
    )
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.delenv("GRPO_RECIPE", raising=False)
    import maxent_grpo.training.cli.trl as cli_trl

    assert cli_trl.parse_grpo_args() == parsed


def test_parse_grpo_args_prefers_explicit_recipe(monkeypatch):
    called = {}
    fake_trl = SimpleNamespace(ModelConfig=type("ModelConfig", (), {}), TrlParser=None)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setenv("GRPO_RECIPE", "env/recipe.yaml")
    monkeypatch.setattr(
        "maxent_grpo.training.cli.trl.load_grpo_recipe",
        lambda path, model_config_cls: called.setdefault(
            "args", (path, model_config_cls)
        ),
    )
    import maxent_grpo.training.cli.trl as cli_trl

    cli_trl.parse_grpo_args("cli/recipe.yaml")
    assert called["args"][0] == "cli/recipe.yaml"
    assert called["args"][1] is fake_trl.ModelConfig


def test_parse_grpo_args_raises_without_trl(monkeypatch):
    import builtins
    import importlib

    monkeypatch.delitem(sys.modules, "trl", raising=False)
    monkeypatch.delenv("GRPO_RECIPE", raising=False)
    orig_import = builtins.__import__

    def _missing(name, *args, **kwargs):
        if name == "trl":
            raise ModuleNotFoundError("no trl")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing)
    import maxent_grpo.training.cli.trl as cli_trl

    with pytest.raises(ImportError) as excinfo:
        importlib.reload(cli_trl).parse_grpo_args()
    assert "pip install trl" in str(excinfo.value)
