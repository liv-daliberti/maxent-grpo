"""
Tests for TRL CLI argument parsing and optimizer helpers.
"""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from maxent_grpo.training.cli import trl as cli_trl
from maxent_grpo.training.optim import configure_accumulation_steps


def _install_trl_stub(monkeypatch, parser_cls=None):
    """Install a lightweight ``trl`` stub into ``sys.modules``."""
    trl_mod = ModuleType("trl")
    trl_mod.__spec__ = SimpleNamespace()

    class _ModelConfig:
        pass

    trl_mod.ModelConfig = _ModelConfig

    if parser_cls is None:

        def parser_cls(args):
            return None

    trl_mod.TrlParser = parser_cls
    monkeypatch.setitem(sys.modules, "trl", trl_mod)
    return trl_mod


def test_parse_grpo_args_raises_when_trl_missing(monkeypatch):
    monkeypatch.delenv("GRPO_RECIPE", raising=False)
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "trl":
            raise ModuleNotFoundError("no trl")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError) as excinfo:
        cli_trl.parse_grpo_args()
    assert "pip install trl" in str(excinfo.value)


def test_parse_grpo_args_prefers_recipe_loader(monkeypatch):
    monkeypatch.delenv("GRPO_RECIPE", raising=False)
    calls = {}

    def _loader(path, model_config_cls):
        calls["path"] = path
        calls["model_cls"] = model_config_cls
        return ("args", "cfg", "model")

    parser_cls = type(
        "Parser",
        (),
        {
            "__init__": lambda self, args: None,
            "parse_args_and_config": lambda self: None,
        },
    )
    trl_mod = _install_trl_stub(monkeypatch, parser_cls)
    monkeypatch.setattr(cli_trl, "load_grpo_recipe", _loader)

    result = cli_trl.parse_grpo_args("recipe.yml")
    assert result == ("args", "cfg", "model")
    assert calls["path"] == "recipe.yml"
    assert calls["model_cls"] is trl_mod.ModelConfig


def test_parse_grpo_args_invokes_trl_parser(monkeypatch):
    monkeypatch.delenv("GRPO_RECIPE", raising=False)
    parsed = ("script_args", "grpo_cfg", "model_cfg")
    captured = {}

    class _Parser:
        def __init__(self, args):
            captured["args"] = args

        def parse_args_and_config(self):
            return parsed

    _install_trl_stub(monkeypatch, _Parser)
    result = cli_trl.parse_grpo_args()
    assert result == parsed
    # Expect the parser to receive the dataclass classes.
    assert len(captured["args"]) == 3


def test_configure_accumulation_steps_handles_setattr_errors():
    class _Target:
        def __init__(self):
            self._val = 1
            self.attempts = 0
            self.gradient_state = None

        @property
        def gradient_accumulation_steps(self):
            return self._val

        @gradient_accumulation_steps.setter
        def gradient_accumulation_steps(self, value):
            self.attempts += 1
            raise TypeError("cannot set")

    accelerator = _Target()
    accelerator.gradient_state = _Target()
    configure_accumulation_steps(accelerator, 3)
    assert accelerator.attempts == 1
    assert accelerator.gradient_state.attempts == 1
    assert accelerator.gradient_accumulation_steps == 1
