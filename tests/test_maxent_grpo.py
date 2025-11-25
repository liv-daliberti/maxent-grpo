"""Tests for the maxent_grpo.main entrypoint helper."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


def test_main_uses_parsed_args(monkeypatch):
    module = importlib.reload(importlib.import_module("maxent_grpo.maxent_grpo"))

    called = {}

    def _parse():
        called["parsed"] = True
        return ("s_args", "t_args", "m_args")

    def _run(s_args, t_args, m_args):
        called["run"] = (s_args, t_args, m_args)
        return "ok"

    monkeypatch.setattr(module, "parse_grpo_args", _parse)
    monkeypatch.setattr(
        module,
        "hydra_cli",
        SimpleNamespace(
            maxent_entry=lambda: (_ for _ in ()).throw(RuntimeError("should not"))
        ),
    )
    stub_mod = ModuleType("maxent_grpo.pipelines.training.maxent")
    stub_mod.run_maxent_training = _run
    monkeypatch.setitem(sys.modules, "maxent_grpo.pipelines.training.maxent", stub_mod)

    result = module.main()
    assert result == "ok"
    assert called["run"] == ("s_args", "t_args", "m_args")


def test_main_fallbacks_to_hydra(monkeypatch):
    module = importlib.reload(importlib.import_module("maxent_grpo.maxent_grpo"))

    def _parse():
        raise ValueError("boom")

    hydra_called = {}
    monkeypatch.setattr(module, "parse_grpo_args", _parse)

    def _hydra_entry():
        hydra_called["called"] = True
        return "hydra"

    monkeypatch.setattr(module, "hydra_cli", SimpleNamespace(maxent_entry=_hydra_entry))

    result = module.main()
    assert hydra_called.get("called") is True
    assert result == "hydra"
