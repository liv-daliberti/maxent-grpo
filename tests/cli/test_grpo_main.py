"""Additional coverage for maxent_grpo.grpo entrypoints."""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
import sys


def _install_baseline_stub(monkeypatch):
    """Install a baseline trainer stub to avoid importing heavy deps."""
    stub = ModuleType("maxent_grpo.pipelines.training.baseline")
    stub.run_baseline_training = lambda *args, **kwargs: ("ran", args, kwargs)
    monkeypatch.setitem(sys.modules, "maxent_grpo.pipelines.training.baseline", stub)
    return stub


def test_main_with_explicit_args(monkeypatch):
    _install_baseline_stub(monkeypatch)
    mod = importlib.reload(importlib.import_module("maxent_grpo.grpo"))
    result = mod.main("s", "t", "m")
    assert result[0] == "ran"
    assert result[1] == ("s", "t", "m")


def test_main_parse_fallback(monkeypatch):
    stub = _install_baseline_stub(monkeypatch)
    mod = importlib.reload(importlib.import_module("maxent_grpo.grpo"))

    def _parse():
        return ("s1", "t1", "m1")

    called = {}
    stub.run_baseline_training = lambda *a, **k: called.setdefault("args", (a, k))
    monkeypatch.setattr(mod, "parse_grpo_args", _parse)
    mod.main()
    assert called["args"][0] == ("s1", "t1", "m1")


def test_main_hydra_fallback(monkeypatch):
    _install_baseline_stub(monkeypatch)
    mod = importlib.reload(importlib.import_module("maxent_grpo.grpo"))
    hydra_called = {}
    monkeypatch.setattr(
        mod, "parse_grpo_args", lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    )

    def _hydra():
        hydra_called["ok"] = True
        return "hydra"

    monkeypatch.setattr(mod, "hydra_cli", SimpleNamespace(baseline_entry=_hydra))
    res = mod.main()
    assert hydra_called.get("ok") is True
    assert res == "hydra"


def test_cli_delegates_to_main(monkeypatch):
    _install_baseline_stub(monkeypatch)
    mod = importlib.reload(importlib.import_module("maxent_grpo.grpo"))
    called = {}
    monkeypatch.setattr(mod, "main", lambda: called.setdefault("main", True) or "x")
    mod.cli()
    assert called["main"] is True
