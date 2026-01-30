"""Unit tests for the baseline GRPO entrypoint module."""

from __future__ import annotations

from types import ModuleType
import sys


def _install_baseline_stub(monkeypatch):
    baseline_mod = ModuleType("maxent_grpo.pipelines.training.baseline")
    baseline_mod.run_baseline_training = lambda *_a, **_k: (_a, _k)
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.pipelines.training.baseline", baseline_mod
    )
    return baseline_mod


def test_main_uses_explicit_args(monkeypatch):
    import importlib

    monkeypatch.delitem(sys.modules, "maxent_grpo.grpo", raising=False)
    baseline_mod = _install_baseline_stub(monkeypatch)
    module = importlib.import_module("maxent_grpo.grpo")
    called = {}
    baseline_mod.run_baseline_training = lambda *a, **k: called.setdefault(
        "args", (a, k)
    )

    result = module.main("s", "t", "m")
    assert called["args"] == (("s", "t", "m"), {})
    assert result == called["args"]


def test_main_parses_when_args_missing(monkeypatch):
    import importlib

    monkeypatch.delitem(sys.modules, "maxent_grpo.grpo", raising=False)
    baseline_mod = _install_baseline_stub(monkeypatch)
    module = importlib.import_module("maxent_grpo.grpo")
    called = {}
    baseline_mod.run_baseline_training = lambda *a, **k: called.setdefault(
        "args", (a, k)
    )
    monkeypatch.setattr(module, "parse_grpo_args", lambda: ("s1", "t1", "m1"))

    module.main()
    assert called["args"] == (("s1", "t1", "m1"), {})


def test_main_falls_back_to_hydra(monkeypatch):
    import importlib

    monkeypatch.delitem(sys.modules, "maxent_grpo.grpo", raising=False)
    _install_baseline_stub(monkeypatch)
    module = importlib.import_module("maxent_grpo.grpo")
    monkeypatch.setattr(
        module, "parse_grpo_args", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    called = {}
    monkeypatch.setattr(
        module.hydra_cli, "baseline_entry", lambda: called.setdefault("hydra", True)
    )

    module.main()
    assert called["hydra"] is True


def test_cli_invokes_main(monkeypatch):
    import importlib

    monkeypatch.delitem(sys.modules, "maxent_grpo.grpo", raising=False)
    module = importlib.import_module("maxent_grpo.grpo")
    called = {}
    monkeypatch.setattr(module, "main", lambda: called.setdefault("main", True))
    module.cli()
    assert called["main"] is True
