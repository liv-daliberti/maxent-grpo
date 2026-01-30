"""
Tests for the GRPO and MaxEnt entrypoint modules (non-shim paths).
"""

from __future__ import annotations

import sys
from types import ModuleType


def test_grpo_main_runs_with_explicit_args(monkeypatch):
    import maxent_grpo.grpo as grpo

    called = {}
    baseline_mod = ModuleType("maxent_grpo.pipelines.training.baseline")
    baseline_mod.run_baseline_training = lambda *args: called.setdefault("args", args)
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.pipelines.training.baseline", baseline_mod
    )

    grpo.main("script", "train", "model")
    assert called["args"] == ("script", "train", "model")


def test_grpo_main_parses_args_when_missing(monkeypatch):
    import maxent_grpo.grpo as grpo

    baseline_mod = ModuleType("maxent_grpo.pipelines.training.baseline")
    called = {}
    baseline_mod.run_baseline_training = lambda *args: called.setdefault("args", args)
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.pipelines.training.baseline", baseline_mod
    )
    monkeypatch.setattr(grpo, "parse_grpo_args", lambda: ("s", "t", "m"))

    grpo.main()
    assert called["args"] == ("s", "t", "m")


def test_grpo_main_falls_back_to_hydra(monkeypatch):
    import maxent_grpo.grpo as grpo

    monkeypatch.setattr(
        grpo, "parse_grpo_args", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    called = {}
    monkeypatch.setattr(
        grpo.hydra_cli, "baseline_entry", lambda: called.setdefault("hydra", True)
    )

    grpo.main()
    assert called["hydra"] is True


def test_maxent_main_runs_with_explicit_args(monkeypatch):
    import maxent_grpo.maxent_grpo as maxent_mod

    called = {}
    maxent_stub = ModuleType("maxent_grpo.pipelines.training.maxent")
    maxent_stub.run_maxent_training = lambda *args, **kwargs: called.setdefault(
        "args", (args, kwargs)
    )
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.pipelines.training.maxent", maxent_stub
    )

    maxent_mod.main("s_args", "t_args", "m_args")
    assert called["args"] == (("s_args", "t_args", "m_args"), {})


def test_maxent_main_parses_args_when_missing(monkeypatch):
    import maxent_grpo.maxent_grpo as maxent_mod

    called = {}
    maxent_stub = ModuleType("maxent_grpo.pipelines.training.maxent")
    maxent_stub.run_maxent_training = lambda *args, **kwargs: called.setdefault(
        "args", (args, kwargs)
    )
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.pipelines.training.maxent", maxent_stub
    )
    monkeypatch.setattr(maxent_mod, "parse_grpo_args", lambda: ("s", "t", "m"))

    maxent_mod.main()
    assert called["args"] == (("s", "t", "m"), {})


def test_maxent_main_falls_back_to_hydra(monkeypatch):
    import maxent_grpo.maxent_grpo as maxent_mod

    monkeypatch.setattr(
        maxent_mod,
        "parse_grpo_args",
        lambda: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    called = {}
    monkeypatch.setattr(
        maxent_mod.hydra_cli, "maxent_entry", lambda: called.setdefault("hydra", True)
    )

    maxent_mod.main()
    assert called["hydra"] is True
