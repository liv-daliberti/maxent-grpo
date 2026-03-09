"""
Tests for the GRPO and MaxEnt entrypoint modules (non-shim paths).
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


def test_grpo_main_runs_with_explicit_args(monkeypatch):
    import maxent_grpo.grpo as grpo

    called = {}
    baseline_mod = ModuleType("maxent_grpo.training.baseline")
    baseline_mod.run_baseline_training = lambda *args: called.setdefault("args", args)
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.training.baseline", baseline_mod
    )

    grpo.main("script", "train", "model")
    assert called["args"] == ("script", "train", "model")


def test_grpo_main_parses_args_when_missing(monkeypatch):
    import maxent_grpo.grpo as grpo

    baseline_mod = ModuleType("maxent_grpo.training.baseline")
    called = {}
    baseline_mod.run_baseline_training = lambda *args: called.setdefault("args", args)
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.training.baseline", baseline_mod
    )
    monkeypatch.setattr(grpo, "parse_grpo_args", lambda: ("s", "t", "m"))

    grpo.main()
    assert called["args"] == ("s", "t", "m")


def test_grpo_main_falls_back_to_hydra(monkeypatch):
    import maxent_grpo.grpo as grpo

    monkeypatch.setattr(
        grpo, "parse_grpo_args", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        grpo.main()


def test_maxent_main_runs_with_explicit_args(monkeypatch):
    import maxent_grpo as maxent_mod

    called = {}
    baseline_stub = ModuleType("maxent_grpo.training.baseline")
    baseline_stub.run_baseline_training = lambda *args, **kwargs: called.setdefault(
        "args", (args, kwargs)
    )
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.training.baseline", baseline_stub
    )

    maxent_mod.main("s_args", "t_args", "m_args")
    assert called["args"] == (("s_args", "t_args", "m_args"), {})


def test_maxent_main_parses_args_when_missing(monkeypatch):
    import maxent_grpo as maxent_mod

    called = {}
    baseline_stub = ModuleType("maxent_grpo.training.baseline")
    baseline_stub.run_baseline_training = lambda *args, **kwargs: called.setdefault(
        "args", (args, kwargs)
    )
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.training.baseline", baseline_stub
    )
    monkeypatch.setattr(maxent_mod, "parse_grpo_args", lambda: ("s", "t", "m"))

    maxent_mod.main()
    assert called["args"] == (("s", "t", "m"), {})


def test_maxent_main_falls_back_to_hydra(monkeypatch):
    import maxent_grpo as maxent_mod

    monkeypatch.setattr(
        maxent_mod,
        "parse_grpo_args",
        lambda: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    called = {}
    monkeypatch.setattr(
        maxent_mod.hydra_cli,
        "_maybe_insert_command",
        lambda command: called.setdefault("command", command),
    )
    monkeypatch.setattr(
        maxent_mod.hydra_cli, "hydra_entry", lambda: called.setdefault("hydra", True)
    )

    maxent_mod.main()
    assert called["command"] == "train-maxent"
    assert called["hydra"] is True
