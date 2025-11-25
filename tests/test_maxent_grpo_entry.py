"""Unit tests for the maxent_grpo entry shim."""

from __future__ import annotations

from types import SimpleNamespace
import sys


import maxent_grpo.maxent_grpo as entry


def test_main_with_explicit_args_calls_training(monkeypatch):
    called = {}

    def _run_maxent(*args):
        called["args"] = args
        return "ok"

    monkeypatch.setitem(
        sys.modules,
        "maxent_grpo.pipelines.training.maxent",
        SimpleNamespace(run_maxent_training=_run_maxent),
    )
    result = entry.main("a", "b", "c")
    assert result == "ok"
    assert called["args"] == ("a", "b", "c")


def test_main_parses_args_when_missing(monkeypatch):
    called = {}
    monkeypatch.setattr(entry, "parse_grpo_args", lambda: ("s", "t", "m"))
    monkeypatch.setitem(
        sys.modules,
        "maxent_grpo.pipelines.training.maxent",
        SimpleNamespace(run_maxent_training=lambda *a: called.setdefault("args", a)),
    )
    entry.main()
    assert called["args"] == ("s", "t", "m")


def test_main_falls_back_to_hydra_on_parse_error(monkeypatch):
    called = {}
    monkeypatch.setattr(
        entry, "parse_grpo_args", lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    )
    entry.hydra_cli = SimpleNamespace(
        maxent_entry=lambda: called.setdefault("hydra", True)
    )
    result = entry.main()
    assert result is True
    assert called.get("hydra") is True
