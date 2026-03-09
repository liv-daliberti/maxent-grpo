"""Unit tests for the lightweight GRPO entrypoints."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
import pytest

import maxent_grpo.grpo as grpo


def test_main_parses_args_when_missing(monkeypatch):
    parsed = (
        SimpleNamespace(script=True),
        SimpleNamespace(train=True),
        SimpleNamespace(model=True),
    )

    # Force parse_grpo_args to be called and return our sentinel.
    monkeypatch.setattr(grpo, "parse_grpo_args", lambda: parsed)

    captured = {}

    def _run_baseline_training(s_args, t_args, m_args):
        captured["args"] = (s_args, t_args, m_args)
        return "ran"

    monkeypatch.setitem(
        sys.modules,
        "maxent_grpo.training.baseline",
        types.SimpleNamespace(run_baseline_training=_run_baseline_training),
    )

    result = grpo.main()
    assert result == "ran"
    assert captured["args"] == parsed


def test_main_parse_error_propagates(monkeypatch):
    monkeypatch.setattr(
        grpo, "parse_grpo_args", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        grpo.main()


def test_main_uses_provided_args(monkeypatch):
    provided = (
        SimpleNamespace(script=False),
        SimpleNamespace(train=False),
        SimpleNamespace(model=False),
    )

    def _run_baseline_training(s_args, t_args, m_args):
        return (s_args, t_args, m_args)

    monkeypatch.setitem(
        sys.modules,
        "maxent_grpo.training.baseline",
        types.SimpleNamespace(run_baseline_training=_run_baseline_training),
    )

    result = grpo.main(*provided)
    assert result == provided


def test_cli_invokes_main(monkeypatch):
    called = {}
    monkeypatch.setattr(grpo, "main", lambda: called.setdefault("hit", True))
    grpo.cli()
    assert called.get("hit") is True
