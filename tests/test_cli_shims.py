"""Tests for the legacy CLI shim modules grpo.py and maxent_grpo.py."""

from __future__ import annotations


def test_grpo_main_runs_with_explicit_args(monkeypatch):
    import maxent_grpo.grpo as grpo

    called = {}
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.baseline.run_baseline_training",
        lambda *args: called.setdefault("args", args),
    )
    grpo.main("script", "train", "model")
    assert called["args"] == ("script", "train", "model")


def test_grpo_main_falls_back_to_hydra(monkeypatch):
    import maxent_grpo.grpo as grpo

    monkeypatch.setattr(grpo, "parse_grpo_args", lambda: (_ for _ in ()).throw(ImportError()))
    called = {}
    monkeypatch.setattr(
        "maxent_grpo.cli.hydra_cli.baseline_entry",
        lambda: called.setdefault("hydra", True),
    )
    grpo.main()
    assert called.get("hydra") is True


def test_maxent_main_runs_with_explicit_args(monkeypatch):
    import maxent_grpo.maxent_grpo as maxent_mod

    called = {}
    monkeypatch.setattr(
        "maxent_grpo.training.run_maxent_grpo",
        lambda *args: called.setdefault("args", args),
    )
    maxent_mod.main("s_args", "t_args", "m_args")
    assert called["args"] == ("s_args", "t_args", "m_args")


def test_maxent_main_falls_back_to_hydra(monkeypatch):
    import maxent_grpo.maxent_grpo as maxent_mod

    def _raise():
        raise ImportError()

    monkeypatch.setattr("maxent_grpo.training.cli.parse_grpo_args", _raise)
    called = {}
    monkeypatch.setattr(
        "maxent_grpo.cli.hydra_cli.maxent_entry",
        lambda: called.setdefault("hydra", True),
    )
    maxent_mod.main()
    assert called.get("hydra") is True
