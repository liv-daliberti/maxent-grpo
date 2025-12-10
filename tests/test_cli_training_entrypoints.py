"""Sanity checks for the GRPO/MaxEnt/InfoSeed CLI entrypoints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
import maxent_grpo.grpo as grpo_cli
import maxent_grpo.maxent_grpo as maxent_cli
import maxent_grpo.infoseed as infoseed_cli


def _stubbed_args(**training_overrides):
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = GRPOConfig(**training_overrides)
    model_args = SimpleNamespace()
    return script_args, training_args, model_args


def test_grpo_cli_dispatches_to_baseline(monkeypatch):
    """--task grpo should use the baseline pipeline even with MaxEnt flags."""

    script_args, training_args, model_args = _stubbed_args(train_grpo_objective=False)
    monkeypatch.setattr(
        grpo_cli, "parse_grpo_args", lambda: (script_args, training_args, model_args)
    )

    called = {}

    def _fake_run(script, training, model):
        called["args"] = (script, training, model)
        return "ok"

    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.baseline.run_baseline_training",
        _fake_run,
    )

    result = grpo_cli.main()
    assert result == "ok"
    assert called["args"] == (script_args, training_args, model_args)


def test_grpo_cli_with_meta_routes_to_maxent(monkeypatch):
    script_args, training_args, model_args = _stubbed_args(
        controller_meta_enabled=True,
        train_grpo_objective=True,
    )
    monkeypatch.setattr(
        grpo_cli, "parse_grpo_args", lambda: (script_args, training_args, model_args)
    )

    baseline_called = {}
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.baseline.run_baseline_training",
        lambda *_a, **_k: baseline_called.setdefault("called", True),
    )
    called = {}

    def _fake_run(script, training, model):
        called["args"] = (script, training, model)
        return "ok"

    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.maxent.run_maxent_training",
        _fake_run,
    )

    result = grpo_cli.main()
    assert result == "ok"
    assert called["args"] == (script_args, training_args, model_args)
    assert baseline_called == {}


@pytest.mark.parametrize("mixed_flags", [True, False])
def test_maxent_cli_dispatches_to_pipeline(monkeypatch, mixed_flags):
    """--task maxent must always hit the MaxEnt pipeline regardless of recipe flags."""

    overrides = {"train_grpo_objective": mixed_flags, "info_seed_enabled": mixed_flags}
    script_args, training_args, model_args = _stubbed_args(**overrides)
    monkeypatch.setattr(
        maxent_cli, "parse_grpo_args", lambda: (script_args, training_args, model_args)
    )

    called = {}

    def _fake_run(script, training, model):
        called["args"] = (script, training, model)
        return "ok"

    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.maxent.run_maxent_training",
        _fake_run,
    )

    result = maxent_cli.main()
    assert result == "ok"
    assert called["args"] == (script_args, training_args, model_args)


def test_infoseed_cli_dispatches_even_when_flag_disabled(monkeypatch):
    """--task infoseed should still call the InfoSeed pipeline if recipe toggles drift."""

    script_args, training_args, model_args = _stubbed_args(info_seed_enabled=False)
    monkeypatch.setattr(
        infoseed_cli, "parse_grpo_args", lambda: (script_args, training_args, model_args)
    )

    called = {}

    def _fake_run(script, training, model):
        called["args"] = (script, training, model)
        return "ok"

    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.infoseed.run_infoseed_training",
        _fake_run,
    )

    result = infoseed_cli.main()
    assert result == "ok"
    assert called["args"] == (script_args, training_args, model_args)
