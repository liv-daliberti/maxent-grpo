"""Tests for run_maxent_grpo helpers."""

from __future__ import annotations

from importlib import import_module, reload
from types import SimpleNamespace

import pytest

from test_run_setup_reference import _load_run_setup


@pytest.fixture
def run_mod(monkeypatch):
    """Import maxent_helpers.run with torch/accelerate stubs applied."""
    _load_run_setup(monkeypatch)
    return reload(import_module("maxent_helpers.run"))


def _ns(report_to):
    return SimpleNamespace(report_to=report_to)


def test_sync_report_to_from_script_args_backfills_none(run_mod):
    script_args = _ns(["wandb"])
    training_args = _ns(None)
    run_mod._sync_report_to_from_script_args(script_args, training_args)
    assert training_args.report_to == ["wandb"]


def test_sync_report_to_does_not_override_existing_training_value(run_mod):
    script_args = _ns(["wandb"])
    training_args = _ns(["tensorboard"])
    run_mod._sync_report_to_from_script_args(script_args, training_args)
    assert training_args.report_to == ["tensorboard"]


def test_sync_report_to_ignores_script_none(run_mod):
    script_args = _ns("none")
    training_args = _ns(None)
    run_mod._sync_report_to_from_script_args(script_args, training_args)
    assert training_args.report_to is None
