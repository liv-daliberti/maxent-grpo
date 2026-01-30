"""Tests for training.runtime.logging helpers."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace


from maxent_grpo.training.runtime import logging as rt_log


def test_report_to_contains_cases():
    assert rt_log._report_to_contains("wandb", "wandb")
    assert rt_log._report_to_contains(["WANDB", "csv"], "wandb")
    assert not rt_log._report_to_contains(None, "wandb")
    assert not rt_log._report_to_contains("tensorboard", "wandb")


def test_wandb_error_types_handles_missing_dependency(monkeypatch):
    monkeypatch.setitem(os.environ, "PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setitem(sys.modules, "wandb.errors", None)
    types = rt_log._wandb_error_types()
    assert isinstance(types, tuple) and types


def test_maybe_init_wandb_respects_report_to(monkeypatch):
    accel = SimpleNamespace(is_main_process=True)
    args = SimpleNamespace(
        report_to=None,
        run_name=None,
        wandb_entity=None,
        wandb_project=None,
        wandb_run_group=None,
    )
    run = rt_log._maybe_init_wandb_run(accel, args, {"a": 1})
    assert run is None

    # When wandb is missing, should warn and return None
    args.report_to = ["wandb"]
    monkeypatch.setitem(sys.modules, "wandb", None)
    run = rt_log._maybe_init_wandb_run(accel, args, {"b": 2})
    assert run is None


def test_log_wandb_guarded(monkeypatch, caplog):
    recorded = {}

    class _Run:
        id = "abc"

        def log(self, metrics, step=None):
            recorded["metrics"] = metrics
            recorded["step"] = step

    rt_log._FIRST_WANDB_LOGGED_RUNS.clear()
    rt_log._log_wandb(_Run(), {"x": 1}, step=5)
    assert recorded == {"metrics": {"x": 1}, "step": 5}
    # No log call when metrics empty
    recorded.clear()
    rt_log._log_wandb(_Run(), {}, step=6)
    assert recorded == {}
