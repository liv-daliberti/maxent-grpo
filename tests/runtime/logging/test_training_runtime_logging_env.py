"""Ensure logging helpers propagate environment overrides into wandb init."""

from __future__ import annotations

from types import SimpleNamespace

from maxent_grpo.training.runtime import logging as rt_logging


def test_maybe_init_wandb_run_uses_env(monkeypatch):
    # Accelerate stub on main process
    accelerator = SimpleNamespace(is_main_process=True)
    # Training args stub with report_to including wandb
    training_args = SimpleNamespace(report_to=["wandb"], run_name="demo-run")
    # Capture init kwargs
    captured = {}

    class _Run:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def log(self, *_a, **_k):
            return None

    wandb_stub = SimpleNamespace(init=lambda **kwargs: _Run(**kwargs))
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_RUN_GROUP", "grp")
    monkeypatch.setenv("WANDB_DIR", "/tmp/wandbdir")
    monkeypatch.setattr(rt_logging, "_optional_dependency", lambda name: wandb_stub)
    run = rt_logging._maybe_init_wandb_run(accelerator, training_args, {"extra": 1})
    assert isinstance(run, _Run)
    assert captured["config"]["extra"] == 1
    assert captured["project"] == "proj"
    assert captured["entity"] == "ent"
    assert captured["group"] == "grp"
    assert captured["dir"] == "/tmp/wandbdir"
