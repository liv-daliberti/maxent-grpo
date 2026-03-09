"""Unit tests for training.runtime.logging helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace


import maxent_grpo.training.runtime.logging as rt_logging


def test_report_to_contains_variants():
    assert rt_logging._report_to_contains(None, "wandb") is False
    assert rt_logging._report_to_contains("WandB", "wandb") is True
    assert rt_logging._report_to_contains(["tensorboard", "wandb"], "WANDB") is True


def test_resolve_run_metadata_caches(monkeypatch):
    monkeypatch.delenv("MAXENT_GIT_SHA", raising=False)
    rt_logging._RUN_META_CACHE.clear()
    meta1 = rt_logging.resolve_run_metadata(SimpleNamespace(recipe_path="r1"))
    assert meta1["run/recipe_path"] == "r1"
    # Subsequent calls should reuse cache even with different args/env
    monkeypatch.setenv("GRPO_RECIPE_USED", "env_recipe")
    meta2 = rt_logging.resolve_run_metadata()
    assert meta2 is meta1


def test_wandb_error_types_and_log(monkeypatch, caplog):
    # No wandb installed -> defaults
    rt_logging._wandb_error_types.cache_clear()
    import sys

    monkeypatch.delitem(sys.modules, "wandb.errors", raising=False)
    errors = rt_logging._wandb_error_types()
    assert RuntimeError in errors
    # Cached result should be reused until cleared.
    assert rt_logging._wandb_error_types() is errors

    # Install wandb stub with Error subclass
    class _WandbError(Exception):
        pass

    wandb_errors = SimpleNamespace(Error=_WandbError)
    monkeypatch.setitem(sys.modules, "wandb.errors", wandb_errors)
    rt_logging._wandb_error_types.cache_clear()
    errors = rt_logging._wandb_error_types()
    assert errors[0] is _WandbError
    # Clearing cache allows fresh detection on next call.
    rt_logging._wandb_error_types.cache_clear()

    caplog.set_level("INFO")
    run = SimpleNamespace(
        id="run1",
        log=lambda metrics, step=None: (_ for _ in ()).throw(_WandbError("boom")),
    )
    rt_logging._FIRST_WANDB_LOGGED_RUNS.clear()
    rt_logging._log_wandb(run, {"a": 1}, step=1)
    assert any("Failed to log metrics" in rec.message for rec in caplog.records)
    # Cache should retain wandb Error inclusion until cleared again.
    errors_again = rt_logging._wandb_error_types()
    assert errors_again[0] is _WandbError


def test_maybe_init_wandb_run_handles_modes(monkeypatch):
    # Setup accelerator and wandb stubs
    accel = SimpleNamespace(is_main_process=True)
    training_args = SimpleNamespace(
        report_to=["wandb"],
        run_name="name",
        wandb_entity=None,
        wandb_project=None,
        wandb_run_group=None,
    )
    called = {}

    class _Run:
        id = "runid"

        def log(self, *args, **kwargs):
            called["logged"] = True

    def _init(**kwargs):
        called.setdefault("init_kwargs", kwargs)
        return _Run()

    wandb_stub = SimpleNamespace(init=_init)
    import sys

    monkeypatch.setitem(sys.modules, "wandb", wandb_stub)
    rt_logging._FIRST_WANDB_LOGGED_RUNS.clear()
    run = rt_logging._maybe_init_wandb_run(accel, training_args, {"x": 1})
    assert run is None or isinstance(run, _Run)

    # Non-main process should set offline mode and return None
    accel2 = SimpleNamespace(is_main_process=False)
    rt_logging._FIRST_WANDB_LOGGED_RUNS.clear()
    run2 = rt_logging._maybe_init_wandb_run(accel2, training_args, {})
    assert run2 is None
    assert os.environ.get("WANDB_MODE") == "offline"
