"""
Unit tests for training.runtime.logging helpers.
"""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace


from maxent_grpo.training.runtime import logging as rt_log


def test_resolve_run_metadata_prefers_env_and_caches(monkeypatch):
    monkeypatch.setenv("MAXENT_GIT_SHA", "envsha")
    monkeypatch.setenv("GRPO_RECIPE_USED", "used.yml")
    rt_log._RUN_META_CACHE.clear()
    meta = rt_log.resolve_run_metadata()
    assert meta["run/git_sha"] == "envsha"
    assert meta["run/recipe_path"] == "used.yml"
    # cached path should be reused even if env changes
    monkeypatch.setenv("MAXENT_GIT_SHA", "ignored")
    assert rt_log.resolve_run_metadata() is meta


def test_resolve_run_metadata_falls_back_to_git(monkeypatch):
    rt_log._RUN_META_CACHE.clear()
    monkeypatch.delenv("MAXENT_GIT_SHA", raising=False)
    monkeypatch.delenv("GRPO_RECIPE_USED", raising=False)
    monkeypatch.delenv("GRPO_RECIPE", raising=False)

    class _Proc:
        def __init__(self):
            self.stdout = "abc123\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    meta = rt_log.resolve_run_metadata()
    assert meta["run/git_sha"] == "abc123"
    assert meta["run/recipe_path"] == "unknown"


def test_maybe_init_wandb_run_skips_when_report_to_missing(monkeypatch):
    called = {}
    training_args = SimpleNamespace(report_to=None)
    accelerator = SimpleNamespace(is_main_process=True)
    monkeypatch.setattr(
        rt_log, "init_wandb_training", lambda *_a, **_k: called.setdefault("init", True)
    )
    assert rt_log._maybe_init_wandb_run(accelerator, training_args, {}) is None
    assert "init" not in called


def test_maybe_init_wandb_run_offline_for_non_main(monkeypatch):
    training_args = SimpleNamespace(report_to="wandb")
    accelerator = SimpleNamespace(is_main_process=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(rt_log, "init_wandb_training", lambda *_a, **_k: None)
    assert rt_log._maybe_init_wandb_run(accelerator, training_args, {}) is None
    assert os.environ.get("WANDB_MODE") == "offline"


def test_maybe_init_wandb_run_injects_metadata(monkeypatch):
    rt_log._RUN_META_CACHE.clear()
    training_args = SimpleNamespace(
        report_to=["wandb"], run_name="run-name", recipe_path="recipe.yaml"
    )
    accelerator = SimpleNamespace(is_main_process=True)
    monkeypatch.setenv("MAXENT_GIT_SHA", "deadbeef")

    class _Run:
        pass

    class _FakeWandb:
        def __init__(self):
            self.kwargs = None

        def init(self, **kwargs):
            self.kwargs = kwargs
            return _Run()

    monkeypatch.setattr(rt_log, "init_wandb_training", lambda *_a, **_k: None)
    wandb_inst = _FakeWandb()
    monkeypatch.setattr(rt_log, "_optional_dependency", lambda *_a, **_k: wandb_inst)
    run = rt_log._maybe_init_wandb_run(accelerator, training_args, {"a": 1})
    assert isinstance(run, _Run)
    assert run is not None
    meta = rt_log._RUN_META_CACHE
    assert meta["run/git_sha"] == "deadbeef"
    assert meta["run/recipe_path"] == "recipe.yaml"
    assert wandb_inst.kwargs["config"]["run/git_sha"] == "deadbeef"
    assert wandb_inst.kwargs["config"]["run/recipe_path"] == "recipe.yaml"


def test_log_run_header_logs_and_returns_meta(monkeypatch, caplog):
    rt_log._RUN_META_CACHE.clear()
    monkeypatch.setenv("MAXENT_GIT_SHA", "gitsha123")
    monkeypatch.setenv("GRPO_RECIPE_USED", "recipe.yml")
    caplog.set_level("INFO")
    meta = rt_log.log_run_header(SimpleNamespace(recipe_path=None))
    assert meta["run/git_sha"] == "gitsha123"
    assert meta["run/recipe_path"] == "recipe.yml"
    assert "Run metadata" in caplog.text


def test_log_wandb_handles_first_and_error_runs(monkeypatch, caplog):
    rt_log._FIRST_WANDB_LOGGED_RUNS.clear()
    caplog.set_level("INFO")
    rt_log._log_wandb(None, {"a": 1}, step=1)

    class _Run:
        def __init__(self):
            self.logged = []
            self.id = "run1"

        def log(self, metrics, step):
            self.logged.append((metrics, step))

    good_run = _Run()
    rt_log._log_wandb(good_run, {"x": 2}, step=3)
    assert good_run.logged == [({"x": 2}, 3)]
    assert "Logging first metrics" in caplog.text

    class _BadRun:
        def __init__(self):
            self.id = "run2"

        def log(self, *_a, **_k):
            raise ValueError("boom")

    caplog.set_level("WARNING")
    rt_log._log_wandb(_BadRun(), {"y": 5}, step=4)
    assert "Failed to log metrics" in caplog.text


def test_log_wandb_skips_empty_metrics():
    rt_log._FIRST_WANDB_LOGGED_RUNS.clear()
    called = {}

    class _Run:
        id = "run3"

        def log(self, *_a, **_k):
            called["logged"] = True

    rt_log._log_wandb(_Run(), {}, step=0)
    assert "logged" not in called
