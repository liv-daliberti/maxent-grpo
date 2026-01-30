"""Additional coverage for training.runtime.logging helpers."""

from __future__ import annotations

from types import SimpleNamespace

import maxent_grpo.training.runtime.logging as rt_logging


def test_log_wandb_skips_when_empty():
    rt_logging._FIRST_WANDB_LOGGED_RUNS.clear()
    rt_logging._log_wandb(None, {"a": 1}, step=1)  # no-op when run is None
    rt_logging._log_wandb(
        SimpleNamespace(id="r1", log=lambda *_a, **_k: None), {}, step=1
    )  # no metrics


def test_wandb_error_types_returns_base_when_missing(monkeypatch):
    """Ensure missing wandb.errors returns the default error tuple."""

    rt_logging._wandb_error_types.cache_clear()
    monkeypatch.delitem(rt_logging.sys.modules, "wandb.errors", raising=False)
    errors = rt_logging._wandb_error_types()
    assert errors == (RuntimeError, ValueError)
