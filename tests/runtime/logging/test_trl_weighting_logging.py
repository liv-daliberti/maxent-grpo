"""Tests for telemetry.trl_logging weighting metric helpers."""

from types import SimpleNamespace

import pytest

from maxent_grpo.telemetry.trl_logging import ensure_weighting_logging


class _DummyTrainer:
    """Minimal trainer stub that captures the last logged metrics."""

    def __init__(self, args=None):
        self.args = args or SimpleNamespace()
        self.logged = None
        self._last_loss_components = None
        self._last_loss_scalar = None

    def log(self, logs, *args, **kwargs):
        _ = args, kwargs
        self.logged = dict(logs)


def _log_with_args(args) -> dict:
    """Helper to invoke logging on a wrapped trainer and return the metrics."""

    Wrapped = ensure_weighting_logging(_DummyTrainer)
    trainer = Wrapped(args=args)
    trainer.state = SimpleNamespace(global_step=10)
    trainer.log({"loss": 1.23})
    return trainer.logged or {}


def test_ensure_weighting_logging_defaults_to_grpo_when_unspecified():
    """Absent flags should default to GRPO (train_grpo_objective=True)."""

    args = SimpleNamespace(maxent_tau=0.2, beta=0.4)
    metrics = _log_with_args(args)
    assert metrics["train/grpo_objective"] == 1.0
    assert metrics["train/maxent_objective"] == 0.0
    assert metrics["train/weighting/tau"] == pytest.approx(0.2)
    assert metrics["train/weighting/beta"] == pytest.approx(0.4)


def test_ensure_weighting_logging_honors_maxent_flag():
    """Explicit maxent flag should flip the objectives and keep metrics."""

    args = SimpleNamespace(
        maxent_tau=0.3,
        beta=0.5,
        maxent_objective=True,
        train_grpo_objective=None,
    )
    metrics = _log_with_args(args)
    assert metrics["train/grpo_objective"] == 0.0
    assert metrics["train/maxent_objective"] == 1.0
    assert metrics["train/weighting/tau"] == pytest.approx(0.3)
    assert metrics["train/weighting/beta"] == pytest.approx(0.5)


def test_ensure_weighting_logging_prefers_train_grpo_flag_over_maxent():
    """train_grpo_objective=False should keep MaxEnt disabled when maxent_objective=False."""

    args = SimpleNamespace(
        maxent_tau=0.1,
        beta=0.2,
        train_grpo_objective=False,
        maxent_objective=False,
    )
    metrics = _log_with_args(args)
    assert metrics["train/grpo_objective"] == 0.0
    assert metrics["train/maxent_objective"] == 0.0
    assert metrics["train/weighting/tau"] == pytest.approx(0.1)
    assert metrics["train/weighting/beta"] == pytest.approx(0.2)
