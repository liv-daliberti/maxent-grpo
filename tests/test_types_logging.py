"""Unit tests for training.types.logging helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.types.logging import (
    LoggingHandles,
    LoggingConfigView,
    LogStepArtifacts,
    RewardComponentStats,
    RewardLoggingView,
    TokenUsageStats,
    TrainingMetricsPayload,
    TrainingScalarStats,
)


class _SpyWriter:
    def __init__(self):
        self.logged = []
        self.flushed = False

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def flush(self):
        self.flushed = True


def test_logging_handles_step_logger_flushes(monkeypatch):
    writer = _SpyWriter()
    handles = LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_: None,
        save_strategy="steps",
        save_steps=1,
        wandb_run=None,
    )
    with handles.step_logger(step=5) as logger:
        logger.log({"a": 1})
    assert writer.logged == [({"a": 1}, 5)]
    assert writer.flushed is True

    # Disabled logger yields noop
    with handles.step_logger(step=10, enabled=False) as logger:
        logger.log({"b": 2})
    assert writer.logged == [({"a": 1}, 5)]


def test_log_step_artifacts_as_dict():
    artifacts = LogStepArtifacts(
        loss_outputs="loss",
        diagnostics="diag",
        grad_norm_scalar=0.5,
        epoch_progress=0.25,
    )
    d = artifacts.as_dict()
    assert d["loss_outputs"] == "loss"
    assert d["grad_norm_scalar"] == 0.5
    assert d["epoch_progress"] == 0.25


def test_training_scalar_stats_token_usage_proxy():
    tokens = TokenUsageStats(
        avg_completion_tokens=1.0,
        num_completion_tokens=2.0,
        num_input_tokens=3.0,
    )
    stats = TrainingScalarStats(
        ref_logp_mean=0.0,
        tokens=tokens,
        current_lr=1e-4,
        grad_norm_scalar=None,
        epoch_progress=0.1,
        vllm_latency_ms=None,
    )
    stats.avg_completion_tokens = 1.5
    stats.num_completion_tokens = 4.0
    stats.num_input_tokens = 5.0
    assert stats.tokens.avg_completion_tokens == 1.5
    assert stats.tokens.num_completion_tokens == 4.0
    assert stats.tokens.num_input_tokens == 5.0


def test_training_metrics_payload_accepts_views():
    reward_stats = RewardLoggingView(
        reward_mean=1.0,
        reward_std=0.1,
        frac_zero_std=0.0,
        advantage_mean=0.2,
        advantage_std=0.05,
        advantage_count=2,
        per_reward={"acc": RewardComponentStats(mean=1.0, std=0.0)},
    )
    weight_stats = SimpleNamespace()  # WeightLoggingView placeholder
    payload = TrainingMetricsPayload(
        reward_stats=reward_stats,
        weight_stats=weight_stats,
        loss_outputs="loss",
        diagnostics="diag",
        length_stats="lengths",
        config=LoggingConfigView(
            weighting="w_cfg",
            clipping="c_cfg",
            schedule="sched",
        ),
        scalars=TrainingScalarStats(
            ref_logp_mean=0.0,
            tokens=TokenUsageStats(
                avg_completion_tokens=1.0,
                num_completion_tokens=2.0,
                num_input_tokens=3.0,
            ),
            current_lr=1e-3,
            grad_norm_scalar=0.1,
            epoch_progress=0.5,
            vllm_latency_ms=1.2,
        ),
    )
    assert payload.reward_stats.reward_mean == 1.0
    assert payload.weight_stats is weight_stats
