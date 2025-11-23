"""Tests for logging helpers."""

from __future__ import annotations

from importlib import import_module, reload
from types import SimpleNamespace

import pytest

from test_run_setup_reference import _load_run_setup


@pytest.fixture
def logging_mod(monkeypatch):
    """Load logging helpers with torch/accelerate stubs."""
    _load_run_setup(monkeypatch)
    return reload(import_module("maxent_helpers.run_logging"))


def test_log_training_metrics_emits_only_scalars(logging_mod):
    from maxent_helpers.run_training_types import (
        BatchDiagnostics,
        LengthStats,
        LoggingConfigView,
        LoggingHandles,
        RewardComponentStats,
        RewardLoggingView,
        TrainingMetricsPayload,
        TrainingScalarStats,
        TokenUsageStats,
        WeightLoggingView,
    )

    captured = {}

    logging_cfg = LoggingHandles(
        log_metrics=lambda metrics, step: captured.setdefault("metrics", metrics),
        save_checkpoint=lambda *_args, **_kwargs: None,
        save_strategy="no",
        save_steps=0,
        wandb_run=None,
    )

    reward_stats = RewardLoggingView(
        reward_mean=1.0,
        reward_std=0.5,
        frac_zero_std=0.0,
        advantage_mean=0.1,
        advantage_std=0.01,
        advantage_count=8,
        per_reward={
            "accuracy": RewardComponentStats(mean=0.2, std=0.05),
            "format": RewardComponentStats(mean=-0.1, std=0.02),
        },
    )
    weight_stats = WeightLoggingView(
        entropy=0.3,
        entropy_min=0.1,
        entropy_max=0.5,
        advantage_entropy_mean=0.04,
        advantage_entropy_std=0.01,
    )
    payload = TrainingMetricsPayload(
        reward_stats=reward_stats,
        weight_stats=weight_stats,
        loss_outputs=SimpleNamespace(
            total_loss_scalar=0.4,
            kl_loss_scalar=0.2,
            weighted_kl_loss_scalar=0.25,
            clip_loss_scalar=None,
        ),
        diagnostics=BatchDiagnostics(
            kl_value=0.2,
            clip_ratio=0.1,
            clip_ratio_low_mean=0.02,
            clip_ratio_low_min=0.01,
            clip_ratio_high_mean=0.15,
            clip_ratio_high_max=0.2,
            clip_ratio_region_mean=0.05,
        ),
        length_stats=LengthStats(
            min_length=5.0,
            mean_length=10.0,
            max_length=20.0,
            clipped_ratio=0.0,
            min_terminated=5.0,
            mean_terminated=10.0,
            max_terminated=20.0,
        ),
        config=LoggingConfigView(
            weighting=SimpleNamespace(beta=0.5, tau=0.2),
            clipping=SimpleNamespace(
                clip_range=0.1,
                clip_adv_baseline=None,
                clip_objective_coef=1.0,
            ),
            schedule=SimpleNamespace(num_generations=4),
        ),
        scalars=TrainingScalarStats(
            ref_logp_mean=-1.0,
            tokens=TokenUsageStats(
                avg_completion_tokens=12.0,
                num_completion_tokens=96.0,
                num_input_tokens=128.0,
            ),
            current_lr=1e-4,
            grad_norm_scalar=0.5,
            epoch_progress=1.5,
            vllm_latency_ms=25.0,
        ),
    )

    metrics = logging_mod.log_training_metrics(logging_cfg, global_step=10, payload=payload)
    metrics = captured["metrics"]
    assert metrics, "No metrics were emitted"
    assert all(not isinstance(val, list) for val in metrics.values())
    assert metrics == logging_mod.build_training_metrics_dict(payload, 10)
