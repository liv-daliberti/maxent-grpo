"""Unit tests for controller objective helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxent_grpo.training.controller_objective import (
    AnalyticControllerObjective,
    ControllerMetaContext,
    ControllerGradients,
    TruncatedBackpropControllerObjective,
    build_controller_objective,
)
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightingSettings,
)
from maxent_grpo.config import GRPOConfig


def _weighting(train_grpo: bool = False) -> WeightingSettings:
    return WeightingSettings(
        tau=0.5,
        beta=0.2,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=True),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=0.4, learning_rate=0.1, minimum_value=0.0, maximum_value=1.0, warmup_steps=0
        ),
        kl_controller=KlControllerSettings(target=0.3, horizon=8, step_size=0.1),
        train_grpo_objective=train_grpo,
    )


def _meta_ctx(weighting: WeightingSettings) -> ControllerMetaContext:
    weight_stats = SimpleNamespace(weight_entropy=0.9)
    loss_outputs = SimpleNamespace(kl_loss_scalar=0.6)
    return ControllerMetaContext(
        weighting=weighting,
        weight_stats=weight_stats,
        loss_outputs=loss_outputs,
        global_step=5,
    )


def test_analytic_objective_returns_differences():
    weighting = _weighting()
    weighting.controller_meta.enabled = True
    ctx = _meta_ctx(weighting)
    grads = AnalyticControllerObjective().compute(ctx)
    assert isinstance(grads, ControllerGradients)
    assert grads.tau_grad == pytest.approx(0.5)
    assert grads.beta_grad == pytest.approx(0.3)


def test_truncated_backprop_uses_callback_when_available():
    weighting = _weighting()
    weighting.controller_meta.enabled = True
    recorded = ControllerGradients(tau_grad=0.1, beta_grad=-0.2)

    def _fake_backprop(steps: int):
        assert steps == 2
        return recorded

    ctx = _meta_ctx(weighting)
    ctx.backprop_fn = _fake_backprop
    grads = TruncatedBackpropControllerObjective(steps=2).compute(ctx)
    assert grads is recorded


def test_truncated_backprop_falls_back_to_analytic():
    weighting = _weighting()
    weighting.controller_meta.enabled = True
    ctx = _meta_ctx(weighting)
    grads = TruncatedBackpropControllerObjective().compute(ctx)
    assert isinstance(grads, ControllerGradients)
    # Falls back to analytic gradients when no callback is provided.
    assert grads.tau_grad == pytest.approx(0.5)


def test_build_controller_objective_handles_unknown_methods():
    cfg = GRPOConfig()
    weighting = _weighting()
    weighting.controller_meta.enabled = True
    weighting.controller_meta.method = "truncated"
    objective = build_controller_objective(cfg, weighting)
    assert isinstance(objective, TruncatedBackpropControllerObjective)
    weighting.controller_meta.method = "unknown"
    objective = build_controller_objective(cfg, weighting)
    assert isinstance(objective, AnalyticControllerObjective)

