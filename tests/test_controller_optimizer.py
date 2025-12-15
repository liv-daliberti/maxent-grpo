"""Tests for the controller meta-optimizer manager."""

from __future__ import annotations

import pytest

from maxent_grpo.config import GRPOConfig
from maxent_grpo.training.controller_optimizer import ControllerMetaManager
from maxent_grpo.training.controller_objective import ControllerGradients
from maxent_grpo.training.weighting.types import (
    WeightingSettings,
    WeightNormalizationSettings,
    QDistributionSettings,
    TauSchedule,
    KlControllerSettings,
    TorchControllerState,
)


def _weighting(train_grpo: bool = False) -> WeightingSettings:
    return WeightingSettings(
        tau=0.5,
        beta=0.2,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=True),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=0.4,
            learning_rate=0.1,
            minimum_value=0.0,
            maximum_value=1.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.3, horizon=8, step_size=0.1),
        train_grpo_objective=train_grpo,
    )


def test_controller_meta_manager_respects_interval():
    cfg = GRPOConfig()
    weighting = _weighting()
    weighting.controller_meta.enabled = True
    weighting.controller_meta.learning_rate = 0.1
    weighting.controller_meta.update_interval = 3
    manager = ControllerMetaManager(cfg, weighting)
    assert manager.should_run(0) is False
    assert manager.should_run(1) is False
    assert manager.should_run(2) is True


def test_controller_meta_manager_optimizer_updates_values(monkeypatch):
    from tests.helpers.run_setup_stubs import install_training_stubs

    install_training_stubs(monkeypatch)
    cfg = GRPOConfig(controller_meta_lr=0.2)
    weighting = _weighting()
    weighting.controller_meta.enabled = True
    weighting.controller_meta.method = "first_order"
    weighting.controller_meta.learning_rate = 0.2
    weighting.controller_meta.update_interval = 1
    manager = ControllerMetaManager(cfg, weighting)
    weighting._meta_entropy_value = 0.9
    weighting._meta_kl_value = 0.1
    backprop_fn = manager.make_backprop_fn()
    if backprop_fn is None:
        grads = ControllerGradients(tau_grad=0.1, beta_grad=-0.05)
    else:
        grads = backprop_fn(1)
        assert grads and grads.has_updates()
    manager.apply_gradients(grads, lr_scale=1.0)
    assert weighting.tau != pytest.approx(0.5)
    assert weighting.beta >= 0.0
    assert getattr(weighting, "_meta_last_tau_grad") != 0.0


def test_controller_meta_manager_analytic_uses_tau_beta_lrs_when_legacy_zero():
    cfg = GRPOConfig(controller_meta_lr=0.0, controller_meta_tau_lr=0.1, controller_meta_beta_lr=0.05)
    weighting = _weighting()
    weighting.controller_meta.enabled = True
    weighting.controller_meta.method = "analytic"
    weighting.controller_meta.learning_rate = 0.0
    weighting.controller_meta.tau_learning_rate = 0.1
    weighting.controller_meta.beta_learning_rate = 0.05
    weighting.controller_meta.update_interval = 1
    manager = ControllerMetaManager(cfg, weighting)
    grads = ControllerGradients(tau_grad=1.0, beta_grad=1.0)
    manager.apply_gradients(grads, lr_scale=1.0)
    assert weighting.tau == pytest.approx(0.4)
    assert weighting.beta == pytest.approx(0.25)


def test_controller_meta_manager_first_order_uses_parameter_grads():
    pytest.importorskip("torch")
    import torch

    try:
        _ = torch.nn.Parameter(torch.tensor(0.0))
    except Exception:
        pytest.skip("torch stub does not support Parameters")

    cfg = GRPOConfig(controller_meta_lr=0.1)
    weighting = _weighting()
    weighting.controller_meta.enabled = True
    weighting.controller_meta.method = "first_order"
    weighting.controller_meta.learning_rate = 0.1
    weighting.controller_state = TorchControllerState(
        torch,
        tau_init=weighting.tau,
        beta_init=weighting.beta,
        requires_grad=True,
    )
    manager = ControllerMetaManager(cfg, weighting)
    state = weighting.controller_state
    assert state is not None
    state.tau_param.grad = torch.tensor(0.5, dtype=torch.float32)
    state.beta_param.grad = torch.tensor(-0.1, dtype=torch.float32)
    backprop_fn = manager.make_backprop_fn()
    assert backprop_fn is not None
    grads = backprop_fn(1)
    assert grads is not None and grads.has_updates()
    manager.apply_gradients(grads, lr_scale=1.0)
    assert weighting.tau != pytest.approx(0.5)
    assert weighting.beta != pytest.approx(0.2)
    assert state.tau_param.grad is None or getattr(state.tau_param.grad, "item", lambda: None)() == 0  # zeroed
