"""Tests for the MaxEnt weighting helpers."""

from __future__ import annotations

from importlib import import_module, reload
from types import SimpleNamespace

import pytest

from test_run_setup_reference import _load_run_setup


@pytest.fixture
def weighting_mod(monkeypatch):
    """Load run_training_weighting with torch/accelerate stubs applied."""

    _load_run_setup(monkeypatch)
    module = reload(import_module("maxent_helpers.run_training_weighting"))
    types_mod = import_module("maxent_helpers.run_training_types")
    return SimpleNamespace(module=module, types=types_mod)


def _build_weighting(
    weighting_mod,
    *,
    beta,
    tau=0.0,
    kl_target,
    kl_horizon,
    kl_ctl_step_size,
    train_grpo_objective=False,
):
    types = weighting_mod.types
    normalization = types.WeightNormalizationSettings(
        denom=1.0 if train_grpo_objective else (beta + tau if (beta + tau) > 0 else 1.0),
        len_norm_ref=True,
    )
    return types.WeightingSettings(
        tau=tau,
        beta=beta,
        normalization=normalization,
        q_distribution=types.QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=types.TauSchedule(
            target_entropy=None,
            learning_rate=0.0,
            minimum_value=0.0,
            maximum_value=0.0,
            warmup_steps=0,
        ),
        kl_controller=types.KlControllerSettings(
            target=kl_target,
            horizon=kl_horizon,
            step_size=kl_ctl_step_size,
        ),
        train_grpo_objective=train_grpo_objective,
    )


def test_maybe_update_beta_uses_controller_settings(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=3.0,
        kl_target=0.07,
        kl_horizon=50000,
        kl_ctl_step_size=0.15,
    )
    weighting_mod.module.maybe_update_beta(weighting, measured_kl=0.14)
    expected_scale = 1.0 + (0.15 / 50000.0)
    assert weighting.beta == pytest.approx(3.0 * expected_scale)
    assert weighting.denom == pytest.approx(weighting.beta + weighting.tau)


def test_grpo_objective_keeps_unit_denom(weighting_mod):
    weighting = _build_weighting(
        weighting_mod,
        beta=3.0,
        tau=0.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        train_grpo_objective=True,
    )
    weighting_mod.module.maybe_update_beta(weighting, measured_kl=0.14)
    weighting.tau_target_entropy = 0.5
    weighting.tau_lr = 0.1
    weighting.tau_warmup_steps = 0
    weighting_mod.module.maybe_update_tau(
        weighting,
        weight_stats=SimpleNamespace(weight_entropy=0.4),
        global_step=10,
    )
    assert weighting.denom == pytest.approx(1.0)
