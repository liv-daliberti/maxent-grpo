from __future__ import annotations

import math

import pytest

from oat_drgrpo.xdr_sac_dual_controller import XdrSacDualController


def _controller(**overrides):
    config = {
        "base_tau": 0.05,
        "min_tau": 0.005,
        "max_tau": 0.5,
        "target_ratio": 0.8,
        "warmup_steps": 2,
        "alpha_lr": 0.1,
        "beta1": 0.0,
        "beta2": 0.0,
    }
    config.update(overrides)
    return XdrSacDualController(**config)


def test_sac_dual_calibrates_target_then_takes_signed_step():
    controller = _controller()

    controller.observe(0.25)
    assert controller.current_tau == pytest.approx(0.05)

    diagnostics = controller.observe(0.25)
    assert controller.target_entropy == pytest.approx(0.20)
    assert controller.log_alpha == pytest.approx(-0.1, abs=1e-6)
    assert controller.current_tau == pytest.approx(0.05 * math.exp(0.1))
    assert diagnostics["xdr_sac_dual_entropy_error"] == pytest.approx(0.05)


def test_sac_dual_is_two_sided_and_accumulates_log_alpha():
    controller = _controller(warmup_steps=1)
    controller.observe(0.25)  # above target: weaken xDr by raising tau
    relaxed_tau = controller.current_tau
    assert relaxed_tau > controller.base_tau

    controller.observe(0.10)  # below target: strengthen xDr by lowering tau
    assert controller.current_tau < relaxed_tau
    assert controller.log_alpha == pytest.approx(0.0, abs=1e-6)

    controller.observe(0.10)
    assert controller.current_tau < controller.base_tau


def test_sac_dual_projects_temperature_bounds():
    low = _controller(warmup_steps=1, alpha_lr=10.0)
    low.observe(0.25)
    assert low.current_tau == pytest.approx(0.5)

    high = _controller(warmup_steps=1, target_ratio=1.0, alpha_lr=10.0)
    high.observe(0.25)
    high.observe(0.0)
    assert high.current_tau == pytest.approx(0.005)


def test_sac_dual_state_round_trip_preserves_optimizer_state():
    original = _controller(warmup_steps=1)
    original.observe(0.25)
    original.observe(0.10)

    restored = _controller(warmup_steps=1)
    restored.load_state_dict(original.state_dict())

    assert restored.state_dict() == original.state_dict()
    assert restored.current_tau == pytest.approx(original.current_tau)


@pytest.mark.parametrize(
    "override",
    [
        {"base_tau": 0.0},
        {"min_tau": 0.0},
        {"min_tau": 0.06},
        {"max_tau": 0.04},
        {"target_ratio": 0.0},
        {"warmup_steps": 0},
        {"alpha_lr": 0.0},
        {"beta1": 1.0},
        {"beta2": 1.0},
    ],
)
def test_sac_dual_rejects_invalid_configuration(override):
    with pytest.raises(ValueError):
        _controller(**override)
