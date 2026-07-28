from __future__ import annotations

import math

import pytest

from oat_drgrpo.xdr_tau_controller import XdrTauController


def _controller(**overrides):
    config = {
        "base_tau": 0.05,
        "min_tau": 0.005,
        "target_ratio": 0.8,
        "warmup_steps": 4,
        "ema_decay": 0.0,
        "gain": 20.0,
    }
    config.update(overrides)
    return XdrTauController(**config)


def test_controller_calibrates_target_during_fixed_tau_warmup():
    controller = _controller()

    for entropy in [0.30, 0.20, 0.30, 0.20]:
        diagnostics = controller.observe(entropy)

    assert controller.target_entropy == pytest.approx(0.20)
    assert controller.current_tau == pytest.approx(0.05)
    assert diagnostics["xdr_tau_control_target_entropy"] == pytest.approx(0.20)


def test_controller_is_one_sided_and_lowers_tau_only_below_target():
    controller = _controller(warmup_steps=1)
    controller.observe(0.25)  # target = 0.20

    controller.observe(0.30)
    assert controller.current_tau == pytest.approx(0.05)

    controller.observe(0.16)
    expected = 0.05 * math.exp(-20.0 * 0.04)
    assert controller.current_tau == pytest.approx(expected)


def test_controller_respects_minimum_tau():
    controller = _controller(warmup_steps=1)
    controller.observe(0.25)
    controller.observe(0.0)

    assert controller.current_tau == pytest.approx(0.005)


def test_controller_ema_smooths_observations():
    controller = _controller(warmup_steps=1, ema_decay=0.9)
    controller.observe(0.25)
    controller.observe(0.05)

    assert controller.entropy_ema == pytest.approx(0.23)
    assert controller.current_tau == pytest.approx(0.05)


def test_controller_state_round_trip_preserves_next_temperature():
    original = _controller(warmup_steps=1)
    original.observe(0.25)
    original.observe(0.16)

    restored = _controller(warmup_steps=1)
    restored.load_state_dict(original.state_dict())

    assert restored.state_dict() == original.state_dict()


@pytest.mark.parametrize(
    "override",
    [
        {"base_tau": 0.0},
        {"min_tau": 0.0},
        {"min_tau": 0.06},
        {"target_ratio": 0.0},
        {"target_ratio": 1.1},
        {"warmup_steps": 0},
        {"ema_decay": 1.0},
        {"gain": 0.0},
    ],
)
def test_controller_rejects_invalid_configuration(override):
    with pytest.raises(ValueError):
        _controller(**override)
