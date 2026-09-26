from __future__ import annotations

import pytest

from oat_drgrpo.maxent_controllers import (
    MaxEntDualController,
    MaxEntInverseController,
    MaxEntProportionalController,
)


def test_inverse_controller_uses_own_warmup_without_projection():
    controller = MaxEntInverseController(
        base_alpha=0.000075,
        warmup_steps=2,
        ema_decay=0.0,
        entropy_units="conditional_content_token_nats_mean_v1",
        observation_metric_key="maxent_conditional_token_entropy",
    )

    controller.observe(2.0)
    warmup = controller.observe(1.0)
    assert controller.current_alpha == pytest.approx(0.000075)
    assert warmup["maxent_inverse_reference_entropy"] == pytest.approx(1.5)

    low = controller.observe(0.15)
    assert controller.current_alpha == pytest.approx(0.00075)
    assert low["maxent_inverse_multiplier"] == pytest.approx(10.0)
    assert low["maxent_inverse_projection_active"] == 0.0

    controller.observe(3.0)
    assert controller.current_alpha == pytest.approx(0.0000375)


def test_inverse_controller_is_unbounded_and_fails_closed_at_zero():
    controller = MaxEntInverseController(
        base_alpha=0.000075,
        warmup_steps=1,
        ema_decay=0.0,
    )
    controller.observe(1.0)
    controller.observe(1e-12)

    assert controller.current_alpha == pytest.approx(75_000_000.0)
    with pytest.raises(ValueError, match="EMA must remain positive"):
        controller.observe(0.0)


def test_inverse_controller_resume_is_exact_and_unit_bound():
    controller = MaxEntInverseController(
        base_alpha=0.000075,
        warmup_steps=2,
        ema_decay=0.9,
        entropy_units="conditional_content_token_nats_mean_v1",
        observation_metric_key="maxent_conditional_token_entropy",
    )
    for entropy in (1.2, 1.0, 0.4):
        controller.observe(entropy)
    state = controller.state_dict()

    restored = MaxEntInverseController(
        base_alpha=0.000075,
        warmup_steps=2,
        ema_decay=0.9,
        entropy_units="conditional_content_token_nats_mean_v1",
        observation_metric_key="maxent_conditional_token_entropy",
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    wrong_units = MaxEntInverseController(
        base_alpha=0.000075,
        warmup_steps=2,
        ema_decay=0.9,
    )
    with pytest.raises(ValueError, match="incompatible MaxEnt entropy units"):
        wrong_units.load_state_dict(state)


def test_proportional_controller_raises_alpha_when_entropy_is_low():
    controller = MaxEntProportionalController(
        base_alpha=0.05,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=2,
        ema_decay=0.0,
        gain=2.0,
    )
    controller.observe(1.0)
    controller.observe(1.0)
    diagnostics = controller.observe(0.5)

    assert controller.target_entropy == pytest.approx(0.8)
    assert controller.current_alpha > 0.05
    assert diagnostics["maxent_control_deficit"] == pytest.approx(0.3)


def test_proportional_controller_never_weakens_fixed_treatment():
    controller = MaxEntProportionalController(
        base_alpha=0.05,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=1,
        ema_decay=0.0,
        gain=2.0,
    )
    controller.observe(1.0)
    controller.observe(1.2)

    assert controller.current_alpha == pytest.approx(0.05)


def test_proportional_controller_uses_relative_deficit_and_reaches_ceiling():
    controller = MaxEntProportionalController(
        base_alpha=0.05,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=64,
        ema_decay=0.0,
        gain=1.0,
        configured_target_entropy=0.032,
    )

    diagnostics = controller.observe(0.0)

    assert controller.current_alpha == pytest.approx(0.5)
    assert diagnostics["maxent_control_relative_deficit"] == pytest.approx(1.0)
    assert diagnostics["maxent_control_observations"] == 1


def test_configured_dual_target_updates_from_first_observation():
    controller = MaxEntDualController(
        base_alpha=0.05,
        min_alpha=0.005,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=64,
        alpha_lr=0.03,
        configured_target_entropy=0.032,
    )

    diagnostics = controller.observe(0.01)

    assert controller.current_alpha > 0.05
    assert diagnostics["maxent_dual_optimizer_steps"] == 1
    assert diagnostics["maxent_dual_target_entropy"] == pytest.approx(0.032)


def test_dual_controller_uses_alpha_directly_with_correct_sign():
    controller = MaxEntDualController(
        base_alpha=0.05,
        min_alpha=0.005,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=1,
        alpha_lr=0.1,
        ema_decay=0.0,
    )
    controller.observe(1.0)
    before = controller.current_alpha
    low = controller.observe(0.5)
    after_low = controller.current_alpha
    high = controller.observe(1.2)

    assert before == pytest.approx(0.05)
    assert after_low > before
    assert low["maxent_dual_entropy_error"] < 0
    assert high["maxent_dual_entropy_error"] > 0


def test_dual_controller_defaults_to_responsive_entropy_ema():
    controller = MaxEntDualController(
        base_alpha=0.05,
        min_alpha=0.005,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=64,
        alpha_lr=0.01,
        configured_target_entropy=1.0,
    )

    first = controller.observe(0.5)
    second = controller.observe(1.5)

    assert controller.ema_decay == pytest.approx(0.7)
    assert first["maxent_dual_observed_entropy"] == pytest.approx(0.5)
    assert first["maxent_dual_entropy_ema"] == pytest.approx(0.5)
    assert second["maxent_dual_observed_entropy"] == pytest.approx(1.5)
    assert second["maxent_dual_entropy_ema"] == pytest.approx(0.8)
    assert second["maxent_dual_entropy_error"] == pytest.approx(-0.2)
    assert second["maxent_dual_ema_decay"] == pytest.approx(0.7)


def test_controller_state_round_trip():
    source = MaxEntDualController(
        base_alpha=0.05,
        min_alpha=0.005,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=1,
        alpha_lr=0.01,
    )
    source.observe(1.0)
    source.observe(0.5)
    restored = MaxEntDualController(
        base_alpha=0.05,
        min_alpha=0.005,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=1,
        alpha_lr=0.01,
    )
    restored.load_state_dict(source.state_dict())

    assert restored.current_alpha == pytest.approx(source.current_alpha)
    assert restored.target_entropy == pytest.approx(source.target_entropy)
    assert restored.entropy_ema == pytest.approx(source.entropy_ema)
    assert restored.observation_count == source.observation_count


def test_dual_controller_rejects_instantaneous_feedback_checkpoint():
    controller = MaxEntDualController(
        base_alpha=0.05,
        min_alpha=0.005,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=1,
        alpha_lr=0.01,
    )
    old_state = controller.state_dict()
    old_state["controller_rule"] = "log_alpha_adam_v1"
    old_state.pop("entropy_ema")
    old_state.pop("ema_decay")

    with pytest.raises(ValueError, match="incompatible MaxEnt dual rule"):
        controller.load_state_dict(old_state)


@pytest.mark.parametrize(
    "old_units", [None, "sequence_nats_per_tmax"]
)
def test_standard_maxent_controller_rejects_old_entropy_units(old_units):
    controller = MaxEntProportionalController(
        base_alpha=0.05,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=1,
        ema_decay=0.0,
        gain=2.0,
    )

    with pytest.raises(ValueError, match="incompatible MaxEnt entropy units"):
        state = {
            "controller_kind": "maxent_proportional",
            "controller_rule": "relative_deficit_log_span_v1",
            "current_alpha": 0.1,
        }
        if old_units is not None:
            state["entropy_units"] = old_units
        controller.load_state_dict(state)
