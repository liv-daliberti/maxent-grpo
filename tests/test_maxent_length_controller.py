from __future__ import annotations

import copy
import math

import pytest

from oat_drgrpo.maxent_length_controller import MaxEntLengthController


def _controller(**overrides) -> MaxEntLengthController:
    values = {
        "target_length": 32.0,
        "init_lambda": 0.001,
        "max_lambda": 0.01,
        "dual_lr": 0.002,
        "ema_decay": 0.5,
    }
    values.update(overrides)
    return MaxEntLengthController(**values)


def test_ema_starts_at_target_and_dual_step_has_correct_sign():
    controller = _controller()

    assert controller.length_ema == pytest.approx(32.0)
    assert controller.current_lambda == pytest.approx(0.001)

    high = controller.observe(64.0)
    assert controller.length_ema == pytest.approx(48.0)
    assert high["maxent_length_relative_violation"] == pytest.approx(0.5)
    assert controller.current_lambda == pytest.approx(0.002)

    low = controller.observe(0.0)
    assert controller.length_ema == pytest.approx(24.0)
    assert low["maxent_length_relative_violation"] == pytest.approx(-0.25)
    assert controller.current_lambda == pytest.approx(0.0015)


def test_observe_returns_exact_telemetry_schema_and_values():
    controller = _controller()

    diagnostics = controller.observe(64.0)

    assert set(diagnostics) == {
        "maxent_length_target",
        "maxent_length_lambda_next",
        "maxent_length_lambda_max",
        "maxent_length_ema",
        "maxent_length_ema_decay",
        "maxent_length_relative_violation",
        "maxent_length_dual_lr",
        "maxent_length_observed_length",
        "maxent_length_observations",
    }
    assert diagnostics == pytest.approx(
        {
            "maxent_length_target": 32.0,
            "maxent_length_lambda_next": 0.002,
            "maxent_length_lambda_max": 0.01,
            "maxent_length_ema": 48.0,
            "maxent_length_ema_decay": 0.5,
            "maxent_length_relative_violation": 0.5,
            "maxent_length_dual_lr": 0.002,
            "maxent_length_observed_length": 64.0,
            "maxent_length_observations": 1.0,
        }
    )


def test_projected_update_hits_both_bounds():
    lower = _controller(
        target_length=10.0,
        init_lambda=0.001,
        dual_lr=0.01,
        ema_decay=0.0,
    )
    lower.observe(0.0)
    assert lower.current_lambda == 0.0

    upper = _controller(
        target_length=10.0,
        init_lambda=0.009,
        dual_lr=0.01,
        ema_decay=0.0,
    )
    upper.observe(20.0)
    assert upper.current_lambda == pytest.approx(0.01)


def test_target_observation_leaves_lambda_unchanged():
    controller = _controller()

    diagnostics = controller.observe(32.0)

    assert diagnostics["maxent_length_relative_violation"] == 0.0
    assert controller.current_lambda == pytest.approx(0.001)


def test_state_round_trip_preserves_next_update():
    original = _controller()
    original.observe(64.0)
    original.observe(40.0)
    state = original.state_dict()

    assert state["controller_kind"] == "maxent_length_dual"
    assert state["controller_rule"] == "projected_relative_ema_v1"
    assert state["length_units"] == "generated_response_tokens_v1"

    restored = _controller()
    restored.load_state_dict(state)

    assert restored.current_lambda == pytest.approx(original.current_lambda)
    assert restored.length_ema == pytest.approx(original.length_ema)
    assert restored.observation_count == original.observation_count
    assert restored.observe(24.0) == pytest.approx(original.observe(24.0))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"target_length": 0.0}, "target_length"),
        ({"target_length": math.inf}, "target_length"),
        ({"init_lambda": -1e-3}, "init_lambda"),
        ({"init_lambda": math.nan}, "init_lambda"),
        ({"max_lambda": 0.0}, "max_lambda"),
        ({"max_lambda": 5e-4}, "max_lambda"),
        ({"dual_lr": 0.0}, "dual_lr"),
        ({"dual_lr": math.inf}, "dual_lr"),
        ({"ema_decay": -0.1}, "ema_decay"),
        ({"ema_decay": 1.0}, "ema_decay"),
    ],
)
def test_rejects_invalid_configuration(override, message):
    with pytest.raises(ValueError, match=message):
        _controller(**override)


@pytest.mark.parametrize("observed", [-1.0, math.nan, math.inf])
def test_rejects_invalid_observations(observed):
    with pytest.raises(ValueError, match="observed_length"):
        _controller().observe(observed)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("controller_kind", "other", "different length controller"),
        ("controller_rule", "other", "incompatible length-control rule"),
        ("length_units", "tokens_v0", "incompatible response-length units"),
        ("target_length", 31.0, "different target_length"),
        ("init_lambda", 0.002, "different init_lambda"),
        ("max_lambda", 0.02, "different max_lambda"),
        ("dual_lr", 0.003, "different dual_lr"),
        ("ema_decay", 0.4, "different ema_decay"),
        ("current_lambda", -0.1, "checkpoint current_lambda"),
        ("current_lambda", 0.02, "checkpoint current_lambda"),
        ("length_ema", -1.0, "checkpoint length_ema"),
        ("observation_count", -1, "observation_count"),
        ("observation_count", 1.5, "observation_count"),
    ],
)
def test_rejects_incompatible_or_invalid_checkpoint_state(key, value, message):
    state = copy.deepcopy(_controller().state_dict())
    state[key] = value

    with pytest.raises(ValueError, match=message):
        _controller().load_state_dict(state)


@pytest.mark.parametrize(
    "missing",
    [
        "target_length",
        "current_lambda",
        "length_ema",
        "observation_count",
    ],
)
def test_rejects_missing_checkpoint_fields(missing):
    state = _controller().state_dict()
    del state[missing]

    with pytest.raises(ValueError, match=missing):
        _controller().load_state_dict(state)


def test_rejects_non_dictionary_checkpoint():
    with pytest.raises(ValueError, match="dictionary"):
        _controller().load_state_dict([])
