import pytest

from oat_drgrpo.online_canonical_controller import (
    OnlineCanonicalDualController,
    OnlineCanonicalPolicyEntropyController,
)


def _controller():
    return OnlineCanonicalDualController(
        base_alpha=0.1,
        min_alpha=0.1,
        max_alpha=0.5,
        target_ratio=0.8,
        alpha_lr=0.01,
        ema_decay=0.0,
    )


def test_online_canonical_dual_moves_alpha_in_haarnoja_direction():
    low = _controller()
    low_diagnostics = low.observe(0.5)
    assert low_diagnostics["online_canonical_dual_entropy_error"] < 0
    assert low.current_alpha > 0.1

    high = OnlineCanonicalDualController(
        base_alpha=0.2,
        min_alpha=0.05,
        max_alpha=0.5,
        target_ratio=0.8,
        alpha_lr=0.01,
        ema_decay=0.0,
    )
    high.observe(0.95)
    assert high.current_alpha < 0.2


def test_online_canonical_dual_idle_does_not_create_observation():
    controller = _controller()
    diagnostics = controller.idle_diagnostics()
    assert controller.observation_count == 0
    assert controller.current_alpha == pytest.approx(0.1)
    assert diagnostics["online_canonical_dual_observation_skipped"] == 1.0


def test_online_canonical_dual_resume_is_exact_and_fail_closed():
    controller = _controller()
    controller.observe(0.6)
    controller.observe(0.7)
    state = controller.state_dict()

    restored = _controller()
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    mismatch = OnlineCanonicalDualController(
        base_alpha=0.1,
        min_alpha=0.1,
        max_alpha=0.5,
        target_ratio=0.7,
        alpha_lr=0.01,
        ema_decay=0.0,
    )
    with pytest.raises(ValueError, match="resume mismatch for target_ratio"):
        mismatch.load_state_dict(state)


def test_online_canonical_dual_supports_no_upper_alpha_projection():
    controller = OnlineCanonicalDualController(
        base_alpha=0.1,
        min_alpha=0.1,
        max_alpha=float("inf"),
        target_ratio=0.8,
        alpha_lr=0.01,
        ema_decay=0.0,
    )
    for _ in range(250):
        controller.observe(0.0)

    assert controller.current_alpha > 0.5
    state = controller.state_dict()
    assert state["max_alpha"] == float("inf")

    restored = OnlineCanonicalDualController(
        base_alpha=0.1,
        min_alpha=0.1,
        max_alpha=float("inf"),
        target_ratio=0.8,
        alpha_lr=0.01,
        ema_decay=0.0,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_policy_entropy_controller_uses_own_warmup_without_projection():
    controller = OnlineCanonicalPolicyEntropyController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.0,
    )
    controller.observe(2.0)
    warmup = controller.observe(1.0)
    assert controller.current_alpha == pytest.approx(0.1)
    assert warmup["online_canonical_policy_entropy_reference"] == pytest.approx(
        1.5
    )

    low = controller.observe(0.3)
    assert controller.current_alpha == pytest.approx(0.5)
    assert low[
        "online_canonical_policy_entropy_normalized_score"
    ] == pytest.approx(5.0)

    controller.observe(3.0)
    assert controller.current_alpha == pytest.approx(0.05)


def test_policy_entropy_controller_inverse_rule_is_unbounded_and_fails_at_zero():
    controller = OnlineCanonicalPolicyEntropyController(
        base_alpha=0.1,
        warmup_steps=1,
        ema_decay=0.0,
    )
    controller.observe(1.0)
    controller.observe(0.001)
    assert controller.current_alpha == pytest.approx(100.0)

    with pytest.raises(ValueError, match="EMA must remain positive"):
        controller.observe(0.0)


def test_policy_entropy_controller_resume_is_exact_and_fail_closed():
    controller = OnlineCanonicalPolicyEntropyController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.5,
    )
    for value in (1.0, 0.8, 0.4):
        controller.observe(value)
    state = controller.state_dict()

    restored = OnlineCanonicalPolicyEntropyController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.5,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    mismatch = OnlineCanonicalPolicyEntropyController(
        base_alpha=0.2,
        warmup_steps=2,
        ema_decay=0.5,
    )
    with pytest.raises(ValueError, match="resume mismatch for base_alpha"):
        mismatch.load_state_dict(state)

    legacy = dict(state)
    legacy["controller_rule"] = (
        "unprojected_warmup_relative_policy_entropy_alpha_v1"
    )
    with pytest.raises(ValueError, match="incompatible policy-entropy rule"):
        restored.load_state_dict(legacy)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_online_canonical_dual_rejects_invalid_ratios(value):
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        _controller().observe(value)
