from __future__ import annotations

import copy

import pytest

from oat_drgrpo.proposal_starvation import ProposalStarvationController


def controller() -> ProposalStarvationController:
    return ProposalStarvationController(
        base_max_attempts=1,
        patience_updates=64,
        fallback_max_attempts=4,
        burst_updates=16,
        cooldown_updates=48,
    )


def fail_once(value: ProposalStarvationController) -> None:
    plan = value.plan()
    value.observe(plan, admitted_new_outcomes=0)


def test_starvation_fallback_is_bounded_and_rearms_after_cooldown() -> None:
    value = controller()
    for _ in range(64):
        assert value.plan().max_attempts == 1
        fail_once(value)

    first = value.plan()
    assert first.max_attempts == 4
    assert first.fallback_active
    assert first.fallback_activation
    value.observe(first, admitted_new_outcomes=0)
    for _ in range(15):
        plan = value.plan()
        assert plan.max_attempts == 4
        assert plan.fallback_active
        assert not plan.fallback_activation
        value.observe(plan, admitted_new_outcomes=0)

    assert value.diagnostics()["burst_remaining"] == 0
    assert value.diagnostics()["cooldown_remaining"] == 48
    for _ in range(48):
        assert value.plan().max_attempts == 1
        fail_once(value)
    rearmed = value.plan()
    assert rearmed.max_attempts == 4
    assert rearmed.fallback_activation


def test_only_verified_admission_resets_starvation() -> None:
    value = controller()
    for _ in range(64):
        fail_once(value)
    plan = value.plan()
    value.observe(plan, admitted_new_outcomes=2)

    diagnostics = value.diagnostics()
    assert diagnostics["stalled_eligible_updates"] == 0
    assert diagnostics["burst_remaining"] == 0
    assert diagnostics["cooldown_remaining"] == 0
    assert diagnostics["admitted_new_outcomes"] == 2
    assert value.plan().max_attempts == 1


def test_non_sampling_verified_admission_also_resets_starvation() -> None:
    value = controller()
    for _ in range(80):
        fail_once(value)
    eligible_before = value.diagnostics()["eligible_updates"]

    value.observe_admission_without_opportunity(admitted_new_outcomes=1)

    diagnostics = value.diagnostics()
    assert diagnostics["eligible_updates"] == eligible_before
    assert diagnostics["stalled_eligible_updates"] == 0
    assert diagnostics["admitted_new_outcomes"] == 1


def test_restart_roundtrip_preserves_the_exact_schedule() -> None:
    original = controller()
    for _ in range(70):
        fail_once(original)
    restored = controller()
    restored.load_state_dict(copy.deepcopy(original.state_dict()))

    assert restored.diagnostics() == original.diagnostics()
    assert restored.plan() == original.plan()
    for admitted in (0, 0, 1, 0):
        original.observe(original.plan(), admitted_new_outcomes=admitted)
        restored.observe(restored.plan(), admitted_new_outcomes=admitted)
        assert restored.state_dict() == original.state_dict()


def test_restart_rejects_configuration_drift() -> None:
    payload = controller().state_dict()
    drifted = ProposalStarvationController(
        base_max_attempts=1,
        patience_updates=63,
        fallback_max_attempts=4,
        burst_updates=16,
        cooldown_updates=48,
    )
    with pytest.raises(ValueError, match="config mismatch"):
        drifted.load_state_dict(payload)


def test_fallback_budget_must_exceed_the_base_budget() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        ProposalStarvationController(
            base_max_attempts=1,
            patience_updates=64,
            fallback_max_attempts=1,
            burst_updates=16,
            cooldown_updates=48,
        )
