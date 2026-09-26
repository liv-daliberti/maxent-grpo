from __future__ import annotations

import math

import pytest

from oat_drgrpo.semantic_rms_controller import SemanticRmsController


def controller(**overrides) -> SemanticRmsController:
    values = dict(
        base_coefficient=0.10,
        target_ratio=0.05,
        warmup_steps=4,
        ema_decay=0.5,
    )
    values.update(overrides)
    return SemanticRmsController(**values)


def drive(ctrl: SemanticRmsController, *, ratio: float, steps: int,
          task_rms: float = 0.20, eligible: float = 0.5) -> float:
    for _ in range(steps):
        ctrl.observe(
            semantic_rms=ratio * task_rms,
            task_rms=task_rms,
            eligible_fraction=eligible,
        )
    return float(ctrl.current_coefficient)


# --------------------------------------------------------------------------
# the singleton trap: a refused observation must not read as a zero ratio
# --------------------------------------------------------------------------


def test_zero_semantic_signal_freezes_rather_than_raising_the_coefficient():
    ctrl = controller()
    start = ctrl.current_coefficient
    for _ in range(500):
        # a prompt with one verified mode: centered surprisal is exactly zero
        ctrl.observe(semantic_rms=0.0, task_rms=0.20, eligible_fraction=0.5)
    assert ctrl.current_coefficient == start
    assert ctrl.observations == 0
    assert ctrl.frozen_observations == 500
    assert ctrl.diagnostics()["semantic_rms_controller_frozen_fraction"] == 1.0


def test_low_eligibility_freezes():
    ctrl = controller(min_eligible_fraction=0.05)
    start = ctrl.current_coefficient
    drive(ctrl, ratio=0.001, steps=200, eligible=0.01)
    assert ctrl.current_coefficient == start
    assert ctrl.observations == 0


def test_degenerate_task_advantage_freezes():
    # an all-correct or all-wrong group centres to ~zero task advantage; the
    # ratio is then meaningless and must not drive the coefficient.
    ctrl = controller()
    start = ctrl.current_coefficient
    drive(ctrl, ratio=0.001, steps=200, task_rms=1e-9)
    assert ctrl.current_coefficient == start
    assert ctrl.observations == 0


def test_non_finite_observations_are_refused():
    ctrl = controller()
    start = ctrl.current_coefficient
    for bad in (float("nan"), float("inf")):
        ctrl.observe(semantic_rms=bad, task_rms=0.2, eligible_fraction=0.5)
        ctrl.observe(semantic_rms=0.01, task_rms=bad, eligible_fraction=0.5)
    assert ctrl.current_coefficient == start
    assert ctrl.observations == 0


def test_a_frozen_run_never_leaves_its_starting_dose():
    # The whole point: if the signal is never usable, the arm degrades to the
    # fixed-coefficient arm rather than to something unbounded.
    ctrl = controller()
    for _ in range(3072):
        ctrl.observe(semantic_rms=0.0, task_rms=0.0, eligible_fraction=0.0)
    assert ctrl.current_coefficient == pytest.approx(0.10)


# --------------------------------------------------------------------------
# convergence toward the registered ratio
# --------------------------------------------------------------------------


def test_under_pressure_raises_the_coefficient_toward_target():
    ctrl = controller()
    # realized 1% against a 5% target: the coefficient must rise
    final = drive(ctrl, ratio=0.01, steps=300)
    assert final > 0.10
    assert final <= ctrl.max_coefficient


def test_over_pressure_lowers_the_coefficient_toward_target():
    ctrl = controller()
    final = drive(ctrl, ratio=0.20, steps=300)
    assert final < 0.10
    assert final >= ctrl.min_coefficient


def test_a_run_already_at_target_barely_moves():
    ctrl = controller()
    final = drive(ctrl, ratio=0.05, steps=300)
    assert final == pytest.approx(0.10, rel=0.05)


def test_realized_ratio_converges_when_pressure_scales_with_the_coefficient():
    # Closed loop: realized pressure is proportional to the coefficient, which
    # is the regime the real estimator is in.
    ctrl = controller(warmup_steps=8, ema_decay=0.9)
    task_rms, k = 0.20, 0.25  # ratio = k * eta; target .05 needs eta = .20
    for _ in range(2000):
        eta = float(ctrl.current_coefficient)
        ctrl.observe(
            semantic_rms=k * eta * task_rms,
            task_rms=task_rms,
            eligible_fraction=0.5,
        )
    realized = k * float(ctrl.current_coefficient)
    assert realized == pytest.approx(0.05, rel=0.10)


# --------------------------------------------------------------------------
# bounds, damping, and reporting
# --------------------------------------------------------------------------


def test_coefficient_never_escapes_its_registered_bounds():
    for ratio in (1e-4, 1e6):
        ctrl = controller()
        drive(ctrl, ratio=ratio, steps=2000)
        assert ctrl.min_coefficient <= ctrl.current_coefficient <= ctrl.max_coefficient


def test_per_step_change_is_capped():
    ctrl = controller(max_step_ratio=1.1, warmup_steps=0, ema_decay=0.0)
    before = float(ctrl.current_coefficient)
    ctrl.observe(semantic_rms=1e-5, task_rms=0.20, eligible_fraction=0.5)
    after = float(ctrl.current_coefficient)
    assert after <= before * 1.1 + 1e-12


def test_bound_pinning_is_reported_so_a_mis_specified_controller_is_visible():
    ctrl = controller()
    drive(ctrl, ratio=1e-4, steps=2000)
    diagnostics = ctrl.diagnostics()
    assert diagnostics["semantic_rms_controller_at_max"] == 1.0
    assert diagnostics["semantic_rms_controller_bound_hit_fraction"] > 0.2


def test_warmup_only_fills_the_emas():
    ctrl = controller(warmup_steps=50)
    drive(ctrl, ratio=0.001, steps=50)
    assert ctrl.current_coefficient == pytest.approx(0.10)
    assert ctrl.observations == 50
    assert ctrl.updates_applied == 0


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def test_state_survives_an_exact_resume():
    original = controller()
    drive(original, ratio=0.01, steps=120)
    resumed = controller()
    resumed.load_state_dict(original.state_dict())
    assert resumed.current_coefficient == original.current_coefficient
    assert resumed.observations == original.observations
    for _ in range(40):
        a = original.observe(semantic_rms=0.004, task_rms=0.2, eligible_fraction=0.5)
        b = resumed.observe(semantic_rms=0.004, task_rms=0.2, eligible_fraction=0.5)
        assert a == pytest.approx(b)


def test_resume_rejects_a_coefficient_outside_the_registered_bounds():
    ctrl = controller()
    state = ctrl.state_dict()
    state["current_coefficient"] = 5.0
    with pytest.raises(ValueError, match="outside its registered bounds"):
        ctrl.load_state_dict(state)


def test_resume_rejects_a_foreign_schema():
    ctrl = controller()
    state = ctrl.state_dict()
    state["schema"] = "something_else_v1"
    with pytest.raises(ValueError, match="schema mismatch"):
        ctrl.load_state_dict(state)


# --------------------------------------------------------------------------
# construction guards
# --------------------------------------------------------------------------


def test_construction_rejects_incoherent_configuration():
    with pytest.raises(ValueError):
        controller(base_coefficient=0.0)
    with pytest.raises(ValueError):
        controller(target_ratio=-1.0)
    with pytest.raises(ValueError):
        controller(min_coefficient=0.5, max_coefficient=0.1)
    with pytest.raises(ValueError):
        controller(base_coefficient=1.0)  # outside [min, max]
    with pytest.raises(ValueError):
        controller(max_step_ratio=1.0)
    with pytest.raises(ValueError):
        controller(ema_decay=1.0)
