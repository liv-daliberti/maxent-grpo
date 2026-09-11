import numpy as np
import pytest

from oat_drgrpo.outcome_collision import (
    add_outcome_collision_outside_centering_advantage,
    compute_outcome_collision_bonuses,
)


def test_collision_penalty_matches_pair_count_definition():
    bonuses, diagnostics = compute_outcome_collision_bonuses(
        ["a", "a", "a", "b"],
        num_samples=4,
        coefficient=0.1,
    )

    assert bonuses == pytest.approx([-0.05, -0.05, -0.05, 0.0])
    assert diagnostics.collision_rate == pytest.approx(0.375)
    assert diagnostics.bonus_mean == pytest.approx(-0.0375)
    assert diagnostics.bonus_min == pytest.approx(-0.05)
    assert diagnostics.bonus_max == pytest.approx(0.0)
    assert diagnostics.distinct_outcomes_mean == pytest.approx(2.0)
    assert diagnostics.distinct_fraction == pytest.approx(0.5)


def test_none_rows_share_one_invalid_outcome():
    bonuses, diagnostics = compute_outcome_collision_bonuses(
        [None, None, "a", "b"],
        num_samples=4,
        coefficient=0.2,
    )

    assert bonuses == pytest.approx([-0.05, -0.05, 0.0, 0.0])
    assert diagnostics.invalid_fraction == pytest.approx(0.5)
    assert diagnostics.parseable_fraction == pytest.approx(0.5)
    assert diagnostics.distinct_outcomes_mean == pytest.approx(3.0)


def test_candidate_groups_do_not_collide_with_each_other():
    bonuses, diagnostics = compute_outcome_collision_bonuses(
        ["a", "b", "a", "c"],
        num_samples=2,
        coefficient=0.1,
    )

    assert bonuses == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert diagnostics.collision_rate == pytest.approx(0.0)
    assert diagnostics.distinct_outcomes_mean == pytest.approx(2.0)


def test_zero_coefficient_returns_bitwise_zero_bonuses():
    bonuses, diagnostics = compute_outcome_collision_bonuses(
        ["same", "same", None, None],
        num_samples=4,
        coefficient=0.0,
    )

    assert bonuses == [0.0, 0.0, 0.0, 0.0]
    assert diagnostics.bonus_mean == 0.0
    assert diagnostics.collision_rate == pytest.approx(0.25)


def test_outside_centering_stays_active_for_fully_collapsed_group():
    num_samples = 16
    task_rewards = np.ones((num_samples, 1), dtype=np.float32)
    centered_task_advantages = task_rewards - task_rewards.mean(
        axis=0, keepdims=True
    )
    bonuses, diagnostics = compute_outcome_collision_bonuses(
        ["same-answer"] * num_samples,
        num_samples=num_samples,
        coefficient=0.1,
    )
    semantic_advantages = np.asarray(bonuses, dtype=np.float32).reshape(-1, 1)

    combined = add_outcome_collision_outside_centering_advantage(
        centered_task_advantages,
        semantic_advantages,
    )
    e37_centered_augmented = (
        task_rewards + semantic_advantages
        - (task_rewards + semantic_advantages).mean(axis=0, keepdims=True)
    )

    assert centered_task_advantages.tolist() == [[0.0]] * num_samples
    assert semantic_advantages == pytest.approx(
        np.full((num_samples, 1), -0.09375, dtype=np.float32)
    )
    assert diagnostics.collision_rate == pytest.approx(15 / 16)
    assert e37_centered_augmented.tolist() == [[0.0]] * num_samples
    assert np.count_nonzero(combined) == num_samples
    assert combined == pytest.approx(semantic_advantages)


def test_outside_centering_rejects_accidental_broadcasting():
    with pytest.raises(ValueError, match="identical shapes"):
        add_outcome_collision_outside_centering_advantage(
            np.zeros((4, 1), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("keys", "num_samples", "coefficient", "message"),
    [
        (["a"], 1, 0.1, "greater than one"),
        (["a", "b", "c"], 2, 0.1, "complete candidate groups"),
        (["a", "b"], 2, -0.1, "finite and non-negative"),
        (["a", "b"], 2, float("nan"), "finite and non-negative"),
        (["a", "b"], 2, float("inf"), "finite and non-negative"),
    ],
)
def test_invalid_collision_configuration_is_rejected(
    keys, num_samples, coefficient, message
):
    with pytest.raises(ValueError, match=message):
        compute_outcome_collision_bonuses(
            keys,
            num_samples=num_samples,
            coefficient=coefficient,
        )
