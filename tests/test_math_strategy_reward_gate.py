import pytest
import torch

from oat_drgrpo.learner.grpo import (
    MATH_VERIFIED_ANSWER_OUTCOME_KEY,
    apply_math_strategy_task_reward_gate,
    math_verified_answer_outcome_keys,
)


def test_math_verified_answer_track_has_one_validator_bound_outcome():
    assert math_verified_answer_outcome_keys(
        [False, True, True, False]
    ) == [
        None,
        MATH_VERIFIED_ANSWER_OUTCOME_KEY,
        MATH_VERIFIED_ANSWER_OUTCOME_KEY,
        None,
    ]


def test_math_strategy_gate_zeros_both_reward_views_only_when_rejected():
    final = torch.tensor([[1.0], [0.0], [1.0], [0.5]])
    task = torch.tensor([[1.0], [0.0], [1.0], [0.5]])
    gated_final, gated_task = apply_math_strategy_task_reward_gate(
        final,
        task,
        [True, True, False, False],
    )
    expected = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    assert torch.equal(gated_final, expected)
    assert torch.equal(gated_task, expected)
    assert torch.equal(final, torch.tensor([[1.0], [0.0], [1.0], [0.5]]))


def test_math_strategy_gate_rejects_shape_and_mask_drift():
    with pytest.raises(ValueError, match="views differ"):
        apply_math_strategy_task_reward_gate(
            torch.ones(2, 1),
            torch.ones(2),
            [True, True],
        )
    with pytest.raises(ValueError, match="mask differs"):
        apply_math_strategy_task_reward_gate(
            torch.ones(2, 1),
            torch.ones(2, 1),
            [True],
        )
