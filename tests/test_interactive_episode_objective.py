import pytest
import torch

from oat_drgrpo.interactive_episode_objective import (
    add_verified_advantage_outside_centering,
    drgrpo_task_advantages,
    length_neutral_clipped_episode_loss,
)


def test_drgrpo_centers_each_prompt_without_variance_scaling():
    rewards = torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    observed = drgrpo_task_advantages(rewards)
    expected = torch.tensor(
        [[-2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [0.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(observed, expected)


def test_verified_advantage_is_added_after_centering_and_detached():
    task = torch.tensor([[0.25, -0.25]], requires_grad=True)
    verified = torch.tensor([[0.1, -0.2]], requires_grad=True)
    combined = add_verified_advantage_outside_centering(task, verified)
    combined.sum().backward()
    torch.testing.assert_close(combined, torch.tensor([[0.35, -0.45]]))
    torch.testing.assert_close(task.grad, torch.ones_like(task))
    assert verified.grad is None


def test_episode_loss_does_not_upweight_longer_trajectories():
    new = torch.tensor([[0.1, 0.0, 0.0], [-0.2, -0.2, -0.2]])
    old = torch.zeros_like(new)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    advantages = torch.tensor([1.0, -1.0])
    observed = length_neutral_clipped_episode_loss(
        new_action_logprobs=new,
        behavior_action_logprobs=old,
        decision_mask=mask,
        episode_advantages=advantages,
        clip_epsilon=0.2,
    )
    expected = -(torch.exp(torch.tensor(0.1)) - torch.exp(torch.tensor(-0.2))) / 2
    torch.testing.assert_close(observed, expected)


def test_episode_objective_rejects_empty_decision_rows():
    with pytest.raises(ValueError, match="at least one decision"):
        length_neutral_clipped_episode_loss(
            new_action_logprobs=torch.zeros((1, 2)),
            behavior_action_logprobs=torch.zeros((1, 2)),
            decision_mask=torch.zeros((1, 2), dtype=torch.bool),
            episode_advantages=torch.ones(1),
            clip_epsilon=0.2,
        )
