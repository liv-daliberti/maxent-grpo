from __future__ import annotations

import torch

from oat_drgrpo.candidate_maxent import (
    candidate_projection_minibatch_loss,
    candidate_projection_loss,
    candidate_projection_row_multipliers,
    reward_target_distribution,
    solve_candidate_maxent_target,
)
from oat_drgrpo.xdr import compute_xdr_row_weights


def test_reward_target_is_group_local_and_supports_probability_floor():
    utilities = torch.tensor([[0.0, 1.0, 2.0], [4.0, 1.0, -3.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    target = reward_target_distribution(
        utilities,
        temperature=0.5,
        epsilon=0.01,
        valid_mask=mask,
    )

    assert torch.allclose(target.sum(dim=1), torch.ones(2))
    assert bool((target[mask] >= 0.01).all())
    assert target[1, 2] == 0


def test_maxent_target_reduces_to_gibbs_weights_without_reference():
    utilities = torch.tensor([[0.1, 0.4, -0.2]])
    q = reward_target_distribution(utilities, temperature=1.0)

    weights = solve_candidate_maxent_target(q, tau=0.25)

    assert torch.allclose(weights, torch.softmax(utilities / 0.25, dim=1))


def test_reference_tilt_recovers_reference_distribution_for_uniform_reward():
    q = torch.full((1, 3), 1.0 / 3.0)
    reference_log_probs = torch.tensor([[-3.0, -1.0, -2.0]])

    weights = solve_candidate_maxent_target(
        q,
        tau=0.0,
        reference_log_probs=reference_log_probs,
        reference_kl_coef=1.0,
    )

    assert torch.allclose(weights, torch.softmax(reference_log_probs, dim=1))


def test_zero_temperature_without_reference_is_uniform_over_argmax_ties():
    q = torch.tensor([[0.45, 0.45, 0.10]])

    weights = solve_candidate_maxent_target(q, tau=0.0)

    assert torch.equal(weights, torch.tensor([[0.5, 0.5, 0.0]]))


def test_maxent_target_preserves_zero_reward_support():
    q = torch.tensor([[0.7, 0.3, 0.0]])

    weights = solve_candidate_maxent_target(q, tau=1.0)

    assert torch.equal(weights, q)


def test_candidate_projection_loss_uses_one_shared_length_scale():
    sequence_log_probs = torch.tensor([[-8.0, -3.0]], requires_grad=True)
    target = torch.tensor([[0.75, 0.25]])

    loss = candidate_projection_loss(
        sequence_log_probs,
        target,
        normalization_constant=4.0,
    )
    loss.backward()

    assert torch.allclose(loss, torch.tensor(1.6875))
    assert torch.allclose(
        sequence_log_probs.grad, torch.tensor([[-0.1875, -0.0625]])
    )


def test_row_multipliers_exactly_decompose_group_projection_after_shuffle():
    sequence_log_probs = torch.tensor([[-8.0, -3.0], [-2.0, -6.0]])
    target = torch.tensor([[0.75, 0.25], [0.40, 0.60]])

    grouped_loss = candidate_projection_loss(
        sequence_log_probs,
        target,
        normalization_constant=4.0,
    )
    multipliers = candidate_projection_row_multipliers(target).reshape_as(target)
    row_loss = -(multipliers * sequence_log_probs / 4.0).reshape(-1).mean()

    assert torch.allclose(row_loss, grouped_loss)


def test_live_minibatch_projection_matches_grouped_objective_and_gradients():
    token_log_probs = torch.tensor(
        [[-2.0, -2.0, 0.0], [-3.0, 0.0, 0.0]], requires_grad=True
    )
    response_masks = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    target = torch.tensor([[0.75, 0.25]])
    multipliers = candidate_projection_row_multipliers(target)

    live_loss = candidate_projection_minibatch_loss(
        token_log_probs,
        response_masks,
        multipliers,
        torch.ones(2),
        normalization_constant=4.0,
    )
    grouped_loss = candidate_projection_loss(
        torch.tensor([[-4.0, -3.0]]),
        target,
        normalization_constant=4.0,
    )
    live_loss.backward()

    assert torch.allclose(live_loss, grouped_loss)
    assert torch.allclose(
        token_log_probs.grad,
        torch.tensor([[-0.1875, -0.1875, 0.0], [-0.0625, 0.0, 0.0]]),
    )


def test_shared_projection_scale_does_not_upweight_shorter_candidates():
    """Every target unit has the same total score-function scale."""
    token_log_probs = torch.tensor(
        [[-1.0, -1.0, -1.0, -1.0], [-1.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    response_masks = torch.tensor(
        [[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32
    )
    target = torch.tensor([[0.5, 0.5]])

    loss = candidate_projection_minibatch_loss(
        token_log_probs,
        response_masks,
        candidate_projection_row_multipliers(target),
        torch.ones(2),
        normalization_constant=4.0,
    )
    loss.backward()

    # Per active token the scale is identical. The old 1/T_i objective made
    # the one-token candidate four times stronger per token and created the
    # observed EOS attractor.
    active_gradients = token_log_probs.grad[response_masks.bool()]
    assert torch.allclose(active_gradients, torch.full_like(active_gradients, -0.125))


def test_tau_point_oh_five_maxent_target_matches_fixed_xdr_weights():
    advantages = torch.tensor([0.2, -0.1, 0.4, -0.3])
    counts = torch.tensor([8.0, 6.0, 7.0, 5.0])
    utilities = (advantages * counts / 16.0).reshape(1, 4)
    reward_target = reward_target_distribution(utilities, temperature=1.0)
    target = solve_candidate_maxent_target(reward_target, tau=0.05)

    projection_multipliers = candidate_projection_row_multipliers(target)
    xdr_multipliers = compute_xdr_row_weights(
        advantages,
        counts,
        num_samples=4,
        tau=0.05,
        t_max=16,
    )

    assert torch.allclose(projection_multipliers, xdr_multipliers)
