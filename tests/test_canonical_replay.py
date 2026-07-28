import math

import pytest
import torch

from oat_drgrpo.canonical_replay import (
    CanonicalReplayLikelihoodController,
    CanonicalReplayInverseController,
    canonical_replay_split_mass_balance_loss,
    canonical_replay_uniform_loss,
    canonical_replay_uniform_verified_likelihood_loss,
    materialize_canonical_replay_batch,
)
from oat_drgrpo.online_canonical_bank import (
    VerifiedCanonicalReplayGroup,
)


def test_uniform_replay_loss_is_zero_only_at_balanced_model_scores():
    balanced = torch.tensor([0.0, 0.0], requires_grad=True)
    result = canonical_replay_uniform_loss(balanced, [2])

    assert result.loss.item() == pytest.approx(0.0, abs=1e-7)
    assert result.normalized_entropy.item() == pytest.approx(1.0)
    assert result.eligible_groups == 1
    assert result.retained_modes == 2


def test_replay_loss_restores_a_near_missing_observed_mode():
    scores = torch.tensor([4.0, -4.0], requires_grad=True)
    result = canonical_replay_uniform_loss(scores, [2])
    result.loss.backward()

    assert result.loss.item() > 3
    assert 0 < result.normalized_entropy.item() < 0.01
    assert scores.grad is not None
    assert torch.allclose(scores.grad, result.score_gradients)
    # Gradient descent lowers the dominant score and raises the missing one.
    assert scores.grad[0].item() > 0
    assert scores.grad[1].item() < 0


def test_replay_loss_averages_prompt_local_banks_without_gold_support():
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    result = canonical_replay_uniform_loss(scores, [2, 3])

    assert result.eligible_groups == 2
    assert result.retained_modes == 5
    assert torch.isfinite(result.loss)
    assert 0 < result.normalized_entropy.item() <= 1


def test_uniform_verified_likelihood_has_common_mass_gradient():
    scores = torch.tensor([-1.0, -1.0], requires_grad=True)
    result = canonical_replay_uniform_verified_likelihood_loss(scores, [2])
    result.loss.backward()

    assert result.loss.item() == pytest.approx(1.0)
    assert result.cross_entropy_excess.item() == pytest.approx(0.0, abs=1e-7)
    assert result.normalized_entropy.item() == pytest.approx(1.0)
    assert scores.grad is not None
    assert torch.allclose(scores.grad, torch.tensor([-0.5, -0.5]))
    assert torch.allclose(scores.grad, result.score_gradients)
    assert scores.grad.sum().item() == pytest.approx(-1.0)


def test_uniform_verified_likelihood_weights_prompts_then_modes_equally():
    scores = torch.tensor(
        [-1.0, -3.0, -2.0, -4.0, -6.0],
        requires_grad=True,
    )
    result = canonical_replay_uniform_verified_likelihood_loss(
        scores,
        [2, 3],
    )
    result.loss.backward()

    assert result.loss.item() == pytest.approx(
        ((1.0 + 3.0) / 2.0 + (2.0 + 4.0 + 6.0) / 3.0) / 2.0
    )
    assert scores.grad is not None
    assert torch.allclose(
        scores.grad,
        torch.tensor([-0.25, -0.25, -1 / 6, -1 / 6, -1 / 6]),
    )
    assert scores.grad[:2].sum().item() == pytest.approx(-0.5)
    assert scores.grad[2:].sum().item() == pytest.approx(-0.5)


def test_uniform_verified_likelihood_anchors_singleton_without_fake_entropy():
    scores = torch.tensor([-2.0], requires_grad=True)
    result = canonical_replay_uniform_verified_likelihood_loss(scores, [1])
    result.loss.backward()

    assert result.loss.item() == pytest.approx(2.0)
    assert result.cross_entropy_excess.item() == pytest.approx(0.0)
    assert result.normalized_entropy.item() == pytest.approx(1.0)
    assert result.eligible_groups == 0
    assert result.retained_modes == 0
    assert result.actuator_groups == 1
    assert result.actuator_modes == 1
    assert scores.grad is not None
    assert scores.grad.item() == pytest.approx(-1.0)


def test_uniform_verified_likelihood_senses_only_multimode_groups():
    scores = torch.tensor([-2.0, -1.0, -3.0])
    result = canonical_replay_uniform_verified_likelihood_loss(
        scores,
        [1, 2],
    )

    assert result.eligible_groups == 1
    assert result.retained_modes == 2
    assert result.actuator_groups == 2
    assert result.actuator_modes == 3
    assert result.score_gradients.tolist() == pytest.approx(
        [-0.5, -0.25, -0.25]
    )


def test_split_replay_keeps_mass_and_balance_gradients_independent():
    scores = torch.tensor([-2.0, 3.0, -3.0], requires_grad=True)
    result = canonical_replay_split_mass_balance_loss(scores, [1, 2])

    assert result.mass_score_gradients.tolist() == pytest.approx(
        [-0.5, -0.25, -0.25]
    )
    assert result.mass_score_gradients.sum().item() == pytest.approx(-1.0)
    assert result.balance_score_gradients[0].item() == pytest.approx(0.0)
    assert result.balance_score_gradients.sum().item() == pytest.approx(
        0.0,
        abs=1e-7,
    )
    assert result.balance_score_gradients[1].item() > 0
    assert result.balance_score_gradients[2].item() < 0
    assert result.balance_eligible_groups == 1
    assert result.balance_retained_modes == 2


def test_split_replay_singleton_has_mass_but_no_fake_balance():
    scores = torch.tensor([-2.0])
    result = canonical_replay_split_mass_balance_loss(scores, [1])

    assert result.mass_loss.item() == pytest.approx(2.0)
    assert result.mass_score_gradients.item() == pytest.approx(-1.0)
    assert result.balance_loss.item() == pytest.approx(0.0)
    assert result.balance_score_gradients.item() == pytest.approx(0.0)
    assert result.balance_eligible_groups == 0


def test_replay_batch_marks_exact_response_labels_after_each_prompt():
    groups = [
        VerifiedCanonicalReplayGroup(
            prompt_token_ids=(10, 11, 12),
            outcome_keys=("a", "b"),
            response_token_ids=((20, 21), (30,)),
        ),
        VerifiedCanonicalReplayGroup(
            prompt_token_ids=(40, 41),
            outcome_keys=("c", "d"),
            response_token_ids=((50,), (60, 61, 62)),
        ),
    ]

    batch = materialize_canonical_replay_batch(
        groups,
        pad_token_id=0,
        device="cpu",
    )

    assert batch.group_sizes == (2, 2)
    assert batch.input_ids.tolist() == [
        [10, 11, 12, 20, 21],
        [10, 11, 12, 30, 0],
        [40, 41, 50, 0, 0],
        [40, 41, 60, 61, 62],
    ]
    assert batch.attention_mask.tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert batch.response_masks.tolist() == [
        [False, False, True, True],
        [False, False, True, False],
        [False, True, False, False],
        [False, True, True, True],
    ]


def test_replay_entropy_sensor_survives_float32_probability_underflow():
    scores = torch.tensor([0.0, -100.0], requires_grad=True)

    result = canonical_replay_uniform_loss(scores, [2])

    assert result.normalized_entropy.item() > 0.0


@pytest.mark.parametrize(
    ("scores", "sizes", "message"),
    (
        (torch.zeros(2, 1), [2], "one-dimensional"),
        (torch.zeros(2), [1, 1], "at least two"),
        (torch.zeros(3), [2], "do not partition"),
        (torch.tensor([0.0, math.inf]), [2], "finite"),
    ),
)
def test_replay_loss_fails_closed_on_malformed_banks(scores, sizes, message):
    with pytest.raises(ValueError, match=message):
        canonical_replay_uniform_loss(scores, sizes)


def test_replay_inverse_controller_is_unprojected_and_model_score_only():
    controller = CanonicalReplayInverseController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.0,
    )

    assert controller.observe(0.8)["canonical_replay_next_alpha"] == 0.1
    warmup = controller.observe(0.6)
    assert warmup["canonical_replay_reference_entropy"] == pytest.approx(0.7)
    low = controller.observe(0.07)
    assert low["canonical_replay_next_alpha"] == pytest.approx(1.0)
    assert low["canonical_replay_projection_active"] == 0.0
    high = controller.observe(1.0)
    assert high["canonical_replay_next_alpha"] == pytest.approx(0.07)
    assert controller.state_dict()["controller_rule"] == (
        "unprojected_warmup_inverse_observed_bank_entropy_v1"
    )


def test_replay_controller_idle_does_not_consume_warmup():
    controller = CanonicalReplayInverseController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.9,
    )

    diagnostics = controller.idle_diagnostics()

    assert diagnostics["canonical_replay_observation_skipped"] == 1.0
    assert controller.observation_count == 0
    assert controller.entropy_ema is None


def test_replay_controller_resume_is_exact_and_configuration_bound():
    controller = CanonicalReplayInverseController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.5,
    )
    controller.observe(0.9)
    controller.observe(0.7)
    controller.observe(0.4)
    state = controller.state_dict()

    restored = CanonicalReplayInverseController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.5,
    )
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    mismatch = CanonicalReplayInverseController(
        base_alpha=0.2,
        warmup_steps=2,
        ema_decay=0.5,
    )
    with pytest.raises(ValueError, match="base_alpha"):
        mismatch.load_state_dict(state)


def test_likelihood_controller_strengthens_only_when_verified_surprisal_rises():
    controller = CanonicalReplayLikelihoodController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.0,
    )

    assert controller.observe(2.0)["canonical_replay_mass_next_alpha"] == 0.1
    warmup = controller.observe(4.0)
    assert warmup[
        "canonical_replay_mass_surprisal_reference"
    ] == pytest.approx(3.0)

    stronger = controller.observe(12.0)
    assert stronger["canonical_replay_mass_next_alpha"] == pytest.approx(0.4)
    assert stronger["canonical_replay_mass_projection_active"] == 0.0
    weaker = controller.observe(1.5)
    assert weaker["canonical_replay_mass_next_alpha"] == pytest.approx(0.05)


def test_likelihood_controller_resume_is_exact():
    controller = CanonicalReplayLikelihoodController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.5,
    )
    controller.observe(2.0)
    controller.observe(4.0)
    controller.observe(8.0)

    state = controller.state_dict()
    restored = CanonicalReplayLikelihoodController(
        base_alpha=0.1,
        warmup_steps=2,
        ema_decay=0.5,
    )
    restored.load_state_dict(state)

    assert restored.state_dict() == state
