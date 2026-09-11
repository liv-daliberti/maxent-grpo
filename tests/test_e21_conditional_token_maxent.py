"""E21's length-neutral free-form MaxEnt objective contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from oat_drgrpo.learner.base import ZeroMathLearnerBaseMixin
from oat_drgrpo.maxent_controllers import MaxEntProportionalController
from oat_drgrpo.on_policy_maxent import mean_active_token_entropy_by_response


class _Scorer(ZeroMathLearnerBaseMixin):
    tokenizer = SimpleNamespace(eos_token_id=2)


def test_conditional_content_entropy_is_invariant_to_eos_logit():
    scorer = _Scorer()
    logits = torch.tensor(
        [[[0.2, -0.1, -8.0, 0.7], [0.5, 0.4, 9.0, -0.2]]],
        requires_grad=True,
    )
    entropy = scorer._chunked_conditional_content_entropy_from_logits(logits)
    changed = logits.detach().clone()
    changed[..., 2] += 1000.0
    changed_entropy = scorer._chunked_conditional_content_entropy_from_logits(
        changed
    )

    assert torch.allclose(entropy, changed_entropy, atol=1e-7, rtol=0)
    entropy.sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[..., 2]).item() == 0


def test_response_mean_does_not_pay_longer_rows_more():
    token_entropy = torch.tensor(
        [
            [2.0, 99.0, 99.0, 99.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    masks = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ]
    )

    by_response = mean_active_token_entropy_by_response(token_entropy, masks)

    assert by_response.tolist() == pytest.approx([2.0, 2.0])
    assert by_response.mean().item() == pytest.approx(2.0)


def test_response_mean_rejects_nonfinite_active_entropy():
    with pytest.raises(ValueError, match="finite"):
        mean_active_token_entropy_by_response(
            torch.tensor([[float("nan")]]),
            torch.ones(1, 1),
        )


def test_conditional_entropy_actuator_has_gradient_at_single_mode_collapse():
    scorer = _Scorer()
    # Token 0 dominates every visited prefix. The direct categorical entropy
    # gradient must remain present even though every sampled completion could
    # be identical and a verified-outcome bank could contain only one key.
    logits = torch.tensor(
        [
            [[7.0, -7.0, 50.0, -7.0], [7.0, -7.0, 50.0, -7.0]],
            [[7.0, -7.0, 50.0, -7.0], [7.0, -7.0, 50.0, -7.0]],
        ],
        requires_grad=True,
    )
    token_entropy = scorer._chunked_conditional_content_entropy_from_logits(
        logits
    )
    response_entropy = mean_active_token_entropy_by_response(
        token_entropy,
        torch.ones_like(token_entropy),
    ).mean()

    assert response_entropy.item() > 0
    response_entropy.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[..., [0, 1, 3]]).item() > 0
    assert torch.count_nonzero(logits.grad[..., 2]).item() == 0


def test_controller_checkpoint_units_bind_conditional_token_sensor():
    controller = MaxEntProportionalController(
        base_alpha=0.000075,
        max_alpha=0.00015,
        target_ratio=0.8,
        warmup_steps=16,
        ema_decay=0.9,
        gain=2.0,
        entropy_units="conditional_content_token_nats_mean_v1",
        observation_metric_key="maxent_conditional_token_entropy",
    )

    assert controller.state_dict()["entropy_units"] == (
        "conditional_content_token_nats_mean_v1"
    )

    incompatible = controller.state_dict()
    incompatible["entropy_units"] = "sequence_nats_v1"
    with pytest.raises(ValueError, match="incompatible MaxEnt entropy units"):
        controller.load_state_dict(incompatible)
