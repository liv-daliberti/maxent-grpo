import pytest
import torch

from oat_drgrpo.interactive_sft import restricted_action_cross_entropy


def test_restricted_action_loss_ignores_logits_outside_policy_support():
    logits = torch.tensor([[1.0, 2.0, 100.0, 3.0]])
    action_ids = torch.tensor([0, 1, 3], dtype=torch.long)
    target = torch.tensor([2], dtype=torch.long)
    observed = restricted_action_cross_entropy(
        logits,
        action_token_ids=action_ids,
        target_action_indices=target,
    )
    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[1.0, 2.0, 3.0]]),
        target,
    )
    torch.testing.assert_close(observed, expected)


def test_restricted_action_loss_rejects_duplicate_support():
    with pytest.raises(ValueError, match="unique"):
        restricted_action_cross_entropy(
            torch.zeros((1, 4)),
            action_token_ids=torch.tensor([1, 1], dtype=torch.long),
            target_action_indices=torch.tensor([0], dtype=torch.long),
        )
