"""Restricted finite-action supervised loss for interactive warm starts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def restricted_action_cross_entropy(
    decision_logits: torch.Tensor,
    *,
    action_token_ids: torch.Tensor,
    target_action_indices: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over the same finite token support used by the policy."""

    if decision_logits.ndim != 2:
        raise ValueError("decision_logits must be [batch, vocabulary]")
    if action_token_ids.ndim != 1 or action_token_ids.numel() < 2:
        raise ValueError("action_token_ids must contain at least two tokens")
    if target_action_indices.ndim != 1 or target_action_indices.shape[0] != (
        decision_logits.shape[0]
    ):
        raise ValueError("one target action index is required per batch row")
    if action_token_ids.dtype != torch.long:
        raise ValueError("action_token_ids must use torch.long")
    if target_action_indices.dtype != torch.long:
        raise ValueError("target_action_indices must use torch.long")
    if len(set(action_token_ids.detach().cpu().tolist())) != action_token_ids.numel():
        raise ValueError("action_token_ids must be unique")
    if int(action_token_ids.min()) < 0 or int(action_token_ids.max()) >= (
        decision_logits.shape[1]
    ):
        raise ValueError("action token ID is outside the model vocabulary")
    if int(target_action_indices.min()) < 0 or int(
        target_action_indices.max()
    ) >= action_token_ids.numel():
        raise ValueError("target action index is outside the finite support")
    restricted_logits = decision_logits.index_select(1, action_token_ids)
    return F.cross_entropy(restricted_logits.float(), target_action_indices)
