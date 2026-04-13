"""Helpers for learner-side semantic feature tensors."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch


def normalize_action_id_sequences(
    action_ids: Sequence[Sequence[int] | torch.Tensor | Any],
) -> list[list[int]]:
    """Return ``action_ids`` as plain Python ``list[int]`` sequences."""

    normalized: list[list[int]] = []
    for ids in action_ids:
        if torch.is_tensor(ids):
            normalized.append([int(token_id) for token_id in ids.tolist()])
            continue
        if hasattr(ids, "tolist"):
            as_list = ids.tolist()
            if isinstance(as_list, list):
                normalized.append([int(token_id) for token_id in as_list])
                continue
        normalized.append([int(token_id) for token_id in ids])
    return normalized


def build_candidate_response_features(
    action_ids: Sequence[Sequence[int] | torch.Tensor | Any],
    *,
    tokenizer: Any,
    template: str,
    formatted_checker: Any | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flat response-length and formatted-indicator tensors."""

    if formatted_checker is None:
        from .math_grader import is_response_formatted_for_reward

        formatted_checker = is_response_formatted_for_reward

    normalized_action_ids = normalize_action_id_sequences(action_ids)
    response_lengths = torch.tensor(
        [len(ids) for ids in normalized_action_ids],
        device=device,
        dtype=dtype,
    )
    responses = tokenizer.batch_decode(
        normalized_action_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    formatted = torch.tensor(
        [
            1.0 if formatted_checker(response, template=template) else 0.0
            for response in responses
        ],
        device=device,
        dtype=dtype,
    )
    return response_lengths, formatted
