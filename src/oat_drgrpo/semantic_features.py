"""Helpers for learner-side semantic feature tensors."""

from __future__ import annotations

from typing import Any, Sequence

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


def truncate_text_to_max_tokens(
    text: str | None,
    *,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[str | None, bool]:
    """Return ``text`` truncated to at most ``max_tokens`` tokenizer tokens."""

    if text is None:
        return None, False
    normalized = str(text)
    if not normalized.strip():
        return None, False
    safe_max_tokens = int(max_tokens)
    if safe_max_tokens <= 0:
        return normalized, False

    token_ids = _encode_text_without_special_tokens(tokenizer, normalized)
    if token_ids is None:
        return normalized, False
    if len(token_ids) <= safe_max_tokens:
        return normalized, False

    truncated_ids = token_ids[:safe_max_tokens]
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        truncated = decode(
            truncated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    else:
        truncated = tokenizer.batch_decode(
            [truncated_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
    truncated = str(truncated).strip()
    return (truncated or None), True


def _encode_text_without_special_tokens(tokenizer: Any, text: str) -> list[int] | None:
    """Best-effort tokenization helper used for response-local token masks."""

    if callable(tokenizer):
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
        except TypeError:
            encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
        else:
            input_ids = getattr(encoded, "input_ids", None)
        if torch.is_tensor(input_ids):
            return [int(token_id) for token_id in input_ids.reshape(-1).tolist()]
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return [int(token_id) for token_id in input_ids]

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            encoded = encode(text, add_special_tokens=False)
        except TypeError:
            encoded = encode(text)
        if torch.is_tensor(encoded):
            return [int(token_id) for token_id in encoded.reshape(-1).tolist()]
        if isinstance(encoded, list):
            return [int(token_id) for token_id in encoded]
    return None


def _count_prefix_tokens_before_answer_tag(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    response_text: str,
) -> int:
    """Return the number of response tokens before the first ``<answer>`` tag."""

    prefix_text = response_text.split("<answer>", 1)[0]
    prefix_ids = _encode_text_without_special_tokens(tokenizer, prefix_text)
    if prefix_ids is not None:
        return max(min(len(prefix_ids), len(token_ids)), 0)

    # Tokenizers used in training should expose encode/__call__. Keep a small
    # decode-based fallback for unit tests and other lightweight stubs.
    for prefix_len in range(len(token_ids) + 1):
        decoded_prefix = tokenizer.batch_decode(
            [list(token_ids[:prefix_len])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        if "<answer>" in decoded_prefix:
            return max(prefix_len - 1, 0)
    return len(token_ids)


def build_pre_answer_response_token_counts(
    action_ids: Sequence[Sequence[int] | torch.Tensor | Any],
    *,
    tokenizer: Any,
    template: str,
    formatted_checker: Any | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return per-response token counts strictly before ``<answer>``.

    Rows that do not satisfy the reward-format gate return zero so semantic
    bonus updates can be disabled cleanly for malformed traces.
    """

    if formatted_checker is None:
        from .math_grader import is_response_formatted_for_reward

        formatted_checker = is_response_formatted_for_reward

    normalized_action_ids = normalize_action_id_sequences(action_ids)
    responses = tokenizer.batch_decode(
        normalized_action_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    counts: list[int] = []
    for token_ids, response in zip(normalized_action_ids, responses):
        if not formatted_checker(response, template=template):
            counts.append(0)
            continue
        if "<answer>" not in response:
            counts.append(0)
            continue
        counts.append(
            _count_prefix_tokens_before_answer_tag(
                tokenizer=tokenizer,
                token_ids=token_ids,
                response_text=response,
            )
        )
    return torch.tensor(counts, device=device, dtype=torch.int64)
