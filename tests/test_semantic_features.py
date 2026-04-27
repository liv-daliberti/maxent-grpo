from __future__ import annotations

import torch

from oat_drgrpo.listwise import reshape_prompt_major_tensor
from oat_drgrpo.semantic_features import (
    build_candidate_response_features,
    build_pre_answer_response_token_counts,
    normalize_action_id_sequences,
    truncate_text_to_max_tokens,
)


class _DummyTokenizer:
    def __init__(
        self,
        mapping: dict[tuple[int, ...], str],
        *,
        encode_mapping: dict[str, list[int]] | None = None,
    ) -> None:
        self._mapping = mapping
        self._encode_mapping = encode_mapping or {}

    def batch_decode(
        self,
        sequences,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return [
            self._mapping[tuple(int(token_id) for token_id in seq)] for seq in sequences
        ]

    def decode(
        self,
        sequence,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return self._mapping[tuple(int(token_id) for token_id in sequence)]

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
    ):
        del add_special_tokens, return_attention_mask
        return {"input_ids": list(self._encode_mapping[text])}


def test_normalize_action_id_sequences_handles_tensor_like_inputs():
    sequences = normalize_action_id_sequences(
        [
            torch.tensor([1, 2, 3]),
            [4, 5],
        ]
    )

    assert sequences == [[1, 2, 3], [4, 5]]


def test_build_candidate_response_features_returns_lengths_and_formatted_bits():
    tokenizer = _DummyTokenizer(
        {
            (1, 2, 3): "reasoning </think> <answer>42</answer>",
            (4, 5): "missing tags",
        }
    )
    lengths, formatted = build_candidate_response_features(
        [torch.tensor([1, 2, 3]), [4, 5]],
        tokenizer=tokenizer,
        template="r1",
        formatted_checker=lambda response, template: "</answer>" in response,
        dtype=torch.float32,
    )

    torch.testing.assert_close(lengths, torch.tensor([3.0, 2.0]))
    torch.testing.assert_close(formatted, torch.tensor([1.0, 0.0]))


def test_candidate_response_features_reshape_cleanly_into_prompt_groups():
    tokenizer = _DummyTokenizer(
        {
            (1,): "a </think> <answer>1</answer>",
            (2, 3): "missing",
            (4, 5, 6): "b </think> <answer>2</answer>",
            (7,): "missing",
        }
    )
    lengths, formatted = build_candidate_response_features(
        [[1], [2, 3], [4, 5, 6], [7]],
        tokenizer=tokenizer,
        template="r1",
        formatted_checker=lambda response, template: "</answer>" in response,
        dtype=torch.float32,
    )

    grouped_lengths = reshape_prompt_major_tensor(lengths, 2)
    grouped_formatted = reshape_prompt_major_tensor(formatted, 2)

    torch.testing.assert_close(
        grouped_lengths,
        torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
    )
    torch.testing.assert_close(
        grouped_formatted,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )


def test_build_pre_answer_response_token_counts_zeroes_malformed_rows():
    tokenizer = _DummyTokenizer(
        {
            (1, 2, 3, 4): "reasoning</think><answer>42</answer>",
            (5, 6): "missing tags",
        },
        encode_mapping={
            "reasoning</think>": [1, 2],
            "reasoning</think><answer>42</answer>": [1, 2, 3, 4],
            "missing tags": [5, 6],
        },
    )

    counts = build_pre_answer_response_token_counts(
        [[1, 2, 3, 4], [5, 6]],
        tokenizer=tokenizer,
        template="r1",
        formatted_checker=lambda response, template: "</think><answer>" in response,
    )

    torch.testing.assert_close(counts, torch.tensor([2, 0], dtype=torch.int64))


def test_truncate_text_to_max_tokens_caps_reasoning_trace_without_special_tokens():
    tokenizer = _DummyTokenizer(
        {
            (1, 2): "x=2\ny=4",
        },
        encode_mapping={
            "x=2\ny=4\nz=5": [1, 2, 3],
        },
    )

    truncated, was_truncated = truncate_text_to_max_tokens(
        "x=2\ny=4\nz=5",
        tokenizer=tokenizer,
        max_tokens=2,
    )

    assert was_truncated is True
    assert truncated == "x=2\ny=4"
