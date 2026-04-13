from __future__ import annotations

import torch

from oat_drgrpo.listwise import reshape_prompt_major_tensor
from oat_drgrpo.semantic_features import (
    build_candidate_response_features,
    normalize_action_id_sequences,
)


class _DummyTokenizer:
    def __init__(self, mapping: dict[tuple[int, ...], str]) -> None:
        self._mapping = mapping

    def batch_decode(
        self,
        sequences,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return [self._mapping[tuple(int(token_id) for token_id in seq)] for seq in sequences]


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
