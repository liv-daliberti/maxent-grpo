from __future__ import annotations

import pytest
import torch

from oat_drgrpo.templates import (
    TEMPLATE_FACTORY,
    apply_no_template,
    apply_qwen_boxed_template,
    apply_qwen_math_template,
    apply_r1_template,
    build_response_token_prefix_mask,
    collate_eval_prompt_items,
)


def test_prompt_templates_expose_expected_prompt_shapes():
    question = "What is 2+2?"

    assert apply_no_template(question) == question
    assert "Do not explain" in apply_qwen_boxed_template(question)
    assert apply_qwen_boxed_template(question).endswith("<|im_start|>assistant\n")
    assert "\\boxed{}" in apply_qwen_math_template(question)
    assert apply_qwen_math_template(question).endswith("<|im_start|>assistant\n")
    assert "<think>" in apply_r1_template(question)
    assert apply_r1_template(question).endswith("\nAssistant: <think>")
    assert TEMPLATE_FACTORY["r1"](question) == apply_r1_template(question)


def test_collate_eval_prompt_items_formats_prompts_and_preserves_answers():
    formatted, problems, answers = collate_eval_prompt_items(
        [
            {"problem": "What is 2+2?", "answer": "4"},
            {"problem": "What is 3+5?", "answer": "8"},
        ],
        prompt_template="no",
    )

    assert formatted == ["What is 2+2?", "What is 3+5?"]
    assert problems == ["What is 2+2?", "What is 3+5?"]
    assert answers == ["4", "8"]


def test_build_response_token_prefix_mask_selects_only_response_prefix():
    response_masks = torch.tensor(
        [
            [False, True, True, False, True],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    token_counts = torch.tensor([2, 0])

    prefix_mask = build_response_token_prefix_mask(response_masks, token_counts)

    torch.testing.assert_close(
        prefix_mask,
        torch.tensor(
            [
                [False, True, True, False, False],
                [False, False, False, False, False],
            ],
            dtype=torch.bool,
        ),
    )


def test_build_response_token_prefix_mask_validates_shapes():
    with pytest.raises(ValueError, match="response_masks"):
        build_response_token_prefix_mask(torch.ones(3), torch.ones(3))

    with pytest.raises(ValueError, match="token_counts"):
        build_response_token_prefix_mask(torch.ones(2, 3), torch.ones(2, 1))
