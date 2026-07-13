"""Prompt templates and response-token mask helpers."""

from __future__ import annotations

from collections.abc import Callable

import torch


def apply_qwen_math_template(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_qwen_boxed_template(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Return only the final answer inside \\boxed{}. Do not explain."
        "<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str) -> str:
    return (
        "A conversation between User and Assistant. The User asks a question, "
        "and the Assistant solves it. The Assistant first thinks about the "
        "reasoning process in the mind and then provides the User with the "
        "answer. The reasoning process is enclosed within <think> </think> "
        "and answer is enclosed within <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here "
        "</answer>.\nUser: " + question + "\nAssistant: <think>"
    )


def apply_no_template(question: str) -> str:
    return question


TEMPLATE_FACTORY: dict[str, Callable[[str], str]] = {
    "qwen_boxed": apply_qwen_boxed_template,
    "qwen_math": apply_qwen_math_template,
    "r1": apply_r1_template,
    "no": apply_no_template,
}


def apply_prompt_template_to_example(
    example: dict,
    *,
    input_key: str,
    prompt_template: str,
) -> dict:
    """Apply a configured prompt template to one dataset row."""

    problem = example[input_key]
    example[input_key] = TEMPLATE_FACTORY[prompt_template](problem)
    return example


def collate_eval_prompt_items(
    item_list: list[dict],
    *,
    prompt_template: str,
) -> tuple[list[str], list[str], list[str]]:
    """Collate eval rows into templated prompts, raw problems, and answers."""

    problems = []
    formatted_problems = []
    answers = []
    for item in item_list:
        problems.append(item["problem"])
        formatted_problems.append(TEMPLATE_FACTORY[prompt_template](item["problem"]))
        answers.append(item["answer"])
    return formatted_problems, problems, answers


def build_response_token_prefix_mask(
    response_masks: torch.Tensor,
    token_counts: torch.Tensor,
) -> torch.Tensor:
    """Select the first ``token_counts`` response tokens in each row."""

    if response_masks.ndim != 2:
        raise ValueError("response_masks must have shape [batch, seq].")
    if token_counts.ndim != 1 or token_counts.shape[0] != response_masks.shape[0]:
        raise ValueError(
            "token_counts must have shape [batch] matching response_masks."
        )
    safe_counts = token_counts.to(
        device=response_masks.device, dtype=torch.int64
    ).clamp(min=0)
    response_positions = response_masks.to(torch.int64).cumsum(dim=1)
    return response_masks.to(torch.bool) & (response_positions <= safe_counts[:, None])
