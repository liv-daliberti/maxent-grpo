"""Prompt templates and response-token mask helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def apply_qwen_math_template(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_qwen_math_route_template(question: str) -> str:
    """Request a boxed answer plus an optional executable numeric route."""

    return (
        "<|im_start|>system\n"
        "Solve the problem step by step and put the final answer within "
        "\\boxed{}. After the boxed answer, also emit exactly one compact "
        "<route>JSON</route> block when the computation fits this schema: "
        '{"version":"math-route-v1","steps":[...],"final":"sN"}. '
        "Step IDs are contiguous s1,s2,... . A source step is "
        '{"id":"s1","op":"source","value":"a number copied from the problem"}. '
        "Other steps contain only id, op, and earlier args; never write a "
        "result field. Allowed ops are neg, abs, square, cube, sqrt, factorial, "
        "percent, add, mul, sub, div, pow, mod, choose, permute, gcd, lcm, min, "
        "max, and average. Every step must contribute to final. If this exact "
        "numeric trace language cannot express the solution, omit the route "
        "block but still give the boxed answer."
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


def apply_qwen_graph_digits_template(question: str) -> str:
    """Request the benchmark's canonical bare graph-color action vector."""

    boxed_suffix = "inside \\boxed{}."
    if question.count(boxed_suffix) != 1 or not question.endswith(boxed_suffix):
        raise ValueError(
            "qwen_graph_digits requires a graph prompt ending in a boxed answer"
        )
    canonical_question = (
        question[: -len(boxed_suffix)]
        + "as one bare digit string with no spaces, punctuation, or other text."
    )
    return (
        "<|im_start|>system\n"
        "Return only the requested bare sequence of digits. Do not explain, "
        "add punctuation, or use LaTeX."
        "<|im_end|>\n<|im_start|>user\n"
        + canonical_question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_qwen_countdown_digits_template(question: str) -> str:
    """Request the audited three-digit Countdown expression code."""

    boxed_suffix = "the expression inside \\boxed{}."
    if question.count(boxed_suffix) != 1 or not question.endswith(boxed_suffix):
        raise ValueError(
            "qwen_countdown_digits requires an easy3 Countdown prompt ending "
            "in the boxed-expression instruction"
        )
    canonical_question = (
        question[: -len(boxed_suffix)]
        + "a three-digit canonical action code using the scheme above."
    )
    return (
        "<|im_start|>system\n"
        "Return only three digits and no other text. The given numbers are "
        "n1,n2,n3 in dataset order. Digit 2 selects singleton s: 1=n1, "
        "2=n2, 3=n3; a,b are the other two numbers in their original order. "
        "Digit 3 makes pair: 1=a+b, 2=a*b, 3=a-b, 4=b-a, 5=a/b, 6=b/a. "
        "Digit 1 combines pair and s: 1=pair+s, 2=pair*s, 3=pair-s, "
        "4=s-pair, 5=pair/s, 6=s/pair."
        "<|im_end|>\n<|im_start|>user\n"
        + canonical_question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def validate_qwen_graph_digits_materialization(
    raw_questions: Sequence[str], formatted_prompts: Sequence[str]
) -> None:
    """Fail closed if dataset mapping did not render the canonical template.

    The exact string comparison deliberately detects stale Hugging Face map
    caches from older prompt templates as well as dropped or reordered rows.
    """

    if len(raw_questions) != len(formatted_prompts):
        raise RuntimeError(
            "canonical prompt materialization changed the row count: "
            f"raw={len(raw_questions)} formatted={len(formatted_prompts)}"
        )
    for row_index, (question, observed) in enumerate(
        zip(raw_questions, formatted_prompts)
    ):
        expected = apply_qwen_graph_digits_template(question)
        if observed != expected:
            raise RuntimeError(
                "canonical prompt materialization mismatch at row "
                f"{row_index}; stale or foreign template cache detected"
            )


def validate_qwen_countdown_digits_materialization(
    raw_questions: Sequence[str], formatted_prompts: Sequence[str]
) -> None:
    """Fail closed on stale or foreign canonical Countdown prompt rows."""

    if len(raw_questions) != len(formatted_prompts):
        raise RuntimeError(
            "canonical prompt materialization changed the row count: "
            f"raw={len(raw_questions)} formatted={len(formatted_prompts)}"
        )
    for row_index, (question, observed) in enumerate(
        zip(raw_questions, formatted_prompts)
    ):
        expected = apply_qwen_countdown_digits_template(question)
        if observed != expected:
            raise RuntimeError(
                "canonical Countdown prompt materialization mismatch at row "
                f"{row_index}; stale or foreign template cache detected"
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
    "qwen_countdown_digits": apply_qwen_countdown_digits_template,
    "qwen_graph_digits": apply_qwen_graph_digits_template,
    "qwen_math": apply_qwen_math_template,
    "qwen_math_route": apply_qwen_math_route_template,
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
