"""Shared prompt templates aligned with the official SEED-GRPO/OAT code."""

from __future__ import annotations

from typing import Callable

SUPPORTED_PROMPT_TEMPLATES = ("qwen_math", "no", "r1")


def apply_qwen_math_template(question: str) -> str:
    """Render the official SEED-GRPO Qwen math prompt."""

    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str) -> str:
    """Render the official SEED-GRPO R1-style prompt."""

    return (
        "A conversation between User and Assistant. The User asks a question, and "
        "the Assistant solves it. The Assistant first thinks about the reasoning "
        "process in the mind and then provides the User with the answer. The "
        "reasoning process is enclosed within <think> </think> and answer is "
        "enclosed within <answer> </answer> tags, respectively, i.e., <think> "
        "reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


def apply_no_template(question: str) -> str:
    """Return the raw question unchanged."""

    return question


PROMPT_TEMPLATE_FACTORY: dict[str, Callable[[str], str]] = {
    "qwen_math": apply_qwen_math_template,
    "no": apply_no_template,
    "r1": apply_r1_template,
}


def resolve_generation_stop_settings(
    prompt_template: str | None,
) -> tuple[list[str] | None, bool]:
    """Return OAT-style stop settings for the active prompt template.

    OAT's math RL actor applies an explicit ``</answer>`` stop only for the
    strict ``r1`` prompt template, and it preserves that stop string in the
    returned text so answer-tag grading still sees ``</answer>``.
    """

    normalized = normalize_prompt_template(prompt_template, default=None)
    if normalized == "r1":
        return ["</answer>"], True
    return None, False


def normalize_prompt_template(
    value: str | None,
    *,
    default: str | None = None,
) -> str | None:
    """Normalize a prompt-template name and validate supported values."""

    if value is None:
        return default
    candidate = str(value).strip().lower()
    if not candidate:
        return default
    if candidate not in PROMPT_TEMPLATE_FACTORY:
        supported = ", ".join(SUPPORTED_PROMPT_TEMPLATES)
        raise ValueError(
            f"prompt_template must be one of: {supported}; got {value!r}"
        )
    return candidate


def render_prompt_template(
    question: str,
    prompt_template: str | None,
    *,
    default: str | None = None,
) -> str:
    """Render ``question`` with the requested prompt template."""

    normalized = normalize_prompt_template(prompt_template, default=default)
    if normalized is None:
        return question
    return PROMPT_TEMPLATE_FACTORY[normalized](question)


__all__ = [
    "PROMPT_TEMPLATE_FACTORY",
    "SUPPORTED_PROMPT_TEMPLATES",
    "apply_no_template",
    "apply_qwen_math_template",
    "apply_r1_template",
    "normalize_prompt_template",
    "render_prompt_template",
    "resolve_generation_stop_settings",
]
