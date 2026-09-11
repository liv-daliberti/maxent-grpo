"""Conservative surface normalization for independently audited MATH answers."""

from __future__ import annotations

import re

from .math_grader import boxed_reward_fn


NORMALIZATION_VERSION = "audited_math_answer_surface_v1"
_UNIT_SUFFIX_RE = re.compile(
    r"""
    \s*
    (?:
        \\text\{\s*
        (?:cm|mm|km|meters?|metres?|inches?|feet|yards?|miles?|
           degrees?|radians?|dollars?|cents?|seconds?|minutes?|hours?|
           days?|weeks?|months?|years?)
        (?:\s*\^\s*[23]|\s*[²³])?
        \s*\}
      |
        (?:cm|mm|km|meters?|metres?|inches?|feet|yards?|miles?|
           degrees?|radians?|dollars?|cents?|seconds?|minutes?|hours?|
           days?|weeks?|months?|years?)
        (?:\s*\^\s*[23]|\s*[²³])?
    )
    \s*\.?\s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_SQRT_RE = re.compile(
    r"√\s*(?:\(([^()]+)\)|\{([^{}]+)\}|([0-9]+(?:\.[0-9]+)?))"
)


def _strip_math_delimiters(value: str) -> str:
    text = value.strip()
    pairs = (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    changed = True
    while changed:
        changed = False
        for left, right in pairs:
            if (
                text.startswith(left)
                and text.endswith(right)
                and len(text) > len(left) + len(right)
            ):
                text = text[len(left) : -len(right)].strip()
                changed = True
    return text


def _normalize_unicode_math(value: str) -> str:
    text = (
        value.replace("−", "-")
        .replace("×", "*")
        .replace("·", "*")
        .replace("π", "pi")
    )

    def replace_sqrt(match: re.Match[str]) -> str:
        radicand = next(
            group for group in match.groups() if group is not None
        )
        return f"sqrt({radicand.strip()})"

    text = _SQRT_RE.sub(replace_sqrt, text)
    text = re.sub(r"(?<=[0-9)\]])\s*(?=sqrt\()", "*", text)
    return text


def audited_answer_candidates(derived_answer: str) -> tuple[str, ...]:
    """Return bounded surface variants without changing mathematical values."""

    raw = str(derived_answer).strip()
    variants = [raw, _strip_math_delimiters(raw)]
    for value in list(variants):
        stripped = _UNIT_SUFFIX_RE.sub("", value).strip()
        variants.append(stripped)
    for value in list(variants):
        variants.append(_normalize_unicode_math(value))
    observed = set()
    result = []
    for value in variants:
        value = value.strip()
        if value and value not in observed:
            observed.add(value)
            result.append(value)
    return tuple(result[:8])


def audited_answer_matches(
    derived_answer: str,
    reference_answer: str,
) -> bool:
    """Use the task verifier over conservative surface-only candidates."""

    for candidate in audited_answer_candidates(derived_answer):
        try:
            _, reward = boxed_reward_fn(
                rf"\boxed{{{candidate}}}",
                reference_answer,
            )
        except Exception:
            continue
        if float(reward) > 0:
            return True
    return False
