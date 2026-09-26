"""Posthoc formatting-only sensitivity analysis; strict grades remain primary.

The rule set was chosen after reading the first 15 test responses. No model
request or original grader is changed. Every changed answer is reverified by
the original executable validator; rules never search for a valid answer.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from frontier_modebench_contract import _identity, grade_response

SCHEMA = "frontier-modebench-posthoc-formatting-v1"
RULE_DESCRIPTIONS = {
    "python_lambda": "Leading LaTeX lambda n: becomes Python lambda n:.",
    "escaped_percent": "LaTeX escaped percent becomes the Python modulo character.",
    "escaped_underscore": "LaTeX escaped underscores become literal identifier underscores.",
    "latex_spacing": "LaTeX backslash-space, comma-space, semicolon-space and negative thin-space commands become ordinary spaces.",
    "sized_parentheses": "LaTeX left/right parenthesis sizing commands become plain parentheses.",
    "latex_fraction": "Balanced frac/dfrac/tfrac numerator and denominator groups become fully parenthesized division, recursively.",
    "implicit_multiplication": "Numeric/parenthesis adjacency becomes explicit multiplication only when its containing group has no division sign.",
}


class _MalformedFraction(ValueError):
    pass


def _braced(text: str, start: int) -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise _MalformedFraction("fraction argument is not a braced group")
    depth = 1
    for position in range(start + 1, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                value = text[start + 1:position]
                if not value.strip():
                    raise _MalformedFraction("empty fraction argument")
                return value, position + 1
    raise _MalformedFraction("unbalanced fraction argument")


def _fractions(text: str, depth: int = 0) -> str:
    if depth > 32:
        raise _MalformedFraction("fraction nesting limit")
    pattern = re.compile(r"\\(?:frac|dfrac|tfrac)(?![A-Za-z])")
    out = []
    start = 0
    for_match = pattern.search(text, start)
    while for_match is not None:
        numerator, end = _braced(text, for_match.end())
        denominator, end = _braced(text, end)
        out.append(text[start:for_match.start()])
        out.append("((" + _fractions(numerator, depth + 1) + ")/(" + _fractions(denominator, depth + 1) + "))")
        start = end
        for_match = pattern.search(text, start)
    out.append(text[start:])
    return "".join(out)


def _implicit_multiplication(text: str) -> str:
    """Skip ambiguous a/b(c) conventions and all non-arithmetic surfaces."""
    if not re.fullmatch(r"[0-9()+*/.\s-]+", text):
        return text
    stack = [-1]
    context_after = []
    groups_with_division = set()
    for position, character in enumerate(text):
        if character == "(":
            stack.append(position)
        elif character == ")":
            if len(stack) == 1:
                return text
            stack.pop()
        elif character == "/":
            groups_with_division.add(stack[-1])
        context_after.append(stack[-1])
    if len(stack) != 1:
        return text
    adjacency = re.compile(r"(?<=[0-9)])\s*(?=\()|(?<=\))\s*(?=[0-9])")
    return adjacency.sub(lambda match: match.group() if context_after[match.start() - 1] in groups_with_division else "*", text)


def _candidate_surface(text: str) -> tuple[str, str, str] | None:
    """Preserve original surrounding prose and the exact final boxed boundary."""
    if "\\boxed" not in text:
        return "", text, ""
    from oat_drgrpo.math_grader import last_boxed_only_string, remove_boxed
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    candidate = remove_boxed(boxed)
    if candidate is None:
        return None
    start = text.rfind("\\boxed")
    return text[:start] + "\\boxed{", candidate, "}" + text[start + len(boxed):]


def normalize_response(level: int | str, domain: str, text: str) -> dict[str, Any]:
    """Normalize only a closed set of mathematical typography conventions."""
    level, domain = _identity(level, domain)
    if not isinstance(text, str):
        raise TypeError("response text must be a string")
    unchanged = {"normalized_text": text, "transformations": []}
    if len(text) > 32768 or domain in {"graph_coloring", "mathir"} or (level == 1 and domain == "pantry_plan"):
        return unchanged
    surface = _candidate_surface(text)
    if surface is None:
        return unchanged
    prefix, candidate, suffix = surface
    rules = []

    def apply(rule, replacement):
        nonlocal candidate
        updated = replacement(candidate)
        if updated != candidate:
            rules.append(rule)
            candidate = updated

    if domain == "python_factors":
        apply("python_lambda", lambda value: re.sub(r"^(\s*)\\lambda(?:\s+|\\,\s*)n\s*:\s*", r"\1lambda n: ", value, count=1))
        apply("escaped_percent", lambda value: value.replace(r"\%", "%"))
    if domain in {"python_factors", "pantry_plan"}:
        apply("escaped_underscore", lambda value: value.replace(r"\_", "_"))
        apply("latex_spacing", lambda value: re.sub(r"\\[ ,;!]", " ", value))
    if domain == "countdown":
        apply("sized_parentheses", lambda value: value.replace(r"\left(", "(").replace(r"\right)", ")"))
        apply("latex_spacing", lambda value: re.sub(r"\\[ ,;!]", " ", value))
        try:
            fraction_result = _fractions(candidate)
        except _MalformedFraction:
            return unchanged
        apply("latex_fraction", lambda value: fraction_result)
        apply("implicit_multiplication", _implicit_multiplication)
    return {"normalized_text": prefix + candidate + suffix, "transformations": rules}


def normalize_and_grade(row: Mapping[str, Any], text: str, *, strict_grade: Mapping[str, Any] | None = None, grader: Callable | None = None) -> dict[str, Any]:
    """Compute a separately labeled diagnostic grade for the identical response.

    The row uses the saved inventory shape (level/domain/problem/answer).
    Strictly verified answers retain their original canonical key unchanged.
    Failed answers are normalized and reverified by the original grader; an
    unchanged failed answer can reuse its supplied strict grade. Pass the
    frozen snapshot contract grader via ``grader`` for the saved experiment.
    """
    if strict_grade is not None and strict_grade.get("verified") is True:
        return {"schema": SCHEMA, "posthoc": True, "original_text": text,
                "normalized_text": text, "transformations": [],
                "already_strictly_verified": True, "original_verified": True,
                "original_canonical_key": strict_grade["canonical_key"],
                **{key: strict_grade[key] for key in ("verified", "canonical_key", "graded_text")}}
    normalization = normalize_response(row["level"], row["domain"], text)
    if normalization["normalized_text"] == text and strict_grade is not None:
        grade = {key: strict_grade[key] for key in ("verified", "canonical_key", "graded_text")}
    else:
        grade = (grader or grade_response)(row["level"], row["domain"], row, normalization["normalized_text"])
    return {"schema": SCHEMA, "posthoc": True, "original_text": text,
            "original_verified": strict_grade.get("verified") if strict_grade is not None else None,
            "original_canonical_key": strict_grade.get("canonical_key") if strict_grade is not None else None,
            **normalization, **grade}
