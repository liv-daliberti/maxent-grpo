"""Hosted-model prompt and verifier adapters for the existing E124 test sets.

Only the chat transport changes: system/user strings are materialized by the
original templates. Hosted generation does not inherit vLLM syntax masks.
Level 1 Pantry retains the registered support-mask environment transition.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "frontier-modebench-e124-prompt-contract-v1"
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")


def _identity(level: int | str, domain: str) -> tuple[int, str]:
    if isinstance(level, str) and level.startswith("level"):
        level = level[5:]
    if isinstance(level, bool) or str(level) not in {"1", "2", "3"}:
        raise ValueError("level must be 1, 2, or 3")
    domain = "pantry_plan" if domain == "pantry" else domain
    if domain not in DOMAINS:
        raise ValueError(f"unsupported ModeBench domain: {domain}")
    return int(level), domain


def profile_metadata(level: int | str, domain: str) -> dict[str, Any]:
    """Record the inherited interface and explicit hosted-generation difference."""
    level, domain = _identity(level, domain)
    mask = level == 1 and domain == "pantry_plan"
    template = ("qwen_pantry_support_mask" if mask else "qwen_boxed") if level == 1 else (
        "qwen_boxed" if domain == "graph_coloring" else
        "qwen_level2_" + ("pantry" if domain == "pantry_plan" else domain)
    )
    parent_syntax = "none" if level == 1 or domain == "graph_coloring" else (
        "countdown_legal_v3" if domain == "countdown" else "domain_legal_v1"
    )
    return {
        "schema": SCHEMA,
        "level": level,
        "domain": domain,
        "template_name": template,
        "prompt_source": "src/oat_drgrpo/templates.py:TEMPLATE_FACTORY",
        "reference_recipe": "ops/exp_scaling/e124_qwen7b_recipes.py:interface",
        "chat_transport": "structured_system_and_user_messages",
        "response_surface": "six_bit_ingredient_support" if mask else "boxed_final_answer",
        "canonical_action_task": "pantry_support_mask" if mask else "none",
        "trusted_quantity_projection": mask,
        "hosted_syntax_constraints": "none",
        "reference_syntax_profile": parent_syntax,
        "reference_fixed_action_mask": mask,
        "verifier": "oat_drgrpo.math_grader.validated_modebench_outcome_key",
    }


def make_messages(level: int | str, domain: str, row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Reproduce original prompt text without sending model-specific role tokens.

    The answer/spec and metadata are never included in the request. Retaining
    the original template function also retains its stale-suffix checks.
    """
    from oat_drgrpo.templates import CHAT_SURFACES, TEMPLATE_FACTORY

    profile = profile_metadata(level, domain)
    problem = row.get("problem")
    if not isinstance(problem, str) or not problem:
        raise ValueError("ModeBench row requires a nonempty problem string")
    if "<|im_start|>" in problem or "<|im_end|>" in problem:
        raise ValueError("unexpected chat role marker in benchmark problem")
    rendered = TEMPLATE_FACTORY[profile["template_name"]](problem)
    system_marker, user_marker, assistant_marker = CHAT_SURFACES["qwen"]
    if not rendered.startswith(system_marker) or not rendered.endswith(assistant_marker):
        raise ValueError("original template no longer uses the registered chat surface")
    body = rendered[len(system_marker):-len(assistant_marker)]
    if body.count(user_marker) != 1:
        raise ValueError("original template is not one system/user turn")
    system, user = body.split(user_marker)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def grade_response(level: int | str, domain: str, row: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Grade with the original executable verifier, retaining the graded surface.

    A Level 1 Pantry mask is decoded by the existing deterministic environment
    transition before verification. It searches only the selected ingredient
    support and never substitutes a different support. Invalid masks fail closed.
    """
    from oat_drgrpo.math_grader import validated_modebench_outcome_key

    level, domain = _identity(level, domain)
    if not isinstance(text, str):
        raise TypeError("response text must be a string")
    answer = row["answer"]
    graded_text = text
    if level == 1 and domain == "pantry_plan":
        from oat_drgrpo.pantry_support_action import decode_pantry_support_mask

        spec = json.loads(answer) if isinstance(answer, str) else answer
        if not isinstance(spec, Mapping):
            raise ValueError("Pantry executable spec must be an object")
        graded_text = decode_pantry_support_mask(text, spec)
    key = validated_modebench_outcome_key(graded_text, answer)
    return {"verified": key is not None, "canonical_key": key, "graded_text": graded_text}
