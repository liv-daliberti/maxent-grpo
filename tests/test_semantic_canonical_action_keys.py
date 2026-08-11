"""The semantic term must bind a canonical-action task's own surfaces.

PantryPlan emits an eight-token canonical action sequence, not a free-form
response with a boxed answer. Decoding those tokens and running the free-form
text extractor over them returns None for every row, so the semantic advantage
is exact zero for the entire run while the verifier and the replay bank both
work normally. That is what happened in E81, E82, and E83: PantryPlan logged a
reward-positive fraction of .544 and a parseable fraction of .000.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRPO = ROOT / "src/oat_drgrpo/learner/grpo.py"
SURFACE_HELPER = "_task_bound_canonicalization_surfaces"


def _call_sites() -> list[ast.Call]:
    tree = ast.parse(GRPO.read_text(encoding="utf-8"), filename=str(GRPO))
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "_seed_answer_keys_grouped":
            sites.append(node)
    return sites


def _enclosing_source(call: ast.Call, *, before: int = 40) -> str:
    lines = GRPO.read_text(encoding="utf-8").splitlines()
    start = max(0, call.lineno - before)
    return "\n".join(lines[start : call.lineno])


def test_every_key_derivation_site_is_accounted_for():
    # Five sites derive answer keys, guarded by five different treatments:
    #   mi_tracker (DIAYN), outcome_collision_coef, semantic_shannon_tracker,
    #   xdr_mode_adaptive, and seed_alpha.
    # Only the outcome-collision and semantic-Shannon sites bind canonical
    # surfaces. The other three carry the same latent gap, but no arm in the
    # current campaign enables them, and repairing them would change the
    # runtime of treatments nobody is running. If a new site appears, this
    # forces a decision rather than letting it pass silently.
    assert len(_call_sites()) == 5


def test_the_semantic_site_binds_canonical_action_surfaces():
    semantic = [
        call
        for call in _call_sites()
        if "semantic_key_kwargs" in _enclosing_source(call)
        and "semantic_shannon" in _enclosing_source(call, before=200)
    ]
    assert semantic, "no semantic-Shannon key-derivation site binds surfaces"
    for call in semantic:
        source = _enclosing_source(call)
        assert "if canonical_actions:" in source
        assert SURFACE_HELPER in source
        assert any(
            kw.arg is None and getattr(kw.value, "id", "") == "semantic_key_kwargs"
            for kw in call.keywords
        ), "semantic site does not forward the canonical surfaces"


def test_the_outcome_collision_site_still_binds_surfaces():
    # The sibling path this fix was modelled on must not regress.
    collision = [
        call
        for call in _call_sites()
        if "outcome_collision" in _enclosing_source(call, before=120)
        and "semantic_key_kwargs" in _enclosing_source(call)
    ]
    assert collision, "outcome-collision site lost its canonical-action branch"


def test_surfaces_are_bound_only_for_canonical_action_tasks():
    # Free-form domains must keep decoding their own responses; binding
    # surfaces unconditionally would break Graph, Countdown, Python, MathIR.
    source = GRPO.read_text(encoding="utf-8")
    assert source.count("if canonical_actions:\n                semantic_key_kwargs") == 2
