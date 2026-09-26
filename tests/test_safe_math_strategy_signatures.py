from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "ops/math_strategy_calibration/safe_math_strategy_signatures.py"
)
SPEC = importlib.util.spec_from_file_location(
    "safe_math_strategy_signatures", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
SIGNATURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIGNATURES
SPEC.loader.exec_module(SIGNATURES)


def _route(label: str, *actions: str) -> dict[str, object]:
    return {
        "label": label,
        "actions": list(actions),
    }


def test_accepts_only_frozen_incompatible_engine_pairs() -> None:
    distinct, left, right = SIGNATURES.safe_distinct_pair(
        _route("Use the Euclidean algorithm", "Take repeated remainders."),
        _route("Use prime factorization", "Intersect prime exponents."),
    )
    assert distinct is True
    assert {left, right} == {"euclidean_algorithm", "prime_factorization"}


def test_collapses_product_rendering_as_uncertain() -> None:
    distinct, left, right = SIGNATURES.safe_distinct_pair(
        _route("Multiply independent choices", "Use 10*9*8*7."),
        _route(
            "Ordered tuple filter",
            "Exhaust all tuples and filter repeated members.",
        ),
    )
    assert distinct is False
    assert left is None
    assert right == "finite_constraint_search"


def test_rejects_mixed_route() -> None:
    distinct, left, right = SIGNATURES.safe_distinct_pair(
        _route(
            "Use either method",
            "Apply the quadratic formula and then use Vieta's relations.",
        ),
        _route("Use Vieta", "Use the sum and product of the roots."),
    )
    assert distinct is False
    assert left is None
    assert right == "vieta_relations"


def test_menu_uses_only_referenced_actions() -> None:
    menu = {
        "actions": [
            {"action_id": "A1", "operation": "Apply the quadratic formula."},
            {"action_id": "A2", "operation": "Use Vieta's formulas."},
            {
                "action_id": "A3",
                "operation": "A distracting Euclidean algorithm action.",
            },
        ],
        "strategies": [
            {"strategy_id": "S1", "action_ids": ["A1"], "plan": ""},
            {"strategy_id": "S2", "action_ids": ["A2"], "plan": ""},
        ],
    }
    distinct, left, right = SIGNATURES.safe_menu_pair(menu)
    assert distinct is True
    assert {left, right} == {"quadratic_formula", "vieta_relations"}
