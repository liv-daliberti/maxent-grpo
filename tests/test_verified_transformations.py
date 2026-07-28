import json

from oat_drgrpo.math_grader import validated_modebench_outcome_key
from oat_drgrpo.verified_transformations import (
    derive_validator_preserving_counterfactuals,
)


def _keys(response, reference):
    original = validated_modebench_outcome_key(response, reference)
    alternatives = derive_validator_preserving_counterfactuals(
        response,
        reference,
    )
    keys = [
        validated_modebench_outcome_key(alternative, reference)
        for alternative in alternatives
    ]
    assert original is not None
    assert alternatives
    assert all(key is not None and key != original for key in keys)
    assert len(keys) == len(set(keys))
    return alternatives, keys


def test_python_factor_cofactor_is_model_self_and_verified():
    reference = json.dumps(
        {
            "verifier": "python_factor_function",
            "python_version": "factor-v1",
            "cases": [6, 10, 14, 22],
            "num_modes": 999,
            "gold_modes": ["must-not-be-read"],
        }
    )
    alternatives, keys = _keys("\\boxed{lambda n: 2}", reference)

    assert any("if n ==" in alternative for alternative in alternatives)
    assert all("n //" not in alternative for alternative in alternatives)
    assert "python_factor:3,5,7,11" in keys


def test_countdown_sign_rewrite_changes_tree_not_value_or_operands():
    reference = json.dumps(
        {
            "verifier": "countdown",
            "numbers": [2, 3, 4],
            "target": 14,
            "num_completions": 999,
            "gold_modes": ["must-not-be-read"],
        }
    )
    alternatives, _ = _keys("\\boxed{2 + 3 * 4}", reference)

    assert any("- -" in alternative or "-(-" in alternative for alternative in alternatives)


def test_graph_local_recolor_respects_public_fixed_vertices():
    reference = json.dumps(
        {
            "verifier": "graph_coloring",
            "n": 4,
            "edges": [[1, 2], [2, 3]],
            "partial_colors": [1, None, None, None],
            "num_completions": 999,
            "gold_modes": ["must-not-be-read"],
        }
    )
    alternatives, keys = _keys("\\boxed{1231}", reference)

    assert all(key.startswith("graph_coloring:1") for key in keys)
    assert alternatives


def test_mathir_commutes_additive_and_scaling_commands():
    reference = json.dumps(
        {
            "verifier": "mathir_algebra",
            "mathir_version": "linear-v0",
            "bindings": {"a": 3, "b": 2, "c": 17},
            "initial_lhs": "add(mul(a,x),b)",
            "initial_rhs": "c",
            "max_steps": 4,
            "support_is_open": True,
            "num_certified_strategies": 999,
            "gold_modes": ["must-not-be-read"],
        }
    )
    alternatives, _ = _keys("\\boxed{sub(b);div(a)}", reference)

    assert any("div(a);sub(div(b,a))" in alternative for alternative in alternatives)


def test_mathir_menu_uses_fixed_local_path_edits_not_declared_support():
    reference = json.dumps(
        {
            "verifier": "mathir_action_menu",
            "mathir_version": "linear-menu-v1",
            "bindings": {"a": 5, "b": 2, "c": 8, "d": 2},
            "initial_lhs": "add(mul(a,x),b)",
            "initial_rhs": "add(mul(d,x),c)",
            "max_steps": 4,
            "actions": {
                "A": "sub(b)",
                "B": "sub(mul(d,x))",
                "C": "div(sub(a,d))",
                "D": "sub(add(mul(d,x),b))",
                "E": "add(b)",
                "F": "div(a)",
            },
            "support_is_open": False,
            "num_completions": 999,
            "gold_modes": ["must-not-be-read"],
        }
    )
    alternatives, _ = _keys("\\boxed{A;B;C}", reference)

    assert "\\boxed{B;A;C}" in alternatives
