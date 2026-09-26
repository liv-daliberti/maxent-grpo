"""Support counts must respect actual mode identity and incomplete certificates."""
import ast
from collections import Counter
from fractions import Fraction
import json

from ops.gpt56_discovery_support import binary_countdown_modes, certify_row, graph_modes


def test_graph_support_counts_fixed_completions_not_unconstrained_colorings():
    spec = {"n": 3, "edges": [[1, 2], [2, 3]], "partial_colors": [1, None, None]}
    modes = graph_modes(spec)
    assert set(modes) == {"graph_coloring:121", "graph_coloring:123",
                          "graph_coloring:131", "graph_coloring:132"}


def test_countdown_commutative_aliases_merge_but_distinct_routes_survive():
    modes = binary_countdown_modes({"numbers": [2, 3, 4], "target": 10})
    assert set(modes) == {"countdown:add(4,mul(2,3))", "countdown:sub(mul(3,4),2)"}
    assert len(binary_countdown_modes({"numbers": [3, 6, 9], "target": 18})) == 8


def test_countdown_witnesses_execute_exactly_with_every_operand_once():
    modes = binary_countdown_modes({"numbers": [2, 4, 6, 7], "target": 18})
    assert len(modes) == 6
    for expression in modes.values():
        tree = ast.parse(expression, mode="eval")
        assert Counter(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)) == Counter([2, 4, 6, 7])
        def execute(n):
            if isinstance(n, ast.Constant):
                return Fraction(n.value)
            a, b = execute(n.left), execute(n.right)
            if isinstance(n.op, ast.Add): return a + b
            if isinstance(n.op, ast.Sub): return a - b
            if isinstance(n.op, ast.Mult): return a * b
            return a / b
        assert execute(tree.body) == 18


def test_countdown_certificate_does_not_promote_binary_grammar_to_total_support():
    spec = {"verifier": "countdown", "numbers": [2, 3], "target": 5, "num_completions": 1}
    def verifier(text, answer):
        if text.startswith("\\boxed{-(-("):
            return "countdown:neg(neg(add(2,3)))"
        return "countdown:add(2,3)"
    ref = certify_row({"level": 2, "domain": "countdown", "row_index": 9,
                       "answer": json.dumps(spec)}, verifier)
    assert ref["support_kind"] == "certified_lower_bound"
    assert ref["support_count"] == 1
    assert not ref["unary_extension_witness"]["included_in_reference_count"]
