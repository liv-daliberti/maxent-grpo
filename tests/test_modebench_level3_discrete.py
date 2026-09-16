"""Independent exact-count checks for the Level 3 discrete generators."""
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'ops', ROOT / 'ops/exp_scaling', ROOT / 'src'):
    sys.path.insert(0, str(path))

from make_exact_countdown_mode_data import _canonical_expression_keys
from make_modebench_data import _countdown_expression_map
from make_python_factor_mode_data import _certified_programs
from modebench_level3_discrete import build_pool, countdown_modes, graph_completion_count, row_identity
from oat_drgrpo.math_grader import _verify_graph_coloring_answer
from oat_drgrpo.python_modebench import python_factor_mode_count
from oat_drgrpo.python_modebench_process import validate_python_factor_function_external


def test_countdown_subset_enumeration_matches_existing_exact_canonicalizer():
    for numbers in ((2, 3, 5), (2, 3, 5, 7)):
        old = _countdown_expression_map(list(numbers))
        actual = countdown_modes(numbers)
        assert set(actual) == set(old)
        for target, expressions in old.items():
            expected = _canonical_expression_keys(list(numbers), target, expressions)
            assert {'countdown:' + key for key in actual[target]} == expected


def test_graph_counter_matches_exhaustive_verifier_with_and_without_partial():
    edges = [[1, 2], [1, 3], [2, 4], [3, 4], [4, 5]]
    for partial in ([None] * 5, [1, None, None, 2, None], [1, 1, None, None, None]):
        spec = {'verifier': 'graph_coloring', 'n': 5, 'edges': edges, 'partial_colors': partial}
        expected = sum(_verify_graph_coloring_answer(''.join(map(str, colors)), spec)
                       for colors in itertools.product((1, 2, 3), repeat=5))
        assert graph_completion_count(5, edges, partial) == expected
        capped = graph_completion_count(5, edges, partial, cap=3)
        assert capped == expected if expected <= 3 else capped > 3


def test_pools_preserve_support_are_disjoint_and_python_is_externally_certified():
    for domain, target in (
        ('countdown', Counter({2: 1, 3: 1, 5: 1, 7: 1})),
        ('graph_coloring', Counter({4: 1, 6: 1, 8: 1, 9: 1, 12: 1})),
        ('python_factors', Counter({16: 1, 32: 1, 144: 1, 420: 1, 1200: 1})),
    ):
        rows = build_pool(domain, target, set(), 800123, 'test_first', 3, multiplier=2)
        assert Counter(row['answer_mode_count'] for row in rows) == Counter({k: 2 * v for k, v in target.items()})
        blocked = {row_identity(domain, row) for row in rows}
        second = build_pool(domain, target, blocked, 800123, 'test_second', 3, multiplier=1)
        assert not blocked & {row_identity(domain, row) for row in second}
        for row in rows:
            assert row['level3_difficulty'] == 3
            spec = json.loads(row['answer'])
            if domain == 'python_factors':
                cases = tuple(spec['cases'])
                assert python_factor_mode_count(cases) == row['answer_mode_count']
                certified = [validate_python_factor_function_external(program, spec)
                             for program in _certified_programs(cases)]
                assert all(certified)
                assert len({validation.canonical_key for validation in certified}) == 2
            elif domain == 'graph_coloring':
                assert graph_completion_count(spec['n'], spec['edges'], spec['partial_colors']) == row['answer_mode_count']
