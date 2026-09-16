from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
from modebench_level3_mathir_v2 import build_pool, family_for_difficulty
from oat_drgrpo.mathir import enumerate_mathir_action_menu_keys, validate_mathir_action_menu


@pytest.mark.parametrize('difficulty', range(4))
def test_broader_mathir_templates_have_exact_five_modes(difficulty):
    rows = build_pool('mathir', Counter({5: 2}), set(), 68400 + difficulty,
                      'test_broad', difficulty, 1)
    family = family_for_difficulty(difficulty)
    for row in rows:
        spec = json.loads(row['answer'])
        assert set(spec['actions']) == set('ABCDEF')
        assert spec['max_steps'] == 4
        by_command = {command: key for key, command in spec['actions'].items()}
        for route in family.certified_routes:
            assert validate_mathir_action_menu(';'.join(by_command[command] for command in route), spec)
    spec = json.loads(rows[1]['answer'])
    keys = enumerate_mathir_action_menu_keys(spec)
    assert len(keys) == 5
    assert hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest() == spec['valid_mode_key_sha256']
    assert build_pool('mathir', Counter({5: 2}), set(), 68400 + difficulty,
                      'test_broad', difficulty, 1) == rows
    excluded = {('mathir', json.loads(row['answer'])['family'],
                 tuple(sorted(json.loads(row['answer'])['bindings'].items()))) for row in rows}
    fresh = build_pool('mathir', Counter({5: 2}), excluded, 68400 + difficulty,
                       'test_broad', difficulty, 1)
    fresh_ids = {('mathir', json.loads(row['answer'])['family'],
                  tuple(sorted(json.loads(row['answer'])['bindings'].items()))) for row in fresh}
    assert not excluded & fresh_ids


@pytest.mark.parametrize('difficulty', (0, 1, 2))
def test_easier_tiers_keep_original_coefficient_and_solution_bounds(difficulty):
    rows = build_pool('mathir', Counter({5: 128}), set(), 77500 + difficulty,
                      'test_bounds', difficulty, 1)
    assert len(rows) == 128
    for row in rows:
        spec = json.loads(row['answer'])
        bindings = spec['bindings']
        a, b, c = (bindings[key] for key in 'abc')
        assert 1 <= abs(a) <= 9
        assert -12 <= b <= 12
        if difficulty == 0:
            assert spec['family'] == 'ax_plus_b_eq_c'
            solution = Fraction(c - b, a)
        elif difficulty == 1:
            solution = Fraction(c, a) - b
        else:
            assert spec['family'] == 'ax_plus_b_eq_dx_plus_c'
            assert 1 <= abs(bindings['d']) <= 9
            solution = Fraction(c - b, a - bindings['d'])
        assert solution.denominator == 1 and 1 <= abs(solution) <= 9


def test_rejects_wrong_domain_and_support():
    with pytest.raises(ValueError, match='MathIR only'):
        build_pool('pantry', Counter({5: 1}), set(), 1, 'test', 0)
    with pytest.raises(ValueError, match='five-mode'):
        build_pool('mathir', Counter({4: 1}), set(), 1, 'test', 0)
