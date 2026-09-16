from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_mathir_v3 as v3
from oat_drgrpo.mathir import enumerate_mathir_action_menu_keys, validate_mathir_action_menu


def identities(rows):
    return {v3.semantic_identity(v3.family_for_difficulty(row['level3_difficulty']),
                                json.loads(row['answer'])['bindings']) for row in rows}


@pytest.mark.parametrize('difficulty', range(4))
def test_original_support_routes_bounds_and_quota_prefix(difficulty):
    small = v3.build_pool('mathir', Counter({5: 5}), set(), 71400, 'test', difficulty, 1)
    large = v3.build_pool('mathir', Counter({5: 19}), set(), 71400, 'test', difficulty, 1)
    assert large[:5] == small
    assert len(identities(large)) == 19
    for row in large:
        spec = json.loads(row['answer'])
        assert list(spec['actions']) == list('ABCDEF')
        assert spec['max_steps'] == 4
        assert row['answer_mode_count'] == spec['num_completions'] == spec['valid_mode_count'] == 5
        assert spec['verifier'] == 'mathir_action_menu'
        assert spec['mathir_version'] == 'linear-menu-v1'
        bindings = spec['bindings']
        if difficulty < 3:
            assert 1 <= abs(bindings['a']) <= 9 and -12 <= bindings['b'] <= 12
            solution = Fraction(bindings['c'] - bindings['b'], bindings['a'])
            assert solution.denominator == 1 and 1 <= abs(solution) <= 9
            bound = v3.CONSTANT_BOUNDS[difficulty]
            assert bound is None or abs(bindings['c']) <= bound
        else:
            assert all(1 <= abs(v) <= 29 for v in bindings.values())
            assert bindings['a'] * bindings['f'] + bindings['d'] * bindings['e']
    spec = json.loads(small[0]['answer'])
    family = v3.family_for_difficulty(difficulty)
    command_ids = {command: key for key, command in spec['actions'].items()}
    for route in family.certified_routes:
        assert validate_mathir_action_menu(';'.join(command_ids[cmd] for cmd in route), spec)
    keys = enumerate_mathir_action_menu_keys(spec)
    assert len(keys) == 5
    assert hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest() == spec['valid_mode_key_sha256']
    fresh = v3.build_pool('mathir', Counter({5: 5}), identities(large), 71400,
                          'test', difficulty, 1)
    assert not identities(fresh) & identities(large)


def test_exact_finite_inventory_and_exhaustion_preserve_the_law():
    assert [len(v3.finite_inventory(d)) for d in range(3)] == [2584, 1428, 8100]
    assert v3.finite_inventory(1) < v3.finite_inventory(0) < v3.finite_inventory(2)
    with pytest.raises(RuntimeError, match='fixed binding law is not widened'):
        v3.build_pool('mathir', Counter({5: 1}), set(v3.finite_inventory(1)),
                      71400, 'test', 1, 1)


def test_menu_stream_is_independent_of_binding_rejection(monkeypatch):
    baseline = v3.build_pool('mathir', Counter({5: 4}), set(), 71400, 'test', 1, 1)
    original = v3.sample_bindings

    def consume_extra_randomness(difficulty, rng):
        rng.random()
        return original(difficulty, rng)

    monkeypatch.setattr(v3, 'sample_bindings', consume_extra_randomness)
    changed = v3.build_pool('mathir', Counter({5: 4}), set(), 71400, 'test', 1, 1)
    assert [json.loads(row['answer'])['actions'] for row in baseline] == [
        json.loads(row['answer'])['actions'] for row in changed]
    assert identities(baseline) != identities(changed)


def test_conditioning_keeps_exact_support_at_numeric_boundaries():
    family = v3.family_for_difficulty(1)
    actions = dict(zip('ABCDEF', family.commands))
    key_sets = []
    for bindings in ({'a': 1, 'b': 0, 'c': 1},
                     {'a': -1, 'b': 1, 'c': 0},
                     {'a': 9, 'b': -12, 'c': 6}):
        spec = v3._spec(family, bindings, actions, 0, 'boundary', 0)
        key_sets.append(enumerate_mathir_action_menu_keys(spec))
    assert all(len(keys) == 5 for keys in key_sets)
    assert key_sets[0] == key_sets[1] == key_sets[2]


def test_audit_reload_rejects_duplicate_identities_and_protects_dependencies():
    import audit_modebench_level3_mathir_v3 as audit
    rows = v3.build_pool('mathir', Counter({5: 3}), set(), 71400, 'test', 1, 1)
    assert all(audit.verify_published_pool(rows, rows, set()).values())
    with pytest.raises(RuntimeError, match='unique_identities'):
        audit.verify_published_pool([rows[0], rows[0], rows[2]], rows, set())
    protected = audit.protected_hashes()
    assert 'ops/make_mathir_action_menu_data.py' in protected
    assert 'ops/exp_scaling/modebench_level3_constraints.py' in protected


@pytest.mark.parametrize('arguments', [
    {'domain': 'pantry'}, {'target': Counter({4: 1})},
    {'difficulty': 4}, {'difficulty': True}, {'multiplier': 0},
])
def test_invalid_requests_fail(arguments):
    kwargs = dict(domain='mathir', target=Counter({5: 1}), excluded=set(),
                  seed=1, tag='test', difficulty=0, multiplier=1)
    kwargs.update(arguments)
    with pytest.raises(ValueError):
        v3.build_pool(**kwargs)
