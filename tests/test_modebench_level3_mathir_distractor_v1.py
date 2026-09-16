from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_mathir_distractor_v1 as distractor
import modebench_level3_mathir_sign_v1 as sign
from make_mathir_action_menu_data import FAMILIES
from oat_drgrpo.mathir import enumerate_mathir_action_menu_keys


def identities(rows):
    return {
        distractor.semantic_identity(
            distractor.family_for_difficulty(row['level3_difficulty']),
            json.loads(row['answer'])['bindings'],
        )
        for row in rows
    }


@pytest.fixture(scope='module')
def original_support():
    family = FAMILIES[0]
    spec = distractor._spec(
        family, {'a': 2, 'b': 3, 'c': 11},
        dict(zip('ABCDEF', family.commands)), 0, 'original_certificate', 0,
    )
    keys = frozenset(enumerate_mathir_action_menu_keys(spec))
    assert len(keys) == 5
    return keys


@pytest.mark.parametrize('difficulty', range(4))
def test_original_family_core_five_and_only_sixth_command_replaced(difficulty):
    family = distractor.family_for_difficulty(difficulty)
    original = FAMILIES[0]
    assert distractor.PROFILE == 'mathir_fixed_distractor_laws_v1'
    assert family.name == original.name == 'ax_plus_b_eq_c'
    assert family.initial_lhs == original.initial_lhs == 'add(mul(a,x),b)'
    assert family.initial_rhs == original.initial_rhs == 'c'
    assert family.display_equation == original.display_equation
    assert family.certified_routes == original.certified_routes
    assert family.commands[:5] == original.commands[:5] == (
        'sub(b)', 'div(a)', 'sub(div(b,a))', 'add(b)', 'mul(a)',
    )
    assert len(family.commands) == len(set(family.commands)) == 6
    assert family.commands[5] == ('sub(a)' if difficulty % 2 == 0 else 'div(b)')


def test_exact_finite_inventories_match_independent_laws():
    coefficients = [value for value in range(-9, 10) if value]
    offsets = [value for value in range(-12, 13) if value]
    # Construct identities directly, independently of the profile's law and
    # identity helpers, to check both the catalog and historical exclusions.
    broad = frozenset(
        ('mathir', 'ax_plus_b_eq_c', (('a', a), ('b', b), ('c', a * x + b)))
        for a, b, x in itertools.product(coefficients, offsets, coefficients)
    )
    capped = frozenset(identity for identity in broad if abs(dict(identity[2])['c']) <= 12)
    assert len(broad) == 18 * 24 * 18 == 7776
    assert len(capped) == 2468
    inventories = [distractor.finite_inventory(difficulty) for difficulty in range(4)]
    assert all(isinstance(inventory, frozenset) for inventory in inventories)
    assert inventories == [broad, broad, capped, capped]
    assert capped < broad
    assert any(dict(identity[2])['c'] == 0 for identity in capped)
    for difficulty, inventory in enumerate(inventories):
        assert all(distractor.law_holds(difficulty, dict(identity[2])) for identity in inventory)


@pytest.mark.parametrize('difficulty', range(4))
def test_law_bounds_and_sample_membership(difficulty):
    accepted = [
        {'a': 1, 'b': -1, 'c': 0},
        {'a': -9, 'b': -9, 'c': 0},
        {'a': 1, 'b': 12, 'c': 11},
        {'a': 1, 'b': 11, 'c': 12},
        {'a': -1, 'b': -11, 'c': -12},
    ]
    for bindings in accepted:
        assert distractor.law_holds(difficulty, bindings)
    for bindings in [
        {'a': 0, 'b': 1, 'c': 2},
        {'a': 1, 'b': 0, 'c': 1},
        {'a': 10, 'b': 1, 'c': 11},
        {'a': 1, 'b': 13, 'c': 14},
        {'a': 1, 'b': 1, 'c': 1},  # Zero solution.
        {'a': 1, 'b': 1, 'c': 11},  # Solution outside +/-1..9.
        {'a': 2, 'b': 1, 'c': 2},  # Nonintegral solution.
        {'a': True, 'b': 1, 'c': 2},
        {'a': 1, 'b': 1, 'c': 2.0},
        {'a': 1, 'b': 1},
        {'a': 1, 'b': 1, 'c': 2, 'd': 1},
    ]:
        assert not distractor.law_holds(difficulty, bindings)
    for bindings in ({'a': 9, 'b': 12, 'c': 93}, {'a': -9, 'b': -12, 'c': -93}):
        assert distractor.law_holds(difficulty, bindings) is (difficulty < 2)
    first_rng, second_rng = random.Random(71570), random.Random(71570)
    for _ in range(8):
        bindings = distractor.sample_bindings(difficulty, first_rng)
        assert bindings == distractor.sample_bindings(difficulty, second_rng)
        solution = Fraction(bindings['c'] - bindings['b'], bindings['a'])
        assert solution.denominator == 1 and 1 <= abs(solution) <= 9
        assert distractor.semantic_identity(
            distractor.family_for_difficulty(difficulty), bindings,
        ) in distractor.finite_inventory(difficulty)


@pytest.mark.parametrize('difficulty', range(4))
def test_original_contract_quota_prefix_and_independent_menu_stream(difficulty, original_support):
    small = distractor.build_pool('mathir', Counter({5: 1}), set(), 71570, 'test', difficulty, 1)
    large = distractor.build_pool('mathir', Counter({5: 4}), set(), 71570, 'test', difficulty, 2)
    assert small == large[:1]
    assert len(large) == len(identities(large)) == 8
    digest = hashlib.sha256('\n'.join(sorted(original_support)).encode()).hexdigest()
    assert distractor.template_support(difficulty) == (5, digest)
    family = distractor.family_for_difficulty(difficulty)
    for index, row in enumerate(large):
        spec = json.loads(row['answer'])
        assert distractor.law_holds(difficulty, spec['bindings'])
        assert list(spec['actions']) == list('ABCDEF') and spec['max_steps'] == 4
        assert set(spec['actions'].values()) == set(family.commands)
        assert spec['verifier'] == row['modebench_task'] == 'mathir_action_menu'
        assert spec['mathir_version'] == 'linear-menu-v1'
        assert spec['family'] == row['mathir_family'] == 'ax_plus_b_eq_c'
        assert spec['initial_lhs'] == family.initial_lhs and spec['initial_rhs'] == 'c'
        assert spec['support_is_open'] is False
        assert row['answer_mode_count'] == spec['num_completions'] == spec['valid_mode_count'] == 5
        assert spec['valid_mode_key_sha256'] == digest
        assert row['level3_generation_profile'] == distractor.PROFILE
        assert row['level3_cell_index'] == index
        distractor.verify_witnesses(spec, difficulty)
    blocked = identities(large)
    fresh = distractor.build_pool('mathir', Counter({5: 8}), blocked, 71570, 'fresh', difficulty, 1)
    assert len(identities(fresh)) == 8
    assert not identities(fresh) & blocked
    # Exclusions and split tags can change bindings, but cannot consume or
    # reseed the independent menu stream for the same row index.
    assert [json.loads(row['answer'])['actions'] for row in fresh] == [
        json.loads(row['answer'])['actions'] for row in large
    ]


@pytest.mark.parametrize('difficulty,bindings', [
    (0, {'a': 1, 'b': -1, 'c': 0}),
    (1, {'a': 9, 'b': 12, 'c': 93}),
    (2, {'a': -9, 'b': -9, 'c': 0}),
    (3, {'a': -9, 'b': 9, 'c': -9}),
])
def test_numeric_boundaries_preserve_full_original_canonical_support(
    difficulty, bindings, original_support,
):
    assert distractor.law_holds(difficulty, bindings)
    family = distractor.family_for_difficulty(difficulty)
    spec = distractor._spec(
        family, bindings, dict(zip('ABCDEF', family.commands)), 0, 'boundary', 0,
    )
    assert frozenset(enumerate_mathir_action_menu_keys(spec)) == original_support
    distractor.verify_witnesses(spec, difficulty)


def test_cached_certificates_and_witness_verification_for_every_row(monkeypatch):
    for difficulty in range(4):
        distractor.template_support(difficulty)

    def unexpected_enumeration(*args, **kwargs):
        pytest.fail('cached templates must not be exhaustively enumerated for each row')

    monkeypatch.setattr(distractor, 'enumerate_mathir_action_menu_validations', unexpected_enumeration)
    checked = []
    verify = distractor.verify_witnesses

    def record_verification(spec, difficulty):
        verify(spec, difficulty)
        checked.append((spec['instance_id'], difficulty))

    monkeypatch.setattr(distractor, 'verify_witnesses', record_verification)
    for difficulty in range(4):
        rows = distractor.build_pool('mathir', Counter({5: 2}), set(), 98, 'cached', difficulty, 1)
        assert checked[-2:] == [(json.loads(row['answer'])['instance_id'], difficulty) for row in rows]
    assert len(checked) == 8


def test_original_profile_and_cross_preset_semantic_exclusions():
    bindings = {'a': 1, 'b': -1, 'c': 0}
    identity = ('mathir', 'ax_plus_b_eq_c', tuple(sorted(bindings.items())))
    assert sign.semantic_identity(FAMILIES[0], bindings) == identity
    irrelevant = {
        ('mathir', 'ax_plus_b_eq_c', (('a', 1), ('b', 0), ('c', 1))),
        ('mathir', 'some_other_family', tuple(sorted(bindings.items()))),
    }
    for difficulty in range(4):
        family = distractor.family_for_difficulty(difficulty)
        assert distractor.semantic_identity(family, bindings) == identity
        inventory = distractor.finite_inventory(difficulty)
        assert distractor.available_capacity(difficulty, irrelevant) == len(inventory)
        assert distractor.available_capacity(difficulty, irrelevant | {identity}) == len(inventory) - 1


@pytest.mark.parametrize('difficulty', range(4))
def test_exact_capacity_exhaustion_and_last_remaining_identity(difficulty):
    inventory = distractor.finite_inventory(difficulty)
    blocked = set(inventory)
    assert distractor.available_capacity(difficulty, blocked) == 0
    with pytest.raises(RuntimeError, match='fixed binding law is not widened'):
        distractor.build_pool('mathir', Counter({5: 1}), blocked, 0, 'test', difficulty, 1)
    last = min(inventory)
    blocked.remove(last)
    assert distractor.available_capacity(difficulty, blocked) == 1
    with pytest.raises(RuntimeError, match='fixed binding law is not widened'):
        distractor.build_pool('mathir', Counter({5: 1}), blocked, 0, 'test', difficulty, 2)
    rows = distractor.build_pool('mathir', Counter({5: 1}), blocked, 0, 'test', difficulty, 1)
    assert identities(rows) == {last}


@pytest.mark.parametrize('changes', [
    {'domain': 'pantry'}, {'target': Counter({4: 1})}, {'target': Counter()},
    {'target': Counter({'5': 1})}, {'target': Counter({5.0: 1})},
    {'target': Counter({True: 1})}, {'joint_target': Counter({5: 1})},
    {'target': Counter({5: -1})}, {'target': Counter({5: True})},
    {'target': Counter({5: 1.5})}, {'difficulty': True}, {'difficulty': 4},
    {'difficulty': 0.0}, {'difficulty': '0'},
    {'multiplier': 0}, {'multiplier': True}, {'multiplier': 1.5},
])
def test_invalid_requests(changes):
    kwargs = dict(domain='mathir', target=Counter({5: 1}), excluded=set(), seed=1,
                  tag='test', difficulty=0, multiplier=1)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        distractor.build_pool(**kwargs)


@pytest.mark.parametrize('mutation', ['zero_a', 'zero_b', 'changed_distractor'])
def test_witness_certificate_rejects_rows_outside_fixed_contract(mutation):
    difficulty = 1
    family = distractor.family_for_difficulty(difficulty)
    spec = distractor._spec(
        family, {'a': 2, 'b': 3, 'c': 11},
        dict(zip('ABCDEF', family.commands)), 0, 'invalid_contract', 0,
    )
    if mutation == 'zero_a':
        spec['bindings']['a'] = 0
    elif mutation == 'zero_b':
        spec['bindings']['b'] = 0
    else:
        spec['actions']['F'] = 'sub(c)'
    with pytest.raises(RuntimeError, match='fixed binding or six-action contract'):
        distractor.verify_witnesses(spec, difficulty)
