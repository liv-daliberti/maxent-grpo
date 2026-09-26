from collections import Counter
from fractions import Fraction
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_mathir_sign_v1 as sign
from oat_drgrpo.mathir import enumerate_mathir_action_menu_keys


def identities(rows):
    return {sign.semantic_identity(sign.family_for_difficulty(row['level3_difficulty']),
                                   json.loads(row['answer'])['bindings']) for row in rows}


@pytest.mark.parametrize('difficulty', range(4))
def test_law_original_contract_all_witnesses_and_quota_prefix(difficulty):
    small = sign.build_pool('mathir', Counter({5: 1}), set(), 71570, 'test', difficulty, 1)
    large = sign.build_pool('mathir', Counter({5: 8}), set(), 71570, 'test', difficulty, 1)
    assert small == large[:1]
    assert len(identities(large)) == 8
    for row in large:
        spec = json.loads(row['answer'])
        assert sign.law_holds(difficulty, spec['bindings'])
        assert list(spec['actions']) == list('ABCDEF') and spec['max_steps'] == 4
        assert set(spec['actions'].values()) == set(sign.family_for_difficulty(difficulty).commands)
        assert spec['verifier'] == row['modebench_task'] == 'mathir_action_menu'
        assert spec['mathir_version'] == 'linear-menu-v1'
        assert row['answer_mode_count'] == spec['num_completions'] == spec['valid_mode_count'] == 5
        sign.verify_witnesses(spec, difficulty)
    assert len(enumerate_mathir_action_menu_keys(json.loads(large[0]['answer']))) == 5
    blocked = identities(large)
    fresh = sign.build_pool('mathir', Counter({5: 8}), blocked, 71570, 'fresh', difficulty, 1)
    assert not identities(fresh) & blocked
    # Exclusions can change binding proposals, but never consume menu randomness.
    assert [json.loads(r['answer'])['actions'] for r in fresh] == [
        json.loads(r['answer'])['actions'] for r in large]


def test_exact_finite_capacity_and_failure():
    inventories = [sign.finite_inventory(d) for d in range(3)]
    assert [len(x) for x in inventories] == [1944, 1944, 3888]
    assert not inventories[0] & inventories[1]
    assert inventories[0] | inventories[1] == inventories[2]
    blocked = set(inventories[0])
    assert sign.available_capacity(0, blocked) == 0
    with pytest.raises(RuntimeError, match='fixed binding law is not widened'):
        sign.build_pool('mathir', Counter({5: 1}), blocked, 0, 'test', 0, 1)
    one_left = blocked - {min(blocked)}
    row = sign.build_pool('mathir', Counter({5: 1}), one_left, 0, 'test', 0, 1)
    assert identities(row) == {min(blocked)}


@pytest.mark.parametrize('bound', [1, 2, 3])
def test_rational_capacity_matches_exhaustive_small_bounds(bound):
    values = [v for v in range(-bound, bound + 1) if v]
    coefficient_count = sum(a*f+d*e != 0 for a, d, e, f in itertools.product(
        values, values, range(1, bound+1), range(1, bound+1)))
    assert sign.rational_capacity(bound) == coefficient_count * len(values) ** 2
    if bound == 3:
        assert sign.rational_capacity() == 9496390344


def test_rational_exclusion_count_requires_exact_law_and_family():
    family = sign.family_for_difficulty(3)
    yes = sign.semantic_identity(family, dict(a=1,b=1,c=1,d=1,e=1,f=1))
    no = sign.semantic_identity(family, dict(a=1,b=1,c=1,d=-1,e=1,f=1))
    negative_e = sign.semantic_identity(family, dict(a=1,b=1,c=1,d=1,e=-1,f=1))
    assert sign.available_capacity(3, {yes, no, negative_e}) == sign.rational_capacity() - 1


def test_numeric_boundaries_keep_exact_canonical_support():
    fixtures = [(0, {'a': 1, 'b': -1, 'c': 0}),
                (1, {'a': -9, 'b': 12, 'c': -69}),
                (2, {'a': 9, 'b': -12, 'c': -93}),
                (3, {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1}),
                (3, {'a': -29, 'b': 29, 'c': -29, 'd': 28, 'e': 29, 'f': 29})]
    for difficulty, bindings in fixtures:
        assert sign.law_holds(difficulty, bindings)
        family = sign.family_for_difficulty(difficulty)
        spec = sign._spec(family, bindings, dict(zip('ABCDEF', family.commands)), 0, 'test', 0)
        assert len(enumerate_mathir_action_menu_keys(spec)) == 5
        sign.verify_witnesses(spec, difficulty)


def test_all_history_and_candidate_pilots_leave_full_future_split_capacity():
    from materialize_modebench_level3 import historical_ids, identity_set
    blocked = historical_ids('mathir')
    for path in sorted((ROOT / 'var/data').glob('modebench_level3*/pools/mathir/*.jsonl')):
        blocked |= identity_set('mathir', [json.loads(line) for line in path.read_text().splitlines()])
    # Exact worst-case capacity after four 128-row pilots: d2 can remove at
    # most128 identities from either disjoint sign half; d0+d1+d2 remove384
    # from their union. The rational family is disjoint from all three.
    maximum_pilot_losses = (256, 256, 384, 128)
    for difficulty, loss in enumerate(maximum_pilot_losses):
        assert sign.available_capacity(difficulty, blocked) - loss >= 512
    # Every nonempty union of presets contains a preset with >=512 remaining
    # identities, a stronger Hall bound than any train/eval mixture needs.


@pytest.mark.parametrize('changes', [
    {'domain': 'pantry'}, {'target': Counter({4: 1})}, {'target': Counter({5: -1})},
    {'difficulty': True}, {'difficulty': 4}, {'multiplier': 0},
])
def test_invalid_requests(changes):
    kwargs = dict(domain='mathir', target=Counter({5: 1}), excluded=set(), seed=1,
                  tag='test', difficulty=0, multiplier=1)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        sign.build_pool(**kwargs)
