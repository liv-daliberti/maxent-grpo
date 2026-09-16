"""Prospective MathIR laws: structural verification without model inference."""
from collections import Counter
import copy
import hashlib
import json
from math import gcd
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_scale_mathir_numeric_candidates as numeric
import modebench_level3_constraints as constraints


def pool(tier=0, count=1, seed=963011001, excluded=(), tag='STRUCTURAL_TEST_ONLY', **kwargs):
    return numeric.build_pool('mathir', Counter({5: count}), set(excluded), seed, tag, tier, **kwargs)


@pytest.fixture(scope='module')
def representatives():
    return {tier: pool(tier)[0] for tier in range(4)}


@pytest.mark.parametrize('tier', range(4))
def test_fixed_laws_and_exact_catalogs(tier):
    rows = pool(tier, count=64)
    lower, upper = numeric.BANDS[tier]
    assert Counter(row['answer_mode_count'] for row in rows) == {5: 64}
    assert len({numeric.identity('mathir', row) for row in rows}) == 64
    signs = {symbol: set() for symbol in 'abcdef'}
    for row in rows:
        bindings = json.loads(row['answer'])['bindings']
        assert numeric.law_holds(tier, bindings)
        assert all(lower <= abs(value) <= upper for value in bindings.values())
        assert bindings['a'] != bindings['d'] * bindings['e']
        for symbol, value in bindings.items():
            signs[symbol].add(value > 0)
        if tier:
            assert gcd(abs(bindings['a']), abs(bindings['e'])) == 1
            assert gcd(abs(bindings['b']), abs(bindings['f'])) == 1
    assert all(values == {True, False} for values in signs.values())
    if tier:
        assert numeric.coprime_pairs(tier) == tuple(
            (a, e) for a in range(lower, upper + 1) for e in range(lower, upper + 1) if gcd(a, e) == 1)
        assert len(numeric.coprime_pairs(tier)) == {1: 156, 2: 546, 3: 6144}[tier]


def test_anchor_matches_original_independent_nonzero_integer_proposal():
    candidate_rng, original_rng = random.Random(12345), random.Random(12345)
    values = [value for value in range(-13, 14) if value]
    accepted = rejected = 0
    for _ in range(1000):
        expected = {name: original_rng.choice(values) for name in 'abcdef'}
        assert numeric._proposal(0, candidate_rng) == expected
        admissible = expected['a'] != expected['d'] * expected['e']
        assert numeric.law_holds(0, expected) is admissible
        accepted += admissible
        rejected += not admissible
    assert accepted and rejected


@pytest.mark.parametrize('tier', range(4))
def test_quota_prefix_determinism_multiplier_and_exclusion(tier):
    one, three = pool(tier, count=1), pool(tier, count=3)
    assert one == pool(tier, count=1)
    assert one == sorted(three, key=lambda row: row['scale_cell_index'])[:1]
    assert three == pool(tier, count=1, multiplier=3)
    blocked = {numeric.identity('mathir', row) for row in three}
    fresh = pool(tier, count=3, excluded=blocked, tag='FRESH_SPLIT')
    assert not blocked & {numeric.identity('mathir', row) for row in fresh}
    assert not {row['problem'] for row in three} & {row['problem'] for row in fresh}
    assert pool(tier, count=0) == []


@pytest.mark.parametrize('tier', range(4))
def test_original_equation_prompt_actions_and_all_five_witnesses(representatives, tier):
    row = representatives[tier]
    spec = json.loads(row['answer'])
    family = constraints._mathir_family(1)
    assert spec['family'] == family.name
    assert spec['initial_lhs'] == family.initial_lhs and spec['initial_rhs'] == family.initial_rhs
    assert tuple(spec['actions']) == tuple('ABCDEF')
    assert Counter(spec['actions'].values()) == Counter(family.commands)
    assert spec['max_steps'] == 4 and spec['support_is_open'] is False
    assert row['problem'] == constraints.mathir_prompt(family, spec['bindings'], spec['actions'])
    keys, paths, digest = numeric._certificate()
    assert len(keys) == len(paths) == spec['num_completions'] == spec['valid_mode_count'] == 5
    assert spec['valid_mode_key_sha256'] == digest == hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()
    report = numeric.verify_rows('mathir', [row])
    assert report['rows_verified'] == 1 and report['witnesses_verified'] == 5
    assert report['origin_tier'] == 1 and report['exact_canonical_support']
    assert report['unchanged_original_prompt'] and report['original_verifier_witnesses']


@pytest.mark.parametrize('field,value', [
    ('problem', 'changed'), ('answer_mode_count', 4), ('answer_mode_count', 5.0),
    ('scale_candidate_tier', True), ('scale_candidate_tier', 4),
    ('scale_candidate_generator', 'wrong'), ('scale_candidate_profile', '{}'),
    ('scale_origin_metadata', '{}'), ('scale_numeric_law_version', 'wrong'),
    ('scale_generation_seed', -1), ('scale_generation_seed', 1),
    ('scale_cell_index', -1), ('scale_cell_index', 10), ('answer_mode_split', ''),
    ('answer_mode_split', 'wrong'), ('mathir_family', 'wrong'), ('modebench_task', 'wrong')])
def test_corrupt_metadata_or_prompt_rejected(representatives, field, value):
    row = copy.deepcopy(representatives[1])
    row[field] = value
    with pytest.raises(RuntimeError, match='audit failed'):
        numeric.verify_rows('mathir', [row])


@pytest.mark.parametrize('mutation', [
    'bindings', 'coprimality', 'bool_binding', 'family', 'lhs', 'rhs', 'actions',
    'action_ids', 'steps', 'open_support', 'source', 'instance_id', 'count', 'digest', 'extra_field'])
def test_self_consistent_prompt_cannot_escape_exact_spec(representatives, mutation):
    row = copy.deepcopy(representatives[1])
    spec = json.loads(row['answer'])
    if mutation == 'bindings':
        spec['bindings']['a'] = 1
    elif mutation == 'coprimality':
        spec['bindings']['a'] = spec['bindings']['e'] = 14
    elif mutation == 'bool_binding':
        spec['bindings']['a'] = True
    elif mutation == 'family':
        spec['family'] = constraints._mathir_family(2).name
    elif mutation in ('lhs', 'rhs'):
        spec['initial_' + mutation] = 'a'
    elif mutation == 'actions':
        spec['actions']['A'] = 'sub(a)'
    elif mutation == 'action_ids':
        spec['actions']['Z'] = spec['actions'].pop('F')
    elif mutation == 'steps':
        spec['max_steps'] = 3
    elif mutation == 'open_support':
        spec['support_is_open'] = True
    elif mutation == 'source':
        spec['source'] = 'other'
    elif mutation == 'instance_id':
        spec['instance_id'] = 'other'
    elif mutation == 'count':
        spec['num_completions'] = spec['valid_mode_count'] = 6
    elif mutation == 'digest':
        spec['valid_mode_key_sha256'] = '0' * 64
    else:
        spec['unregistered_parameter'] = 1
    row['answer'] = json.dumps(spec, sort_keys=True, separators=(',', ':'))
    row['problem'] = constraints.mathir_prompt(numeric.FAMILY, spec['bindings'], spec['actions'])
    with pytest.raises(RuntimeError, match='audit failed'):
        numeric.verify_rows('mathir', [row])


def test_duplicate_identity_is_rejected_even_after_menu_permutation(representatives):
    row = representatives[2]
    duplicate = copy.deepcopy(row)
    spec = json.loads(duplicate['answer'])
    commands = list(spec['actions'].values())
    spec['actions'] = dict(zip('ABCDEF', commands[1:] + commands[:1]))
    duplicate['answer'] = json.dumps(spec, sort_keys=True, separators=(',', ':'))
    duplicate['problem'] = constraints.mathir_prompt(numeric.FAMILY, spec['bindings'], spec['actions'])
    assert duplicate['problem'] != row['problem']
    assert numeric.identity('mathir', duplicate) == numeric.identity('mathir', row)
    with pytest.raises(RuntimeError, match='duplicate semantic identity'):
        numeric.verify_rows('mathir', [row, duplicate])


@pytest.mark.parametrize('changes', [
    {'tier': True}, {'tier': 4}, {'seed': -1}, {'seed': 1.2}, {'seed': True},
    {'multiplier': False}, {'multiplier': 0}, {'tag': ''}, {'tag': None},
    {'target': {5: True}}, {'target': {5.0: 1}}, {'target': {5: -1}},
    {'target': {4: 1}}, {'target': [5]}, {'target': None}, {'excluded': []},
    {'joint_target': {}}, {'domain': 'pantry'}])
def test_invalid_inputs_fail_closed(changes):
    args = dict(domain='mathir', target={5: 1}, excluded=set(), seed=1, tag='INVALID_TEST', tier=0)
    args.update(changes)
    with pytest.raises(ValueError):
        numeric.build_pool(**args)


def test_exhaustion_fails_without_a_partial_pool(monkeypatch):
    monkeypatch.setattr(numeric, 'MAX_PROPOSALS_PER_ROW', 2)
    monkeypatch.setattr(numeric, '_proposal', lambda tier, rng: dict.fromkeys('abcdef', 1))
    with pytest.raises(RuntimeError, match='exhausted fixed proposal budget'):
        pool(0)


def test_source_inventory_pins_original_generators_and_verifier():
    paths = numeric.source_paths()
    assert Path(numeric.__file__).resolve() in paths
    assert Path(numeric.original.__file__).resolve() in paths
    assert Path(constraints.__file__).resolve() in paths
    assert ROOT / 'src/oat_drgrpo/mathir.py' in paths
    assert len(paths) == len(set(paths)) and all(path.is_file() for path in paths)


def test_arrow_roundtrip_preserves_four_profiles(representatives, tmp_path):
    from datasets import Dataset, DatasetDict, load_from_disk
    rows = [representatives[tier] for tier in range(4)]
    path = tmp_path / 'nonproduction-mathir'
    DatasetDict({'multi_answer': Dataset.from_list(rows)}).save_to_disk(str(path))
    assert [dict(row) for row in load_from_disk(str(path))['multi_answer']] == rows
