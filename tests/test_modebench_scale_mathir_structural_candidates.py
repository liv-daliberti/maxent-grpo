"""Prospective split-denominator MathIR: original parser and verifier only."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_scale_mathir_structural_candidates as structural
import modebench_level3_constraints as constraints


def pool(tier=0, count=1, seed=963011001, excluded=(), tag='STRUCTURAL_TEST_ONLY', **kwargs):
    return structural.build_pool('mathir', Counter({5: count}), set(excluded), seed, tag, tier, **kwargs)


@pytest.fixture(scope='module')
def representatives():
    return {tier: pool(tier)[0] for tier in range(4)}


CERTIFICATE_DIGESTS = (
    '417414a741a76129558315a57af4b20920b69c959666dd63e53e1cc167c1143c',
    '81559288b63102d25d9c74de1127dc49b14d08698b666ee4ebd6573e5455b0c0',
    '89e296fa400d1c8e18ed0ab71f37535b2119fba149c8a3f28f5a8d44fc920d17',
    '0c329c76b6755743cd380effd4537893fbc25f91fb21ed095b07d0a679deda71',
)
DENOMINATORS = (('e', 'f'), ('add(a,e)', 'f'), ('e', 'sub(b,f)'), ('add(a,e)', 'sub(b,f)'))


@pytest.mark.parametrize('tier', range(4))
def test_fixed_structural_laws(tier):
    rows = pool(tier, count=64)
    family = structural.family_for_tier(tier)
    d1, d2 = DENOMINATORS[tier]
    assert family.initial_lhs == f'add(div(mul(a,x),{d1}),div(b,{d2}))'
    assert family.initial_rhs == 'add(mul(d,x),c)'
    assert Counter(row['answer_mode_count'] for row in rows) == {5: 64}
    assert len({structural.identity('mathir', row) for row in rows}) == 64
    signs = {symbol: set() for symbol in 'abcdef'}
    for row in rows:
        bindings = json.loads(row['answer'])['bindings']
        assert structural.law_holds(tier, bindings)
        assert all(1 <= abs(value) <= 13 for value in bindings.values())
        first = bindings['a'] + bindings['e'] if tier in (1, 3) else bindings['e']
        second = bindings['b'] - bindings['f'] if tier in (2, 3) else bindings['f']
        assert structural.denominator_values(tier, bindings) == (first, second)
        assert first != 0 and second != 0 and bindings['a'] != bindings['d'] * first
        for symbol, value in bindings.items():
            signs[symbol].add(value > 0)
    assert all(values == {True, False} for values in signs.values())


@pytest.mark.parametrize('tier,updates', [
    (0, {'a': 6, 'd': 2, 'e': 3}),
    (1, {'a': 5, 'e': -5}), (3, {'a': 5, 'e': -5}),
    (2, {'b': 2, 'f': 2}), (3, {'b': 2, 'f': 2}),
    (1, {'a': -6, 'e': 3, 'd': 2}), (3, {'a': -6, 'e': 3, 'd': 2}),
    (2, {'a': 6, 'd': 2, 'e': 3}), (0, {'a': 0}),
    (1, {'c': 14}), (2, {'a': True}), (3, {'b': '11'}),
])
def test_nonzero_denominator_effective_coefficient_and_binding_guards(tier, updates):
    bindings = dict(a=7, b=11, c=13, d=-3, e=5, f=2)
    bindings.update(updates)
    assert not structural.law_holds(tier, bindings)


def test_anchor_matches_original_independent_nonzero_integer_proposal():
    candidate_rng, original_rng = random.Random(12345), random.Random(12345)
    values = [value for value in range(-13, 14) if value]
    accepted = rejected = 0
    for _ in range(1000):
        expected = {name: original_rng.choice(values) for name in 'abcdef'}
        assert structural._proposal(0, candidate_rng) == expected
        admissible = expected['a'] != expected['d'] * expected['e']
        assert structural.law_holds(0, expected) is admissible
        accepted += admissible
        rejected += not admissible
    assert accepted and rejected


@pytest.mark.parametrize('tier', range(4))
def test_quota_prefix_determinism_multiplier_and_exclusion(tier):
    one, three = pool(tier, count=1), pool(tier, count=3)
    assert one == pool(tier, count=1)
    assert one == sorted(three, key=lambda row: row['scale_cell_index'])[:1]
    assert three == pool(tier, count=1, multiplier=3)
    blocked = {structural.identity('mathir', row) for row in three}
    fresh = pool(tier, count=3, excluded=blocked, tag='FRESH_SPLIT')
    assert not blocked & {structural.identity('mathir', row) for row in fresh}
    assert not {row['problem'] for row in three} & {row['problem'] for row in fresh}
    assert pool(tier, count=0) == []


@pytest.mark.parametrize('tier', range(4))
def test_each_exhaustive_template_and_original_prompt_parser_and_witnesses(representatives, tier):
    row = representatives[tier]
    spec = json.loads(row['answer'])
    family = structural.family_for_tier(tier)
    assert spec['family'] == family.name
    assert spec['initial_lhs'] == family.initial_lhs and spec['initial_rhs'] == family.initial_rhs
    assert tuple(spec['actions']) == tuple('ABCDEF')
    assert Counter(spec['actions'].values()) == Counter(family.commands)
    assert spec['max_steps'] == 4 and spec['support_is_open'] is False
    assert row['problem'] == constraints.mathir_prompt(family, spec['bindings'], spec['actions'])
    keys, paths, digest = structural._certificate(tier)
    assert len(keys) == len(paths) == spec['num_completions'] == spec['valid_mode_count'] == 5
    assert spec['valid_mode_key_sha256'] == digest == CERTIFICATE_DIGESTS[tier]
    assert digest == hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()
    counts = [structural._expr_node_count(structural.parse_mathir_program(
        command, allowed_symbols='abcdefx', max_steps=1)[0].argument) for command in family.commands]
    assert max(counts) == (7, 7, 9, 9)[tier] and max(counts) <= 11
    report = structural.verify_rows('mathir', [row])
    assert report['rows_verified'] == 1 and report['witnesses_verified'] == 5
    assert report['exact_canonical_support'] and report['unchanged_original_prompt']
    assert report['original_verifier_witnesses'] and report['mathir_structural_profile']


@pytest.mark.parametrize('tier', range(4))
def test_five_witnesses_across_admissible_sign_and_numeric_patterns(representatives, tier):
    family = structural.family_for_tier(tier)
    patterns = [dict(a=7, b=11, c=13, d=-3, e=5, f=2),
                dict(a=-7, b=-11, c=-13, d=-3, e=-5, f=-2),
                dict(a=-7, b=11, c=-13, d=3, e=5, f=-2),
                dict(a=1, b=-2, c=3, d=-1, e=2, f=1)]
    rows = []
    for index, bindings in enumerate(patterns):
        assert structural.law_holds(tier, bindings)
        row = copy.deepcopy(representatives[tier])
        row['scale_cell_index'] = index
        commands = list(family.commands)
        commands = commands[index:] + commands[:index]
        actions = dict(zip('ABCDEF', commands))
        spec = structural._spec(bindings, actions, row['answer_mode_split'],
                                row['scale_generation_seed'], tier, index)
        row['answer'] = json.dumps(spec, sort_keys=True, separators=(',', ':'))
        row['problem'] = constraints.mathir_prompt(family, bindings, actions)
        rows.append(row)
    report = structural.verify_rows('mathir', rows)
    assert report['rows_verified'] == 4 and report['witnesses_verified'] == 20


def test_anchor_and_family_identity_are_truthful(representatives):
    assert structural.family_for_tier(0) == constraints._mathir_family(1)
    assert len({structural.family_for_tier(tier).name for tier in range(4)}) == 4
    for tier, row in representatives.items():
        spec = json.loads(row['answer'])
        assert structural.identity('mathir', row) == (
            'mathir', structural.family_for_tier(tier).name, tuple(sorted(spec['bindings'].items())))


@pytest.mark.parametrize('field,value', [
    ('problem', 'changed'), ('answer_mode_count', 4), ('answer_mode_count', 5.0),
    ('scale_candidate_tier', True), ('scale_candidate_tier', 4),
    ('scale_candidate_generator', 'wrong'), ('scale_candidate_profile', '{}'),
    ('scale_origin_metadata', '{}'), ('scale_structural_law_version', 'wrong'),
    ('scale_generation_seed', -1), ('scale_generation_seed', 1),
    ('scale_cell_index', -1), ('scale_cell_index', 10), ('answer_mode_split', ''),
    ('answer_mode_split', 'wrong'), ('mathir_family', 'wrong'), ('modebench_task', 'wrong')])
def test_corrupt_metadata_or_prompt_rejected(representatives, field, value):
    row = copy.deepcopy(representatives[1])
    row[field] = value
    with pytest.raises(RuntimeError, match='audit failed'):
        structural.verify_rows('mathir', [row])


@pytest.mark.parametrize('mutation', [
    'bindings', 'zero_denominator', 'bool_binding', 'family', 'lhs', 'rhs', 'actions',
    'action_ids', 'steps', 'open_support', 'source', 'instance_id', 'count', 'digest', 'extra_field'])
def test_self_consistent_prompt_cannot_escape_exact_spec(representatives, mutation):
    row = copy.deepcopy(representatives[1])
    spec = json.loads(row['answer'])
    if mutation == 'bindings':
        spec['bindings']['a'] = 14
    elif mutation == 'zero_denominator':
        spec['bindings']['a'] = 5
        spec['bindings']['e'] = -5
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
    row['problem'] = constraints.mathir_prompt(structural.family_for_tier(row['scale_candidate_tier']), spec['bindings'], spec['actions'])
    with pytest.raises(RuntimeError, match='audit failed'):
        structural.verify_rows('mathir', [row])


def test_duplicate_identity_is_rejected_even_after_menu_permutation(representatives):
    row = representatives[2]
    duplicate = copy.deepcopy(row)
    spec = json.loads(duplicate['answer'])
    commands = list(spec['actions'].values())
    spec['actions'] = dict(zip('ABCDEF', commands[1:] + commands[:1]))
    duplicate['answer'] = json.dumps(spec, sort_keys=True, separators=(',', ':'))
    duplicate['problem'] = constraints.mathir_prompt(structural.family_for_tier(row['scale_candidate_tier']), spec['bindings'], spec['actions'])
    assert duplicate['problem'] != row['problem']
    assert structural.identity('mathir', duplicate) == structural.identity('mathir', row)
    with pytest.raises(RuntimeError, match='duplicate semantic identity'):
        structural.verify_rows('mathir', [row, duplicate])


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
        structural.build_pool(**args)


def test_exhaustion_fails_without_a_partial_pool(monkeypatch):
    monkeypatch.setattr(structural, 'MAX_PROPOSALS_PER_ROW', 2)
    monkeypatch.setattr(structural, '_proposal', lambda tier, rng: dict.fromkeys('abcdef', 1))
    with pytest.raises(RuntimeError, match='exhausted fixed proposal budget'):
        pool(0)


def test_source_inventory_pins_original_generators_and_verifier():
    paths = structural.source_paths()
    assert Path(structural.__file__).resolve() in paths
    assert Path(structural.original.__file__).resolve() in paths
    assert Path(constraints.__file__).resolve() in paths
    assert ROOT / 'src/oat_drgrpo/mathir.py' in paths
    assert len(paths) == len(set(paths)) and all(path.is_file() for path in paths)


def test_arrow_roundtrip_preserves_four_profiles(representatives, tmp_path):
    from datasets import Dataset, DatasetDict, load_from_disk
    rows = [representatives[tier] for tier in range(4)]
    path = tmp_path / 'nonproduction-mathir'
    DatasetDict({'multi_answer': Dataset.from_list(rows)}).save_to_disk(str(path))
    assert [dict(row) for row in load_from_disk(str(path))['multi_answer']] == rows
