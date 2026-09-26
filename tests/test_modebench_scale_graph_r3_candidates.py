"""Pure structural scratch tests: no registered pools, models, or graders."""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ops/exp_scaling'))
import modebench_scale_graph_r3_candidates as candidate


def brute_count(spec):
    hidden = [i for i, c in enumerate(spec['partial_colors']) if c is None]
    count = 0
    for values in product((1, 2, 3), repeat=len(hidden)):
        colors = list(spec['partial_colors'])
        for i, value in zip(hidden, values):
            colors[i] = value
        count += all(colors[a-1] != colors[b-1] for a, b in spec['edges'])
    return count


def key(spec):
    return ('graph_coloring', 6, tuple(map(tuple, spec['edges'])),
            ''.join('?' if c is None else str(c) for c in spec['partial_colors']))


def test_catalog_exhausts_all_structures_and_exact_supports_independently():
    expected = defaultdict(set)
    pairs = list(combinations(range(3), 2))
    for mask in range(8):
        for domains in product(range(1, 8), repeat=3):
            edges = [[4+a, 4+b] for bit, (a, b) in enumerate(pairs) if mask & (1 << bit)]
            edges += [[color, 4+h] for h, domain in enumerate(domains) for color in (1, 2, 3)
                      if not domain & (1 << (color-1))]
            spec = {'n': 6, 'edges': sorted(edges), 'partial_colors': [1, 2, 3, None, None, None]}
            support = brute_count(spec)
            if support in candidate.SUPPORTS:
                expected[support].add((mask, domains))
                actual = candidate.instantiate(mask, domains, list(range(1, 7)))
                assert actual == spec
                assert candidate.structural_signature(actual) == (mask, domains, support)
    assert {s: set(v) for s, v in candidate.catalog().items()} == dict(expected)
    assert {s: len(v) for s, v in expected.items()} == {4: 387, 5: 36, 6: 208, 8: 108, 9: 27, 12: 57, 18: 12}


def test_high_supports_remain_sparse_and_all_four_tiers_have_capacity():
    for tier in range(4):
        for support in candidate.SUPPORTS:
            options, weights = candidate.proposal_profiles(tier, support)
            assert options and all(type(w) is int and w > 0 for w in weights)
            if support in (9, 18):
                assert all(mask.bit_count() <= 1 for mask, domains in options)
            if support == 5:
                assert all(mask.bit_count() == 2 for mask, domains in options)


@pytest.mark.parametrize('support', [5, 9, 18])
def test_stated_tier_invariant_distributions_are_exact(support):
    reference = None
    for tier in range(4):
        options, weights = candidate.proposal_profiles(tier, support)
        distribution = tuple(Fraction(w, sum(weights)) for w in weights)
        reference = reference or distribution
        assert distribution == reference


@pytest.mark.parametrize('support', [4, 6, 8, 12])
def test_only_intrinsic_path_weight_increases(support):
    masses = []
    for tier, multiplier in enumerate(candidate.PATH_MULTIPLIERS):
        options, weights = candidate.proposal_profiles(tier, support)
        assert all(w == (multiplier if mask.bit_count() == 2 else 1)
                   for (mask, domains), w in zip(options, weights))
        masses.append(Fraction(sum(w for (mask, domains), w in zip(options, weights) if mask.bit_count() == 2), sum(weights)))
    assert masses == sorted(set(masses))


def test_integer_tickets_realize_exact_weighted_catalog(monkeypatch):
    class Ticket:
        def __init__(self, value):
            self.value = value
        def randrange(self, maximum):
            assert self.value < maximum
            return self.value
        def shuffle(self, values):
            assert values == list(range(1, 7))
    tier, support = 2, 12
    options, weights = candidate.proposal_profiles(tier, support)
    seen = Counter()
    for ticket in range(sum(weights)):
        spec = candidate._proposal(tier, support, Ticket(ticket))
        mask, domains, observed = candidate.structural_signature(spec)
        assert observed == support
        seen[(mask, domains)] += 1
    assert seen == dict(zip(options, weights))


def test_uniform_relabeling_has_constant_sixfold_preimages_for_support18():
    identities = Counter()
    for mask, domains in candidate.catalog()[18]:
        for permutation in permutations(range(1, 7)):
            identities[key(candidate.instantiate(mask, domains, permutation))] += 1
    assert len(identities) == 1440
    assert set(identities.values()) == {6}


@pytest.fixture(scope='module')
def scratch_pools():
    target = Counter({4: 62, 5: 4, 6: 28, 8: 24, 9: 6, 12: 21, 18: 1})
    blocked, result = set(), {}
    for tier in range(4):
        rows = candidate.build_pool('graph_coloring', target, blocked, 193001, 'SCRATCH_STRUCTURAL_ONLY', tier)
        ids = {candidate.identity('graph_coloring', row) for row in rows}
        assert len(ids) == len(rows) and not ids & blocked
        blocked |= ids
        result[tier] = rows
    return target, blocked, result


@pytest.mark.parametrize('tier', range(4))
def test_full_support_quota_native_prompt_and_pure_completion_count(scratch_pools, tier):
    target, blocked, pools = scratch_pools
    rows = pools[tier]
    assert Counter(row['answer_mode_count'] for row in rows) == target
    assert candidate.verify_structure(rows) == {'graph_r3_structural_profile': True, 'rows': 146}
    for row in rows:
        spec = json.loads(row['answer'])
        assert brute_count(spec) == row['answer_mode_count']
        assert row['problem'] == candidate.original._graph_prompt(spec['n'], spec['edges'], spec['partial_colors'])
        assert 'exactly 3 digits' in row['problem']
        assert key(spec) == candidate.identity('graph_coloring', row)


def test_future_split_fixtures_exclude_every_prior_tier_without_new_production_files(scratch_pools):
    target, blocked, pools = scratch_pools
    blocked = set(blocked)
    for split in ('train', 'eval'):
        for tier in range(4):
            rows = candidate.build_pool('graph_coloring', Counter({s: 3 for s in target}), blocked,
                                        193002, 'SCRATCH_'+split, tier)
            ids = {candidate.identity('graph_coloring', row) for row in rows}
            assert len(ids) == len(rows) and not ids & blocked
            blocked |= ids


@pytest.mark.parametrize('tier', range(4))
def test_determinism_quota_independence_and_exclusion(tier):
    args = dict(domain='graph_coloring', excluded=set(), seed=193003, tag='SCRATCH_PREFIX', tier=tier)
    one = candidate.build_pool(target=Counter({4: 1}), **args)
    two = candidate.build_pool(target=Counter({4: 2}), **args)
    assert one == sorted(two, key=lambda r: r['scale_cell_index'])[:1]
    assert two == candidate.build_pool(target=Counter({4: 2}), **args)
    wider = candidate.build_pool(target=Counter({4: 2, 12: 2}), **args)
    assert [r for r in wider if r['answer_mode_count'] == 4] == two
    args['excluded'] = {candidate.identity('graph_coloring', r) for r in two}
    fresh = candidate.build_pool(target=Counter({4: 2}), **args)
    assert not args['excluded'] & {candidate.identity('graph_coloring', r) for r in fresh}


def test_multiplier_only_changes_quota():
    args = dict(domain='graph_coloring', excluded=set(), seed=193004, tag='SCRATCH_MULTIPLIER', tier=0)
    assert candidate.build_pool(target=Counter({5: 2}), **args) == candidate.build_pool(target=Counter({5: 1}), multiplier=2, **args)


def test_fixed_exhaustion_never_changes_law_or_reuses_excluded_identity(monkeypatch):
    spec = candidate.instantiate(*candidate.catalog()[5][0], list(range(1, 7)))
    monkeypatch.setattr(candidate, '_proposal', lambda *args: deepcopy(spec))
    monkeypatch.setattr(candidate, 'MAX_PROPOSALS_PER_ROW', 2)
    with pytest.raises(RuntimeError, match='exhausted fixed proposal budget'):
        candidate.build_pool('graph_coloring', Counter({5: 1}), {key(spec)}, 193005, 'SCRATCH_EXHAUSTION', 0)


@pytest.mark.parametrize('field,value', [('answer_mode_count', 99), ('problem', 'changed'),
    ('scale_candidate_profile', '{}'), ('scale_graph_topology_law_version', 'changed'), ('scale_origin_metadata', 'changed')])
def test_structural_audit_rejects_profile_prompt_support_drift(scratch_pools, field, value):
    row = deepcopy(scratch_pools[2][0][0])
    row[field] = value
    with pytest.raises(RuntimeError, match='changed'):
        candidate.verify_structure([row])


@pytest.mark.parametrize('kind', ['known_edge', 'repeat_known_color', 'four_hidden', 'duplicate_edge', 'bad_endpoint', 'boolean_color'])
def test_structural_law_rejects_wrong_graphs(kind):
    spec = candidate.instantiate(*candidate.catalog()[4][0], list(range(1, 7)))
    if kind == 'known_edge':
        spec['edges'] = sorted(spec['edges']+[[1, 2]])
    elif kind == 'repeat_known_color':
        spec['partial_colors'][1] = 1
    elif kind == 'four_hidden':
        spec['partial_colors'][0] = None
    elif kind == 'duplicate_edge':
        spec['edges'] = sorted(spec['edges']+[spec['edges'][0]])
    elif kind == 'bad_endpoint':
        spec['edges'] = [[0, 4]]
    else:
        spec['partial_colors'][0] = True
    with pytest.raises(ValueError):
        candidate.structural_signature(spec)


@pytest.mark.parametrize('kwargs', [{'tier': True}, {'tier': 4}, {'seed': -1}, {'seed': False},
    {'multiplier': 0}, {'target': {7: 1}}, {'target': {4: True}}, {'target': {4: -1}},
    {'joint_target': {}}, {'domain': 'python_factors'}])
def test_invalid_fixed_inputs_fail(kwargs):
    values = dict(domain='graph_coloring', target=Counter({4: 1}), excluded=set(), seed=193006,
                  tag='SCRATCH_INVALID', tier=0)
    values.update(kwargs)
    with pytest.raises(ValueError):
        candidate.build_pool(**values)


def test_eventual_native_qualification_is_delegated_but_not_executed_in_tests(scratch_pools, monkeypatch):
    calls = []
    def sentinel(domain, rows):
        calls.append((domain, rows))
        return {'native_gate_delegated': True}
    monkeypatch.setattr(candidate.original, 'verify_rows', sentinel)
    rows = scratch_pools[2][0][:1]
    result = candidate.verify_rows('graph_coloring', rows)
    assert result['native_gate_delegated'] and result['graph_r3_structural_profile']
    assert calls == [('graph_coloring', rows)]
