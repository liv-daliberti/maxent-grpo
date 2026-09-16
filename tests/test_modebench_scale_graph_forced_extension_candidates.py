"""Scratch graph construction and pure native syntax/counting; no graders/models."""
from collections import Counter
from copy import deepcopy
from itertools import combinations, product
import json
from pathlib import Path
import random
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ops/exp_scaling'))
import modebench_scale_graph_forced_extension_candidates as c


def all_choices(extra):
    choices = [[pair] for pair in combinations((1, 2, 3), 2)]
    for _ in range(extra-1):
        grown = []
        for values in choices:
            color = 6-sum(values[0])
            for anchor in values[1:]:
                color = 6-color-anchor
            grown += [values+[anchor] for anchor in (1, 2, 3) if anchor != color]
        choices = grown
    return choices


def completions(spec):
    hidden = [i for i, color in enumerate(spec['partial_colors']) if color is None]
    result = set()
    for values in product((1, 2, 3), repeat=len(hidden)):
        colors = list(spec['partial_colors'])
        for index, value in zip(hidden, values):
            colors[index] = value
        if all(colors[a-1] != colors[b-1] for a, b in spec['edges']):
            result.add(tuple(colors))
    return result


@pytest.mark.parametrize('tier', range(4))
def test_every_abstract_base_retains_every_registered_support(tier):
    extra = c.EXTRA_VERTICES[tier]
    choices = all_choices(extra)[-1]
    for support, options in c.base.catalog().items():
        for mask, domains in options:
            spec, witness = c.extend(mask, domains, extra, choices, list(range(1, 7+extra)))
            assert spec['n'] == 7+tier
            assert c.original.graph_completion_count(spec['n'], spec['edges'], spec['partial_colors']) == support
            assert c.projection_support(c.induced(spec, range(1, 7))) == support
    assert set(c.base.catalog()) == {4, 5, 6, 8, 9, 12, 18}


@pytest.mark.parametrize('support', c.SUPPORTS)
def test_every_chain_attachment_has_an_exact_unique_extension_bijection(support):
    mask, domains = c.base.catalog()[support][0]
    base_spec = c.base.instantiate(mask, domains, list(range(1, 7)))
    base_solutions = completions(base_spec)
    for extra in range(1, 5):
        for choices in all_choices(extra):
            permutation = list(range(1, 7+extra))
            random.Random(812+extra).shuffle(permutation)
            spec, _ = c.extend(mask, domains, extra, choices, permutation)
            full = completions(spec)
            restrictions = Counter(tuple(colors[v-1] for v in permutation[:6]) for colors in full)
            assert set(restrictions) == base_solutions
            assert set(restrictions.values()) == {1}
            assert len(full) == support


@pytest.mark.parametrize('tier', range(4))
def test_common_development_informed_base_and_all_profiles_actually_changed(tier):
    assert c.BASE_TIER == 1 and c.base.PATH_MULTIPLIERS[c.BASE_TIER] == 4
    p = c.PROFILES['graph_coloring'][tier]
    assert p['extra_vertices'] == tier+1 and p['vertices'] == tier+7
    assert p['base_choice'] == 'development_informed_previously_selected_failed_r3_tier1'
    assert p['difficulty_ordering'] == 'unmeasured_no_monotonicity_claim'


def test_exact_ticket_and_attachment_law_before_rejection():
    options, weights = c.base.proposal_profiles(1, 18)
    class Draws:
        def __init__(self, ticket, first, second):
            self.values = iter((ticket, first, second))
        def randrange(self, maximum):
            value = next(self.values)
            assert 0 <= value < maximum
            return value
        def shuffle(self, values):
            assert values == list(range(1, 9))
    observed = Counter()
    for ticket in range(sum(weights)):
        for first in range(3):
            for second in range(2):
                spec, witness = c._proposal(1, 18, Draws(ticket, first, second))
                observed[(witness['base_edge_mask'], tuple(witness['base_domains']),
                    tuple(witness['attachment_anchor_colors'][0]), witness['attachment_anchor_colors'][1])] += 1
    expected = {}
    for (mask, domains), weight in zip(options, weights):
        for choices in all_choices(2):
            expected[(mask, domains, tuple(choices[0]), choices[1])] = weight
    assert observed == expected


@pytest.fixture(scope='module')
def rows_by_tier():
    blocked, rows = set(), {}
    for tier in range(4):
        got = c.build_pool('graph_coloring', Counter({s: 3 for s in c.SUPPORTS}), blocked,
            491001, 'SCRATCH_ONLY', tier)
        rows[tier] = got
        blocked.update(c.identity('graph_coloring', row) for row in got)
    return rows


@pytest.mark.parametrize('tier', range(4))
def test_native_prompt_parser_identity_and_structural_verification(rows_by_tier, tier, monkeypatch):
    from oat_drgrpo import math_grader
    # Trap actual grader entrypoints; only the original variable-length parser is used.
    for name in ('_verify_graph_coloring_colors', '_verify_graph_coloring_answer'):
        monkeypatch.setattr(math_grader, name, lambda *a, **k: pytest.fail('no native grader allowed'))
    rows = rows_by_tier[tier]
    assert c.verify_structure(rows) == {'graph_forced_extension_structural_profile': True, 'rows': 21}
    assert Counter(r['answer_mode_count'] for r in rows) == {s: 3 for s in c.SUPPORTS}
    for row in rows:
        spec = json.loads(row['answer'])
        assert row['problem'] == c.original._graph_prompt(spec['n'], spec['edges'], spec['partial_colors'])
        assert f'exactly {4+tier} digits' in row['problem']
        assert c.graph_key(spec) == c.identity('graph_coloring', row)
        answer = next(iter(completions(spec)))
        missing = ''.join(str(v) for v, old in zip(answer, spec['partial_colors']) if old is None)
        assert math_grader._graph_coloring_from_candidate(r'\boxed{'+missing+'}', spec) == list(answer)


def test_history_longer_graphs_and_same_base_different_extras_are_excluded():
    mask, domains = c.base.catalog()[18][0]
    known_base = c.base.instantiate(mask, domains, list(range(1, 7)))
    for extra in range(1, 5):
        extended, _ = c.extend(mask, domains, extra, all_choices(extra)[0], list(range(1, 7+extra)))
        assert c.graph_key(known_base) in c.base_projections(c.graph_key(extended))
    # A historical graph can have an extra shown anchor of a repeated color.
    historical = deepcopy(extended)
    historical['n'] += 1
    historical['partial_colors'].append(1)
    assert c.graph_key(known_base) in c.base_projections(c.graph_key(historical))
    assert c.graph_key(known_base) in c.blocked_projections({c.graph_key(historical)})


def test_alternate_undeclared_projection_blocks_candidate(monkeypatch):
    # The native full identity contains no witness, so alternate induced bases
    # must be rejected even when the declared original base itself is fresh.
    found = None
    rng = random.Random(498)
    for _ in range(100):
        spec, witness = c._proposal(3, 4, rng)
        keys = c.base_projections(c.graph_key(spec))
        selected = c.graph_key(c.induced(spec, witness['slot_to_vertex'][:6]))
        if keys-{selected}:
            found = spec, witness, next(iter(keys-{selected}))
            break
    assert found is not None
    spec, witness, alternate = found
    monkeypatch.setattr(c, '_proposal', lambda *args: deepcopy((spec, witness)))
    monkeypatch.setattr(c, 'MAX_PROPOSALS_PER_ROW', 2)
    with pytest.raises(RuntimeError, match='exhausted fixed proposal budget'):
        c.build_pool('graph_coloring', {4: 1}, {alternate}, 491002, 'SCRATCH_REJECT', 3)


def test_cross_tier_and_cross_split_projection_freshness(rows_by_tier):
    full, projections = set(), set()
    for rows in rows_by_tier.values():
        for row in rows:
            key = c.identity('graph_coloring', row)
            new = c.base_projections(key)
            assert key not in full and not new & projections
            full.add(key)
            projections.update(new)
    for split in ('train', 'eval'):
        for tier in range(4):
            rows = c.build_pool('graph_coloring', {s: 2 for s in c.SUPPORTS}, full, 491003,
                'SCRATCH_'+split, tier)
            for row in rows:
                key = c.identity('graph_coloring', row)
                new = c.base_projections(key)
                assert key not in full and not new & projections
                full.add(key)
                projections.update(new)


@pytest.mark.parametrize('tier', range(4))
def test_reproducibility_and_multiplier(tier):
    args = dict(domain='graph_coloring', excluded=set(), seed=491004, tag='SCRATCH', tier=tier)
    first = c.build_pool(target={9: 2}, **args)
    assert first == c.build_pool(target={9: 2}, **args)
    assert first == c.build_pool(target={9: 1}, multiplier=2, **args)
    args['excluded'] = {c.identity('graph_coloring', r) for r in first}
    fresh = c.build_pool(target={9: 2}, **args)
    assert not c.blocked_projections(args['excluded']) & c.blocked_projections({c.identity('graph_coloring', r) for r in fresh})


@pytest.mark.parametrize('field,value', [('answer_mode_count', 99), ('problem', 'changed'),
    ('scale_candidate_profile', '{}'), ('scale_graph_forced_extension_law_version', 'changed')])
def test_tampered_rows_rejected(rows_by_tier, field, value):
    row = deepcopy(rows_by_tier[0][0])
    row[field] = value
    with pytest.raises(RuntimeError, match='changed'):
        c.verify_structure([row])


def test_tampered_witness_cannot_certify_different_graph(rows_by_tier):
    row = deepcopy(rows_by_tier[3][0])
    witness = json.loads(row['scale_origin_metadata'])
    witness['slot_to_vertex'][0], witness['slot_to_vertex'][1] = witness['slot_to_vertex'][1], witness['slot_to_vertex'][0]
    row['scale_origin_metadata'] = json.dumps(witness)
    with pytest.raises(RuntimeError, match='changed'):
        c.verify_structure([row])


@pytest.mark.parametrize('kwargs', [{'tier': True}, {'tier': 4}, {'seed': False}, {'seed': -1},
    {'multiplier': 0}, {'target': {7: 1}}, {'target': {4: True}}, {'target': {4: -1}},
    {'joint_target': {}}, {'domain': 'python_factors'}])
def test_invalid_native_inputs(kwargs):
    args = dict(domain='graph_coloring', target={4: 1}, excluded=set(), seed=1, tag='SCRATCH', tier=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        c.build_pool(**args)


def test_native_gate_delegated_only_to_scratch_sentinel(rows_by_tier, monkeypatch):
    called = []
    monkeypatch.setattr(c.original, 'verify_rows', lambda domain, rows: called.append((domain, rows)) or {'native_stub': True})
    rows = rows_by_tier[0][:1]
    assert c.verify_rows('graph_coloring', rows)['native_stub']
    assert called == [('graph_coloring', rows)]
