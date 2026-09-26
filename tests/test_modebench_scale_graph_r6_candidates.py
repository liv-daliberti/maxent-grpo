"""Structural tests for the Level5 graph r6 structural-grading law.

r6 keeps r5's construction, which is what produces near-target heterogeneity
gaps, and grades difficulty by two statistics of the skeleton conditioned on
support: how many hidden vertices are pinned to a single colour, and how many
hidden-hidden edges there are. Neither changes the support, so the registered
histogram is untouched.

The invariants worth pinning are that a forced vertex really is free of the
support, that the declared fallback is applied and recorded wherever a cell is
unpopulated, and that tier 3 still reproduces r5's four-free-vertex control.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

import modebench_scale_graph_r5_candidates as r5  # noqa: E402
import modebench_scale_graph_r6_candidates as law  # noqa: E402


def test_schema_and_tier_table():
    assert law.SCHEMA == 'modebench_scale_graph_r6_structural_grading_laws_v1'
    assert law.TIERS == ((2, 'any'), (1, 'low'), (1, 'high'), (0, 'any'))
    assert law.HIDDEN == 4


def test_a_forced_vertex_contributes_a_factor_of_one_to_the_support():
    """This is why the knob is free of the registered histogram."""
    catalog = law.catalog()
    for support in law.SUPPORTS:
        for edge_mask, domains in catalog[support][:200]:
            assert law.completion_count(edge_mask, domains) == support
            assert law.forced_count(domains) == sum(1 for d in domains if d.bit_count() == 1)


def test_the_knob_varies_at_constant_support():
    """The design property: the same support is reachable at several forced counts.

    This is what lets difficulty move without touching the registered histogram.
    Supports 4 and 6 carry 79 of the 128 dev rows between them.
    """
    catalog = law.catalog()
    for support in (4, 6):
        available = {law.forced_count(domains) for _, domains in catalog[support]}
        assert {0, 1, 2} <= available, (support, available)
    for support in law.SUPPORTS:
        available = {law.forced_count(domains) for _, domains in catalog[support]}
        assert {0, 1} <= available, (support, available)


def test_a_forced_vertex_costs_a_digit_without_adding_an_answer():
    """Every row has four hidden digits regardless of how many are forced."""
    for tier in range(4):
        rows = law.build_pool('graph_coloring', {4: 2}, set(), 4004, 'digits', tier)
        for row in rows:
            spec = json.loads(row['answer'])
            assert spec['partial_colors'].count(None) == law.HIDDEN
            assert spec['num_completions'] == row['answer_mode_count']


def test_catalog_covers_every_registered_support():
    catalog = law.catalog()
    assert set(catalog) == set(law.SUPPORTS)
    assert all(catalog[s] for s in law.SUPPORTS)


def test_tier_three_reproduces_the_r5_control():
    """Tier 3 is r5 tier 2: four hidden vertices, none forced, measured 0.0993 / 0.2734."""
    assert law.TIERS[3] == (0, 'any')
    assert r5.TIERS[2][0] == law.HIDDEN


@pytest.mark.parametrize('tier', range(4))
def test_resolution_is_declared_and_recorded(tier):
    for support in law.SUPPORTS:
        options, used = law.resolve(tier, support)
        assert options, (tier, support)
        profile = law._profile(tier, support)
        assert profile['realized_forced_hidden_vertices'] == used['forced']
        assert profile['realized_hidden_edge_band'] == used['band']
        assert profile['fallback_applied'] == used['relaxed']
        assert profile['requested_forced_hidden_vertices'] == law.TIERS[tier][0]
        assert profile['requested_hidden_edge_band'] == law.TIERS[tier][1]
        if used['relaxed'] is None:
            assert used['forced'] == law.TIERS[tier][0] and used['band'] == law.TIERS[tier][1]


def test_the_clean_tiers_need_no_fallback_anywhere():
    """Tiers 1 and 3 carry the level-finding weight, so they must be undiluted."""
    for tier in (1, 3):
        for support in law.SUPPORTS:
            _, used = law.resolve(tier, support)
            assert used['relaxed'] is None, (tier, support)


def test_every_resolved_option_matches_its_recorded_statistics():
    for tier in range(4):
        for support in law.SUPPORTS:
            options, used = law.resolve(tier, support)
            low, high = law.BANDS[used['band']]
            for edge_mask, domains in options:
                assert law.forced_count(domains) == used['forced']
                assert low <= edge_mask.bit_count() <= high
                assert law.completion_count(edge_mask, domains) == support


@pytest.mark.parametrize('tier', range(4))
def test_rows_generate_verify_and_match_their_profile(tier):
    rows = law.build_pool('graph_coloring', {s: 2 for s in law.SUPPORTS}, set(), 606, 'test', tier)
    assert rows
    law.verify_structure(rows)
    for row in rows:
        spec = json.loads(row['answer'])
        profile = json.loads(row['scale_candidate_profile'])
        edge_mask, domains, support = law.structural_signature(spec)
        assert support == row['answer_mode_count'] == profile['answer_mode_count']
        assert spec['n'] == law.SHOWN + law.HIDDEN
        assert spec['partial_colors'].count(None) == law.HIDDEN
        assert law.forced_count(domains) == profile['realized_forced_hidden_vertices']
        low, high = law.BANDS[profile['realized_hidden_edge_band']]
        assert low <= edge_mask.bit_count() <= high


@pytest.mark.parametrize('tier', range(4))
def test_support_histogram_and_identity_uniqueness(tier):
    target = {4: 6, 6: 3, 12: 2}
    rows = law.build_pool('graph_coloring', target, set(), 77, 'hist', tier)
    counts = {}
    for row in rows:
        counts[row['answer_mode_count']] = counts.get(row['answer_mode_count'], 0) + 1
    assert counts == target
    assert len({law.identity('graph_coloring', r) for r in rows}) == len(rows)


def test_exclusions_are_honoured():
    first = law.build_pool('graph_coloring', {4: 4}, set(), 909, 'excl', 1)
    blocked = {law.identity('graph_coloring', r) for r in first}
    second = law.build_pool('graph_coloring', {4: 4}, blocked, 909, 'excl', 1)
    assert not blocked & {law.identity('graph_coloring', r) for r in second}


def test_verify_structure_rejects_a_row_from_another_tier():
    """A tier-3 row has no forced vertex, so it cannot pass a tier-1 audit."""
    rows = law.build_pool('graph_coloring', {4: 2}, set(), 31, 'cross', 3)
    bad = dict(rows[0], scale_candidate_tier=1)
    with pytest.raises(RuntimeError):
        law.verify_structure([bad])


def test_build_pool_rejects_bad_arguments():
    with pytest.raises(ValueError):
        law.build_pool('pantry', {4: 1}, set(), 1, 'x', 0)
    with pytest.raises(ValueError):
        law.build_pool('graph_coloring', {4: 1}, set(), 1, 'x', 4)
    with pytest.raises(ValueError):
        law.build_pool('graph_coloring', {7: 1}, set(), 1, 'x', 0)
    with pytest.raises(ValueError):
        law.build_pool('graph_coloring', {4: 1}, set(), 1, 'x', 0, joint_target={})
