"""Structural tests for the Level5 graph r7 density-graded law.

r7 exists because four knobs measured inert and twelve configurations collapsed
onto two levels set entirely by hidden-vertex count. The governing variable is
answer density, support / 3**hidden, which the registered histogram and the
integer hidden count together pin. r7 assigns the hidden count per cell so the
weighted mean density becomes tunable.

The invariants worth pinning are that the support histogram is untouched, that
the per-cell assignment comes from the declared target density rather than a
hand-written table, and that the catalog agrees with r5's independent
enumeration of the same abstract construction.
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
import modebench_scale_graph_r7_candidates as law  # noqa: E402

DEV_HISTOGRAM = {4: 54, 6: 25, 8: 24, 9: 4, 12: 21}
INTENDED_DENSITY = (0.1892, 0.1522, 0.1233, 0.0816)


def test_schema_and_targets():
    assert law.SCHEMA == 'modebench_scale_graph_r7_density_graded_laws_v1'
    assert law.TARGET_DENSITY == (0.21, 0.152, 0.123, 0.0816)
    assert law.HIDDEN_CHOICES == (3, 4)


def test_density_definition():
    assert law.cell_density(4, 3) == pytest.approx(4 / 27)
    assert law.cell_density(4, 4) == pytest.approx(4 / 81)
    assert law.cell_density(12, 4) == pytest.approx(12 / 81)


def test_assignment_comes_from_the_declared_target():
    """Each cell takes whichever hidden count is nearest its tier's target density."""
    for tier in range(4):
        for support in law.SUPPORTS:
            chosen = law.assign_hidden(tier, support)
            other = 4 if chosen == 3 else 3
            near = abs(law.cell_density(support, chosen) - law.TARGET_DENSITY[tier])
            far = abs(law.cell_density(support, other) - law.TARGET_DENSITY[tier])
            assert near <= far, (tier, support)


@pytest.mark.parametrize('tier', range(4))
def test_weighted_density_matches_the_intended_ladder(tier):
    assert law.weighted_density(tier, DEV_HISTOGRAM) == pytest.approx(INTENDED_DENSITY[tier], abs=1e-3)


def test_the_ladder_brackets_the_target_density():
    """The target pass@1 of 0.2078 implies a density near 0.150 at the measured ratio."""
    densities = [law.weighted_density(t, DEV_HISTOGRAM) for t in range(4)]
    assert min(densities) < 0.2078 / 1.388 < max(densities)
    assert densities == sorted(densities, reverse=True), 'density must fall monotonically by tier'


def test_tier_three_is_the_all_four_hidden_control():
    assert all(law.assign_hidden(3, s) == 4 for s in law.SUPPORTS)
    assert law.weighted_density(3, DEV_HISTOGRAM) == pytest.approx(
        sum(w * s / 81 for s, w in DEV_HISTOGRAM.items()) / sum(DEV_HISTOGRAM.values()))


def test_at_least_one_tier_is_genuinely_mixed():
    """A tier that renders every cell at one hidden count cannot tune density."""
    mixed = [t for t in range(4) if len({law.assign_hidden(t, s) for s in law.SUPPORTS}) > 1]
    assert len(mixed) >= 3


def test_catalog_agrees_with_the_r5_enumeration():
    """Same abstract construction, independently enumerated; a drift check, not a coupling."""
    for hidden in law.HIDDEN_CHOICES:
        mine = {s: len(v) for s, v in law.catalog(hidden).items()}
        theirs = {s: len(v) for s, v in r5.catalog(hidden).items()}
        assert mine == theirs, hidden


def test_catalog_covers_every_support_at_both_hidden_counts():
    for hidden in law.HIDDEN_CHOICES:
        catalog = law.catalog(hidden)
        assert set(catalog) == set(law.SUPPORTS)
        for support, options in catalog.items():
            assert options, (hidden, support)
            edge_mask, domains = options[0]
            assert law.completion_count(hidden, edge_mask, domains) == support


@pytest.mark.parametrize('tier', range(4))
def test_rows_match_their_cell_assignment(tier):
    rows = law.build_pool('graph_coloring', {s: 2 for s in law.SUPPORTS}, set(), 303, 'test', tier)
    assert rows
    law.verify_structure(rows)
    for row in rows:
        support = row['answer_mode_count']
        hidden = law.assign_hidden(tier, support)
        spec = json.loads(row['answer'])
        profile = json.loads(row['scale_candidate_profile'])
        assert spec['n'] == law.SHOWN + hidden == profile['vertices']
        assert spec['partial_colors'].count(None) == hidden == profile['hidden_vertices']
        assert profile['cell_answer_density'] == pytest.approx(law.cell_density(support, hidden))
        assert profile['target_answer_density'] == law.TARGET_DENSITY[tier]
        assert spec['num_completions'] == support


@pytest.mark.parametrize('tier', range(4))
def test_the_support_histogram_is_untouched(tier):
    """The whole point: density moves, the registered composition does not."""
    rows = law.build_pool('graph_coloring', DEV_HISTOGRAM, set(), 404, 'hist', tier)
    counts = {}
    for row in rows:
        counts[row['answer_mode_count']] = counts.get(row['answer_mode_count'], 0) + 1
    assert counts == DEV_HISTOGRAM
    assert len({law.identity('graph_coloring', r) for r in rows}) == len(rows)


def test_a_pool_may_mix_vertex_counts():
    """Cells at different hidden counts coexist in one pool; that is the mechanism."""
    rows = law.build_pool('graph_coloring', {4: 3, 12: 3}, set(), 505, 'mixed', 1)
    sizes = {json.loads(r['answer'])['n'] for r in rows}
    assert sizes == {6, 7}


def test_exclusions_are_honoured():
    first = law.build_pool('graph_coloring', {4: 4}, set(), 606, 'excl', 1)
    blocked = {law.identity('graph_coloring', r) for r in first}
    second = law.build_pool('graph_coloring', {4: 4}, blocked, 606, 'excl', 1)
    assert not blocked & {law.identity('graph_coloring', r) for r in second}


def test_verify_structure_rejects_a_row_from_another_tier():
    """Tier 3 renders support 4 on seven vertices, tier 1 on six."""
    rows = law.build_pool('graph_coloring', {4: 2}, set(), 707, 'cross', 3)
    bad = dict(rows[0], scale_candidate_tier=1)
    with pytest.raises((RuntimeError, ValueError)):
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
    with pytest.raises(ValueError):
        law.catalog(5)
