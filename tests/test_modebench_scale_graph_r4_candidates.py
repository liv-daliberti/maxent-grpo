"""Structural tests for the Level5 graph r4 per-cell candidate law.

The regression these guard is the screening pilot's provenance defect: rows that
recorded a profile computed from a module global at import time while generation
read a patched value, so every row was labelled one rung off from how it was
built. Here the recorded profile must agree with the row's own structure.
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

import modebench_scale_graph_r4_candidates as law  # noqa: E402

TARGET_DEV_HISTOGRAM = {4: 54, 6: 25, 8: 24, 9: 4, 12: 21}


def test_schema_and_domain_are_the_r4_per_cell_law():
    assert law.SCHEMA == 'modebench_scale_l5_graph_r4_per_cell_candidate_laws_v1'
    assert law.DOMAINS == ('graph_coloring',)


def test_every_registered_support_has_a_preset_in_every_tier():
    assert len(law.CELL_PRESETS) == 4
    for tier in range(4):
        assert set(law.CELL_PRESETS[tier]) == set(law.SUPPORTS)
        for vertices, hidden, ordering in law.CELL_PRESETS[tier].values():
            assert 0 < hidden < vertices
            assert ordering in ('known_first', 'random')


def test_borrowed_cells_follow_their_declared_neighbour():
    for tier in range(4):
        for cell, source in law.BORROWED.items():
            assert law.CELL_PRESETS[tier][cell] == law.CELL_PRESETS[tier][source]
            assert law._profile(tier, cell)['borrowed_from_cell'] == source
        for support in set(law.SUPPORTS) - set(law.BORROWED):
            assert law._profile(tier, support)['borrowed_from_cell'] is None


def test_the_law_is_graded_per_cell_not_per_tier():
    """At least one tier must use different presets across cells, or this is r2 again."""
    assert any(len({law.CELL_PRESETS[tier][s] for s in law.SUPPORTS}) > 1 for tier in range(4))


def test_tiers_are_distinct_assignments():
    seen = {tuple(sorted((s, law.CELL_PRESETS[tier][s]) for s in law.SUPPORTS)) for tier in range(4)}
    assert len(seen) == 4, 'a degenerate ladder of repeated rungs defeats tier mixing'


@pytest.mark.parametrize('tier', range(4))
def test_generated_rows_match_their_own_recorded_profile(tier):
    rows = law.build_pool('graph_coloring', {support: 2 for support in law.SUPPORTS},
                          set(), 4242, 'test', tier)
    assert rows
    for row in rows:
        spec = json.loads(row['answer'])
        profile = json.loads(row['scale_candidate_profile'])
        support = row['answer_mode_count']
        assert profile['answer_mode_count'] == support
        assert spec['n'] == profile['vertices']
        assert spec['partial_colors'].count(None) == profile['hidden_vertices']
        assert spec['num_completions'] == support
        assert len(spec['edges']) >= law.MINIMUM_EDGES
        if profile['vertex_order'] == 'known_first':
            known = spec['partial_colors'][:spec['n'] - profile['hidden_vertices']]
            assert all(colour is not None for colour in known)
        assert row['scale_candidate_tier'] == tier
        assert row['scale_candidate_generator'] == law.SCHEMA


def test_profile_is_recomputed_at_call_time_not_cached_at_import(monkeypatch):
    """The pilot defect: patching the table must not leave stale recorded profiles."""
    patched = dict(law.CELL_PRESETS[0])
    patched[4] = (7, 3, 'random')
    monkeypatch.setattr(law, 'CELL_PRESETS', (patched, *law.CELL_PRESETS[1:]))
    rows = law.build_pool('graph_coloring', {4: 2}, set(), 99, 'patched', 0)
    for row in rows:
        spec = json.loads(row['answer'])
        profile = json.loads(row['scale_candidate_profile'])
        assert spec['n'] == 7 == profile['vertices']
    law.verify_rows('graph_coloring', rows)


@pytest.mark.parametrize('tier', range(4))
def test_build_pool_respects_the_exact_support_histogram(tier):
    target = {4: 6, 6: 3, 12: 2}
    rows = law.build_pool('graph_coloring', target, set(), 77, 'hist', tier)
    counts = {}
    for row in rows:
        counts[row['answer_mode_count']] = counts.get(row['answer_mode_count'], 0) + 1
    assert counts == target
    identities = {law.identity('graph_coloring', row) for row in rows}
    assert len(identities) == len(rows)


def test_exclusions_are_honoured():
    first = law.build_pool('graph_coloring', {4: 4}, set(), 5150, 'excl', 0)
    blocked = {law.identity('graph_coloring', row) for row in first}
    second = law.build_pool('graph_coloring', {4: 4}, blocked, 5150, 'excl', 0)
    assert not blocked & {law.identity('graph_coloring', row) for row in second}


def test_verify_rows_rejects_a_tampered_profile():
    rows = law.build_pool('graph_coloring', {4: 2}, set(), 31337, 'tamper', 0)
    law.verify_rows('graph_coloring', rows)
    bad = dict(rows[0])
    bad['scale_candidate_profile'] = json.dumps({'vertices': 99}, sort_keys=True, separators=(',', ':'))
    with pytest.raises(RuntimeError):
        law.verify_rows('graph_coloring', [bad])


def test_verify_rows_rejects_a_row_whose_structure_left_its_cell():
    rows = law.build_pool('graph_coloring', {4: 2}, set(), 606, 'struct', 0)
    bad = dict(rows[0])
    spec = json.loads(bad['answer'])
    spec['n'] = spec['n'] + 1
    bad['answer'] = json.dumps(spec, sort_keys=True)
    with pytest.raises(RuntimeError):
        law.verify_rows('graph_coloring', [bad])


def test_build_pool_rejects_unregistered_supports_and_bad_arguments():
    with pytest.raises(ValueError):
        law.build_pool('pantry', {4: 1}, set(), 1, 'x', 0)
    with pytest.raises(ValueError):
        law.build_pool('graph_coloring', {4: 1}, set(), 1, 'x', 4)
    with pytest.raises(ValueError):
        law.build_pool('graph_coloring', {4: 1}, set(), 1, 'x', 0, joint_target={})
    with pytest.raises(ValueError):
        law.build_pool('graph_coloring', {7: 1}, set(), 1, 'x', 0)


def test_dev_histogram_cells_are_all_covered():
    assert set(TARGET_DEV_HISTOGRAM) <= set(law.SUPPORTS)
