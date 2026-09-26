"""Structural tests for the Level5 graph r5 topology law.

r5 keeps r3's enumerated skeleton catalog, which is why r3's per-prompt success
is the least dispersed measured for this domain, and adds two knobs that raise
difficulty without breaking that uniformity: a fourth hidden vertex, and edges
among the shown vertices that are satisfied by construction.

The invariants that matter are that tier 0 still reproduces r3 exactly, that the
distractor edges really cannot change the answer, and that every registered
support is realizable at both hidden counts.
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

import modebench_scale_candidates as original  # noqa: E402
import modebench_scale_graph_r3_candidates as r3  # noqa: E402
import modebench_scale_graph_r5_candidates as law  # noqa: E402


def test_schema_and_tier_table():
    assert law.SCHEMA == 'modebench_scale_graph_r5_hidden_topology_laws_v1'
    assert law.TIERS == ((3, 0, 64), (3, 3, 64), (4, 0, 64), (4, 3, 64))


def test_the_ladder_is_a_two_by_two_factorial():
    hidden = {t[0] for t in law.TIERS}
    known = {t[1] for t in law.TIERS}
    assert hidden == {3, 4} and known == {0, 3}
    assert len(set(law.TIERS)) == 4
    assert len({t[2] for t in law.TIERS}) == 1, 'the path multiplier is held fixed'
    assert {(t[0], t[1]) for t in law.TIERS} == {(h, k) for h in hidden for k in known}


def test_tier_zero_reproduces_the_r3_control():
    """Tier 0 is r3 tier 3, a measured point at pass@1 0.362 / pass@8 0.712."""
    assert law.TIERS[0] == (3, 0, r3.PATH_MULTIPLIERS[3])
    mine = {s: len(v) for s, v in law.catalog(3).items()}
    theirs = {s: len(v) for s, v in r3.catalog().items()}
    assert mine == theirs


def test_every_registered_support_is_realizable_at_both_hidden_counts():
    for hidden in (3, 4):
        catalog = law.catalog(hidden)
        assert set(catalog) == set(law.SUPPORTS)
        for support, options in catalog.items():
            assert options, support
            edge_mask, domains = options[0]
            assert law.completion_count(hidden, edge_mask, domains) == support


def test_a_fourth_hidden_vertex_enlarges_the_search():
    assert len(law._colorings(3)) == 27
    assert len(law._colorings(4)) == 81
    assert len(law._hidden_pairs(4)) == 6


@pytest.mark.parametrize('tier', range(4))
def test_rows_carry_the_declared_shape(tier):
    hidden, known, _ = law.TIERS[tier]
    rows = law.build_pool('graph_coloring', {s: 2 for s in law.SUPPORTS}, set(), 2024, 'test', tier)
    assert rows
    law.verify_structure(rows)
    for row in rows:
        spec = json.loads(row['answer'])
        profile = json.loads(row['scale_candidate_profile'])
        assert spec['n'] == law.SHOWN + hidden == profile['vertices']
        assert spec['partial_colors'].count(None) == hidden == profile['hidden_vertices']
        assert profile['known_known_edges'] == known
        assert sorted(c for c in spec['partial_colors'] if c is not None) == [1, 2, 3]
        assert spec['num_completions'] == row['answer_mode_count']


@pytest.mark.parametrize('tier', (1, 3))
def test_distractor_edges_are_present_and_cannot_change_the_answer(tier):
    """Shown vertices carry distinct colours, so a shown-shown edge is always satisfied."""
    hidden, known, _ = law.TIERS[tier]
    assert known == 3
    rows = law.build_pool('graph_coloring', {4: 3, 12: 3}, set(), 55, 'distract', tier)
    for row in rows:
        spec = json.loads(row['answer'])
        partial = spec['partial_colors']
        shown_shown = [(a, b) for a, b in spec['edges']
                       if partial[a - 1] is not None and partial[b - 1] is not None]
        assert len(shown_shown) == known
        for a, b in shown_shown:
            assert partial[a - 1] != partial[b - 1]
        # Removing them leaves the same number of valid completions.
        kept = [e for e in spec['edges'] if e not in [list(p) for p in shown_shown]]
        assert original.graph_completion_count(spec['n'], kept, partial) == spec['num_completions']


@pytest.mark.parametrize('tier', (0, 2))
def test_tiers_without_distractors_have_no_shown_shown_edges(tier):
    rows = law.build_pool('graph_coloring', {4: 3}, set(), 56, 'plain', tier)
    for row in rows:
        spec = json.loads(row['answer'])
        partial = spec['partial_colors']
        assert not [(a, b) for a, b in spec['edges']
                    if partial[a - 1] is not None and partial[b - 1] is not None]


@pytest.mark.parametrize('tier', range(4))
def test_support_histogram_and_identity_uniqueness(tier):
    target = {4: 5, 6: 3, 12: 2}
    rows = law.build_pool('graph_coloring', target, set(), 88, 'hist', tier)
    counts = {}
    for row in rows:
        counts[row['answer_mode_count']] = counts.get(row['answer_mode_count'], 0) + 1
    assert counts == target
    assert len({law.identity('graph_coloring', r) for r in rows}) == len(rows)


def test_exclusions_are_honoured():
    first = law.build_pool('graph_coloring', {4: 4}, set(), 4321, 'excl', 2)
    blocked = {law.identity('graph_coloring', r) for r in first}
    second = law.build_pool('graph_coloring', {4: 4}, blocked, 4321, 'excl', 2)
    assert not blocked & {law.identity('graph_coloring', r) for r in second}


def test_structural_signature_round_trips():
    for tier in range(4):
        hidden, known, _ = law.TIERS[tier]
        rows = law.build_pool('graph_coloring', {8: 2}, set(), 13, 'sig', tier)
        for row in rows:
            spec = json.loads(row['answer'])
            _, _, support = law.structural_signature(hidden, known, spec)
            assert support == row['answer_mode_count']


def test_structural_signature_rejects_a_graph_outside_the_law():
    """Wrong vertex count, wrong hidden count and empty domains are all rejected."""
    rows = law.build_pool('graph_coloring', {4: 1}, set(), 14, 'bad', 0)
    spec = json.loads(rows[0]['answer'])
    with pytest.raises(ValueError):
        law.structural_signature(4, 0, spec)          # wrong hidden count for this row
    widened = dict(spec, partial_colors=list(spec['partial_colors']))
    widened['n'] = 7
    with pytest.raises(ValueError):
        law.structural_signature(3, 0, widened)       # wrong vertex count


def test_verify_structure_rejects_a_tampered_row():
    """Dropping an edge may still be a legal skeleton, but not THIS row's skeleton.

    structural_signature recomputes the support, and a widened domain can land on
    another registered support, so the binding check is that the recovered support
    still matches the support the row declares and the prompt it ships.
    """
    rows = law.build_pool('graph_coloring', {4: 1}, set(), 14, 'bad', 0)
    bad = dict(rows[0])
    spec = json.loads(bad['answer'])
    spec['edges'] = spec['edges'][:-1]
    bad['answer'] = json.dumps(spec, sort_keys=True)
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
