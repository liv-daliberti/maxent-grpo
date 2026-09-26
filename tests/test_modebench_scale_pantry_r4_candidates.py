"""Structural tests for the Level5 pantry r4 re-cut ladder.

r4 exists because the r3 pilot measured a cliff: pass@1 fell from 0.0742 at tier 1
to 0.0048 at tier 2 across a single rung that moved all four knobs at once, with
the 0.0586 target inside that gap, and tiers 2 and 3 came back dead at dead
fractions 0.90 and 0.92. r4 keeps tiers 0 and 1 at exactly their measured r3
settings and replaces the two dead rungs with milder steps.

These tests pin the re-cut itself, so the file cannot drift away from r3 in any
respect other than the tier table.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

import modebench_scale_pantry_r3_candidates as r3  # noqa: E402
import modebench_scale_pantry_r4_candidates as r4  # noqa: E402

R3_PATH = ROOT / 'ops/exp_scaling/modebench_scale_pantry_r3_candidates.py'
R4_PATH = ROOT / 'ops/exp_scaling/modebench_scale_pantry_r4_candidates.py'
CONSTANTS = ('MENU_SIZES', 'HEADROOM', 'FORBIDDEN_PROBABILITY', 'TIGHT_INTERVALS')


def test_schema_is_distinct_from_r3():
    assert r4.SCHEMA == 'modebench_scale_l5_pantry_r4_candidate_laws_v1'
    assert r4.SCHEMA != r3.SCHEMA


def test_r4_differs_from_r3_only_in_the_tier_table():
    """Everything outside the docstring, schema and four tier constants is identical."""
    def normalise(text):
        text = re.sub(r'^""".*?"""', '"""DOC"""', text, count=1, flags=re.S)
        text = text.replace('_r3_', '_rN_').replace('_r4_', '_rN_')
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if any(stripped.startswith(name + ' =') for name in CONSTANTS):
                continue
            lines.append(line)
        return '\n'.join(lines)
    assert normalise(R4_PATH.read_text()) == normalise(R3_PATH.read_text())


def test_tiers_zero_and_one_reproduce_their_measured_r3_settings():
    for tier in (0, 1):
        assert r4.MENU_SIZES[tier] == r3.MENU_SIZES[tier]
        assert r4.HEADROOM[tier] == r3.HEADROOM[tier]
        assert r4.FORBIDDEN_PROBABILITY[tier] == r3.FORBIDDEN_PROBABILITY[tier]
        assert r4.TIGHT_INTERVALS[tier] == r3.TIGHT_INTERVALS[tier]
        assert r4.AVAILABLE_GRAMS[tier] == r3.AVAILABLE_GRAMS[tier]


def test_the_dead_rungs_are_gone():
    """Tiers 2 and 3 of r3 were dead; r4 must not simply repeat them."""
    for tier in (2, 3):
        assert (r4.MENU_SIZES[tier], r4.HEADROOM[tier], r4.FORBIDDEN_PROBABILITY[tier],
                r4.TIGHT_INTERVALS[tier]) != (
            r3.MENU_SIZES[tier], r3.HEADROOM[tier], r3.FORBIDDEN_PROBABILITY[tier],
            r3.TIGHT_INTERVALS[tier])


def test_only_headroom_and_menu_size_move_below_tier_one():
    """The knobs the cliff was attributed to are held fixed across tiers 1..3."""
    assert r4.TIGHT_INTERVALS == (False, False, False, False)
    assert len(set(r4.FORBIDDEN_PROBABILITY[1:])) == 1


def test_the_ladder_is_not_degenerate():
    rungs = {(r4.MENU_SIZES[t], r4.HEADROOM[t], r4.FORBIDDEN_PROBABILITY[t], r4.TIGHT_INTERVALS[t])
             for t in range(4)}
    assert len(rungs) == 4


def test_availability_stays_aligned_to_the_step():
    for tier, available in enumerate(r4.AVAILABLE_GRAMS):
        assert available == tuple(r4.STEP_G * (r4.MINIMUM_STEPS + slack) for slack in r4.HEADROOM[tier])
        for grams in available:
            assert grams % r4.STEP_G == 0
            assert grams >= r4.STEP_G * r4.MINIMUM_STEPS


def test_menu_size_never_exceeds_the_smallest_family_pool():
    assert max(r4.MENU_SIZES) <= 8


@pytest.mark.parametrize('tier', range(4))
def test_rows_generate_and_verify(tier):
    target = {8: 2, 13: 2}
    rows = r4.build_pool('pantry', target, set(), 909, 'test', tier,
                         joint_target={(8, 'low_sodium_pantry_meal'): 2,
                                       (13, 'plant_protein_bowl'): 2})
    assert rows
    r4.verify_rows('pantry', rows)
    for row in rows:
        profile = json.loads(row['scale_candidate_profile'])
        assert profile['menu_size'] == r4.MENU_SIZES[tier]
        assert profile['available_g_choices'] == list(r4.AVAILABLE_GRAMS[tier])
        assert profile['interval_width_law'] == 'original_tier0'
        assert row['scale_candidate_tier'] == tier
