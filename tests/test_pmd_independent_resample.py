"""The resampling exists to replace eleven overlapping streams with thirty-two.

These tests hold that property directly, in both directions: the original seed
policy must still reproduce the eleven-stream overlap the audit found, and the
independent policy must yield fully disjoint blocks. A regression that quietly
returned the two to agreement would invalidate the whole correction while every
other check kept passing.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'ops', ROOT / 'ops' / 'resample'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from resample_terminal_mode_coverage import draw_seed_plan  # noqa: E402
from draw_tail import tail_records  # noqa: E402

CELL = {
    'domain': 'countdown',
    'eval_config': {'eval_mode_coverage_seed': 610200, 'eval_mode_coverage_k': 8,
                    'eval_mode_coverage_draws': 4},
}
ROWS = [{'problem': f'problem {i}'} for i in range(5)]


def test_original_policy_reproduces_the_eleven_stream_overlap():
    plan = draw_seed_plan(CELL, ROWS, 'reproduce')
    assert plan['request_seeds'][0] == [610200, 610201, 610202, 610203]
    # Four draws of eight consecutive children span s..s+10, not 32 streams.
    assert plan['distinct_child_streams_per_prompt'] == 11
    multiplicity = {}
    for block in plan['child_seeds'][0]:
        for child in block:
            multiplicity[child] = multiplicity.get(child, 0) + 1
    counts = [multiplicity[seed] for seed in sorted(multiplicity)]
    assert counts == [1, 2, 3, 4, 4, 4, 4, 4, 3, 2, 1]


def test_independent_policy_gives_disjoint_blocks():
    plan = draw_seed_plan(CELL, ROWS, 'independent')
    assert plan['distinct_child_streams_per_prompt'] == 32
    assert plan['distinct_child_seeds'] == plan['total_child_seeds']
    assert plan['total_child_seeds'] == len(ROWS) * 4 * 8


def test_independent_blocks_are_aligned_and_prompt_specific():
    plan = draw_seed_plan(CELL, ROWS, 'independent')
    for row in plan['request_seeds']:
        for seed in row:
            assert seed % 8 == 0, 'request seeds must start an aligned eight-block'
    first, second = plan['request_seeds'][0], plan['request_seeds'][1]
    assert not set(first) & set(second), 'different prompts must not share a block'


def test_independent_policy_ignores_the_original_seed_base():
    shifted = {**CELL, 'eval_config': {**CELL['eval_config'],
                                       'eval_mode_coverage_seed': 999999}}
    assert (draw_seed_plan(CELL, ROWS, 'independent')['request_seeds']
            == draw_seed_plan(shifted, ROWS, 'independent')['request_seeds'])


def test_original_policy_tracks_the_recorded_seed_base():
    shifted = {**CELL, 'eval_config': {**CELL['eval_config'],
                                       'eval_mode_coverage_seed': 76299}}
    assert draw_seed_plan(shifted, ROWS, 'reproduce')['request_seeds'][0] == [
        76299, 76300, 76301, 76302]


def test_tail_reader_drops_the_truncated_first_line(tmp_path):
    path = tmp_path / 'draws.jsonl'
    lines = [json.dumps({'n': i, 'pad': 'x' * 4096}) for i in range(40)]
    path.write_text('\n'.join(lines) + '\n')
    records = tail_records(path, tail_bytes=20_000)
    assert records, 'a bounded tail must still return whole records'
    assert records[-1]['n'] == 39
    assert all('n' in record for record in records)


def test_tail_reader_reads_a_short_file_whole(tmp_path):
    path = tmp_path / 'draws.jsonl'
    path.write_text('\n'.join(json.dumps({'n': i}) for i in range(3)) + '\n')
    assert [r['n'] for r in tail_records(path)] == [0, 1, 2]


@pytest.mark.parametrize('mode', ('reproduce', 'independent'))
def test_plan_shape_matches_the_prompt_and_draw_grid(mode):
    plan = draw_seed_plan(CELL, ROWS, mode)
    assert len(plan['request_seeds']) == len(ROWS)
    assert all(len(row) == 4 for row in plan['request_seeds'])
    assert all(len(block) == 8 for row in plan['child_seeds'] for block in row)


# A cross-level cell reads a harder split than the run trained on, so the split
# can carry prompts the run's own surface never admitted. Training drops those
# rows (learner/run.py raises rather than keep one), and the replay has to make
# the same choice or it either crashes the cell or silently widens the context
# the run was evaluated under. These pin the rule, not the implementation.

def admit(lengths: list[int], budget: int) -> list[int]:
    """Indices training would keep: rows at or under the prompt budget."""
    return [i for i, n in enumerate(lengths) if n <= budget]


def test_prompts_within_the_budget_are_all_kept():
    assert admit([10, 200, 256], 256) == [0, 1, 2]


def test_the_boundary_prompt_is_admitted_not_dropped():
    # prompt_max_length is a maximum, not an exclusive bound: a prompt of
    # exactly the budget trained, so it must still be measured.
    assert admit([256], 256) == [0]
    assert admit([257], 256) == []


def test_over_budget_prompts_are_dropped_before_the_engine_sees_them():
    # The Pantry budget is 640 while Level 5 renders prompts past 700. Dropping
    # is what keeps the cell measurable; passing them on is what crashed it.
    lengths = [600, 641, 712, 640]
    assert admit(lengths, 640) == [0, 3]


def test_the_rule_depends_on_the_budget_alone_not_on_the_arm():
    # prompt_max_length is a property of the domain, so every arm in a domain
    # drops the same prompts and the cells stay comparable. Two arms that see
    # the same lengths must therefore admit the same indices.
    lengths = [300, 700, 500]
    assert admit(lengths, 640) == admit(lengths, 640) == [0, 2]
