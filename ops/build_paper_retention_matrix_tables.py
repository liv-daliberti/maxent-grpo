#!/usr/bin/env python3
"""Table bodies for the retention comparator matrix.

The two panels used to be one plate. At the width the main body allows, a
6x11 grid of two-decimal numbers was unreadable, so the same record is
rendered as two tables instead: one across model scales, one across the
alternatives at Qwen2.5-0.5B. Nothing is recomputed here --- every number comes
from ``paper/figures/experiment1_retention_comparator_matrix.json``, which the
composite builder writes, so the tables and the machine-readable record cannot
drift apart.

Twelve numeric columns leave no room for a paired interval beside every mean,
so the intervals stay in the record and the numerical appendix, and the space
buys precision instead: cells print three decimals rather than the plate's two. A cell resting on fewer than five paired seeds
prints that count as a superscript; a cell where too few prompts return a
verified pair on both sides prints an em dash.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'paper/figures/experiment1_retention_comparator_matrix.json'
PANEL_A_OUT = ROOT / 'paper/results/retention_matrix_panel_a_table_body.tex'
PANEL_B_OUT = ROOT / 'paper/results/retention_matrix_panel_b_table_body.tex'

SCHEMA = 'paper-experiment1-retention-comparator-matrix-v4'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
AVERAGE = 'average'
COLUMNS = DOMAINS + (AVERAGE,)
METRICS = (('pass8', r'$\Delta$ \texttt{pass@8}'), ('pmd', r'$\Delta$ \pmd{}'))

PANEL_A_ROWS = (
    ('Qwen2.5-0.5B', r'\qwenmark{}2.5-0.5B'),
    ('Falcon3-1B', r'\falconmark{}3-1B'),
    ('Qwen2.5-3B', r'\qwenmark{}2.5-3B'),
)
#: Our two arms lead, the untrained reference sits on its own between rules
#: because it is the starting point rather than a competitor, and the other
#: objectives follow. ``RULE`` draws the separator.
RULE = object()
PANEL_B_ROWS = (
    ('replay_drgrpo', r'Re:Dr \textit{(ours)}'),
    ('replay_maxrl', r'Re:Max \textit{(ours)}'),
    RULE,
    ('before_training', 'Before training'),
    RULE,
    ('maxrl', 'MaxRL (no replay)'),
    ('ucpo', 'UCPO'),
    ('rlep_dr', 'RLEP-Dr'),
    ('semantic_maxent', 'Semantic-MaxEnt'),
    ('grpo', 'GRPO'),
)
#: The untrained reference is a starting point, not a competitor, so it is
#: excluded when marking the best alternative in a column.
PANEL_B_BEST_ROWS = tuple(entry[0] for entry in PANEL_B_ROWS
                          if entry is not RULE and entry[0] != 'before_training')

GAP = r'\textemdash{}'
#: Red through orange and yellow to green, as light tints so black digits stay
#: legible in print. Zero is pinned to the orange knot, so the red arm covers
#: the losses and the yellow-green arm the gains; reading a sign off the colour
#: alone is never necessary, since every cell also prints its number.
RAMP = ((0.00, (0xF0, 0xA5, 0x8F)),
        (0.33, (0xF7, 0xCC, 0x9C)),
        (0.66, (0xF3, 0xEB, 0xA6)),
        (1.00, (0xA9, 0xD6, 0xA0)))
ZERO_KNOT = 0.33


def _fixed(value: float) -> str:
    """Signed three decimals with the leading zero dropped, as elsewhere.

    A value that rounds to zero prints unsigned: a displayed ``-.000`` reads as
    a direction the measurement does not support.
    """
    if abs(value) < 5e-4:
        return '.000'
    return f'{value:+.3f}'.replace('0.', '.', 1)


def _ramp(position: float) -> str:
    position = min(max(position, 0.0), 1.0)
    for (low, left), (high, right) in zip(RAMP, RAMP[1:]):
        if position <= high:
            weight = 0.0 if high == low else (position - low) / (high - low)
            channels = (round(a + (b - a) * weight) for a, b in zip(left, right))
            return ''.join(f'{value:02X}' for value in channels)
    return ''.join(f'{value:02X}' for value in RAMP[-1][1])


def _shade(value: float, low: float, high: float) -> str:
    """Colour for one effect, with zero pinned to the ramp's orange knot."""
    if value > 0:
        span = high if high > 0 else 1.0
        position = ZERO_KNOT + (1.0 - ZERO_KNOT) * (value / span)
    elif value < 0 and low < 0:
        position = ZERO_KNOT * (1.0 - value / low)
    else:
        position = ZERO_KNOT
    return rf'\cellcolor[HTML]{{{_ramp(position)}}}'


def _cell(summary: dict, *, best: bool, bounds: tuple[float, float]) -> str:
    if summary['mean'] is None:
        return GAP
    mean = float(summary['mean'])
    body = _fixed(mean)
    if best:
        body = rf'\mathbf{{{body}}}'
    if summary['n'] < 5:
        body += rf'^{{{summary["n"]}}}'
    return f'{_shade(mean, *bounds)}${body}$'


def _summary(cells: dict, row: str, column: str, metric: str) -> dict:
    return cells[row][column]['summaries'][metric]


def _rows(cells: dict, rows: tuple, best_rows: tuple) -> list[str]:
    drawn = tuple(row for row in rows if row is not RULE)
    best_value = {}
    # Each metric's shading spans only what that metric reaches in this table,
    # so a .006 breadth gain is not washed out by a .979 correctness gain.
    bounds = {}
    for metric, _ in METRICS:
        present = [float(_summary(cells, row, column, metric)['mean'])
                   for row, _ in drawn for column in COLUMNS
                   if _summary(cells, row, column, metric)['mean'] is not None]
        bounds[metric] = (min(present, default=0.0), max(present, default=0.0))
        for column in COLUMNS:
            values = [float(_summary(cells, row, column, metric)['mean'])
                      for row in best_rows
                      if _summary(cells, row, column, metric)['mean'] is not None]
            best_value[(metric, column)] = max(values) if values else None
    lines = []
    for entry in rows:
        if entry is RULE:
            lines.append(r'    \midrule')
            continue
        row, label = entry
        pieces = [label]
        for metric, _ in METRICS:
            for column in COLUMNS:
                summary = _summary(cells, row, column, metric)
                mean = summary['mean']
                reference = best_value[(metric, column)]
                best = (row in best_rows and mean is not None
                        and reference is not None
                        and abs(float(mean) - reference) < 1e-12)
                pieces.append(_cell(summary, best=best, bounds=bounds[metric]))
        lines.append('    ' + ' & '.join(pieces) + r' \\')
    return lines


def _body(cells: dict, rows: tuple, best_rows: tuple = ()) -> str:
    # The two metrics sit side by side as column groups, so the caller owns the
    # header and this file opens with an ordinary cell: \input's file hooks
    # insert tokens at both ends, and a \multicolumn or a \bottomrule left in
    # the caller would land inside a cell the hook had already started.
    out = _rows(cells, rows, best_rows)
    out.append(r'    \bottomrule')
    return '\n'.join(out) + '\n'


def main() -> None:
    record = json.loads(RECORD.read_text(encoding='utf-8'))
    if record.get('schema') != SCHEMA:
        raise RuntimeError(f'retention matrix record schema drifted: {record.get("schema")}')
    PANEL_A_OUT.write_text(
        '% Generated by build_paper_retention_matrix_tables.py; do not hand edit.\n'
        + _body(record['panel_a']['cells'], PANEL_A_ROWS), encoding='utf-8')
    PANEL_B_OUT.write_text(
        '% Generated by build_paper_retention_matrix_tables.py; do not hand edit.\n'
        + _body(record['panel_b']['cells'], PANEL_B_ROWS, PANEL_B_BEST_ROWS),
        encoding='utf-8')
    print(json.dumps({'event': 'built',
                      'panel_a': str(PANEL_A_OUT.relative_to(ROOT)),
                      'panel_b': str(PANEL_B_OUT.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
