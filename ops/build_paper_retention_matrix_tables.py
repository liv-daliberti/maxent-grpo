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
import statistics
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
#: Our two arms lead, followed by trained alternatives. The untrained
#: checkpoint is a separate reference below all methods. ``RULE`` draws each
#: separator.
RULE = object()
PANEL_B_ROWS = (
    ('replay_drgrpo', r'Re:Dr \textit{(ours)}'),
    ('replay_maxrl', r'Re:Max \textit{(ours)}'),
    RULE,
    ('maxrl', 'MaxRL (no replay)'),
    ('gapo', 'GAPO'),
    ('setpo', 'SetPO'),
    ('ucpo', 'UCPO'),
    ('rlep_dr', 'RLEP-Dr'),
    ('semantic_maxent', 'Semantic-MaxEnt'),
    ('grpo', 'GRPO'),
    RULE,
    ('before_training', 'Untrained checkpoint'),
)
#: Only the trained alternatives are sorted by their mean \pmd{} effect;
#: our two arms and the checkpoint reference retain their fixed positions.
ORDER_BY = 'pmd'


def _ordered_panel_b(cells: dict) -> tuple:
    """Sort trained alternatives by mean effect, leaving references separate.

    A row whose average is undefined sorts last; it has no claim to a rank.
    """
    groups = [[]]
    for entry in PANEL_B_ROWS:
        if entry is RULE:
            groups.append([])
        else:
            groups[-1].append(entry)
    ours, alternatives, references = groups

    def rank(entry):
        summary = cells.get(entry[0], {}).get(AVERAGE, {}).get('summaries', {})
        mean = summary.get(ORDER_BY, {}).get('mean')
        return (mean is None, -(mean or 0.0))

    return (tuple(ours) + (RULE,) + tuple(sorted(alternatives, key=rank))
            + (RULE,) + tuple(references))
#: The untrained reference is a starting point, not a competitor, so it is
#: excluded when marking the best alternative in a column.
PANEL_B_BEST_ROWS = tuple(entry[0] for entry in PANEL_B_ROWS
                          if entry is not RULE and entry[0] != 'before_training')

GAP = r'\textemdash{}'
#: Red for losses, yellow at no change, green for gains, as light tints so
#: black digits stay legible in print. Zero sits at the middle knot and the two
#: arms are scaled alike, so equal gains and losses read equally strongly and a
#: negative number can never come out green.
RAMP = ((0.00, (0xF0, 0xA5, 0x8F)),
        (0.50, (0xF8, 0xF0, 0xA6)),
        (1.00, (0xA9, 0xD6, 0xA0)))
ZERO_KNOT = 0.50
#: Smallest extent a column's ramp may cover. Five points is the smallest
#: effect the results describe as real, so a column whose entries are all
#: noise stays pale rather than being stretched to the full ramp: without this
#: floor the +.027 that tops \pmd{} MathIR would print the same green as the
#: +.591 that tops \pmd{} PantryPlan.
MIN_SPAN = 0.05


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


def _shade(value: float, extent: float) -> str:
    """Colour for one effect against its column's extent, zero at the middle.

    The extent is the largest magnitude in the column, so both arms use one
    scale: +.3 and -.3 are equally saturated, and the sign is what picks green
    or red.
    """
    position = ZERO_KNOT + ZERO_KNOT * (value / extent if extent else 0.0)
    return rf'\cellcolor[HTML]{{{_ramp(position)}}}'


def _cell(summary: dict, *, best: bool, extent: float) -> str:
    if summary['mean'] is None:
        return GAP
    mean = float(summary['mean'])
    body = _fixed(mean)
    if best:
        body = rf'\mathbf{{{body}}}'
    if summary['n'] < 5:
        # A bare superscript digit runs into the value at \scriptsize: the
        # Python cell printed ``.000^{2}`` and read as the number .0002, and
        # ``+.032^{4}`` as +.0324. The thin space keeps the seed count legible
        # as a marker rather than a fourth decimal.
        body += rf'^{{\,{summary["n"]}}}'
    return f'{_shade(mean, extent)}${body}$'


def _common_basis(cells: dict, row: str, metric: str) -> tuple[str, ...]:
    """Domains of ``row`` that every one of its seeds defines for ``metric``.

    The stored ``average`` cell is a within-seed mean over whatever domains
    that seed happens to define, so its basis moves from seed to seed: the
    Falcon \pmd{} row averaged three domains on seeds 55--57, four on seed 58,
    and dropped seed 59 entirely. That is the cross-domain mean over a shifting
    population the text declines to take, printed as though it were the equally
    weighted five-domain average the caption describes. Restricting to the
    domains the row's fullest seed set defines makes one fixed basis per row,
    which is comparable down a column and reproducible from the printed cells.
    """
    seeds = {domain: frozenset(cells[row][domain]['summaries'][metric]['per_seed'])
             for domain in DOMAINS}
    full = max(seeds.values(), key=len, default=frozenset())
    return tuple(domain for domain in DOMAINS if seeds[domain] == full and full)


def _basis_average(cells: dict, row: str, metric: str) -> dict:
    """Equal-weight mean over the common basis, within seed then across seeds."""
    basis = _common_basis(cells, row, metric)
    if not basis:
        return {'mean': None, 'n': 0, 'basis': ()}
    per_seed = {}
    for seed in cells[row][basis[0]]['summaries'][metric]['per_seed']:
        values = [float(cells[row][domain]['summaries'][metric]['per_seed'][seed])
                  for domain in basis]
        per_seed[seed] = statistics.fmean(values)
    return {'mean': statistics.fmean(per_seed.values()), 'n': len(per_seed),
            'basis': basis}


#: Only \pmd{} needs the recomputed average. Every domain defines
#: ``pass8``, so the stored average there is already one fixed five-domain
#: basis, taken over the seeds all five share -- Falcon's is n=4 because
#: Countdown is, not because a domain drops out. \pmd{} is where domains
#: vanish per seed, so it is the column whose basis has to be pinned.
BASIS_METRICS = ('pmd',)


def _summary(cells: dict, row: str, column: str, metric: str) -> dict:
    if column == AVERAGE and metric in BASIS_METRICS:
        return _basis_average(cells, row, metric)
    return cells[row][column]['summaries'][metric]


def _rows(cells: dict, rows: tuple, best_rows: tuple) -> list[str]:
    drawn = tuple(row for row in rows if row is not RULE)
    best_value = {}
    # Shading is scaled within a column, which is the axis the table is read
    # down: every row is one method on the same domain and metric, so a column
    # is the comparison, and the bold mark already picks a column's best.
    # Scaling within a row instead made the colour answer nothing a reader
    # asks -- a row's own maximum always went full green, so GRPO's +.213 on
    # Graph and Re:Dr's +.646 on Graph printed the identical shade.
    # A table-wide ramp is the other failure: domains differ in size by an
    # order of magnitude, and Graph's half-point effects flatten \pmd{} MathIR
    # to one shade. Per column keeps both the sign and the within-domain
    # ordering readable; comparing shades across columns is what the printed
    # numbers are for.
    extents = {}
    for metric, _ in METRICS:
        for column in COLUMNS:
            present = [abs(float(_summary(cells, row, column, metric)['mean']))
                       for row, _ in drawn
                       if _summary(cells, row, column, metric)['mean'] is not None]
            extents[(metric, column)] = max(*present, MIN_SPAN) if present else MIN_SPAN
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
                pieces.append(_cell(summary, best=best,
                                    extent=extents[(metric, column)]))
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
        + _body(record['panel_b']['cells'],
                _ordered_panel_b(record['panel_b']['cells']),
                PANEL_B_BEST_ROWS),
        encoding='utf-8')
    print(json.dumps({'event': 'built',
                      'panel_a': str(PANEL_A_OUT.relative_to(ROOT)),
                      'panel_b': str(PANEL_B_OUT.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
