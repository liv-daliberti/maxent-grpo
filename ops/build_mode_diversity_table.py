#!/usr/bin/env python3
"""Emit the appendix table body pairing the registered endpoints with PMD.

One row per domain and level, one column group per model scale. Each group
reports \\texttt{pass@8} and \\pmd{} so a reader can see directly that the two
do not track each other, and so the registered endpoint stays visible next to
the statistic that carries the breadth axis. Cells without enough verified
pairs print an em dash rather than a number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / 'paper/results/mode_diversity_base_grid.json'
OUT = ROOT / 'paper/results/mode_diversity_base_grid_table_body.tex'
COVERAGE = ROOT / 'paper/results/mode_diversity_coverage.tex'

MODELS = ('05b', '3b', '7b', '14b')
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry')
DOMAIN_LABELS = {'graph_coloring': 'Graph', 'countdown': 'Countdown',
                 'python_factors': 'Python', 'mathir': 'MathIR', 'pantry': 'Pantry'}
LEVELS = ('level1', 'level2', 'level3')


def fmt(value: float | None, *, reportable: bool = True) -> str:
    if value is None or not reportable:
        return '---'
    text = f'{value:.3f}'
    return text[1:] if text.startswith('0.') else text


def build(payload: dict) -> str:
    index = {(c['model_label'], c['level'], c['domain']): c for c in payload['cells']}
    lines = []
    for domain in DOMAINS:
        for position, level in enumerate(LEVELS):
            label = DOMAIN_LABELS[domain] if position == 0 else ''
            cells = [f'{label}', f'{position + 1}']
            for model in MODELS:
                cell = index.get((model, level, domain))
                if cell is None:
                    cells += ['---', '---']
                    continue
                cells.append(fmt(cell['pass8']))
                cells.append(fmt(cell['pmd'], reportable=cell['reportable']))
            lines.append(' & '.join(cells) + r' \\')
        if domain != DOMAINS[-1]:
            lines.append(r'\addlinespace')
    # The rule closes the body file rather than the manuscript: a body ending
    # in a bare row separator leaves \bottomrule stranded outside the \cr.
    lines.append(r'    \bottomrule')
    return '\n'.join(lines) + '\n'


def build_coverage(payload: dict) -> str:
    """Macros for the support counts, so prose cannot go stale during collection.

    The grid grows as cells finish, and any count written by hand in the
    manuscript is wrong by the next rebuild. These are regenerated with the
    payload and expanded at compile time.
    """
    from collections import Counter
    cells = payload['cells']
    reportable = [c for c in cells if c['reportable']]
    gaps = Counter(c['domain'] for c in cells if not c['reportable'])
    top, top_n = gaps.most_common(1)[0] if gaps else ('none', 0)
    labels = {'graph_coloring': 'Graph', 'countdown': 'Countdown',
              'python_factors': 'Python', 'mathir': 'MathIR', 'pantry': 'Pantry'}
    def corr(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** .5
        return num / den

    def fmt(v):
        # LaTeX-friendly leading-point form, e.g. .924 and -.185.
        text = f'{v:.3f}'
        return text.replace('0.', '.', 1) if text.startswith(('0.', '-0.')) else text

    # distinct@8 needs no support gate, so it uses every cell; PMD uses the
    # cells where it is defined.
    distinct = corr([c['pass8'] for c in cells], [c['distinct8'] for c in cells])
    pmd = corr([c['pass8'] for c in reportable], [c['pmd'] for c in reportable])
    # How far hardening a task moves each axis, holding model, domain and the
    # notion of mode fixed: the median absolute step between consecutive levels
    # of the same cell. Reported for the family the main-body figure plots.
    import statistics
    import evaluate_modebench_base_grid as grid
    family = {m for m, f in grid.MODEL_FAMILY.items() if f == 'Qwen2.5'}
    indexed = {(c['model_label'], c['domain'], c['level']): c for c in reportable}
    ladder = sorted({c['level'] for c in cells})
    pass_steps, pmd_steps = [], []
    for (model, domain, level), cell in indexed.items():
        if model not in family:
            continue
        position = ladder.index(level)
        if position + 1 >= len(ladder):
            continue
        nxt = indexed.get((model, domain, ladder[position + 1]))
        if nxt is None:
            continue
        pass_steps.append(abs(nxt['pass8'] - cell['pass8']))
        pmd_steps.append(abs(nxt['pmd'] - cell['pmd']))
    level_pass = statistics.median(pass_steps) if pass_steps else float('nan')
    level_pmd = statistics.median(pmd_steps) if pmd_steps else float('nan')

    # The hosted deployments answer nearly every prompt at every level, so their
    # PMD decline with level cannot be a side effect of falling success.
    frontier_path = ROOT / 'paper/results/mode_diversity_frontier_points.json'
    front_first = front_last = front_pass = float('nan')
    if frontier_path.is_file():
        fcells = [c for c in json.loads(frontier_path.read_text())['cells']
                  if c.get('reportable')]
        flevels = sorted({c['level'] for c in fcells})
        if flevels:
            def mean_at(level, field):
                vals = [c[field] for c in fcells if c['level'] == level]
                return statistics.mean(vals) if vals else float('nan')
            front_first = mean_at(flevels[0], 'pmd')
            front_last = mean_at(flevels[-1], 'pmd')
            front_pass = statistics.mean(c['pass8'] for c in fcells)

    lines = ['% Generated by build_mode_diversity_table.py; do not hand edit.',
             fr'\newcommand{{\MDfrontierpmdlow}}{{{fmt(front_last)}}}',
             fr'\newcommand{{\MDfrontierpmdhigh}}{{{fmt(front_first)}}}',
             fr'\newcommand{{\MDfrontierpass}}{{{fmt(front_pass)}}}',
             fr'\newcommand{{\MDlevelpass}}{{{fmt(level_pass)}}}',
             fr'\newcommand{{\MDlevelpmd}}{{{fmt(level_pmd)}}}',
             fr'\newcommand{{\MDlevelsteps}}{{{len(pass_steps)}}}',
             fr'\newcommand{{\MDdistinctcorr}}{{{fmt(distinct)}}}',
             fr'\newcommand{{\MDpmdcorr}}{{{fmt(pmd)}}}',
             fr'\newcommand{{\MDcells}}{{{len(cells)}}}',
             fr'\newcommand{{\MDreportable}}{{{len(reportable)}}}',
             fr'\newcommand{{\MDgaps}}{{{len(cells) - len(reportable)}}}',
             fr'\newcommand{{\MDtopgapdomain}}{{{labels.get(top, top)}}}',
             fr'\newcommand{{\MDtopgapcount}}{{{top_n}}}']
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=PAYLOAD)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    args.output.write_text(build(payload))
    COVERAGE.write_text(build_coverage(payload))
    gaps = sum(1 for c in payload['cells'] if not c['reportable'])
    print(json.dumps({'event': 'built', 'output': str(args.output),
                      'coverage': str(COVERAGE),
                      'rows': len(DOMAINS) * len(LEVELS), 'gap_cells': gaps}))


if __name__ == '__main__':
    main()
