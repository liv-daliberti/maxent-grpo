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


def _seven_b_range_macros(reportable, fmt) -> list[str]:
    """Ranges the construction caption quotes for the 7B Python/Graph contrast.

    The grid is still filling, so these move; quoting them by hand guarantees a
    stale caption. Emitted empty when 7B has no reportable cell yet, in which
    case the caption's macros expand to nothing and the omission is visible.
    """
    def span(domain):
        vals = [(c['pass8'], c['pmd']) for c in reportable
                if c['model_label'] == '7b' and c['domain'] == domain]
        if not vals:
            return None
        return (min(v[0] for v in vals), max(v[0] for v in vals),
                min(v[1] for v in vals), max(v[1] for v in vals))

    out = []
    for domain, tag in (('python_factors', 'Py'), ('graph_coloring', 'Graph')):
        got = span(domain)
        if got is None:
            continue
        lo_p, hi_p, lo_d, hi_d = got
        out += [fr'\newcommand{{\MDsevenb{tag}passlo}}{{{fmt(lo_p)}}}',
                fr'\newcommand{{\MDsevenb{tag}passhi}}{{{fmt(hi_p)}}}',
                fr'\newcommand{{\MDsevenb{tag}pmdlo}}{{{fmt(lo_d)}}}',
                fr'\newcommand{{\MDsevenb{tag}pmdhi}}{{{fmt(hi_d)}}}']
    return out


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
    front_extra: dict[str, dict] = {}
    front_models = 0
    if frontier_path.is_file():
        fcells = [c for c in json.loads(frontier_path.read_text())['cells']
                  if c.get('reportable')]
        # The level-to-level decline must be read across levels the whole cohort
        # covers. Level 4 is one deployment on four domains, so folding it into
        # this comparison would silently change what the endpoints mean.
        cohort = len({c['model'] for c in fcells})
        front_models = cohort
        full = [lev for lev in sorted({c['level'] for c in fcells})
                if len({c['model'] for c in fcells if c['level'] == lev}) == cohort]
        if full:
            def mean_at(level, field):
                vals = [c[field] for c in fcells if c['level'] == level]
                return statistics.mean(vals) if vals else float('nan')
            front_first = mean_at(full[0], 'pmd')
            front_last = mean_at(full[-1], 'pmd')
            front_pass = statistics.mean(c['pass8'] for c in fcells
                                         if c['level'] in full)
            partial = [lev for lev in sorted({c['level'] for c in fcells})
                       if lev not in full]
            for lev in partial:
                vals = [c for c in fcells if c['level'] == lev]
                front_extra[lev] = {
                    'pmd': statistics.mean(c['pmd'] for c in vals),
                    'models': len({c['model'] for c in vals}),
                    'domains': len({c['domain'] for c in vals})}

    # The hosted deployments against the small local models: the comparison the
    # text makes is that being far more accurate does not buy more modes.
    front_mean = (statistics.mean(c['pmd'] for c in fcells)
                  if frontier_path.is_file() and fcells else float('nan'))
    small = {}
    for model in grid.MODEL_PARAMS:
        if grid.MODEL_PARAMS[model] >= 2.0:
            continue
        vals = [c['pmd'] for c in reportable if c['model_label'] == model]
        if vals:
            small[model] = statistics.mean(vals)
    best = max(small, key=small.get) if small else None
    small_pmd = small[best] if best else float('nan')
    small_pass = (statistics.mean(c['pass8'] for c in reportable
                                  if c['model_label'] == best) if best else float('nan'))

    lines = ['% Generated by build_mode_diversity_table.py; do not hand edit.',
             fr'\newcommand{{\MDfrontierpmdmean}}{{{fmt(front_mean)}}}',
             fr'\newcommand{{\MDsmallbestpmd}}{{{fmt(small_pmd)}}}',
             fr'\newcommand{{\MDsmallbestpass}}{{{fmt(small_pass)}}}',
             fr'\newcommand{{\MDfrontierpmdlow}}{{{fmt(front_last)}}}',
             fr'\newcommand{{\MDfrontierpmdhigh}}{{{fmt(front_first)}}}',
             fr'\newcommand{{\MDfrontierpass}}{{{fmt(front_pass)}}}',
             fr'\newcommand{{\MDfrontierlevels}}{{{len(full) if frontier_path.is_file() and full else 0}}}',
             fr'\newcommand{{\MDfrontierdeployments}}{{{front_models}}}',
             *_seven_b_range_macros(reportable, fmt),
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
    # LaTeX control sequences are letters only, so the level is spelled out.
    words = {'1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five'}
    for lev, info in sorted(front_extra.items()):
        tag = words.get(lev.replace('level', ''))
        if tag is None:
            continue
        lines += [fr'\newcommand{{\MDfrontierLpmd{tag}}}{{{fmt(info["pmd"])}}}',
                  fr'\newcommand{{\MDfrontierLmodels{tag}}}{{{info["models"]}}}',
                  fr'\newcommand{{\MDfrontierLdomains{tag}}}{{{info["domains"]}}}']
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
