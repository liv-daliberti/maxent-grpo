#!/usr/bin/env python3
"""Emit the appendix table body pairing the registered endpoints with PCMD.

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

# The Qwen ramp and the ladder are read off the grid, not written here. Hard
# tuples went stale twice: the 1.5B scale was drawn in the figure while every
# macro on this page described six scales, and Level 5 was released without
# entering the counts. A level joins once the ramp covers it completely, so a
# half-measured level cannot move a quoted number, and joins on its own when it
# fills.
QWEN_RAMP = ('05b', 'qwen15b', '3b', '7b', '14b', 'qwen32b', 'qwen72b')
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry')
DOMAIN_LABELS = {'graph_coloring': 'Graph', 'countdown': 'Countdown',
                 'python_factors': 'Python', 'mathir': 'MathIR', 'pantry': 'Pantry'}
ALL_LEVELS = ('level1', 'level2', 'level3', 'level4', 'level5')


def grid_axes(cells):
    """The scales present, and the levels the ramp covers for every domain."""
    present = {(c['model_label'], c['domain'], c['level']) for c in cells}
    models = tuple(m for m in QWEN_RAMP
                   if any(k[0] == m for k in present))
    levels = tuple(level for level in ALL_LEVELS
                   if all((m, d, level) in present
                          for m in models for d in DOMAINS))
    return models, levels


MODELS = QWEN_RAMP
LEVELS = ALL_LEVELS[:4]


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


SPLIT_OUT = ROOT / 'paper/results/mode_diversity_base_grid_split_body.tex'

# Pale wash so the printed number stays legible: low values red, high values
# green, near-white in the middle. Both panels use the same direction, so the
# reader sees pass@8 warming to the right while PCMD cools.
LOW = (0.94, 0.58, 0.54)
MID = (0.99, 0.99, 0.97)
HIGH = (0.58, 0.84, 0.60)
BLANK = (0.93, 0.93, 0.93)
# Fixed cell width: the two panels then share one geometry, so neither has
# to be scaled to fit and both print at the same size.
CELL = r'\makebox[2.25em][c]{%s}'


def _wash(t: float) -> tuple[float, float, float]:
    """Interpolate LOW -> MID -> HIGH at t in [0, 1]."""
    t = min(1.0, max(0.0, t))
    if t <= 0.5:
        a, b, u = LOW, MID, t / 0.5
    else:
        a, b, u = MID, HIGH, (t - 0.5) / 0.5
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))


def _paint(rgb) -> str:
    return r'\cellcolor[rgb]{%.3f,%.3f,%.3f}' % rgb


def build_split(payload: dict) -> str:
    """One row per domain and level, carrying both metrics side by side.

    The row labels appear once and serve both column groups. Each domain is
    washed from its own low to its own peak, separately per metric, so a row is
    read against the rest of its domain rather than against Python's
    near-ceiling accuracy. Values below the support bar are printed in
    parentheses and washed at reduced strength instead of being dropped: the
    gap is more legible as a weak number than as absence.
    """
    index = {(c['model_label'], c['level'], c['domain']): c for c in payload['cells']}

    def bounds(domain, field):
        seen = [index[(m, l, domain)][field]
                for l in LEVELS for m in MODELS
                if (m, l, domain) in index
                and index[(m, l, domain)][field] is not None]
        lo, hi = (min(seen), max(seen)) if seen else (0.0, 1.0)
        return lo, (hi - lo) or 1.0

    def group(domain, level, field):
        lo, span = bounds(domain, field)
        out = []
        for model in MODELS:
            cell = index.get((model, level, domain))
            if cell is None or cell[field] is None:
                out.append(_paint(BLANK) + CELL % '---')
                continue
            text = fmt(cell[field])
            rgb = _wash((cell[field] - lo) / span)
            if field == 'pmd' and not cell['reportable']:
                # A noise-dominated estimate should not carry the same visual
                # weight as a measurement that clears the bar.
                text = f'({text})'
                rgb = tuple(c + (1.0 - c) * 0.55 for c in rgb)
            out.append(_paint(rgb) + CELL % text)
        return out

    lines = []
    for domain in DOMAINS:
        for position, level in enumerate(LEVELS):
            label = DOMAIN_LABELS[domain] if position == 0 else ''
            row = [label, str(position + 1)]
            row += group(domain, level, 'pass8')
            row += group(domain, level, 'pmd')
            lines.append(' & '.join(row) + r' \\')
        if domain != DOMAINS[-1]:
            lines.append(r'\addlinespace')
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
    # Shadow the module defaults with the axes the payload actually carries, so
    # a newly released scale or level is counted instead of silently skipped.
    MODELS, LEVELS = grid_axes(cells)
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

    # distinct@8 needs no support gate, so it uses every cell; PCMD uses the
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
    # PCMD decline with level cannot be a side effect of falling success.
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

    # The appendix scale table: how many frozen base models it prints, and the
    # scale trend its caption quotes. A block supports the comparison when PCMD
    # is reportable at two or more scales; it falls when PCMD at the largest such
    # scale sits below PCMD at the smallest. Both move as the ladder grows, so
    # neither may be written by hand.
    indexed_grid = {(c['model_label'], c['domain'], c['level']): c for c in reportable}
    scale_fall = scale_support = 0
    for domain in DOMAINS:
        for level in LEVELS:
            run = [indexed_grid[(m, domain, level)] for m in MODELS
                   if (m, domain, level) in indexed_grid]
            if len(run) < 2:
                continue
            scale_support += 1
            scale_fall += run[-1]['pmd'] < run[0]['pmd']

    # The ladder's own claim, and its limit. A construction ladder is supposed to
    # order difficulty, not breadth: pass@8 should fall from the first level to
    # the last, while PCMD should be left to the model. Both numbers move as the
    # grid fills, so B.2 quotes them through macros rather than by hand.
    # pass@8 needs no support gate -- it is defined wherever the cell exists --
    # so the ladder claim indexes every measured cell, not the PCMD-reportable
    # subset. Gating it would silently drop the hardest cells, which are exactly
    # the ones the claim is about.
    indexed_all = {(c['model_label'], c['domain'], c['level']): c for c in cells}
    ladder_fall = ladder_support = 0
    for domain in DOMAINS:
        for model in MODELS:
            run = [indexed_all[(model, domain, level)] for level in LEVELS
                   if (model, domain, level) in indexed_all]
            if len(run) < 2:
                continue
            ladder_support += 1
            ladder_fall += run[-1]['pass8'] < run[0]['pass8']

    # The ladder's span, on a balanced population. Only (scale, domain) series
    # measured at every level enter, so the comparison cannot be moved by which
    # cells happen to exist at the hardest level. pass@8 needs no support gate.
    ladder_balanced = [(m, d) for m in MODELS for d in DOMAINS
                       if all((m, d, level) in indexed_all for level in LEVELS)]
    def _ladder_mean(level, pairs):
        vals = [indexed_all[(m, d, level)]['pass8'] for m, d in pairs]
        return statistics.fmean(vals) if vals else float('nan')
    ladder_pass_first = _ladder_mean(LEVELS[0], ladder_balanced)
    ladder_pass_last = _ladder_mean(LEVELS[-1], ladder_balanced)
    # The body asserts the mean falls at every rung, so the rungs are counted
    # here and the claim fails closed: if a level ever stops lowering the mean,
    # this raises instead of letting the sentence quietly go stale.
    ladder_means = [_ladder_mean(level, ladder_balanced) for level in LEVELS]
    ladder_steps_total = len(ladder_means) - 1
    ladder_steps_down = sum(ladder_means[i + 1] < ladder_means[i]
                            for i in range(ladder_steps_total))
    if ladder_balanced and ladder_steps_down != ladder_steps_total:
        raise ValueError(
            'mean pass@8 no longer falls at every rung of the ladder '
            f'({ladder_steps_down} of {ladder_steps_total}); '
            f'means {[round(x, 3) for x in ladder_means]}. '
            'Update the Section 2.2 sentence before regenerating.')
    ladder_domains = ladder_domains_total = 0
    for domain in DOMAINS:
        pairs = [(m, d) for m, d in ladder_balanced if d == domain]
        if not pairs:
            continue
        ladder_domains_total += 1
        ladder_domains += _ladder_mean(LEVELS[-1], pairs) < _ladder_mean(LEVELS[0], pairs)

    # How far PCMD travels when only the level changes, against how far it
    # travels when only the scale changes. Medians over the series that carry at
    # least three reportable points, so one sparse row cannot set the range.
    level_ranges, scale_ranges = [], []
    for domain in DOMAINS:
        for model in MODELS:
            run = [indexed_grid[(model, domain, level)]['pmd'] for level in LEVELS
                   if (model, domain, level) in indexed_grid]
            if len(run) >= 3:
                level_ranges.append(max(run) - min(run))
        for level in LEVELS:
            run = [indexed_grid[(model, domain, level)]['pmd'] for model in MODELS
                   if (model, domain, level) in indexed_grid]
            if len(run) >= 3:
                scale_ranges.append(max(run) - min(run))
    level_span = statistics.median(level_ranges) if level_ranges else float('nan')
    scale_span = statistics.median(scale_ranges) if scale_ranges else float('nan')

    # The sharpest case of the split the axis argument makes: at 7B and above,
    # Python is answered almost perfectly and almost always the same way. Both
    # move as the grid fills, so A.2 quotes them through macros.
    py = [c for c in reportable
          if c['domain'] == 'python_factors' and c['level'] in LEVELS[:3]
          and c['model_label'] in ('7b', '14b', 'qwen32b', 'qwen72b')]
    py_pass = min((c['pass8'] for c in py), default=float('nan'))
    py_pmd = max((c['pmd'] for c in py), default=float('nan'))

    lines = ['% Generated by build_mode_diversity_table.py; do not hand edit.',
             fr'\newcommand{{\MDgridscales}}{{{len(MODELS)}}}',
             fr'\newcommand{{\MDpyPasslo}}{{{fmt(py_pass)}}}',
             fr'\newcommand{{\MDpyPmdhi}}{{{fmt(py_pmd)}}}',
             fr'\newcommand{{\MDscalefall}}{{{scale_fall}}}',
             fr'\newcommand{{\MDladderfall}}{{{ladder_fall}}}',
             fr'\newcommand{{\MDladdersupport}}{{{ladder_support}}}',
             fr'\newcommand{{\MDladderpassfirst}}{{{fmt(ladder_pass_first)}}}',
             fr'\newcommand{{\MDladderpasslast}}{{{fmt(ladder_pass_last)}}}',
             fr'\newcommand{{\MDladderdomains}}{{{ladder_domains}}}',
             fr'\newcommand{{\MDladderdomainstotal}}{{{ladder_domains_total}}}',
             fr'\newcommand{{\MDladderseries}}{{{len(ladder_balanced)}}}',
             fr'\newcommand{{\MDladderrungs}}{{{len(LEVELS)}}}',
             fr'\newcommand{{\MDladdersteps}}{{{ladder_steps_total}}}',
             fr'\newcommand{{\MDlevelspan}}{{{fmt(level_span)}}}',
             fr'\newcommand{{\MDscalespan}}{{{fmt(scale_span)}}}',
             fr'\newcommand{{\MDscalesupport}}{{{scale_support}}}',
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
    SPLIT_OUT.write_text(build_split(payload))
    COVERAGE.write_text(build_coverage(payload))
    gaps = sum(1 for c in payload['cells'] if not c['reportable'])
    print(json.dumps({'event': 'built', 'output': str(args.output),
                      'coverage': str(COVERAGE),
                      'rows': len(DOMAINS) * len(LEVELS), 'gap_cells': gaps}))


if __name__ == '__main__':
    main()
