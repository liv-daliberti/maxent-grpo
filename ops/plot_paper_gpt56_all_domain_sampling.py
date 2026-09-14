#!/usr/bin/env python3
"""Plot all five domains from a completed, authenticated GPT-5.6 Sol analysis.

Collection, native authentication, grading, and bootstrap calculations belong to
that analysis. This renderer validates its complete plotted grid and preserves
its provenance. It draws no extrapolation beyond each cell's observed budget.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/modebench_discovery_five_domains_sol_20260912/analysis/analysis.json'
OUTPUT = ROOT / 'paper/figures/gpt56_all_domain_sampling_budget'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LABELS = ('Graph', 'Countdown', 'Python factors', 'MathIR', 'PantryPlan')
LEVELS = (2, 3)
COLORS = {2: '#00509E', 3: '#C76A3A'}
FIGSIZE = (6.4, 2.15)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def binding(path):
    return {'path': relative(path), 'sha256': file_sha(path)}


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def required_grid(maximum):
    require(type(maximum) is int and maximum >= 64, 'Every cell needs at least 64 observed draws')
    values = [2 ** power for power in range(maximum.bit_length())]
    return values if values[-1] == maximum else [*values, maximum]


def validate_cells(cells):
    require(isinstance(cells, list) and len(cells) == 10, 'All five domains and both levels are required')
    expected = {(domain, level) for domain in DOMAINS for level in LEVELS}
    require({(c['domain'], c['level']) for c in cells} == expected,
            'Missing, duplicate, or unexpected domain-level cell')
    for cell in cells:
        require(cell['n_prompts'] == 16, 'Retain all 16 registered prompts in every cell')
        grid = required_grid(cell['max_draws'])
        require([p['k'] for p in cell['points']] == grid,
                'Plotted points must cover the complete observed sampling grid through the endpoint')
        support = cell['support']
        bounds = support['range']
        require(support['kind'] in ('exact', 'certified_lower_bound')
                and finite(support['mean']) and support['mean'] > 0
                and isinstance(bounds, list) and len(bounds) == 2
                and all(type(v) is int and v > 0 for v in bounds)
                and bounds[0] <= support['mean'] <= bounds[1],
                'Known support must retain its exact-count or certified-lower-bound meaning')
        for metric in ('distinct', 'pass'):
            previous = -1.0
            for point in cell['points']:
                value = point[metric]
                estimate, ci = value['estimate'], value['ci95']
                maximum = point['k'] if metric == 'distinct' else 1
                require(finite(estimate) and isinstance(ci, list) and len(ci) == 2
                        and all(finite(v) for v in ci)
                        and -1e-10 <= ci[0] <= estimate + 1e-10
                        and estimate - 1e-10 <= ci[1] <= maximum + 1e-10
                        and estimate >= previous - 1e-10,
                        f'Invalid {metric} estimate, pointwise interval, or monotonic sampling curve')
                previous = estimate
        for point in cell['points']:
            require(point['distinct']['estimate'] + 1e-10 >= point['pass']['estimate'],
                    'Expected distinct verified modes cannot be below pass probability')
    return cells


def build_record(source=SOURCE):
    source = Path(source)
    report = read_json(source)
    require(report.get('schema') == 'gpt56-all-domain-discovery-v1'
            and report.get('status') == 'complete'
            and report.get('model') == 'gpt-5.6-sol'
            and report.get('grading') == 'normalized_secondary'
            and report.get('prompt_arm', report.get('arm', 'original')) == 'original',
            'Use the complete original-wording, normalized GPT-5.6 Sol five-domain analysis')
    cells = validate_cells(deepcopy(report['cells']))
    cells.sort(key=lambda c: (DOMAINS.index(c['domain']), c['level']))
    return {'schema': 'paper-gpt56-all-domain-sampling-budget-v1', 'status': 'complete',
            'source': binding(source), 'renderer': binding(__file__), 'model': report['model'],
            'grading': report['grading'], 'prompt_arm': 'original',
            'cells': cells, 'analysis_provenance': {k: deepcopy(v) for k, v in report.items() if k != 'cells'},
            'display': {'figure_inches': list(FIGSIZE), 'domains': list(DOMAINS), 'levels': list(LEVELS),
                        'x': 'Samples per prompt, k', 'x_scale': 'log2; shared across all five domains',
                        'x_range': [1, max(cell['max_draws'] for cell in cells)],
                        'y': 'Mean distinct verified modes', 'y_scale': 'linear, separate per domain',
                        'curves': 'Original prompt wording; L2 blue solid circles, L3 orange dashed squares.',
                        'intervals': 'Pointwise 95% intervals from the authenticated analysis.',
                        'support': 'Grey lines retain each level\'s mean known-support reference, including exact versus certified-lower-bound labels.',
                        'endpoints': 'Each curve stops at that cell\'s largest observed budget; no extrapolation.'},
            'validation': {'all_five_domains_and_two_levels': True, 'all_16_prompts_per_cell': True,
                           'complete_observed_sampling_grids': True, 'support_semantics_preserved': True,
                           'native_authentication_and_statistics': 'Responsibility of bound source analysis.',
                           'api_calls': 0}}


def support_label(cells):
    means = sorted({cell['support']['mean'] for cell in cells})
    lower_bound = any(cell['support']['kind'] == 'certified_lower_bound' for cell in cells)

    def number(value):
        if float(value).is_integer():
            return f'{value:,.0f}'
        # Rounding a lower bound downward preserves its meaning.
        value = math.floor(value * 100) / 100 if lower_bound else value
        return f'{value:,.2f}'.rstrip('0').rstrip('.')

    count = number(means[0]) if len(means) == 1 else f'{number(means[0])}–{number(means[-1])}'
    prefix = '≥ ' if lower_bound else ''
    variable = any(cell['support']['range'][0] != cell['support']['range'][1] for cell in cells)
    return f'{prefix}{count}\nknown modes (mean)' if variable else f'{prefix}{count} known modes'


def build_figure(record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, MaxNLocator, StrMethodFormatter

    rc = {'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7.5,
          'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'pdf.fonttype': 42,
          'ps.fonttype': 42, 'text.color': '#19324A', 'axes.labelcolor': '#19324A'}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 5, figsize=FIGSIZE, sharex=True)
        maximum_budget = max(cell['max_draws'] for cell in record['cells'])
        fig.subplots_adjust(left=.076, right=.99, bottom=.23, top=.74, wspace=.40)
        for ax, domain, title in zip(axes, DOMAINS, LABELS):
            cells = [cell for cell in record['cells'] if cell['domain'] == domain]
            support_max = max(cell['support']['mean'] for cell in cells)
            maximum_modes = max(support_max, max(p['distinct']['ci95'][1] for cell in cells for p in cell['points']))
            for cell in cells:
                level = cell['level']
                xs = [p['k'] for p in cell['points']]
                ax.fill_between(xs, [p['distinct']['ci95'][0] for p in cell['points']],
                                [p['distinct']['ci95'][1] for p in cell['points']],
                                color=COLORS[level], alpha=.13, linewidth=0)
                ax.plot(xs, [p['distinct']['estimate'] for p in cell['points']],
                        color=COLORS[level], linewidth=1.2, linestyle='-' if level == 2 else '--',
                        marker='o' if level == 2 else 's', markersize=2.2,
                        markerfacecolor=COLORS[level] if level == 2 else 'white', zorder=3)
                ax.axhline(cell['support']['mean'], color='#71808C',
                           linestyle=(0, (3, 2)), linewidth=.75, zorder=1)
            ax.set_xscale('log', base=2)
            ticks = [k for k in (1, 4, 16, 64, 256, 1024) if k <= maximum_budget]
            if maximum_budget not in ticks:
                if len(ticks) > 1 and maximum_budget / ticks[-1] < 3:
                    ticks.pop()  # Leave room for a three-digit measured endpoint.
                ticks.append(maximum_budget)
            ax.set_xlim(.95, maximum_budget * 1.15)
            ax.set_ylim(0, maximum_modes * 1.34)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.annotate(support_label(cells), (1.08, support_max), xytext=(0, 3),
                        textcoords='offset points', fontsize=5.8, color='#566574',
                        ha='left', va='bottom', linespacing=1.05)
            ax.tick_params(length=2, width=.6, colors='#607487', pad=2)
            ax.grid(axis='y', color='#D8E2EA', linewidth=.45, alpha=.65)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color('#AABAC7')
                ax.spines[side].set_linewidth(.6)
            ax.set_title(title, fontsize=7.6, fontweight='bold', pad=8)
        handles = [Line2D([], [], color=COLORS[level], linestyle='-' if level == 2 else '--',
                          marker='o' if level == 2 else 's', markersize=2.5,
                          markerfacecolor=COLORS[level] if level == 2 else 'white',
                          linewidth=1.2, label=f'Level {level}') for level in LEVELS]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.99, 1.025),
                   ncol=2, frameon=False, fontsize=7, handlelength=1.8,
                   handletextpad=.4, columnspacing=1.2)
        fig.text(.076, .969, 'GPT-5.6 Sol · original wording', ha='left', va='top',
                 fontsize=8, fontweight='bold')
        fig.text(.02, .49, 'Mean distinct\nverified modes', rotation=90,
                 ha='center', va='center', fontsize=7.5)
        fig.text(.54, .05, 'Samples per prompt, $k$', ha='center', va='center', fontsize=7.5)
    return fig


def render(record, output=OUTPUT):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(record)
    result = deepcopy(record)
    result['outputs'] = {}
    for extension in ('pdf', 'png'):
        path = output.with_suffix('.' + extension)
        kwargs = {'dpi': 300} if extension == 'png' else {'metadata': {'CreationDate': None, 'ModDate': None}}
        figure.savefig(path, facecolor='white', **kwargs)
        result['outputs'][extension] = binding(path)
    plt.close(figure)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def check(output=OUTPUT, source=SOURCE):
    output = Path(output)
    expected = build_record(source)
    retained = read_json(output.with_suffix('.json'))
    outputs = retained.pop('outputs')
    require(retained == expected, 'Retained five-domain figure differs from the completed source analysis')
    require(set(outputs) == {'pdf', 'png'}, 'Incomplete five-domain rendering')
    for extension, record in outputs.items():
        path = output.with_suffix('.' + extension)
        require(record == binding(path), f'Changed five-domain {extension} rendering')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        record = check(args.output, args.source)
    else:
        record = render(build_record(args.source), args.output)
    print(json.dumps({'status': 'pass', 'mode': 'check' if args.check else 'render',
                      'output': relative(args.output), 'cells': len(record['cells']),
                      'budgets': sorted({c['max_draws'] for c in record['cells']})}))


if __name__ == '__main__':
    main()
