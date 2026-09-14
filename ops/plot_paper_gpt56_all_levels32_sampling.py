#!/usr/bin/env python3
"""Plot the completed 32-problem-per-cell, 512-draw GPT-5.6 Sol study.

No publication asset is written from a partial cell, omitted level, shorter
sampling pool, or extrapolated curve. The bound analyzer authenticates native
responses; this renderer independently checks the finite-pool statistics from
all retained per-prompt mode counts under both grading conventions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/modebench_discovery_all_levels_sol32x512_20260913/analysis/analysis.json'
OUTPUT = ROOT / 'paper/figures/gpt56_all_levels32_sampling_budget'
SCHEMA = 'gpt56-all-levels-discovery-v1'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LABELS = ('Graph', 'Countdown', 'Python factors', 'MathIR', 'PantryPlan')
LEVELS = (1, 2, 3)
DRAWS = 512
PROMPTS_PER_CELL = 32
TOTAL_PROMPTS = len(DOMAINS) * len(LEVELS) * PROMPTS_PER_CELL
K_GRID = tuple(2 ** power for power in range(10))
X_TICKS = (1, 8, 64, 512)
STYLES = {
    1: {'color': '#7B1FA2', 'linestyle': '-.', 'marker': '^', 'markersize': 2.9},
    2: {'color': '#00509E', 'linestyle': '-', 'marker': 'o', 'markersize': 2.3},
    3: {'color': '#C76A3A', 'linestyle': '--', 'marker': 's', 'markersize': 2.5},
}
FIGSIZE = (6.4, 1.90)
# Exact Figure 2 card washes, also used by Figure 3's domain panels.
# Source: plot_paper_modebench_examples.DOMAIN_PANEL (A through E).
DOMAIN_BACKGROUNDS = dict(zip(DOMAINS, ('#E8F1FA', '#E7FBF6', '#EFFBE7', '#E7FBEE', '#E7ECFB')))



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


def close(actual, expected, label):
    require(finite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10),
            f'{label} differs from complete-pool reconstruction')


@lru_cache(maxsize=8192)
def seen_probability(responses, count, k):
    return 1.0 - math.comb(responses - count, k) / math.comb(responses, k)


def rarefaction(mode_counts, responses, k):
    require(type(responses) is int and responses > 0 and type(k) is int and 0 <= k <= responses,
            'Invalid finite-pool sampling budget')
    require(all(type(n) is int and n > 0 for n in mode_counts) and sum(mode_counts) <= responses,
            'Invalid canonical mode-count inventory')
    return {'distinct': math.fsum(seen_probability(responses, n, k) for n in mode_counts),
            'pass': seen_probability(responses, sum(mode_counts), k)}


def validate_estimate(item, expected, maximum, label):
    close(item['estimate'], expected, label)
    ci = item['ci95']
    require(isinstance(ci, list) and len(ci) == 2 and all(finite(v) for v in ci)
            and -1e-10 <= ci[0] <= expected + 1e-10
            and expected - 1e-10 <= ci[1] <= maximum + 1e-10,
            f'Invalid pointwise interval: {label}')


def validate_pool_cells(cells, draws):
    """Shared finite-pool arithmetic; callers enforce their own study budget."""
    require(type(draws) is int and draws >= 2 and draws & (draws - 1) == 0,
            "Expected an explicit power-of-two pool size")
    grid = tuple(2 ** power for power in range(draws.bit_length()))
    midpoint = draws // 2
    expected_cells = {(domain, level) for domain in DOMAINS for level in LEVELS}
    require(isinstance(cells, list) and len(cells) == len(expected_cells)
            and {(cell['domain'], cell['level']) for cell in cells} == expected_cells,
            'All five domains and all three levels require fifteen unique cells')
    for cell in cells:
        identity = cell['domain'], cell['level']
        require(cell['max_draws'] == draws and cell['n_prompts'] == PROMPTS_PER_CELL,
                f'Every domain-level cell must retain thirty-two prompts and all {draws} draws')
        require([p['k'] for p in cell['points']] == list(grid),
                f'Every curve must retain the full observed grid from one through {draws} draws')
        support = cell['support']
        kind = 'exact' if cell['domain'] == 'graph_coloring' else 'certified_lower_bound'
        require(support['kind'] == kind and finite(support['mean']) and support['mean'] > 0
                and isinstance(support['range'], list) and len(support['range']) == 2
                and all(type(v) is int and v > 0 for v in support['range'])
                and support['range'][0] <= support['mean'] <= support['range'][1],
                'Support must retain exact Graph counts and certified lower bounds elsewhere')
        prompts = cell['prompts']
        require(len(prompts) == PROMPTS_PER_CELL
                and len({p['row_index'] for p in prompts}) == PROMPTS_PER_CELL,
                'Every cell must retain all thirty-two distinct registered problems')
        reconstructed = []
        support_counts = []
        for prompt in prompts:
            require((prompt['domain'], prompt['level']) == identity and prompt['responses'] == draws,
                    'Prompt belongs to another cell or an incomplete response pool')
            counts = prompt['mode_counts']
            require(sum(counts) == prompt['correct_draws'],
                    'Canonical mode counts do not sum to the complete correctness inventory')
            reference = prompt['support']
            require(reference['support_kind'] == kind
                    and type(reference['support_count']) is int and reference['support_count'] > 0,
                    'Prompt support semantics differ from the cell reference')
            if kind == 'exact':
                require(len(counts) <= reference['support_count'],
                        'Observed Graph modes exceed the claimed exhaustive support')
            support_counts.append(reference['support_count'])
            values = {k: rarefaction(counts, draws, k) for k in grid}
            reconstructed.append(values)
            prefix = prompt['prefix']
            require(set(prefix) == {str(k) for k in grid}, 'Incomplete ordered-prefix sensitivity')
            previous_distinct = 0
            for k in grid:
                ordered = prefix[str(k)]
                distinct, passed = ordered['distinct'], ordered['pass']
                require(finite(distinct) and float(distinct).is_integer()
                        and previous_distinct <= distinct <= min(k, len(counts))
                        and finite(passed) and passed == float(distinct > 0),
                        'Invalid ordered-prefix mode or success count')
                previous_distinct = distinct
            close(prefix[str(draws)]['distinct'], len(counts), 'Full-pool prefix distinct')
            close(prefix[str(draws)]['pass'], float(bool(counts)), 'Full-pool prefix pass')
        close(support['mean'], math.fsum(support_counts) / PROMPTS_PER_CELL, 'Mean support reference')
        require(support['range'] == [min(support_counts), max(support_counts)],
                'Support range differs from the retained thirty-two problems')
        means = {k: {metric: math.fsum(p[k][metric] for p in reconstructed) / PROMPTS_PER_CELL
                     for metric in ('distinct', 'pass')} for k in grid}
        for point in cell['points']:
            k = point['k']
            for metric in ('distinct', 'pass'):
                validate_estimate(point[metric], means[k][metric], k if metric == 'distinct' else 1,
                                  f'{identity}/{metric}/k{k}')
            if 'breadth' in point:
                validate_estimate(point['breadth'], means[k]['distinct'] - means[k]['pass'], k,
                                  f'{identity}/breadth/k{k}')
        tail = cell['tail']
        require(tail['from_k'] == midpoint and tail['to_k'] == draws,
                f'The final-doubling contrast must use {midpoint} versus {draws} within the final pool')
        for metric in ('distinct', 'pass'):
            validate_estimate(tail[metric], means[draws][metric] - means[midpoint][metric],
                              draws if metric == 'distinct' else 1, f'{identity}/tail/{metric}')
        if 'breadth' in tail:
            value = ((means[draws]['distinct'] - means[draws]['pass'])
                     - (means[midpoint]['distinct'] - means[midpoint]['pass']))
            validate_estimate(tail['breadth'], value, draws, f'{identity}/tail/breadth')
    return cells


def validate_cells(cells):
    """The final publication still requires exactly 512 draws in every cell."""
    return validate_pool_cells(cells, DRAWS)

def validate_controls(report):
    protocol = report['protocol']
    require(protocol['reasoning'] == 'medium' and protocol['max_output_tokens'] == 8192
            and protocol['temperature_and_top_p'] == 'omitted', 'Changed native generation controls')
    bootstrap = protocol['bootstrap']
    require(bootstrap['pointwise'] is True and bootstrap['unit'] == 'whole prompt'
            and bootstrap['replicates'] == 20000 and type(bootstrap['seed']) is int,
            'Expected 20,000 pointwise whole-problem bootstrap resamples')


def validate_grading_conventions(report):
    strict = {(c['domain'], c['level']): c for c in report['strict_cells']}
    for normal in report['cells']:
        raw = strict[normal['domain'], normal['level']]
        require(normal['support'] == raw['support'], 'Grading conventions changed support references')
        normal_prompts = {p['row_index']: p for p in normal['prompts']}
        strict_prompts = {p['row_index']: p for p in raw['prompts']}
        require(normal_prompts.keys() == strict_prompts.keys(),
                'Both grading conventions must retain the identical complete prompt cohort')
        for index, prompt in normal_prompts.items():
            original = strict_prompts[index]
            require(prompt['row_sha256'] == original['row_sha256'] and prompt['support'] == original['support']
                    and prompt['correct_draws'] >= original['correct_draws']
                    and len(prompt['mode_counts']) >= len(original['mode_counts']),
                    'Normalization changed prompt identity, certified support, or a strict success')
        for point, original in zip(normal['points'], raw['points']):
            require(all(point[m]['estimate'] + 1e-10 >= original[m]['estimate']
                        for m in ('distinct', 'pass')),
                    'Normalization cannot remove strict verified answers or modes')


def validate_report(report):
    require(report.get('schema') == SCHEMA and report.get('status') == 'complete'
            and report.get('model') == 'gpt-5.6-sol'
            and report.get('grading') == 'normalized_secondary'
            and report.get('prompt_arm') == 'original',
            'Use the complete original-wording, three-level GPT-5.6 Sol 512-draw analysis')
    require(report.get('levels', list(LEVELS)) == list(LEVELS)
            and report.get('prompts') == TOTAL_PROMPTS
            and report.get('responses') == TOTAL_PROMPTS * DRAWS,
            'The study must retain all 480 problems and 245,760 responses')
    validate_controls(report)
    validate_cells(report['cells'])
    validate_cells(report['strict_cells'])
    validate_grading_conventions(report)
    return report

def build_record(source=SOURCE):
    source = Path(source)
    report = validate_report(read_json(source))
    cells = deepcopy(report['cells'])
    cells.sort(key=lambda c: (DOMAINS.index(c['domain']), c['level']))
    return {'schema': 'paper-gpt56-all-levels-sampling-budget-v1', 'status': 'complete',
            'source': binding(source), 'renderer': binding(__file__), 'model': report['model'],
            'grading': report['grading'], 'prompt_arm': 'original', 'cells': cells,
            'analysis_provenance': {k: deepcopy(v) for k, v in report.items()
                                    if k not in ('cells', 'strict_cells')},
            'display': {'figure_inches': list(FIGSIZE), 'domains': list(DOMAINS), 'levels': list(LEVELS),
                        'x': 'Samples per prompt, k', 'x_scale': 'log2; shared across all five domains',
                        'x_range': [1, DRAWS], 'x_ticks': list(X_TICKS),
                        'domain_backgrounds': deepcopy(DOMAIN_BACKGROUNDS),
                        'domain_palette_source': 'Figure 2 DOMAIN_PANEL; identical to Figure 3',
                        'y': 'Mean distinct verified modes', 'y_scale': 'linear; separate per domain; includes every support reference',
                        'curves': {str(level): {'label': f'Level {level}', **style}
                                   for level, style in STYLES.items()},
                        'intervals': 'Pointwise 95% whole-problem bootstrap intervals from the bound analysis.',
                        'support': 'Three grey lines retain the three level-specific mean support references; labels list their means in Level 1 / Level 2 / Level 3 order. Graph exact; other domains certified lower bounds.',
                        'endpoints': 'All fifteen curves use complete 512-draw pools; no extrapolation.'},
            'validation': {'all_five_domains_and_three_levels': True, 'all_480_prompts_retained': True,
                           'all_245760_responses_retained': True, 'both_grading_pools_reconstructed': True,
                           'all_point_and_tail_means_reconstructed': True, 'support_means_and_ranges_reconstructed': True,
                           'native_authentication_and_bootstrap': 'Responsibility of bound source analysis.',
                           'api_calls': 0}}


def support_label(cells):
    means = [cell['support']['mean'] for cell in sorted(cells, key=lambda c: c['level'])]
    lower = cells[0]['support']['kind'] == 'certified_lower_bound'

    def number(value):
        if float(value).is_integer():
            return f'{value:,.0f}'
        value = math.floor(value * 100) / 100 if lower else value
        return f'{value:,.2f}'.rstrip('0').rstrip('.')

    count = ' / '.join(number(value) for value in means)
    return ('≥ ' if lower else '') + count + '\nknown modes, L1–L3'


def build_figure(record):
    return build_budget_figure(record, draws=DRAWS, x_ticks=X_TICKS, title=None)


def build_budget_figure(record, *, draws, x_ticks, title):
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
        fig.subplots_adjust(left=.076, right=.99, bottom=.30, top=.87, wspace=.16)
        for ax, domain, domain_title in zip(axes, DOMAINS, LABELS):
            ax.set_facecolor(DOMAIN_BACKGROUNDS[domain])
            cells = {cell['level']: cell for cell in record['cells'] if cell['domain'] == domain}
            support_max = max(cell['support']['mean'] for cell in cells.values())
            maximum = max(support_max, max(p['distinct']['ci95'][1]
                          for cell in cells.values() for p in cell['points']))
            # Dashed/open traces over the solid trace expose coincident levels
            # without displacing any observed value from its actual coordinates.
            for level in (2, 3, 1):
                cell, style = cells[level], STYLES[level]
                xs = [p['k'] for p in cell['points']]
                ax.fill_between(xs, [p['distinct']['ci95'][0] for p in cell['points']],
                                [p['distinct']['ci95'][1] for p in cell['points']],
                                color=style['color'], alpha=.09, linewidth=0)
                ax.plot(xs, [p['distinct']['estimate'] for p in cell['points']],
                        **style, linewidth=1.15, markerfacecolor=style['color'] if level == 2 else 'none', zorder=3)
                ax.axhline(cell['support']['mean'], color='#71808C', linestyle=(0, (3, 2)),
                           linewidth=.7, zorder=1)
            ax.set_xscale('log', base=2)
            ax.set_xlim(.95, draws * 1.15)
            ax.set_ylim(0, maximum * 1.34)
            ax.xaxis.set_major_locator(FixedLocator(x_ticks))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.annotate(support_label(list(cells.values())), (1.08, maximum), xytext=(0, 3),
                        textcoords='offset points', fontsize=5.7, color='#566574', ha='left', va='bottom',
                        linespacing=1.05)
            ax.tick_params(length=2, width=.6, colors='#607487', pad=2)
            ax.grid(axis='y', color='#D8E2EA', linewidth=.45, alpha=.65)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color('#AABAC7')
                ax.spines[side].set_linewidth(.6)
            ax.set_title(domain_title, fontsize=7.6, fontweight='bold', pad=4)
        handles = [Line2D([], [], **STYLES[level], linewidth=1.15,
                          markerfacecolor=STYLES[level]['color'] if level == 2 else 'none',
                          label=f'Level {level}') for level in LEVELS]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, -.015), ncol=3,
                   frameon=False, fontsize=7.1, handlelength=2.0, handletextpad=.4, columnspacing=1.4)
        if title:
            fig.text(.076, .985, title, ha='left', va='top', fontsize=8, fontweight='bold')
        fig.text(.02, .58, 'Mean distinct\nverified modes', rotation=90, ha='center', va='center', fontsize=7.5)
        fig.text(.54, .155, 'Samples per prompt, $k$', ha='center', va='center', fontsize=7.5)
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
    require(retained == expected, 'Retained all-level figure differs from the completed source or renderer')
    require(set(outputs) == {'pdf', 'png'}, 'Incomplete all-level rendering')
    for extension, item in outputs.items():
        require(item == binding(output.with_suffix('.' + extension)), 'Changed all-level figure rendering')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    record = check(args.output, args.source) if args.check else render(build_record(args.source), args.output)
    print(json.dumps({'status': 'pass', 'mode': 'check' if args.check else 'render',
                      'output': relative(args.output), 'cells': len(record['cells']), 'draws_per_prompt': DRAWS}))


if __name__ == '__main__':
    main()
