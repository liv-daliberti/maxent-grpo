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
# Same domain wording and weight as Figure 3's panels, so a reader moving
# between the two figures is not asked to re-learn the labels.
LABELS = ('Graph', 'Countdown', 'Python', 'MathIR', 'Pantry')
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
ICON = ROOT / 'paper/icons/openai.png'
# Every curve on this plate is one deployment, so the plate says which one rather
# than leaving it to the caption. A provider mark may only stand beside a single
# provider's points; plates spanning several carry a per-model icon instead.
LOGO_HEIGHT_IN = 0.088
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

ROWS_ROOT = ROOT / 'artifacts/modebench_base_level_grid_20260911/rows'
ROW_DOMAIN = {'graph_coloring': 'graph_coloring', 'countdown': 'countdown',
              'python_factors': 'python_factors', 'mathir': 'mathir',
              'pantry_plan': 'pantry'}


def registered_support(domain, level, row_indices):
    """Enumerated modes for the prompts a cell used, from the registered rows.

    The bound analysis reported `num_externally_certified_modes`, two witnesses
    it re-executed rather than the enumerated support. For four domains that
    matches the benchmark's own `answer_mode_count`; for PythonFactors it is 2
    against a median of 104-128, which made a model finding four modes of many
    hundreds look as though it had exhausted the available ones. Read the count
    the benchmark registers, so every domain is described the same way.
    """
    import sys as _sys
    if str(ROOT / 'ops') not in _sys.path:
        _sys.path.insert(0, str(ROOT / 'ops'))
    import evaluate_modebench_base_grid as grid
    rows, _ = grid.load_rows({'rows_jsonl': str(ROWS_ROOT / f'level{level}'
                                                / f'{ROW_DOMAIN[domain]}.jsonl'),
                             'row_offset': 0, 'row_limit': 0})
    counts = [int(rows[i]['answer_mode_count']) for i in sorted(row_indices)]
    require(counts and all(c > 0 for c in counts), 'registered support must be positive')
    return {'kind': 'registered_enumerated', 'mean': math.fsum(counts) / len(counts),
            'range': [min(counts), max(counts)],
            'source': 'answer_mode_count of the registered evaluation rows'}


def build_record(source=SOURCE):
    source = Path(source)
    report = validate_report(read_json(source))
    cells = deepcopy(report['cells'])
    cells.sort(key=lambda c: (DOMAINS.index(c['domain']), c['level']))
    for cell in cells:
        cell['certified_support'] = cell['support']
        cell['support'] = registered_support(
            cell['domain'], cell['level'], [p['row_index'] for p in cell['prompts']])
    return {'schema': 'paper-gpt56-all-levels-sampling-budget-v1', 'status': 'complete',
            'source': binding(source), 'renderer': binding(__file__), 'model': report['model'],
            'icons': {report['model']: binding(ICON)},
            'grading': report['grading'], 'prompt_arm': 'original', 'cells': cells,
            'analysis_provenance': {k: deepcopy(v) for k, v in report.items()
                                    if k not in ('cells', 'strict_cells')},
            'display': {'figure_inches': list(FIGSIZE), 'domains': list(DOMAINS), 'levels': list(LEVELS),
                        'x': 'Samples per prompt, k', 'x_scale': 'log2; shared across all five domains',
                        'x_range': [1, DRAWS], 'x_ticks': list(X_TICKS),
                        'domain_backgrounds': deepcopy(DOMAIN_BACKGROUNDS),
                        'domain_palette_source': 'Figure 2 DOMAIN_PANEL; identical to Figure 3',
                        'y': 'Share of certified / enumerated modes found',
                        'y_scale': 'linear 0-1, shared by all five domains; 1.0 is every enumerated mode',
                        'curves': {str(level): {'label': f'Level {level}', **style}
                                   for level, style in STYLES.items()},
                        'intervals': 'Pointwise 95% whole-problem bootstrap intervals from the bound analysis.',
                        'support': 'Printed as counts and used as each curve\'s denominator: the mean number of modes the registered rows enumerate for the prompts each level used, in Level 1 / Level 2 / Level 3 order. The bound analysis\'s conservative certificate is retained per cell as certified_support.',
                        'normalization': 'Each plotted value is the cell\'s reconstructed mean distinct verified modes divided by its mean enumerated support. This is a display transform of the retained estimates; no statistic is recomputed.',
                        'shortfall': 'The shaded band runs from the best level reached at each budget up to 1.0, so its height is the share of certified / enumerated modes no level ever produced.',
                        'endpoints': 'All fifteen curves use complete 512-draw pools; no extrapolation.'},
            'validation': {'all_five_domains_and_three_levels': True, 'all_480_prompts_retained': True,
                           'all_245760_responses_retained': True, 'both_grading_pools_reconstructed': True,
                           'all_point_and_tail_means_reconstructed': True, 'support_means_and_ranges_reconstructed': True,
                           'native_authentication_and_bootstrap': 'Responsibility of bound source analysis.',
                           'api_calls': 0}}


def support_label(cells):
    means = [cell['support']['mean'] for cell in sorted(cells, key=lambda c: c['level'])]
    count = ' / '.join(f'{round(value):,d}' for value in means)
    return count + '\ncertified / enumerated\nmodes, L1–L3'


def coverage(cell, point, bound=None):
    """Express one retained estimate as a share of the cell's enumerated support."""
    support = cell['support']['mean']
    require(support > 0, 'A cell without enumerated support cannot be normalized')
    value = point['distinct']['estimate'] if bound is None else point['distinct']['ci95'][bound]
    return value / support


def build_figure(record):
    return build_budget_figure(record, draws=DRAWS, x_ticks=X_TICKS, title=None)


def provider_logo(figure, anchor, *, height_in=LOGO_HEIGHT_IN, align=(0.0, 0.5), path=ICON):
    """Draw the provider mark at a figure-fraction anchor; return its width there.

    Figure coordinates keep the mark at its printed size however the axes are laid
    out, and the returned width lets the label that follows be positioned without
    measuring the canvas.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    image = plt.imread(str(path))
    zoom = height_in * figure.dpi / image.shape[0]
    figure.add_artist(AnnotationBbox(
        OffsetImage(image, zoom=zoom), anchor, xycoords='figure fraction',
        frameon=False, box_alignment=align, annotation_clip=False))
    return image.shape[1] * height_in / (image.shape[0] * figure.get_figwidth())


def build_budget_figure(record, *, draws, x_ticks, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, PercentFormatter, StrMethodFormatter

    rc = {'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7.5,
          'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'pdf.fonttype': 42,
          'ps.fonttype': 42, 'text.color': '#19324A', 'axes.labelcolor': '#19324A'}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 5, figsize=FIGSIZE, sharex=True, sharey=True)
        fig.subplots_adjust(left=.076, right=.99, bottom=.30, top=.79, wspace=.16)
        for index, (ax, domain, domain_title) in enumerate(zip(axes, DOMAINS, LABELS)):
            ax.set_facecolor(DOMAIN_BACKGROUNDS[domain])
            cells = {cell['level']: cell for cell in record['cells'] if cell['domain'] == domain}
            grid = [p['k'] for p in cells[1]['points']]
            # The shaded band is the point of the panel: its height at each
            # budget is the share of certified / enumerated modes that no level ever found.
            best = [max(coverage(cells[level], cells[level]['points'][position])
                        for level in LEVELS) for position in range(len(grid))]
            # A single deeper wash rather than a tint plus hatching: the
            # shortfall has to read the same over five different domain
            # backgrounds, and the hatch lines competed with the traces.
            ax.fill_between(grid, best, 1.0, facecolor='#C0392B', alpha=.22,
                            linewidth=0, zorder=1)
            ax.axhline(1.0, color='#7C4B52', linewidth=.7, linestyle=(0, (4, 2.4)), zorder=2)
            # Dashed/open traces over the solid trace expose coincident levels
            # without displacing any observed value from its actual coordinates.
            for level in (2, 3, 1):
                cell, style = cells[level], STYLES[level]
                xs = [p['k'] for p in cell['points']]
                ax.fill_between(xs, [coverage(cell, p, 0) for p in cell['points']],
                                [coverage(cell, p, 1) for p in cell['points']],
                                color=style['color'], alpha=.09, linewidth=0)
                ax.plot(xs, [coverage(cell, p) for p in cell['points']],
                        **style, linewidth=1.15, markerfacecolor=style['color'] if level == 2 else 'none', zorder=3)
            ax.set_xscale('log', base=2)
            ax.set_xlim(.95, draws * 1.15)
            ax.set_ylim(0, 1.17)
            ax.xaxis.set_major_locator(FixedLocator(x_ticks))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
            ax.yaxis.set_major_locator(FixedLocator([0, .25, .5, .75, 1.0]))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            if index:
                ax.tick_params(labelleft=False)
            ax.annotate(support_label(list(cells.values())), (1.08, 1.0), xytext=(0, 3),
                        textcoords='offset points', fontsize=5.7, color='#566574', ha='left', va='bottom',
                        linespacing=1.05)
            # One number per panel says how much the best level still misses.
            ax.annotate(f'{1 - best[-1]:.0%} never\nproduced', (draws, 1.0), xytext=(-2, -3),
                        textcoords='offset points', fontsize=5.9, color='#9E3A2C', ha='right', va='top',
                        linespacing=1.05, zorder=4)
            ax.tick_params(length=2, width=.6, colors='#607487', pad=2)
            ax.grid(axis='y', color='#D8E2EA', linewidth=.45, alpha=.65)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color('#AABAC7')
                ax.spines[side].set_linewidth(.6)
            ax.set_title(domain_title, fontsize=8.5, pad=20)
        handles = [Line2D([], [], **STYLES[level], linewidth=1.15,
                          markerfacecolor=STYLES[level]['color'] if level == 2 else 'none',
                          label=f'Level {level}') for level in LEVELS]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, -.015), ncol=3,
                   frameon=False, fontsize=7.1, handlelength=2.0, handletextpad=.4, columnspacing=1.4)
        if title:
            fig.text(.076, .985, title, ha='left', va='top', fontsize=8, fontweight='bold')
        width = provider_logo(fig, (.076, .063))
        fig.text(.076 + width + .008, .063, record['model'].replace('gpt-5.6-sol', 'GPT-5.6 Sol'),
                 ha='left', va='center', fontsize=7.1, fontweight='bold')
        fig.text(.02, .58, 'Share of certified /\nenumerated modes found', rotation=90, ha='center', va='center', fontsize=7.5)
        fig.text(.54, .155, 'Samples per prompt, $k$', ha='center', va='center', fontsize=7.5)
    try:
        from ops.paper_domain_figure_typography import apply_domain_typography
    except ModuleNotFoundError:
        from paper_domain_figure_typography import apply_domain_typography
    apply_domain_typography(fig)
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
