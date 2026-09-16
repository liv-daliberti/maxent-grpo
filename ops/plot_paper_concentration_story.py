#!/usr/bin/env python3
"""Render the main concentration comparison from the bound saved-output analysis.

No samples are collected or regraded. The two panels have distinct matched
prompt populations and are never joined into an absolute-probability trajectory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'paper/results/conditional_concentration_20260912.json'
DEFAULT_OUTPUT = ROOT / 'paper/figures/concentration_story'
FONT = 7.6  # printed size: the plate is included at its own width.
# Overrides so this plate can be matched to the levels plate when the two
# are printed side by side. None keeps the geometry this figure had alone.
HEIGHT_IN = None
XLIM = None
MARKER = 5.0
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
ALL_SCALES = ('qwen05b', 'falcon1b', 'qwen3b')
# The main plate carries one scale across all five domains. At three scales the
# same plate needs fifteen rows, which is an appendix-sized object; the scale
# comparison is the appendix plate's job (--scales all).
MAIN_SCALE = ('qwen3b',)
SCALES = MAIN_SCALE


def geometry(rows: int) -> dict:
    """Canvas and axes for a plate of ``rows`` domain/scale rows.

    Every row keeps the printed height it had in the original six-row plate, and
    the margins keep their inch sizes rather than their figure fractions, so the
    one-scale and three-scale plates print the same row and the same type.
    """
    row_in = .292
    height = row_in * rows + 0.92
    if HEIGHT_IN is not None:
        height = HEIGHT_IN
        row_in = (HEIGHT_IN - 0.92) / rows
    axes_height = row_in * rows / height
    axes_bottom = .250 * 2.95 / height
    return {
        # Matched to the levels plate so the two sit side by side without
        # one being magnified into larger type by \includegraphics.
        'figsize': (3.95, round(height, 3)),
        'axes_height': axes_height,
        'axes_bottom': axes_bottom,
        'title_y': axes_bottom + axes_height + .082 * 2.95 / height,
        'legend_top_y': axes_bottom + axes_height + .018 * 2.95 / height,
        'xlabel_y': .105 * 2.95 / height,
        'legend_y': -.010,
    }


DOMAIN_LABELS = {'graph_coloring': 'Graph', 'countdown': 'Countdown',
                 'python_factors': 'Python', 'mathir': 'MathIR',
                 'pantry_plan': 'PantryPlan'}
SCALE_LABELS = {'qwen05b': 'Qwen 0.5B', 'falcon1b': 'Falcon 1B', 'qwen3b': 'Qwen 3B'}
FULL_SCALE_LABELS = {'qwen05b': 'Qwen2.5-0.5B', 'falcon1b': 'Falcon3-1B', 'qwen3b': 'Qwen2.5-3B'}
# One panel, not two. The replay-minus-control panel showed the same contrast
# Figure~\ref{fig:retention-comparator-matrix} shows across every scale and
# domain, on a narrower slice of it; this figure is kept for the thing nothing
# else in the paper measures, the change *within* a training run.
PANELS = (
    ('before_after', ('drgrpo', 'grpo', 'maxrl'), 'Final − initial'),
)
METHOD_LABELS = {'drgrpo': 'Dr.GRPO', 'grpo': 'GRPO', 'maxrl': 'MaxRL'}


def _geo() -> dict:
    return geometry(len(DOMAINS) * len(SCALES))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _summary(summary: dict[str, Any]) -> dict[str, Any]:
    counts = list(summary['eligible_counts'].values())
    values = list(summary['values'].values())
    if summary['n'] != len(values) or any(not math.isfinite(x) for x in values):
        raise ValueError('invalid source seed summary')
    return {
        'mean': summary['mean'], 'ci95': summary['ci95'], 'n': summary['n'],
        'registered_n': summary['registered_n'], 'seed_estimates': deepcopy(summary['values']),
        'eligible_counts': deepcopy(summary['eligible_counts']),
        'eligible_count_range': [min(counts), max(counts)] if counts else None,
        'coverage_range': summary['coverage_range'],
        'delta_percentage_points': None if summary['mean'] is None else 100 * summary['mean'],
        'ci95_percentage_points': None if summary['ci95'] is None else [100 * x for x in summary['ci95']],
        'uncertainty': summary['uncertainty'],
    }


def build_metadata(source_path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source_path = Path(source_path)
    raw = source_path.read_bytes()
    record = json.loads(raw)
    if record.get('schema') != 'paper-conditional-concentration-v1' or record.get('status') != 'analyzed':
        raise ValueError('expected the analyzed saved-output concentration artifact')
    indexed = {(b['level'], b['scale'], b['domain'], b['kind'], b['method']): b for b in record['blocks']}
    if len(indexed) != len(record['blocks']):
        raise ValueError('duplicate source comparison block')
    displayed = []
    for kind, methods, _ in PANELS:
        for domain in DOMAINS:
            for scale in SCALES:
                for method in methods:
                    key = ('level1', scale, domain, kind, method)
                    if key not in indexed:
                        raise ValueError(f'missing prespecified main-figure comparison: {key}')
                    block = indexed[key]
                    primary = _summary(block['summaries']['distinct_streams'])
                    # Dr.GRPO and GRPO are complete at five seeds everywhere they
                    # are drawn; MaxRL is not, so a block that the source admits
                    # with fewer is drawn open and without an interval rather
                    # than dropped. Dropping it would hide which comparisons the
                    # cohort actually covers.
                    complete = primary['n'] == 5 and primary['ci95'] is not None
                    defined = primary['mean'] is not None
                    for seed, seed_data in block['per_seed'].items():
                        if seed_data['distinct_streams']['n_total'] != 128:
                            raise ValueError(f'prompt denominator changed: {key}, seed {seed}')
                    displayed.append({
                        'level': 'level1', 'scale': scale, 'domain': domain,
                        'kind': kind, 'method': method,
                        'block_id': '/'.join(key), 'admitted_seeds': block['admitted_seeds'],
                        'complete': complete, 'defined': defined,
                        'primary': primary,
                        'orientation0': _summary(block['summaries']['orientation0']),
                        'orientation1': _summary(block['summaries']['orientation1']),
                        'original_metrics_on_primary_eligible': deepcopy(
                            block['summaries']['distinct_streams']['original_metrics_on_eligible']),
                    })
    # Drawn positions, not magnitudes: the axis is negated once at draw time so
    # left reads narrower, and the range should follow where the marks actually
    # land. A symmetric range spends half the plate on a region the measurements
    # never reach.
    drawn = []
    for block in displayed:
        if not block['defined']:
            continue
        for name in ('primary', 'orientation0', 'orientation1'):
            value = block[name]['delta_percentage_points']
            if value is not None:
                drawn.append(-value)
        drawn.extend(-value
                     for value in (block['primary']['ci95_percentage_points'] or []))
    # The narrowing side is sized to the data. The broadening side keeps a
    # visible margin even when nothing reaches it, so "almost nothing broadened"
    # stays a thing the reader can see rather than a side that was cropped away.
    low = -10 * math.ceil((max(0.0, -min(drawn)) + 3) / 10)
    high = 10 * math.ceil((max(0.0, max(drawn)) + 3) / 10)
    high = max(high, int(round(abs(low) * .28 / 10)) * 10, 10)
    limit = high
    limits = [
        'Graph and PantryPlan were the existing initial-breadth illustration domains: their initial checkpoints permit observable correct-key breadth. This is not selection by the signs of the concentration effects. All five domains and all 135 defined/undefined blocks remain in the linked appendix report.',
        'Each point averages per-prompt correct-key pair-collision changes over prompts with at least two correct representatives in both conditions, then averages five measured seed estimates. The population differs across contrasts and sensitivity orientations.',
        'The 32 saved positions correspond to 11 nominal child-seed streams under the audited vLLM V0 n=8 mapping. Deterministic earliest draw/output representatives were selected without using output values. The mapping remains conditional where historical runtime identity is unavailable.',
        record['stream_source_audit']['historical_source_limit'],
        'Distinct seed identifiers do not establish iid sampling. The usual per-prompt collision U-statistic identity requires iid correct labels. Shared RNG and joint eligibility make these comparisons descriptive; no iid prompt-population inference is claimed.',
        'The two disjoint nominal-stream orientations are no longer drawn; they reduce same-prompt cross-condition stream overlap, use their own eligible populations, and are retained in the linked appendix report.',
        'Intervals are nominal unadjusted 95% intervals over five measured seed estimates, not prompt-level confidence guarantees.',
        'The left and right panels use separately matched eligible populations. Their estimates cannot be joined into one absolute-collision trajectory or subtracted as if their populations were equal.',
        'Replay-associated reductions in collision coexist with changes in correctness. All displayed replay Graph/PantryPlan blocks lower eligible-prompt mean@8; the linked report retains simultaneous original K8 metrics.',
        'Training collision weights eligible prompts equally; hosted collision pools correct-pair counts. Those different aggregation weights are intentional.',
        'The completed-cohort extension was declared after the initial concentration results were inspected. It adds all newly admitted source cases and withdraws source conflicts according to the documented amended census.',
    ]
    caption = (
        'Training-induced concentration and matched replay effects on Level-1 Graph and PantryPlan, '
        'the two existing initial-breadth illustration domains. Left: final minus initial for Dr.GRPO and GRPO. '
        'Right: replay minus its matched terminal control for Dr.GRPO and MaxRL. '
        'Symbols show equal-prompt changes in pairwise correct-mode diversity, the negated collision change, so that left is narrower and right is broader in both panels; bars are nominal unadjusted 95% intervals over five measured seeds. '
        
        'The 32 saved positions map to 11 nominal child-seed streams under the conditional historical vLLM mapping; '
        'distinct seed IDs do not prove iid sampling, and shared-RNG eligibility makes these descriptive comparisons. '
        'The panels have different matched populations and do not form one absolute trajectory. '
        'Correctness is not held fixed. All five domains, undefined blocks, correctness tradeoffs, source assumptions '
        'and the retrospective completed-cohort extension are retained in Appendix app:conditional-concentration '
        'and paper/results/conditional_concentration_20260912/report.md.'
    )
    return {
        'schema': 'paper-concentration-story-v1',
        'source': {'path': _relative(source_path), 'sha256': hashlib.sha256(raw).hexdigest(),
                   'schema': record['schema'], 'analysis_code_sha256': record['analysis_code_sha256']},
        'renderer': {'path': _relative(Path(__file__)), 'sha256': _sha256(Path(__file__))},
        'style_source': {'path': 'ops/paper_style.py', 'sha256': _sha256(ROOT / 'ops/paper_style.py')},
        'cohort_extension': deepcopy(record.get('cohort_extension')),
        'appendix': {'label': 'app:conditional-concentration',
                     'all_domain_report': 'paper/results/conditional_concentration_20260912/report.md',
                     'tradeoffs': 'paper/results/conditional_concentration_20260912/metric_tradeoffs.md',
                     'analyzed_block_count': len(record['blocks'])},
        'display': {'domains': list(DOMAINS), 'scales': list(SCALES),
                    'full_scale_labels': FULL_SCALE_LABELS, 'prompt_denominator': 128,
                    'blocks': displayed, 'limits': limits, 'caption': caption},
        'figure': {'size_inches': list(_geo()['figsize']), 'minimum_font_points': FONT,
                   'minimum_font_at_5p5_in_width': FONT * 5.5 / _geo()['figsize'][0],
                   'scales_drawn': list(SCALES),
                   'x_limits_percentage_points': list(XLIM) if XLIM else [low, high],
                   'layout': 'Separate training and replay panels; six domain/scale rows and two methods per row.',
                   'primary_marks': {'drgrpo': 'circle', 'grpo': 'square', 'maxrl': 'diamond'},
                   'sensitivity_marks': {'orientation0': 'open left triangle', 'orientation1': 'open right triangle'},
                   'gutter': 'Defined seed n | minimum–maximum primary-eligible prompt count, out of 128.'},
    }


def _count_label(summary: dict[str, Any]) -> str:
    low, high = summary['eligible_count_range']
    counts = str(low) if low == high else f'{low}–{high}'
    return f"{summary['n']} | {counts}"


def build_figure(source_path: Path = DEFAULT_SOURCE):
    """Return (Figure, provenance/measurement metadata) without writing artifacts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if str(ROOT / 'ops') not in sys.path:
        sys.path.insert(0, str(ROOT / 'ops'))
    import paper_style as style

    metadata = build_metadata(source_path)
    blocks = {(b['kind'], b['domain'], b['scale'], b['method']): b for b in metadata['display']['blocks']}
    colors = {'drgrpo': style.CONTROL, 'grpo': style.METHOD, 'maxrl': style.COMPARATOR}
    markers = {'drgrpo': 'o', 'grpo': 's', 'maxrl': 'D'}
    metadata['figure']['method_colors'] = colors
    rc = {'font.family': 'DejaVu Sans', 'font.size': FONT, 'axes.titlesize': FONT,
          'axes.labelsize': FONT, 'xtick.labelsize': FONT, 'ytick.labelsize': FONT,
          'text.color': style.INK, 'axes.labelcolor': style.INK,
          'xtick.color': style.MUTED, 'ytick.color': style.INK,
          'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(rc):
        g = _geo()
        fig = plt.figure(figsize=g['figsize'])
        # Stacked rather than side by side: in a wrapped column two panels
        # abreast leave no room for the scale labels. The count gutters are gone
        # too; those numbers live in the caption and the linked report.
        # The gutter carries a rotated domain label and a scale name side by
        # side; at the old .330 of a narrower canvas they overlapped.
        axes = [fig.add_axes((.300, g['axes_bottom'], .655, g['axes_height']))]
        for panel_index, (kind, methods, title) in enumerate(PANELS):
            ax = axes[panel_index]
            ax.set_xlim(metadata['figure']['x_limits_percentage_points'])
            ax.set_ylim(-.60, len(DOMAINS) * len(SCALES) - .40)
            lo, hi = metadata['figure']['x_limits_percentage_points']
            ax.set_xticks((lo, 0, hi),
                          (f'\u2212{abs(lo):g}', '0', f'+{hi:g}'))
            ax.set_yticks([])
            low, high = metadata['figure']['x_limits_percentage_points']
            ax.axvspan(low, 0, color='#FBEFEF', zorder=0, lw=0)
            ax.axvspan(0, high, color='#EDF6EE', zorder=0, lw=0)
            ax.axvline(0, color=style.MUTED, linewidth=.65, zorder=1)
            for edge in range(1, len(DOMAINS)):
                ax.axhline(len(DOMAINS) * len(SCALES) - 1 - edge * len(SCALES) + .5,
                           # Neutral grey, not comparator blue: the rule is
                           # structure, and in blue it reads as a MaxRL mark.
                           color='#5B6066',
                           linewidth=1.5 if len(SCALES) > 1 else 1.1, zorder=3)
            for group_index, (domain, scale) in enumerate((d, s) for d in DOMAINS for s in SCALES):
                y = len(DOMAINS) * len(SCALES) - 1 - group_index

                span = .29 if len(methods) > 2 else .26
                for method_index, method in enumerate(methods):
                    offset = (0 if len(methods) == 1
                              else span - 2 * span * method_index / (len(methods) - 1))
                    row_y = y + offset
                    block = blocks[(kind, domain, scale, method)]
                    if not block['defined']:
                        # Draw the objective's own shape, unfilled, in the
                        # paper's convention for a measurement that has no
                        # height: the reader sees which objectives could not be
                        # measured, and the mark sits in the row rather than at
                        # a value it does not have.
                        undefined = [m for m in methods
                                     if not blocks[(kind, domain, scale, m)]['defined']]
                        if method == undefined[0]:
                            for slot, m in enumerate(undefined):
                                ax.plot(.055 + slot * .045, y, marker=markers[m],
                                        transform=ax.get_yaxis_transform(),
                                        markersize=MARKER, markeredgewidth=.9,
                                        markerfacecolor='none', markeredgecolor=colors[m],
                                        linestyle='none', zorder=6, clip_on=False)
                            ax.text(.055 + len(undefined) * .045 + .015, y,
                                    'initial solves <2 per prompt',
                                    transform=ax.get_yaxis_transform(),
                                    fontsize=FONT - 3.0, color=style.MUTED,
                                    ha='left', va='center', style='italic', zorder=6)
                        continue
                    primary = block['primary']
                    # Collision is negated once, here, so the axis reads as
                    # diversity: left is narrower, right is broader, in both
                    # panels. Every other figure in the paper reads that way.
                    if primary['ci95_percentage_points'] is not None:
                        interval = [-value for value in reversed(primary['ci95_percentage_points'])]
                        ax.plot(interval, [row_y, row_y], color=colors[method],
                                linewidth=1.3, zorder=2)
                    face = colors[method] if block['complete'] else 'none'
                    ax.plot(-primary['delta_percentage_points'], row_y, marker=markers[method],
                            markersize=MARKER, markeredgewidth=.9, markerfacecolor=face,
                            markeredgecolor=colors[method], linestyle='none', zorder=4)
                # One scale per row means the scale name is the same on every
                # row, so the row is named by what actually varies.
                label = DOMAIN_LABELS[domain] if len(SCALES) == 1 else SCALE_LABELS[scale]
                ax.text(-.055, y, label, transform=ax.get_yaxis_transform(),
                        fontsize=FONT, ha='right', va='center', clip_on=False)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(style.GRID)
            ax.spines['bottom'].set_linewidth(.7)
            ax.tick_params(axis='x', length=2.4, width=.6, pad=2)
            handles = [Line2D([], [], marker=markers[m], linestyle='none', markersize=MARKER - .5,
                              color=colors[m], label=METHOD_LABELS[m]) for m in methods]
            fig.legend(handles=handles, loc='lower center',
                       bbox_to_anchor=(.640, g['legend_top_y']), ncol=len(methods),
                       frameon=False, fontsize=FONT - 1.1, handletextpad=.3,
                       columnspacing=.85, borderaxespad=0)
        rows = len(DOMAINS) * len(SCALES)
        if len(SCALES) > 1:
            for domain_index, domain in enumerate(DOMAINS):
                centre = rows - 1 - domain_index * len(SCALES) - (len(SCALES) - 1) / 2
                fig.text(.042, g['axes_bottom'] + g['axes_height'] * (centre + .60) / (rows + .20),
                         DOMAIN_LABELS[domain], fontsize=FONT - .6, fontweight='bold',
                         rotation=90, ha='center', va='center')
        # Two lines: the single-line form is wider than the wrapped canvas.
        fig.text(.640, g['xlabel_y'],
                 'Δ PCMD (pp)\n← same answer    different answers →',
                 fontsize=FONT - 1.4, ha='center', va='center', linespacing=1.85)

        # Reject accidental invisible truncation before any publication file is written.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        canvas = fig.bbox
        for text in fig.findobj(match=matplotlib.text.Text):
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < canvas.x0-1 or box.x1 > canvas.x1+1 or box.y0 < canvas.y0-1 or box.y1 > canvas.y1+1:
                raise ValueError(f'text falls outside fixed publication canvas: {text.get_text()!r}')
    return fig, metadata


def render(source_path: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, metadata = build_figure(source_path)
    bindings = {}
    for suffix in ('.pdf', '.png'):
        path = output.with_suffix(suffix)
        options = {'dpi': 250} if suffix == '.png' else {'metadata': {'CreationDate': None, 'ModDate': None}}
        fig.savefig(path, facecolor='white', **options)
        bindings[suffix[1:]] = {'path': _relative(path), 'sha256': _sha256(path)}
    plt.close(fig)
    if _sha256(Path(source_path)) != metadata['source']['sha256']:
        raise ValueError('source changed during figure rendering')
    metadata['outputs'] = bindings
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--height', type=float, default=None)
    parser.add_argument('--xlow', type=float, default=None)
    parser.add_argument('--xhigh', type=float, default=None)
    parser.add_argument('--markersize', type=float, default=5.0)
    parser.add_argument('--scales', choices=('main', 'all'), default='main',
                        help="'main' draws Qwen2.5-3B alone; 'all' is the appendix plate")
    args = parser.parse_args()
    global SCALES, HEIGHT_IN, XLIM, MARKER
    SCALES = ALL_SCALES if args.scales == 'all' else MAIN_SCALE
    HEIGHT_IN = args.height
    XLIM = (args.xlow, args.xhigh) if args.xlow is not None else None
    MARKER = args.markersize
    metadata = render(args.source, args.output)
    print(json.dumps({'output': str(args.output), 'displayed_blocks': len(metadata['display']['blocks']),
                      'size_inches': _geo()['figsize'], 'source_sha256': metadata['source']['sha256']}))


if __name__ == '__main__':
    main()
