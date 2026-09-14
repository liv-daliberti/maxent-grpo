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
DEFAULT_SOURCE = ROOT / 'paper/results/conditional_concentration_20260911.json'
DEFAULT_OUTPUT = ROOT / 'paper/figures/concentration_story'
FIGSIZE = (6.7, 3.1)
FONT = 9.75  # 8.0 pt when a 6.7-in canvas is included at 5.5-in text width.
DOMAINS = ('graph_coloring', 'pantry_plan')
SCALES = ('qwen05b', 'falcon1b', 'qwen3b')
DOMAIN_LABELS = {'graph_coloring': 'Graph', 'pantry_plan': 'PantryPlan'}
SCALE_LABELS = {'qwen05b': 'Qwen 0.5B', 'falcon1b': 'Falcon 1B', 'qwen3b': 'Qwen 3B'}
FULL_SCALE_LABELS = {'qwen05b': 'Qwen2.5-0.5B', 'falcon1b': 'Falcon3-1B', 'qwen3b': 'Qwen2.5-3B'}
PANELS = (
    ('before_after', ('drgrpo', 'grpo'), '(a) Training: final − initial'),
    ('replay_effect', ('drgrpo', 'maxrl'), '(b) Replay − control'),
)
METHOD_LABELS = {'drgrpo': 'Dr.GRPO', 'grpo': 'GRPO', 'maxrl': 'MaxRL'}


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
                    if primary['n'] != 5 or primary['ci95'] is None:
                        raise ValueError(f'main-figure five-seed census changed: {key}')
                    for seed, seed_data in block['per_seed'].items():
                        if seed_data['distinct_streams']['n_total'] != 128:
                            raise ValueError(f'prompt denominator changed: {key}, seed {seed}')
                    displayed.append({
                        'level': 'level1', 'scale': scale, 'domain': domain,
                        'kind': kind, 'method': method,
                        'block_id': '/'.join(key), 'admitted_seeds': block['admitted_seeds'],
                        'primary': primary,
                        'orientation0': _summary(block['summaries']['orientation0']),
                        'orientation1': _summary(block['summaries']['orientation1']),
                        'original_metrics_on_primary_eligible': deepcopy(
                            block['summaries']['distinct_streams']['original_metrics_on_eligible']),
                    })
    displayed_values = []
    for block in displayed:
        for name in ('primary', 'orientation0', 'orientation1'):
            value = block[name]['delta_percentage_points']
            if value is not None:
                displayed_values.append(abs(value))
        displayed_values.extend(abs(value) for value in block['primary']['ci95_percentage_points'])
    limit = 10 * math.ceil((max(displayed_values) + 3) / 10)
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
        'Symbols show equal-prompt changes in pairwise modal diversity, the negated collision change, so that left is narrower and right is broader in both panels; bars are nominal unadjusted 95% intervals over five measured seeds. '
        'Gutters give defined seed n and the range of jointly eligible prompt counts, each out of 128. '
        'The 32 saved positions map to 11 nominal child-seed streams under the conditional historical vLLM mapping; '
        'distinct seed IDs do not prove iid sampling, and shared-RNG eligibility makes these descriptive comparisons. '
        'The panels have different matched populations and do not form one absolute trajectory. '
        'Correctness is not held fixed. All five domains, undefined blocks, correctness tradeoffs, source assumptions '
        'and the retrospective completed-cohort extension are retained in Appendix app:conditional-concentration '
        'and paper/results/conditional_concentration_20260911/report.md.'
    )
    return {
        'schema': 'paper-concentration-story-v1',
        'source': {'path': _relative(source_path), 'sha256': hashlib.sha256(raw).hexdigest(),
                   'schema': record['schema'], 'analysis_code_sha256': record['analysis_code_sha256']},
        'renderer': {'path': _relative(Path(__file__)), 'sha256': _sha256(Path(__file__))},
        'style_source': {'path': 'ops/paper_style.py', 'sha256': _sha256(ROOT / 'ops/paper_style.py')},
        'cohort_extension': deepcopy(record.get('cohort_extension')),
        'appendix': {'label': 'app:conditional-concentration',
                     'all_domain_report': 'paper/results/conditional_concentration_20260911/report.md',
                     'tradeoffs': 'paper/results/conditional_concentration_20260911/metric_tradeoffs.md',
                     'analyzed_block_count': len(record['blocks'])},
        'display': {'domains': list(DOMAINS), 'scales': list(SCALES),
                    'full_scale_labels': FULL_SCALE_LABELS, 'prompt_denominator': 128,
                    'blocks': displayed, 'limits': limits, 'caption': caption},
        'figure': {'size_inches': list(FIGSIZE), 'minimum_font_points': FONT,
                   'minimum_font_at_5p5_in_width': FONT * 5.5 / FIGSIZE[0],
                   'x_limits_percentage_points': [-limit, limit],
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
    colors = {'drgrpo': style.CONTROL, 'grpo': style.COMPARATOR, 'maxrl': style.COMPARATOR}
    markers = {'drgrpo': 'o', 'grpo': 's', 'maxrl': 'D'}
    metadata['figure']['method_colors'] = colors
    rc = {'font.family': 'DejaVu Sans', 'font.size': FONT, 'axes.titlesize': FONT,
          'axes.labelsize': FONT, 'xtick.labelsize': FONT, 'ytick.labelsize': FONT,
          'text.color': style.INK, 'axes.labelcolor': style.INK,
          'xtick.color': style.MUTED, 'ytick.color': style.INK,
          'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=FIGSIZE)
        axes = [fig.add_axes((left, .20, .245, .59)) for left in (.155, .60)]
        counts_x = (.463, .910)
        for panel_index, (kind, methods, title) in enumerate(PANELS):
            ax = axes[panel_index]
            ax.set_xlim(metadata['figure']['x_limits_percentage_points'])
            ax.set_ylim(-.60, 5.60)
            ax.set_xticks((-100, 0, 100), ('−100', '0', '+100'))
            ax.set_yticks([])
            ax.axvline(0, color=style.MUTED, linewidth=.65, zorder=1)
            ax.axhline(2.5, color=style.GRID, linewidth=.7, zorder=1)
            for group_index, (domain, scale) in enumerate((d, s) for d in DOMAINS for s in SCALES):
                y = 5 - group_index
                if group_index % 2 == 0:
                    ax.axhspan(y-.47, y+.47, color='#F3F6F9', zorder=0)
                for method_index, method in enumerate(methods):
                    row_y = y + (.26 if method_index == 0 else -.26)
                    block = blocks[(kind, domain, scale, method)]
                    primary = block['primary']
                    # Collision is negated once, here, so the axis reads as
                    # diversity: left is narrower, right is broader, in both
                    # panels. Every other figure in the paper reads that way.
                    interval = [-value for value in reversed(primary['ci95_percentage_points'])]
                    ax.plot(interval, [row_y, row_y], color=colors[method], linewidth=1.3, zorder=2)
                    ax.plot(-primary['delta_percentage_points'], row_y, marker=markers[method],
                            markersize=5.0, markeredgewidth=.5, color=colors[method], linestyle='none', zorder=4)
                    fig.text(counts_x[panel_index], .20 + .59 * (row_y + .60) / 6.20,
                             _count_label(primary), fontsize=FONT, ha='center', va='center', color=colors[method])
                if panel_index == 0:
                    ax.text(-.055, y, SCALE_LABELS[scale], transform=ax.get_yaxis_transform(),
                            fontsize=FONT, ha='right', va='center', clip_on=False)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(style.GRID)
            ax.spines['bottom'].set_linewidth(.7)
            ax.tick_params(axis='x', length=2.4, width=.6, pad=2)
            fig.text((.155, .60)[panel_index], .98, title, fontsize=FONT+.3, fontweight='bold', ha='left', va='top')
            handles = [Line2D([], [], marker=markers[m], linestyle='none', markersize=4.5,
                              color=colors[m], label=METHOD_LABELS[m]) for m in methods]
            fig.legend(handles=handles, loc='upper left', bbox_to_anchor=((.145, .59)[panel_index], .922),
                       ncol=2, frameon=False, fontsize=FONT, handletextpad=.35, columnspacing=.9, borderaxespad=0)
            fig.text(counts_x[panel_index], .815, 'n | prompts', fontsize=FONT, ha='center', va='bottom')
        for label, y in (('Graph', 4), ('PantryPlan', 1)):
            fig.text(.014, .20 + .59 * (y + .60) / 6.20, label, fontsize=FONT,
                     fontweight='bold', rotation=90, ha='center', va='center')
        fig.text(.53, .102, 'Δ pairwise modal diversity (pp):  ← narrower    broader →',
                 fontsize=FONT, ha='center', va='center')
        handles = [Line2D([], [], marker='o', color=style.INK, markersize=4.5,
                          linewidth=1.2, label='Mean; 95% seed CI')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.535, -.008),
                   ncol=1, frameon=False, fontsize=FONT, handletextpad=.4, columnspacing=.9)
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
    args = parser.parse_args()
    metadata = render(args.source, args.output)
    print(json.dumps({'output': str(args.output), 'displayed_blocks': len(metadata['display']['blocks']),
                      'size_inches': FIGSIZE, 'source_sha256': metadata['source']['sha256']}))


if __name__ == '__main__':
    main()
