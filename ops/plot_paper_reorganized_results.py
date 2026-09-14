#!/usr/bin/env python3
"""Compact replay figures from frozen paired-effect summaries only.

This renderer never reads live endpoints, recomputes an interval, resamples a
seed, or changes source cohorts. ``build_figures()`` returns a mapping from
output stem to ``(matplotlib.figure.Figure, metadata)``; the three individual
``build_*`` functions expose the same pair for isolated reconstruction.
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
sys.path.insert(0, str(ROOT / 'ops'))
import paper_style as style

DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LABELS = ('Graph', 'Countdown', 'Python', 'MathIR', 'Pantry')
MODELS = (
    ('qwen05b', 'Qwen2.5-0.5B', [43, 44, 45, 46, 47]),
    ('falcon1b', 'Falcon3-1B', [55, 56, 57, 58, 59]),
    ('qwen3b', 'Qwen2.5-3B', [70, 71, 72, 73, 74]),
)
FONT = 9.75  # 8.0 pt at the manuscript's 5.5-in text width.
FACTORIAL_SIZE = (6.7, 3.1)
COMPACT_SIZE = (6.7, 1.8)
SOURCES = {
    'drgrpo': 'paper/figures/experiment1_retention_comparator_matrix.json',
    'maxrl': 'paper/figures/e118_all_scale_factorial_progress.json',
    'weighting': 'paper/results/e120_primary_breadth.json',
    'level2': 'paper/results/level2_factorial_contrasts_20260912.json',
}
OBJECTIVES = {
    'drgrpo': {'label': 'Dr.GRPO', 'color': style.CONTROL, 'marker': 'o', 'offset': -.16},
    'maxrl': {'label': 'MaxRL', 'color': style.COMPARATOR, 'marker': 's', 'offset': .16},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(source: str, root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = root / source
    raw = path.read_bytes()
    return json.loads(raw), {'path': source, 'sha256': hashlib.sha256(raw).hexdigest()}


def _summary(raw: dict[str, Any], seeds: list[int], interval_key: str,
             interval_type: str) -> dict[str, Any]:
    """Copy source numbers exactly, checking their declared seed population."""
    n = raw.get('n', len(seeds))
    if n != len(seeds) or len(set(seeds)) != n:
        raise ValueError('inconsistent paired seed count')
    interval = deepcopy(raw.get(interval_key))
    if n == 5 and interval is None:
        raise ValueError('missing frozen five-seed interval')
    if n < 5 and interval is not None:
        raise ValueError('partial block unexpectedly has an interval')
    values = [raw['mean']] + ([] if interval is None else interval)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('nonfinite frozen summary')
    if interval is not None and not interval[0] <= raw['mean'] <= interval[1]:
        raise ValueError('source mean outside interval')
    out = {'mean': raw['mean'], 'ci95': interval, 'n': n, 'seeds': deepcopy(seeds),
           'interval_type': interval_type if interval is not None else None,
           'partial': n < 5}
    if 'per_seed' in raw:
        if set(raw['per_seed']) != {str(s) for s in seeds}:
            raise ValueError('summary seed identities differ from source cohort')
        out['per_seed'] = deepcopy(raw['per_seed'])
    return out


def _metadata(stem: str, size: tuple[float, float], sources: list[dict[str, str]],
              rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'schema': 'paper-reorganized-replay-figure-v1', 'figure': stem,
        'builder': {'path': 'ops/plot_paper_reorganized_results.py', 'sha256': _sha(Path(__file__))},
        'sources': sources, 'source_sha256': {s['path']: s['sha256'] for s in sources},
        'figure_size_inches': list(size), 'minimum_font_pt': FONT,
        'terminal_step': 3072, 'sample_budget_k': 8,
        'domain_order': list(DOMAINS),
        'display_metrics': {
            'pass8': {'label': 'pass@8', 'unit': 'probability points', 'source_scale_multiplier': 100},
            'distinct8': {'label': 'distinct@8', 'unit': 'expected verified keys', 'source_scale_multiplier': 1},
            'breadth8': {'label': 'extra@8 = distinct@8 - pass@8', 'unit': 'expected extra verified keys', 'source_scale_multiplier': 1},
        },
        'inference_note': 'Intervals copied exactly from frozen summaries; unadjusted descriptive estimation intervals, not equivalence tests or simultaneous intervals.',
        'rows': rows,
    }


def factorial_metadata(root: Path = ROOT) -> dict[str, Any]:
    dr, dr_source = _load(SOURCES['drgrpo'], root)
    mx, mx_source = _load(SOURCES['maxrl'], root)
    if dr['panel_a']['description'] != 'Re:Dr.GRPO minus Dr.GRPO across scale':
        raise ValueError('Dr.GRPO source contrast changed')
    rows = []
    counts = {'drgrpo': 0, 'maxrl': 0}
    for scale, model, registered in MODELS:
        for domain in DOMAINS:
            dr_cell = dr['panel_a']['cells'][model][domain]
            mx_cell = mx['cells'][scale][domain]
            for method, cell, seeds in (
                ('drgrpo', dr_cell, dr_cell['seeds']),
                ('maxrl', mx_cell['replay_maxrl_minus_maxrl'], mx_cell['matched_seeds']),
            ):
                expected = registered[:-1] if (method, scale, domain) == ('drgrpo', 'falcon1b', 'countdown') else registered
                if seeds != expected:
                    raise ValueError(f'frozen primary cohort changed: {method}/{scale}/{domain}')
                counts[method] += len(seeds)
                summaries = {}
                for metric in ('pass8', 'distinct8'):
                    summary = _summary(cell['summaries'][metric], seeds, 'student_t_95', 'paired Student-t 95%')
                    per_seed = cell.get('per_seed', {})
                    if per_seed:
                        if set(per_seed) != {str(s) for s in seeds}:
                            raise ValueError('paired-effect source seed identities changed')
                        summary['per_seed'] = {str(s): per_seed[str(s)][metric] for s in seeds}
                    summaries[metric] = summary
                rows.append({'scale': scale, 'model': model, 'domain': domain, 'objective': method,
                             'contrast': f'replay_{method} minus {method}', 'summaries': summaries})
    if counts != {'drgrpo': 74, 'maxrl': 75}:
        raise ValueError('primary terminal census changed')
    out = _metadata('replay_factorial_effects', FACTORIAL_SIZE, [dr_source, mx_source], rows)
    out.update({'scope': 'Level 1 matched terminal replay effects; each objective retains its own source-admitted paired seeds.',
                'paired_seed_counts': counts, 'plotted_metrics': ['pass8', 'distinct8'],
                'partial_blocks': [{'scale': 'falcon1b', 'domain': 'countdown', 'objective': 'drgrpo',
                                    'seeds': [55, 56, 57, 58], 'n': 4, 'ci95': None}],
                'contrast_direction': 'replay minus the same base objective',
                'source_endpoint_snapshot': deepcopy(mx['endpoint_snapshot'])})
    return out


def weighting_metadata(root: Path = ROOT) -> dict[str, Any]:
    data, source = _load(SOURCES['weighting'], root)
    if data['schema'] != 'e120-primary-breadth-frozen-v1' or data['seeds'] != [43, 44, 45, 46, 47]:
        raise ValueError('frozen E120 primary contract changed')
    if data['contrast'] != 'uniform key-balanced replay minus fresh-frequency replay':
        raise ValueError('weighting contrast direction changed')
    rows = []
    for domain in DOMAINS:
        cell = data['rows'][domain]
        if cell['n'] != 5:
            raise ValueError('incomplete E120 primary block')
        summaries = {metric: _summary(cell['uniform_minus_frequency'][metric], data['seeds'],
                                      'paired_bootstrap_percentile_95', 'paired-seed percentile bootstrap 95%')
                     for metric in ('pass8', 'breadth8')}
        rows.append({'scale': 'qwen05b', 'model': 'Qwen2.5-0.5B', 'domain': domain,
                     'contrast': data['contrast'], 'summaries': summaries})
    out = _metadata('replay_key_weighting', COMPACT_SIZE, [source], rows)
    out.update({'scope': data['scope'], 'contrast_direction': data['contrast'],
                'bootstrap': deepcopy(data['bootstrap']), 'plotted_metrics': ['pass8', 'breadth8'],
                'paired_seed_counts_by_domain': {domain: 5 for domain in DOMAINS},
                'distinct_paired_seeds': deepcopy(data['seeds']),
                'extra8_definition': 'distinct@8 minus pass@8; extra verified keys beyond the first. This quantity remains coupled to correctness.',
                'registration': {'path': data['preregistration'], 'sha256': data['preregistration_sha256']}})
    return out


def level2_metadata(root: Path = ROOT) -> dict[str, Any]:
    data, source = _load(SOURCES['level2'], root)
    if data['schema'] != 'paper-level2-factorial-contrasts-v1' or data['complete_five_seed_blocks'] != 5:
        raise ValueError('frozen Level 2 contract changed')
    blocks = {b['domain']: b for b in data['blocks']}
    if len(blocks) != len(data['blocks']) or set(blocks) != set(DOMAINS):
        raise ValueError('Level 2 domain census changed')
    rows = []
    for domain in DOMAINS:
        block = blocks[domain]
        seeds = block['paired_seeds']
        if seeds != [43, 44, 45, 46, 47]:
            raise ValueError('Level 2 four-arm paired intersection changed')
        for method in OBJECTIVES:
            contrast = f'replay_{method}_minus_{method}'
            summaries = {metric: _summary(block['contrasts'][contrast]['summaries'][metric], seeds,
                                          'student_t_95', 'paired Student-t 95%')
                         for metric in ('pass8', 'distinct8')}
            rows.append({'scale': 'qwen05b', 'model': data['model'], 'domain': domain, 'objective': method,
                         'contrast': contrast, 'terminal_seeds_by_arm': deepcopy(block['terminal_seeds_by_arm']),
                         'summaries': summaries})
    out = _metadata('replay_level2_effects', COMPACT_SIZE, [source], rows)
    out.update({'scope': data['scope'], 'level': 2,
                'contrast_direction': 'replay minus the same base objective within Level 2',
                'cohort_rule': 'all-four-arm paired seed intersection within each domain',
                'paired_seed_counts': {'drgrpo': 21, 'maxrl': 21}, 'plotted_metrics': ['pass8', 'distinct8'],
                'partial_blocks': [{'domain': 'pantry_plan', 'seeds': [43], 'n': 1, 'ci95': None}],
                'interpretation': 'A within-Level-2 replay comparison; no cross-level absolute-mean contrast or isolated difficulty intervention.',
                'source_endpoint_snapshot': deepcopy(data['source_audit'])})
    return out


def _plot_context():
    import matplotlib as mpl
    mpl.use('Agg')
    return mpl.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': FONT, 'axes.labelsize': FONT,
        'axes.titlesize': FONT + .5, 'xtick.labelsize': FONT, 'ytick.labelsize': FONT,
        'legend.fontsize': FONT, 'text.color': style.INK, 'axes.labelcolor': style.INK,
        'xtick.color': style.MUTED, 'ytick.color': style.INK,
        'axes.edgecolor': style.GRID, 'axes.linewidth': .6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def _forest_axis(ax, labels: bool = True):
    ax.set_ylim(4.55, -.55)
    ax.set_yticks(range(5), LABELS if labels else [''] * 5)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.tick_params(axis='x', length=2.5, pad=2)
    for spine in ('left', 'right', 'top'):
        ax.spines[spine].set_visible(False)
    ax.axvline(0, color=style.MUTED, linewidth=.75, zorder=1)
    ax.grid(axis='x', color=style.GRID, linewidth=.5, zorder=0)
    for y in (0, 2, 4):
        ax.axhspan(y - .46, y + .46, color='#F4F7F9', lw=0, zorder=0)


def _point(ax, summary, y, metric, *, color, marker):
    multiplier = 100 if metric == 'pass8' else 1
    value = summary['mean'] * multiplier
    interval = summary['ci95']
    displayed = [value] + ([] if interval is None else [x * multiplier for x in interval])
    if min(displayed) < ax.get_xlim()[0] or max(displayed) > ax.get_xlim()[1]:
        raise ValueError('frozen effect or interval falls outside figure axis')
    if interval is not None:
        low, high = (x * multiplier for x in interval)
        ax.plot([low, high], [y, y], color=color, linewidth=1.25, zorder=3, solid_capstyle='round')
        ax.plot([low, low], [y - .075, y + .075], color=color, linewidth=.8, zorder=3)
        ax.plot([high, high], [y - .075, y + .075], color=color, linewidth=.8, zorder=3)
    ax.plot(value, y, marker=marker, markersize=4.2, markeredgewidth=.95,
            markerfacecolor='white' if summary['partial'] else color,
            markeredgecolor=color, linestyle='none', zorder=4)


def _legend(fig, *, anchor=(.52, 1.0)):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=v['marker'], color=v['color'], markersize=4.4,
                      linewidth=1.2, label=v['label']) for v in OBJECTIVES.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=anchor, ncol=2,
               frameon=False, borderaxespad=0, columnspacing=1.3, handlelength=1.6,
               handletextpad=.5, labelspacing=0)


def build_factorial(root: Path = ROOT):
    metadata = factorial_metadata(root)
    with _plot_context():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=FACTORIAL_SIZE)
        fig.subplots_adjust(left=.155, right=.985, bottom=.20, top=.82, wspace=.14, hspace=.60)
        index = {(r['scale'], r['domain'], r['objective']): r for r in metadata['rows']}
        for col, (scale, model, _) in enumerate(MODELS):
            for row, metric in enumerate(('pass8', 'distinct8')):
                ax = axes[row, col]
                _forest_axis(ax, col == 0)
                ax.set_xlim((-43, 112) if metric == 'pass8' else (-.33, 2.36))
                ax.set_xticks([-40, 0, 50, 100] if metric == 'pass8' else [0, 1, 2])
                ax.set_xlabel('Δpass@8 (prob. points)' if metric == 'pass8' else 'Δdistinct@8 (exp. keys)', labelpad=2)
                for y, domain in enumerate(DOMAINS):
                    for method, appearance in OBJECTIVES.items():
                        summary = index[scale, domain, method]['summaries'][metric]
                        _point(ax, summary, y + appearance['offset'], metric,
                               color=appearance['color'], marker=appearance['marker'])
                if row == 0:
                    ax.set_title(model, pad=5, fontweight='bold')
        fig.text(.155, .979, 'Replay − base objective', ha='left', va='top', fontsize=FONT)
        _legend(fig, anchor=(.785, .985))
        fig.text(.5, .008, '95% t intervals; n = 5. Open: Falcon Countdown, Dr.GRPO n = 4 (no interval).',
                 ha='center', va='bottom', fontsize=FONT, color=style.MUTED)
    return fig, metadata


def build_weighting(root: Path = ROOT):
    metadata = weighting_metadata(root)
    with _plot_context():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=COMPACT_SIZE)
        fig.subplots_adjust(left=.155, right=.985, bottom=.35, top=.82, wspace=.24)
        for ax, metric in zip(axes, ('pass8', 'breadth8')):
            _forest_axis(ax, ax is axes[0])
            ax.set_xlim((-77, 67) if metric == 'pass8' else (-.045, .82))
            ax.set_xticks([-60, 0, 60] if metric == 'pass8' else [0, .3, .6])
            ax.set_xlabel('Δpass@8 (probability points)' if metric == 'pass8' else 'Δextra@8 (expected extra keys)', labelpad=1.5)
            for y, row in enumerate(metadata['rows']):
                _point(ax, row['summaries'][metric], y, metric, color=style.METHOD, marker='D')
                # Exact mean labels preserve legibility beside Python's wide interval.
                if metric == 'pass8':
                    ax.text(1.02, y, f"{100 * row['summaries'][metric]['mean']:+.1f}", ha='left', va='center',
                            transform=ax.get_yaxis_transform(), fontsize=FONT, color=style.INK)
        fig.text(.155, .975, 'Uniform − frequency replay  |  Qwen2.5-0.5B', ha='left', va='top',
                 fontsize=FONT + .5, fontweight='bold')
        fig.text(.5, .01, 'Paired-bootstrap 95% intervals; five paired seeds per domain.',
                 ha='center', va='bottom', fontsize=FONT, color=style.MUTED)
    return fig, metadata


def build_level2(root: Path = ROOT):
    metadata = level2_metadata(root)
    with _plot_context():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=COMPACT_SIZE)
        fig.subplots_adjust(left=.155, right=.985, bottom=.35, top=.82, wspace=.24)
        index = {(r['domain'], r['objective']): r for r in metadata['rows']}
        for ax, metric in zip(axes, ('pass8', 'distinct8')):
            _forest_axis(ax, ax is axes[0])
            ax.set_xlim((-32, 108) if metric == 'pass8' else (-.34, 1.24))
            ax.set_xticks([-25, 0, 50, 100] if metric == 'pass8' else [0, .5, 1])
            ax.set_xlabel('Δpass@8 (probability points)' if metric == 'pass8' else 'Δdistinct@8 (expected keys)', labelpad=1.5)
            for y, domain in enumerate(DOMAINS):
                for method, appearance in OBJECTIVES.items():
                    _point(ax, index[domain, method]['summaries'][metric], y + appearance['offset'], metric,
                           color=appearance['color'], marker=appearance['marker'])
        fig.text(.155, .975, 'Level 2  |  Qwen2.5-0.5B', ha='left', va='top', fontsize=FONT + .5, fontweight='bold')
        _legend(fig, anchor=(.795, .98))
        fig.text(.5, .01, 'Replay − base; 95% t intervals. Four-arm cohorts: n = 5, Pantry n = 1 (open).',
                 ha='center', va='bottom', fontsize=FONT, color=style.MUTED)
    return fig, metadata


def build_figures(root: Path = ROOT) -> dict[str, tuple[Any, dict[str, Any]]]:
    return {
        'replay_factorial_effects': build_factorial(root),
        'replay_key_weighting': build_weighting(root),
        'replay_level2_effects': build_level2(root),
    }


def render(output_dir: Path = ROOT / 'paper/figures', root: Path = ROOT) -> dict[str, dict[str, Any]]:
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for stem, (figure, metadata) in build_figures(root).items():
        target = output_dir / stem
        figure.canvas.draw()
        from matplotlib.text import Text
        for text in figure.findobj(match=Text):
            if text.get_visible() and text.get_text():
                box = text.get_window_extent(figure.canvas.get_renderer())
                if box.x0 < -1 or box.y0 < -1 or box.x1 > figure.bbox.x1 + 1 or box.y1 > figure.bbox.y1 + 1:
                    raise ValueError(f'text falls outside publication canvas: {text.get_text()!r}')
        figure.savefig(target.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
        figure.savefig(target.with_suffix('.png'), dpi=200)
        for source in metadata['sources']:
            if _sha(root / source['path']) != source['sha256']:
                raise ValueError('frozen source changed during rendering')
        metadata['outputs'] = {}
        for ext in ('pdf', 'png'):
            path = target.with_suffix('.' + ext).resolve()
            try:
                relative = str(path.relative_to(ROOT))
            except ValueError:
                relative = str(path)
            metadata['outputs'][ext] = {'path': relative, 'sha256': _sha(path)}
        target.with_suffix('.json').write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + '\n')
        plt.close(figure)
        result[stem] = {'rows': len(metadata['rows']), 'figure_size_inches': metadata['figure_size_inches'],
                        'source_sha256': metadata['source_sha256']}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'paper/figures')
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
