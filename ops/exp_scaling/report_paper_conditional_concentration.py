#!/usr/bin/env python3
"""Render all frozen conditional-concentration blocks without selecting results.

Reads one analyzed JSON artifact, never run logs or model APIs. Exports a main
Level-1 figure, complete block/seed tables, a lossless numeric block archive,
and a report containing every defined and undefined primary comparison.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'paper/results/conditional_concentration_20260911.json'
DEFAULT_OUTPUT = ROOT / 'paper/results/conditional_concentration_20260911'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
SCALES = ('qwen05b', 'falcon1b', 'qwen3b')
DOMAIN_LABELS = dict(zip(DOMAINS, ('Graph', 'Countdown', 'Python', 'MathIR', 'PantryPlan')))
SCALE_LABELS = dict(zip(SCALES, ('Qwen2.5-0.5B', 'Falcon3-1B', 'Qwen2.5-3B')))
METHOD_LABELS = {'drgrpo': 'Dr.GRPO', 'grpo': 'GRPO', 'maxrl': 'MaxRL',
                 'replay_drgrpo': 'Re:Dr.GRPO', 'replay_maxrl': 'Re:MaxRL'}
METRICS = ('collision', 'mean8', 'pass8', 'distinct8', 'extra8')
DIRECT_ANALYSES = ('distinct_streams', 'orientation0', 'orientation1', 'naive_reused_streams_32')
MAIN_CONTRASTS = (('before_after', 'drgrpo'), ('before_after', 'grpo'),
                  ('before_after', 'maxrl'), ('replay_effect', 'drgrpo'), ('replay_effect', 'maxrl'))
ANALYSIS_LABELS = {
    'distinct_streams': 'Nominal stream representatives',
    'orientation0': 'Disjoint: A lower / B upper',
    'orientation1': 'Disjoint: A upper / B lower',
    'orientation_mean_descriptive': 'Mean of both defined orientations',
    'intact_k8': 'Mean of eligible intact K8 draws within prompt',
    'first_k8': 'First intact K8 draw',
    'naive_reused_streams_32': 'All 32 positions, reused streams (diagnostic)',
    'change_difference': 'Four-condition change difference',
}
LIMITS = [
    'C is correct-key pair collision on the stated observable prompt population. Positive delta C means greater concentration; it does not establish extinction of unobserved modes.',
    'The 32 saved positions map to 11 nominal child-seed streams under the audited vLLM V0 n=8 rule (parent seed plus output index). The earliest draw/output representative is selected without inspecting values.',
    'Historical runtime sources were unavailable for 250 of the 400 cells in the original four-method runtime audit. Applying that child-stream mapping to those cells remains an assumption; different seed IDs alone do not prove iid sampling.',
    'The per-prompt collision U-statistic has its usual conditional-distribution identity under iid correct labels. Shared nominal streams across compared conditions make jointly eligible primary means descriptive.',
    'The two disjoint orientations allocate lower/upper nominal streams to opposite conditions (five/six when eleven streams are shared). They reduce same-prompt cross-condition coupling, have different eligibility populations, and are shown separately. Reused seed IDs across prompts still limit an iid population interpretation.',
    'Training estimates weight jointly eligible prompts equally. Hosted collision pools correct-pair counts and therefore gives greater weight to prompts with more correct pairs; the two aggregates are intentionally different.',
    'Collision uses the selected nominal streams. Original mean@8, pass@8, distinct@8 and extra-mode metrics retain the original intact K8 groups, even when reported on the collision-eligible prompt population.',
    'Intervals describe variability among five measured training-seed estimates. They are nominal and unadjusted, not a simultaneous guarantee or proof of sampler independence. Partial and undefined blocks remain visible.',
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_extension(record: dict):
    return record.get('cohort_extension') or record.get('source_metadata', {}).get('cohort_extension')


def cohort_note(record: dict) -> str:
    if cohort_extension(record):
        return ('This artifact includes a documented cohort extension performed after the initial '
                'concentration results were inspected. Its amended source census determines all '
                'additions and withdrawals; the extension is not presented as an analysis frozen '
                'before those first results. Original and extension provenance remain bound in the source JSON.')
    return ('This artifact uses the saved-output cohort frozen before the initial analysis. '
            'Later published endpoint or trajectory-snapshot refreshes do not change its membership; '
            'partial blocks here need not match a newer manuscript table.')


def evidence_limits(record: dict) -> list[str]:
    items = list(LIMITS)
    historical = record.get('stream_source_audit', {}).get('historical_source_limit')
    if historical:
        items = [historical + ' The nominal child-stream interpretation therefore remains conditional '
                 'where historical sampling identity is not established.'
                 if item.startswith('Historical runtime sources') else item for item in items]
    return items


def block_id(block: dict) -> str:
    return '/'.join(str(block[key]) for key in ('level', 'scale', 'domain', 'kind', 'method'))


def base_fields(block: dict) -> dict:
    return {**{k: block[k] for k in ('kind', 'level', 'scale', 'domain', 'method')},
            'block_id': block_id(block), 'registered_n': block['registered_n'],
            'admitted_n': len(block['admitted_seeds']),
            'admitted_seeds': ';'.join(map(str, block['admitted_seeds']))}


def ordered_blocks(record: dict) -> list[dict]:
    kind_order = {'before_after': 0, 'replay_effect': 1}
    methods = tuple(METHOD_LABELS)
    return sorted(record['blocks'], key=lambda b: (
        b['level'], SCALES.index(b['scale']), DOMAINS.index(b['domain']),
        kind_order[b['kind']], methods.index(b['method'])))


def validate(record: dict) -> None:
    if record.get('schema') != 'paper-conditional-concentration-v1' or record.get('status') != 'analyzed':
        raise ValueError('expected an analyzed conditional-concentration artifact')
    blocks = record['blocks']
    if not blocks or len({block_id(b) for b in blocks}) != len(blocks):
        raise ValueError('empty or duplicate concentration blocks')
    for block in blocks:
        if block['registered_n'] != 5 or len(set(block['admitted_seeds'])) != len(block['admitted_seeds']):
            raise ValueError('invalid registered seed metadata')
        for name, summary in block['summaries'].items():
            values = list(summary['values'].values())
            if summary['n'] != len(values) or any(not math.isfinite(v) for v in values):
                raise ValueError(f'{block_id(block)}/{name}: invalid seed summary')
            mean = statistics.mean(values) if values else None
            if mean != summary['mean']:
                raise ValueError(f'{block_id(block)}/{name}: seed mean mismatch')
            if summary['ci95'] is not None and summary['n'] != 5:
                raise ValueError('partial block has a five-seed interval')


def summary_rows(record: dict) -> list[dict]:
    rows = []
    for block in ordered_blocks(record):
        base = base_fields(block)
        for name, summary in block['summaries'].items():
            metrics = {'collision': summary, **summary.get('original_metrics_on_eligible', {})}
            for metric, stats in metrics.items():
                interval, spread = stats.get('ci95'), stats.get('range')
                rows.append({**base, 'analysis': name, 'metric': metric,
                    'status': 'defined' if stats['mean'] is not None else 'undefined',
                    'defined_seed_n': stats['n'], 'mean_a': summary.get('a_mean') if metric == 'collision' else None,
                    'mean_b': summary.get('b_mean') if metric == 'collision' else None,
                    'delta_mean': stats['mean'], 'ci95_lower': interval[0] if interval else None,
                    'ci95_upper': interval[1] if interval else None,
                    'seed_range_lower': spread[0] if spread else None,
                    'seed_range_upper': spread[1] if spread else None,
                    'coverage_min': (summary.get('coverage_range') or [None, None])[0],
                    'coverage_max': (summary.get('coverage_range') or [None, None])[1],
                    'eligible_counts_by_seed': json.dumps(summary.get('eligible_counts'), sort_keys=True),
                    'uncertainty': stats.get('uncertainty'), 'issue_count': len(block['issues'])})
        for name, fixed in block.get('fixed_across_seed_population', {}).items():
            stats = fixed.get('summary', {})
            interval = stats.get('ci95')
            rows.append({**base, 'analysis': 'fixed_across_seed_population/' + name, 'metric': 'collision',
                'status': 'defined' if stats.get('mean') is not None else 'undefined',
                'defined_seed_n': stats.get('n', 0), 'delta_mean': stats.get('mean'),
                'ci95_lower': interval[0] if interval else None, 'ci95_upper': interval[1] if interval else None,
                'fixed_eligible_n': fixed.get('n_eligible'), 'reason': fixed.get('reason'),
                'uncertainty': stats.get('uncertainty'), 'issue_count': len(block['issues'])})
    return rows


def seed_records(block: dict):
    for seed, data in sorted(block['per_seed'].items(), key=lambda item: int(item[0])):
        for name in DIRECT_ANALYSES:
            if name in data:
                yield seed, name, data[name]
        intact = data.get('intact_k8', {})
        for i, item in enumerate(intact.get('individual_draws', [])):
            yield seed, f'intact_k8/draw{i}', item
        for name in ('distinct_streams_same_prompt_set', 'naive32_same_prompt_set'):
            if name in intact:
                yield seed, 'intact_k8/' + name, intact[name]
        if 'change_difference' in data:
            item = data['change_difference']
            yield seed, 'change_difference', {**item, 'delta': {'collision': item['delta']}}
    for name, fixed in block.get('fixed_across_seed_population', {}).items():
        for seed, item in sorted(fixed.get('per_seed', {}).items(), key=lambda pair: int(pair[0])):
            yield seed, 'fixed_across_seed_population/' + name, item


def seed_rows(record: dict) -> list[dict]:
    rows = []
    for block in ordered_blocks(record):
        base = base_fields(block)
        for seed, analysis, item in seed_records(block):
            for metric, delta in item.get('delta', {}).items():
                rows.append({**base, 'seed': seed, 'analysis': analysis, 'metric': metric,
                    'status': 'defined' if delta is not None else 'undefined',
                    'a': item.get('a', {}).get(metric), 'b': item.get('b', {}).get(metric), 'delta': delta,
                    'full_population_a': item.get('full_a', {}).get(metric),
                    'full_population_b': item.get('full_b', {}).get(metric),
                    'n_total': item.get('n_total'), 'n_eligible': item.get('n_eligible'),
                    'coverage': item.get('coverage'),
                    **{'eligible_' + key: value for key, value in item.get('eligibility', {}).items()},
                    'pooled_collision_a': item.get('pooled_collision_a'),
                    'pooled_collision_b': item.get('pooled_collision_b'),
                    'pooled_all_eligible_a': item.get('pooled_all_eligible_a'),
                    'pooled_all_eligible_b': item.get('pooled_all_eligible_b'),
                    'selected_correct_a': item.get('selected_correct_a'),
                    'selected_correct_b': item.get('selected_correct_b'),
                    'selected_total_a': json.dumps(item.get('selected_total_a')),
                    'selected_total_b': json.dumps(item.get('selected_total_b')),
                    'stream_selection': json.dumps(item.get('stream_selection'), sort_keys=True)})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'refuse an empty table: {path}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_numeric_archive(path: Path, record: dict) -> int:
    """Every numeric/bool/null block scalar, including all sensitivity counts."""
    def walk(value, pointer=''):
        if isinstance(value, dict):
            for key, item in value.items():
                if key != 'eligible_ids':
                    yield from walk(item, pointer + '/' + str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from walk(item, pointer + '/' + str(index))
        elif value is None or isinstance(value, (bool, int, float)):
            yield pointer, value
    count = 0
    with path.open('w', newline='', encoding='utf-8') as handle:
        fields = ['block_id', 'level', 'scale', 'domain', 'kind', 'method', 'field_path', 'value', 'value_type']
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for block in ordered_blocks(record):
            base = {key: base_fields(block)[key] for key in fields[:6]}
            for pointer, value in walk(block):
                writer.writerow({**base, 'field_path': pointer,
                    'value': 'null' if value is None else json.dumps(value, allow_nan=False),
                    'value_type': type(value).__name__})
                count += 1
    return count


def contrast_label(kind: str, method: str, *, short=False) -> str:
    method_label = METHOD_LABELS[method]
    if kind == 'before_after':
        return ('Initial → ' if short else 'Final minus initial: ') + method_label
    return (method_label + ' → Replay') if short else ('Replay minus no replay: ' + method_label)


def format_delta(value, *, interval=None, percent=True) -> str:
    if value is None:
        return 'undefined'
    scale = 100 if percent else 1
    result = f'{value * scale:+.1f}'
    if interval is not None:
        result += f' [{interval[0] * scale:+.1f}, {interval[1] * scale:+.1f}]'
    return result


def coverage_text(summary: dict) -> str:
    coverage = summary.get('coverage_range')
    if coverage is None:
        return '—'
    low, high = coverage
    return f'{100*low:.0f}%' if low == high else f'{100*low:.0f}–{100*high:.0f}%'


def draw_figure(record: dict, output: Path, *, level='level1', all_before_after=False) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    indexed = {(b['level'], b['scale'], b['domain'], b['kind'], b['method']): b for b in record['blocks']}
    scales = [scale for scale in SCALES if any(b['level'] == level and b['scale'] == scale for b in record['blocks'])]
    contrasts = ([('before_after', method) for method in METHOD_LABELS] if all_before_after else list(MAIN_CONTRASTS))
    contrasts = [pair for pair in contrasts if any(b['level'] == level and (b['kind'], b['method']) == pair for b in record['blocks'])]
    blocks = [block for key, block in indexed.items() if key[0] == level and key[3:] in contrasts]
    if not blocks:
        raise ValueError('no registered blocks for requested figure')
    limit = 20.0
    for block in blocks:
        summary = block['summaries']['distinct_streams']
        for value in [summary['mean'], *(summary['ci95'] or [])]:
            if value is not None:
                limit = max(limit, 100*abs(value))
        for name in ('orientation0', 'orientation1'):
            if block['summaries'][name]['mean'] is not None:
                limit = max(limit, 100*abs(block['summaries'][name]['mean']))
    limit = 20 * math.ceil((limit + 2) / 20)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(DOMAINS), len(scales), squeeze=False,
        figsize=(4.3*len(scales)+1.8, 10.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17 if len(scales)>1 else .29, right=.91,
                        top=.855, bottom=.15 if len(scales)>1 else .20, hspace=.40, wspace=.67)
    primary, low, high = '#243F70', '#BA5A24', '#087E83'
    counts = {'shown_defined': 0, 'shown_undefined': 0, 'not_registered': 0}
    for row, domain in enumerate(DOMAINS):
        for col, scale in enumerate(scales):
            ax = axes[row, col]
            ax.axvline(0, color='#B6BDC6', linewidth=.8, zorder=0)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-.55, len(contrasts)-.35)
            ax.set_xticks([-limit, -limit/2, 0, limit/2, limit])
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', length=0, labelsize=8)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#CED3D8')
            ys = list(reversed(range(len(contrasts))))
            ax.set_yticks(ys)
            ax.set_yticklabels([contrast_label(*pair, short=True) for pair in contrasts])
            for y, pair in zip(ys, contrasts):
                block = indexed.get((level, scale, domain, *pair))
                if block is None:
                    ax.text(0, y, 'not registered', ha='center', va='center', fontsize=7, color='#777777')
                    counts['not_registered'] += 1
                    continue
                summary = block['summaries']['distinct_streams']
                mean, interval = summary['mean'], summary['ci95']
                if mean is None:
                    ax.text(0, y, 'undefined', ha='center', va='center', fontsize=7, color='#777777')
                    counts['shown_undefined'] += 1
                else:
                    counts['shown_defined'] += 1
                    if interval is not None:
                        ax.plot([100*interval[0], 100*interval[1]], [y, y], color=primary, linewidth=1.3, zorder=2)
                    ax.plot(100*mean, y, 'o', markersize=4.7, color=primary,
                            markerfacecolor=primary if summary['n']==5 else 'white', zorder=4)
                for name, offset, marker, color in [('orientation0', -.18, '<', low), ('orientation1', .18, '>', high)]:
                    sensitivity = block['summaries'][name]
                    if sensitivity['mean'] is not None:
                        ax.plot(100*sensitivity['mean'], y+offset, marker=marker, linestyle='none',
                                markersize=4.0, markerfacecolor='white', markeredgecolor=color, zorder=3)
                ax.text(1.035, y, f"n={summary['n']}; {coverage_text(summary)}", transform=ax.get_yaxis_transform(),
                        va='center', ha='left', fontsize=7.4, color='#333333', clip_on=False)
            if row == 0:
                ax.set_title(SCALE_LABELS[scale], pad=10, fontweight='bold')
            if col == 0:
                ax.text(-.83 if len(scales)>1 else -.40, .5, DOMAIN_LABELS[domain],
                        transform=ax.transAxes, rotation=90, ha='center', va='center', fontweight='bold', fontsize=10)
            if row == len(DOMAINS)-1:
                ax.set_xlabel('ΔC (percentage points)')
    handles = [Line2D([0],[0],color=primary,marker='o',markersize=5,label=('Nominal stream representatives; nominal 95% seed interval' if len(scales)>1
            else 'Nominal stream representatives;\nnominal 95% seed interval')),
        Line2D([0],[0],color=low,marker='<',linestyle='none',markerfacecolor='white',label='Disjoint A lower / B upper'),
        Line2D([0],[0],color=high,marker='>',linestyle='none',markerfacecolor='white',label='Disjoint A upper / B lower')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52,.954), ncol=1, frameon=False, fontsize=8.5)
    title = 'Correct-output concentration: all initial-to-final comparisons' if all_before_after else 'Correct-output concentration: training changes and replay effects'
    shown_title = title if len(scales)>1 else title.replace(': ', ':\n')
    fig.suptitle(shown_title + f" · {'Level 1' if level=='level1' else 'Level 2'}",
                 fontsize=14 if len(scales)>1 else 11, fontweight='bold', y=.992)
    footer = ('Positive ΔC: more concentrated correct outputs. Negative ΔC: less concentrated.\n'
              'Labels show defined seed count and the primary eligible-prompt coverage range across seeds.\n'
              'Open circles: fewer than five defined seeds. Sensitivity points use their own eligible prompts.\n'
              'Correctness can change on these selected prompts; the companion table reports all original K8 metric changes.')
    if len(scales)==1:
        import textwrap
        footer = '\n'.join(textwrap.fill(line, width=72) for line in footer.splitlines())
    fig.text(.5,.055 if len(scales)>1 else .065, footer,
             ha='center', va='center', fontsize=8.5 if len(scales)>1 else 7.3, linespacing=1.5)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', metadata={'Title':title})
    fig.savefig(output.with_suffix('.png'), dpi=210, bbox_inches='tight')
    plt.close(fig)
    return {'level': level, 'all_before_after': all_before_after, 'contrasts': contrasts,
            'scales': scales, 'domain_order': list(DOMAINS), 'x_limit_percentage_points': limit, **counts}



def findings_report(record: dict) -> list[str]:
    """Compute the complete prespecified domain groups, retaining their exceptions."""
    blocks = ordered_blocks(record)
    s_of = lambda block: block['summaries']['distinct_streams']
    def subset(**conditions):
        return [b for b in blocks if all(b[k] in v if isinstance(v, tuple) else b[k] == v
                                         for k, v in conditions.items())]
    def signed(value, sign):
        return value is not None and sign * value > 0
    def both(block, sign):
        return all(signed(block['summaries'][name]['mean'], sign)
                   for name in ('orientation0', 'orientation1'))
    def ci_signed(block, sign):
        ci = s_of(block)['ci95']
        return ci is not None and all(signed(value, sign) for value in ci)
    def range_pp(values):
        values = [value for value in values if value is not None]
        return 'undefined' if not values else f'{100*min(values):+.1f} to {100*max(values):+.1f} pp'
    original = subset(level='level1', domain=('graph_coloring', 'pantry_plan'),
                      kind='before_after', method=('drgrpo', 'grpo'))
    replay = subset(level='level1', domain=('graph_coloring', 'pantry_plan'),
                    kind='replay_effect', method='drgrpo')
    lines = ['## Findings across the complete reported groups', '',
        f"- **Training concentration:** {sum(signed(s_of(b)['mean'], 1) for b in original)}/{len(original)} Level-1 Graph/PantryPlan Dr.GRPO and GRPO initial-to-final blocks increase collision; {sum(ci_signed(b, 1) for b in original)}/{len(original)} have nominal intervals entirely above zero. Both disjoint orientations are positive in {sum(both(b, 1) for b in original)}/{len(original)}. The full tables identify the orientation exceptions.",
        f"- **Dr.GRPO replay effect:** {sum(signed(s_of(b)['mean'], -1) for b in replay)}/{len(replay)} Graph/PantryPlan blocks decrease collision; {sum(ci_signed(b, -1) for b in replay)}/{len(replay)} have nominal intervals entirely below zero, and {sum(both(b, -1) for b in replay)}/{len(replay)} have two negative disjoint orientations."]
    max3 = subset(level='level1', scale='qwen3b', kind='replay_effect', method='maxrl')
    details = []
    for b in max3:
        summary = s_of(b)
        split = [b['summaries'][name]['mean'] for name in ('orientation0', 'orientation1')]
        direction = ('both splits negative' if both(b, -1) else
                     'both splits positive' if both(b, 1) else
                     'both splits zero' if split == [0, 0] else
                     'a split is undefined' if None in split else 'split directions differ')
        details.append(f"{DOMAIN_LABELS[b['domain']]} {format_delta(summary['mean'], interval=summary['ci95'])} pp, n={summary['n']} ({direction})")
    if details:
        lines.append('- **Qwen2.5-3B MaxRL replay across every domain:** ' + '; '.join(details) + '.')
    replay_initial = subset(level='level1', domain='pantry_plan', kind='before_after',
                            method=('replay_drgrpo', 'replay_maxrl'))
    lines.append(f"- **Mitigation does not imply a return to the initial collision profile:** {sum(signed(s_of(b)['mean'], 1) for b in replay_initial)}/{len(replay_initial)} PantryPlan replay-arm initial-to-final primary estimates remain positive ({range_pp([s_of(b)['mean'] for b in replay_initial])}); {sum(s_of(b)['n'] < 5 for b in replay_initial)} are partial. These before/after and replay contrasts use their respective eligible populations and must not be subtracted as if they shared one population.")
    all_replay = subset(level='level1', domain=('graph_coloring', 'pantry_plan'), kind='replay_effect')
    correct = [s_of(b)['original_metrics_on_eligible']['mean8']['mean'] for b in all_replay]
    lines.append(f"- **Simultaneous correctness tradeoff:** eligible-prompt mean@8 decreases in {sum(signed(value, -1) for value in correct)}/{len(correct)} Graph/PantryPlan replay blocks ({range_pp(correct)}). Lower collision here is not evidence that correctness was held fixed. The companion table also reports pass@8, distinct@8 and extra modes.")
    pantry2 = subset(level='level2', domain='pantry_plan', kind='replay_effect')
    details = []
    for b in pantry2:
        summary = s_of(b)
        splits = sum(b['summaries'][name]['mean'] is not None for name in ('orientation0', 'orientation1'))
        details.append(f"{METHOD_LABELS[b['method']]} {format_delta(summary['mean'])} pp, n={summary['n']}, coverage {coverage_text(summary)}, {splits}/2 defined splits")
    if details:
        lines.append('- **Harder-level boundary:** PantryPlan replay estimates are sparse: ' + '; '.join(details) + '. These retained results do not support a uniform replay-improvement claim.')
    return lines + ['']


def readable_report(record: dict, source: Path, *, source_sha256: str | None = None) -> str:
    blocks = ordered_blocks(record)
    defined = sum(b['summaries']['distinct_streams']['mean'] is not None for b in blocks)
    lines = ['# Conditional concentration from saved verified outputs', '',
        f"This report includes all {len(blocks)} analyzed blocks: {defined} have a defined primary mean and {len(blocks)-defined} are undefined. No block is selected by effect direction or significance.", '', *findings_report(record), cohort_note(record), '',
        'The primary estimate averages per-prompt correct-key collision differences on prompts with at least two correct nominal-stream representatives in both conditions, then averages the measured seed estimates. ΔC is B minus A. For initial-to-final comparisons B is the final checkpoint; for replay comparisons B is the replay arm.', '',
        '## Reading the figures and tables', '',
        '- `conditional_concentration_overview.pdf`: the three original objectives before/after training and both paired replay effects, for every Level-1 domain and scale.',
        '- `all_before_after_level1.pdf`: all five methods, including the replay arms, before/after training.',
        '- `conditional_concentration_level2.pdf`: all registered main contrast types on the harder level.',
        '- `block_summaries.csv`: every summary, metric, interval, defined seed count and available coverage field.',
        '- `seed_metrics.csv`: seed-level metrics, selected and full-population values, eligible counts and pair-weighted diagnostics.',
        '- `all_numeric_scalars.csv`: every numeric, Boolean and null scalar under every analyzed block, including sensitivities and fixed-across-seed populations. Prompt identity lists remain in the source JSON.',
        '- `metric_tradeoffs.md`: collision and all four original K8 metric changes side by side for every block, on the same collision-eligible prompt population.', '',
        'These comparisons do not hold correctness fixed. Read collision changes alongside the original mean@8, pass@8, distinct@8 and extra-mode changes in `metric_tradeoffs.md`; lower collision can occur together with lower correctness on the selected prompts. The K8 metrics retain their original groups and are not recomputed as eleven-sample metrics.', '',
        'Numbers below are percentage-point changes in collision. Brackets are nominal 95% seed intervals when all five seed estimates are defined. `n` is the number of defined seed estimates; coverage is the range of eligible-prompt fractions across available seeds. Separate disjoint orientations have their own eligibility and counts.', '']
    for level in sorted({b['level'] for b in blocks}):
        for scale in SCALES:
            selected = [b for b in blocks if b['level']==level and b['scale']==scale]
            if not selected:
                continue
            lines += [f"## {level.replace('level','Level ')} · {SCALE_LABELS[scale]}", '',
                '| Domain | Comparison | Primary ΔC [95%] | n / admitted | Coverage | A low / B high: ΔC; n; coverage | A high / B low: ΔC; n; coverage |',
                '|---|---|---:|---:|---:|---:|---:|']
            for block in selected:
                primary = block['summaries']['distinct_streams']
                sensitivities = []
                for name in ('orientation0','orientation1'):
                    s = block['summaries'][name]
                    sensitivities.append(f"{format_delta(s['mean'])}; {s['n']}; {coverage_text(s)}")
                lines.append(f"| {DOMAIN_LABELS[block['domain']]} | {contrast_label(block['kind'],block['method'])} | "
                    f"{format_delta(primary['mean'],interval=primary['ci95'])} | {primary['n']} / {len(block['admitted_seeds'])} | "
                    f"{coverage_text(primary)} | {sensitivities[0]} | {sensitivities[1]} |")
            lines += ['']
    lines += ['## Eligibility, missing estimates and source issues', '']
    any_issue = False
    for block in blocks:
        primary = block['summaries']['distinct_streams']
        zero = [s for s,n in primary.get('eligible_counts',{}).items() if n == 0]
        missing = sorted(set(map(str,block['admitted_seeds'])) - set(block['per_seed']), key=int)
        if block['issues'] or zero or missing:
            any_issue = True
            lines.append(f"- `{block_id(block)}`: admitted seeds {block['admitted_seeds']}; missing paired endpoints {missing}; zero eligible prompts in seeds {zero}; issues `{json.dumps(block['issues'],sort_keys=True)}`.")
    if not any_issue:
        lines.append('No block-level endpoint issue or zero-eligibility seed was recorded.')
    source_issues = [row for row in record.get('cell_availability',[]) if row.get('sample_issues')]
    lines += ['', f"The source artifact retains sample-integrity issues for {len(source_issues)} cells. All source checkpoint statuses and issue messages are exported in `source_availability.csv`.", '',
        '## Interpretation limits', ''] + ['- '+item for item in evidence_limits(record)]
    lines += ['', '## Provenance', '', f'- Source: `{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`.',
        f'- Source SHA-256: `{source_sha256 or sha(source)}`.',
        f"- Analysis code SHA-256: `{record['analysis_code_sha256']}`.",
        '- The source JSON binds the protocol, stream amendment, any cohort extension, sample cache and source manifests. The renderer does not change any measurements or regrade outputs.', '']
    return '\n'.join(lines)


def tradeoff_report(record: dict) -> str:
    lines = ['# Collision and correctness changes on the same eligible prompts', '',
        'Each row uses the primary nominal-stream collision population, then reports original intact-K8 metrics on those prompts. These are simultaneous changes; correctness has not been held fixed. Collision, mean@8 and pass@8 changes are percentage points; distinct@8 and extra-mode changes are expected counts. All blocks, including undefined and partial blocks, are retained.', '']
    blocks = ordered_blocks(record)
    for level in sorted({b['level'] for b in blocks}):
        for scale in SCALES:
            selected = [b for b in blocks if b['level']==level and b['scale']==scale]
            if not selected:
                continue
            lines += [f"## {level.replace('level','Level ')} · {SCALE_LABELS[scale]}", '',
                '| Domain | Comparison | n | Coverage | ΔC, pp | Δmean@8, pp | Δpass@8, pp | Δdistinct@8 | Δextra modes |',
                '|---|---|---:|---:|---:|---:|---:|---:|---:|']
            for block in selected:
                summary = block['summaries']['distinct_streams']
                other = summary['original_metrics_on_eligible']
                values = [format_delta(summary['mean'])]
                for metric in METRICS[1:]:
                    value = other[metric]['mean']
                    if metric in ('mean8','pass8'):
                        values.append(format_delta(value))
                    else:
                        values.append('undefined' if value is None else f'{value:+.3f}')
                lines.append(f"| {DOMAIN_LABELS[block['domain']]} | {contrast_label(block['kind'],block['method'])} | "
                    f"{summary['n']} | {coverage_text(summary)} | " + ' | '.join(values) + ' |')
            lines.append('')
    lines += ['Full-128-prompt metric values, eligible counts, seed-level changes, and uncertainty are preserved in the CSVs and the source JSON. Correctness on this collision-eligible subset is not the complete held-out benchmark average.', '']
    return '\n'.join(lines)


def build(source: Path, output: Path) -> dict:
    raw = source.read_bytes(); source_digest = hashlib.sha256(raw).hexdigest()
    record = json.loads(raw); validate(record)
    output.mkdir(parents=True, exist_ok=True)
    summaries, seeds = summary_rows(record), seed_rows(record)
    write_csv(output/'block_summaries.csv', summaries)
    write_csv(output/'seed_metrics.csv', seeds)
    numeric_count = write_numeric_archive(output/'all_numeric_scalars.csv', record)
    availability = [{'level': row['cell'][0], 'scale': row['cell'][1], 'domain': row['cell'][2],
                    'method': row['cell'][3], 'seed': row['cell'][4],
                    **{'checkpoint_'+step: available for step,available in row['checkpoints'].items()},
                    'sample_issue_count':len(row['sample_issues']),
                    'sample_issues':json.dumps(row['sample_issues'],sort_keys=True)} for row in record['cell_availability']]
    write_csv(output/'source_availability.csv', availability)
    figures = [draw_figure(record,output/'conditional_concentration_overview'),
               draw_figure(record,output/'all_before_after_level1',all_before_after=True)]
    if any(b['level']=='level2' for b in record['blocks']):
        figures.append(draw_figure(record,output/'conditional_concentration_level2',level='level2'))
    (output/'report.md').write_text(readable_report(record,source,source_sha256=source_digest),encoding='utf-8')
    (output/'metric_tradeoffs.md').write_text(tradeoff_report(record),encoding='utf-8')
    (output/'figure_caption.txt').write_text(
        'Correct-output concentration on matched eligible prompts. Points show equal-prompt B-minus-A collision changes across measured seeds. '
        'Bars are nominal unadjusted 95% intervals only for five defined seed estimates; open circles identify partial blocks. '
        'Correctness is not held fixed; the companion tradeoff table reports simultaneous original K8 metric changes. '
        'Side markers show both disjoint nominal-stream orientations separately. Right-hand labels give primary n and eligible-prompt coverage. '
        'The 32 saved positions contain 11 nominal child-seed streams under the audited V0 mapping; this mapping is assumed where historical runtime evidence is missing. '
        'Distinct seed IDs do not establish iid sampling. Common-eligibility and disjoint-subset means remain descriptive under the recorded shared RNG design. '
        'Hosted pair-weighted collision and this equal-prompt training statistic use different aggregation weights. '
        + cohort_note(record) + '\n',encoding='utf-8')
    if sha(source) != source_digest:
        raise ValueError('analysis source changed during report rendering; no build manifest published')
    manifest = {'schema':'paper-conditional-concentration-report-v1', 'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source':{'path':str(source),'sha256':source_digest},
        'renderer':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},
        'cohort_extension':cohort_extension(record),
        'block_count':len(record['blocks']),'summary_rows':len(summaries),'seed_metric_rows':len(seeds),
        'numeric_scalar_rows':numeric_count,'figure_designs':figures,
        'outputs':{path.name:sha(path) for path in sorted(output.iterdir()) if path.is_file() and path.name!='build_manifest.json'}}
    (output/'build_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    result=build(args.source.resolve(),args.output_dir.resolve())
    print(json.dumps({key:result[key] for key in ('block_count','summary_rows','seed_metric_rows','numeric_scalar_rows')},sort_keys=True))


if __name__=='__main__':
    main()
