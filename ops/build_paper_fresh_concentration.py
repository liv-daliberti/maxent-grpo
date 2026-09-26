#!/usr/bin/env python3
"""Publish the frozen fresh Graph/Pantry evaluation in the paper's PCMD units.

This offline adapter never samples responses or regrades them. --check verifies
the frozen report, every derived statistic, and the retained generated assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/modebench_fresh_concentration_20260912/fresh_panel/report.json'
SOURCE_SHA256 = 'c9d6f8d4b5a21cc1af4b6cd6f5162e3d5d960c9b0f19657fa0169ff2c84a2190'
OUT = ROOT / 'paper/results/fresh_concentration_20260912'
FIGURE = ROOT / 'paper/figures/fresh_concentration_pcmd_20260912'
SCALES = ('qwen05b', 'falcon1b', 'qwen3b')
DOMAINS = ('graph_coloring', 'pantry_plan')
SCALE_LABELS = {'qwen05b': 'Qwen2.5-0.5B', 'falcon1b': 'Falcon3-1B', 'qwen3b': 'Qwen2.5-3B'}
SHORT_LABELS = {'qwen05b': 'Qwen 0.5B', 'falcon1b': 'Falcon 1B', 'qwen3b': 'Qwen 3B'}
DOMAIN_LABELS = {'graph_coloring': 'Graph', 'pantry_plan': 'PantryPlan'}
METHODS = ('drgrpo', 'replay_drgrpo', 'maxrl', 'replay_maxrl')
METHOD_LABELS = {'drgrpo': 'Dr.GRPO', 'replay_drgrpo': 'Re:Dr', 'maxrl': 'MaxRL', 'replay_maxrl': 'Re:Max'}
CONTRASTS = tuple(m + '_minus_initial' for m in METHODS) + (
    'replay_drgrpo_minus_drgrpo', 'replay_maxrl_minus_maxrl')
T_DF4_975 = 2.7764451051977987


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f'inconsistent derived statistic: {actual} != {expected}')


def build_metadata() -> dict:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError('fresh evaluation report differs from the frozen scientific review')
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'complete'
    assert source['scope']['fresh_draws_per_prompt'] == 64
    assert source['scope']['prompts_per_cell'] == 128
    assert source['completeness_audit']['authenticated_tasks'] == 150
    assert source['completeness_audit']['authenticated_response_slots'] == 1228800
    indexed = {(b['model_scale'], b['domain'], b['contrast']): b for b in source['contrasts']}
    assert len(indexed) == len(source['contrasts']) == 36
    reconciled = {(b['identity']['model_scale'], b['identity']['domain'], b['identity']['contrast']): b
                  for b in source['population_reconciliation']}
    blocks = []
    for scale in SCALES:
        for domain in DOMAINS:
            for contrast in CONTRASTS:
                original = indexed[scale, domain, contrast]
                assert original['level'] == 1 and original['wording'] == 'original' and original['grading'] == 'strict'
                summary = original['summary']
                assert summary['n_defined'] == summary['n_expected'] == 5
                if original['left_method'] == 'initial':
                    assert summary['initial_weights_shared'] and summary['independent_initial_checkpoints'] == 1
                collision = summary['joint_population_effects']['collision']
                assert collision['interval_method'] == 'paired_Student_t_df4_nominal'
                seeds = {s: -v for s, v in summary['seed_effects'].items()}
                mean = statistics.mean(seeds.values())
                halfwidth = T_DF4_975 * statistics.stdev(seeds.values()) / math.sqrt(5)
                equal(mean, -collision['mean'])
                interval = [-collision['ci95'][1], -collision['ci95'][0]]
                equal(interval[0], mean - halfwidth)
                equal(interval[1], mean + halfwidth)
                counts = summary['eligible_prompt_counts']
                assert len(counts) == 5 and all(2 <= n <= 128 for n in counts.values())
                full = summary['all_prompt_effects_equal_seed']
                full_seed = {str(s['training_seed']): s['populations']['all_prompts']['delta'] for s in original['seeds']}
                for metric, value in full.items():
                    if value is not None:
                        equal(value, statistics.mean(s[metric] for s in full_seed.values()))
                recon = reconciled[scale, domain, contrast]
                block = {
                    'scale': scale, 'domain': domain, 'contrast': contrast,
                    'left_method': original['left_method'], 'right_method': original['right_method'],
                    'pcmd_delta': mean, 'ci95': interval, 'seed_deltas': seeds,
                    'eligible_prompt_counts': counts,
                    'eligible_prompt_range': [min(counts.values()), max(counts.values())],
                    'full_cohort_effects': full, 'full_cohort_seed_effects': full_seed,
                    'eligible_correctness_delta': summary['joint_population_effects']['mean_correct']['mean'],
                    'sensitivity_pcmd': {
                        'joint_pair_weighted': -summary['pooled_pairs_delta_equal_seed'],
                        'fixed_prompt_intersection': -recon['fixed_common_population']['equal_prompt_delta_equal_seed'],
                        'fixed_prompt_count': recon['fixed_common_prompts'],
                        'own_population_pair_weighted': -recon['own_population_pair_pooled_then_equal_seed'],
                    },
                }
                blocks.append(block)

    dr = [b for b in blocks if b['contrast'] == 'drgrpo_minus_initial']
    replay = [b for b in blocks if b['contrast'] in CONTRASTS[-2:]]
    initial = [b for b in blocks if b['left_method'] == 'initial']
    pantry_initial = [b for b in initial if b['domain'] == 'pantry_plan']
    for b, direction in [(b, -1) for b in dr] + [(b, 1) for b in replay]:
        assert all(direction * v > 0 for v in b['seed_deltas'].values())
        assert all(direction * v > 0 for v in b['ci95'])
        assert all(direction * b['sensitivity_pcmd'][s] > 0 for s in (
            'joint_pair_weighted', 'fixed_prompt_intersection', 'own_population_pair_weighted'))
    assert len(dr) == 6 and len(replay) == 12 and len(initial) == 24
    assert all(b['full_cohort_effects']['mean_correct'] > 0 for b in initial)
    assert all(s['mean_correct'] > 0 for b in initial for s in b['full_cohort_seed_effects'].values())
    assert all(s['pass_all'] < 0 for b in pantry_initial for s in b['full_cohort_seed_effects'].values())
    for b in replay:
        assert all(b['full_cohort_effects'][metric] > 0 for metric in
                   ('pass8_rarefied', 'distinct8_rarefied', 'pass_all', 'distinct_all'))
        assert all(s['distinct_all'] > 0 for s in b['full_cohort_seed_effects'].values())
    assert sum(b['full_cohort_effects']['mean_correct'] > 0 for b in replay) == 7
    assert sum(b['eligible_correctness_delta'] > 0 for b in replay) == 1
    for b in initial:
        if b['right_method'].startswith('replay_'):
            assert b['pcmd_delta'] < 0 if b['domain'] == 'pantry_plan' else b['pcmd_delta'] > 0
        if b['right_method'] == 'maxrl' and b['domain'] == 'graph_coloring' and b['scale'] != 'qwen05b':
            assert b['ci95'][0] < 0 < b['ci95'][1]
    example = next(b for b in dr if b['scale'] == 'qwen3b' and b['domain'] == 'graph_coloring')
    assert all(s['distinct8_rarefied'] > 0 and s['distinct_all'] < 0
               for s in example['full_cohort_seed_effects'].values())
    assert round(100 * example['full_cohort_effects']['mean_correct'], 2) == 14.49
    assert round(example['full_cohort_effects']['distinct8_rarefied'], 3) == .112
    assert round(100 * example['pcmd_delta'], 2) == -11.20
    assert round(example['full_cohort_effects']['distinct_all'], 3) == -.220
    assert min(b['eligible_prompt_range'][0] for b in blocks) == 37
    assert max(b['eligible_prompt_range'][1] for b in blocks) == 127
    assert all(b['sensitivity_pcmd']['fixed_prompt_count'] == 5 for b in blocks
               if b['scale'] == 'qwen05b' and b['domain'] == 'graph_coloring'
               and b['contrast'] in ('drgrpo_minus_initial', 'replay_drgrpo_minus_drgrpo'))
    return {
        'schema': 'paper-fresh-concentration-pcmd-v1',
        'source': {'path': str(SOURCE.relative_to(ROOT)), 'sha256': SOURCE_SHA256},
        'renderer': {'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': sha256(Path(__file__))},
        'style_source': {'path': 'ops/paper_style.py', 'sha256': sha256(ROOT / 'ops/paper_style.py')},
        'metric': 'PCMD = 1 - collision; positive differences mean more diverse correct modes',
        'estimator': 'equal-prompt paired differences on joint R>=2 prompts, then equal weight over five seeds',
        'interval': 'nominal paired Student t, df=4, fixed prompt cohorts, no multiplicity adjustment',
        'scope': {'level': 1, 'domains': list(DOMAINS), 'scales': list(SCALES),
                  'prompts_per_domain': 128, 'responses_per_prompt': 64,
                  'initial_reference': 'five sampling replicas of the same initial weights, shared across method contrasts'},
        'claims': {'dr_negative_blocks': 6, 'dr_negative_seed_effects': 30,
                   'replay_positive_blocks': 12, 'replay_positive_seed_effects': 60,
                   'core_intervals_excluding_zero': 18,
                   'replay_positive_full_cohort_correctness_blocks': 7,
                   'replay_positive_eligible_correctness_blocks': 1},
        'blocks': blocks,
    }


def signed(value: float) -> str:
    return f'{value:+.3f}'


def table(metadata: dict, *, replay: bool) -> str:
    selected = [b for b in metadata['blocks'] if (b['left_method'] != 'initial') == replay]
    label = 'replay' if replay else 'initial'
    caption = (
        r'64-response evaluation: replay minus its matched trained control.' if replay else
        r'64-response evaluation: trained policy minus initialization.')
    caption += (r' Changes in \pmd{} are in probability units; positive values indicate more diverse correct modes.'
                r' Brackets give nominal 95\% paired Student-$t$ intervals ($n=5$, four degrees of freedom).'
                r' The last column gives the range of jointly eligible prompts across seeds, out of 128.'
                r' Each row uses its own matched prompt population.')
    lines = [r'% Generated by ops/build_paper_fresh_concentration.py; do not hand edit.',
             r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\caption{' + caption + '}', r'\label{tab:fresh-concentration-' + label + '}',
             r'\begin{tabular}{llrr}', r'\toprule',
             r'Domain & Contrast & $\Delta\pmd$ [95\% interval] & Joint prompts \\', r'\midrule']
    prev_scale = None
    prev_domain = None
    for b in selected:
        if b['scale'] != prev_scale:
            if prev_scale is not None:
                lines.append(r'\midrule')
            lines.append(r'\multicolumn{4}{l}{\textbf{' + SCALE_LABELS[b['scale']] + r'}} \\')
            prev_scale, prev_domain = b['scale'], None
        domain = r'\texttt{' + DOMAIN_LABELS[b['domain']] + '}' if b['domain'] != prev_domain else ''
        right = METHOD_LABELS[b['right_method']]
        left = 'initial' if b['left_method'] == 'initial' else METHOD_LABELS[b['left_method']]
        lo, hi = b['ci95']
        low_n, high_n = b['eligible_prompt_range']
        n_text = str(low_n) if low_n == high_n else f'{low_n}--{high_n}'
        lines.append(f'{domain} & {right} $-$ {left} & ${signed(b["pcmd_delta"])}\;[{signed(lo)}, {signed(hi)}]$ & {n_text} ' + r'\\')
        prev_domain = b['domain']
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)


def render_figure(metadata: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, str(ROOT / 'ops'))
    import paper_style as style

    style.apply_rcparams(font_size=8)
    indexed = {(b['scale'], b['domain'], b['contrast']): b for b in metadata['blocks']}
    rows = [(scale, domain) for domain in DOMAINS for scale in SCALES]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.65), sharey=True)
    fig.subplots_adjust(left=.175, right=.98, bottom=.21, top=.81, wspace=.14)
    settings = [
        [('drgrpo_minus_initial', 'Dr.GRPO', style.CONTROL, 'o', -.13),
         ('maxrl_minus_initial', 'MaxRL', style.ABLATION, 's', .13)],
        [('replay_drgrpo_minus_drgrpo', 'Re:Dr', style.METHOD, 'o', -.13),
         ('replay_maxrl_minus_maxrl', 'Re:Max', style.METHOD, 's', .13)],
    ]
    for panel, (ax, methods) in enumerate(zip(axes, settings)):
        style.style_axis(ax, grid='both')
        ax.axvline(0, color=style.INK, lw=.9, zorder=1)
        ax.axhspan(2.5, 5.5, facecolor=style.PANEL, alpha=.48, zorder=-2)
        for contrast, name, color, marker, shift in methods:
            for row, (scale, domain) in enumerate(rows):
                b = indexed[scale, domain, contrast]
                value = 100 * b['pcmd_delta']
                lo, hi = [100 * v for v in b['ci95']]
                ax.errorbar(value, row + shift, xerr=[[value-lo], [hi-value]],
                            fmt=marker, ms=4.1, color=color, ecolor=color,
                            markerfacecolor='white' if marker == 's' else color,
                            markeredgewidth=.95, capsize=2, elinewidth=1.05, zorder=3)
        ax.set_ylim(5.5, -.5)
        ax.tick_params(axis='both', labelsize=8, length=2)
        ax.set_xlabel(r'$\Delta$ PCMD (percentage points)', fontsize=8, labelpad=5)
        ax.set_title('Final − initial' if panel == 0 else 'Replay − matched control',
                     fontsize=9, fontweight='bold', pad=29)
        handles = [Line2D([], [], linestyle='', marker=m, markersize=4,
                          color=c, markerfacecolor='white' if m == 's' else c, label=n)
                   for _, n, c, m, _ in methods]
        ax.legend(handles=handles, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.01),
                  frameon=False, fontsize=8, handletextpad=.35, columnspacing=.9, borderaxespad=0)
        ax.set_yticks(range(6))
        if panel == 0:
            ax.set_yticklabels([SHORT_LABELS[s] for s, _ in rows], fontsize=8)
            ax.set_xlim(-100, 15)
            ax.set_xticks([-100, -75, -50, -25, 0])
        else:
            ax.set_xlim(-2, 85)
            ax.set_xticks([0, 20, 40, 60, 80])
            ax.tick_params(labelleft=False)
    fig.text(.012, .649, 'Graph', rotation=90, va='center', fontsize=8, fontfamily='monospace')
    fig.text(.012, .349, 'PantryPlan', rotation=90, va='center', fontsize=8, fontfamily='monospace')
    fig.text(.575, .045, 'More negative: narrower correct modes     More positive: broader correct modes',
             ha='center', fontsize=7.5, color=style.MUTED)
    # Fix metadata so rerendering identical inputs produces identical artifacts.
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    from paper_domain_figure_typography import apply_domain_typography
    apply_domain_typography(fig)
    fig.savefig(FIGURE.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'ModeBench'},
                facecolor='white')
    fig.savefig(FIGURE.with_suffix('.png'), dpi=180, metadata={'Software': 'ModeBench'}, facecolor='white')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify outputs without changing files')
    args = parser.parse_args()
    metadata = build_metadata()
    texts = {
        OUT.with_name(OUT.name + '_initial_table.tex'): table(metadata, replay=False),
        OUT.with_name(OUT.name + '_replay_table.tex'): table(metadata, replay=True),
    }
    if args.check:
        retained = json.loads(OUT.with_suffix('.json').read_text())
        generated = dict(retained)
        assets = generated.pop('assets')
        assert generated == metadata, 'generated scientific metadata is stale'
        for path, text in texts.items():
            assert path.read_text() == text, f'stale generated table: {path}'
        for entry in assets:
            assert sha256(ROOT / entry['path']) == entry['sha256'], f'changed asset: {entry["path"]}'
        print('Fresh PCMD assets verified: 36 contrasts, 18 core intervals, 90 core seed effects; source SHA-256 matches.')
        return
    for path, text in texts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    render_figure(metadata)
    paths = list(texts) + [FIGURE.with_suffix('.pdf'), FIGURE.with_suffix('.png')]
    metadata['assets'] = [{'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)} for path in paths]
    OUT.with_suffix('.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print('Wrote fresh PCMD appendix tables, figure, and scientific metadata.')


if __name__ == '__main__':
    main()
