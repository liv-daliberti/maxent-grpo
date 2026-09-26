#!/usr/bin/env python3
"""Render fixed QA checkpoint bounds from the immutable exact-format summary.

This script only draws independently reduced population bounds. It does not
estimate seed uncertainty, select checkpoints, or modify experiment evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plot_data(path: Path):
    before = digest(path)
    data = json.loads(path.read_text())
    if data.get('status') != 'complete':
        raise ValueError('Summary must be complete')
    sources = data.get('sources', {})
    if not sources:
        raise ValueError('Summary must bind its source files')
    for name, expected in sources.items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Summary source changed: {name}')
    expected_names = {'base'} | {f'{arm}_{step}' for arm in ('maxrl', 'remax') for step in (32, 64, 128)}
    plot = {}
    for cohort, size in (('train', 8), ('dev', 32)):
        rows = data['cohorts'][cohort]['checkpoints']
        if set(rows) != expected_names:
            raise ValueError(f'Unexpected checkpoint set for {cohort}')
        task_ids = None
        plot[cohort] = {}
        for checkpoint, metrics in rows.items():
            tasks = [t for t in data['tasks'] if t['checkpoint'] == checkpoint and t['split'] == cohort]
            ids = {t['task_id'] for t in tasks}
            if metrics['tasks'] != size or len(tasks) != size or len(ids) != size:
                raise ValueError(f'Incomplete or duplicate {cohort}/{checkpoint} task cohort')
            if task_ids is not None and task_ids != ids:
                raise ValueError('Checkpoint task cohorts changed')
            task_ids = ids
            plot[cohort][checkpoint] = {'tasks': size, 'task_ids': sorted(ids)}
            for metric in ('pcmd', 'ed32'):
                lo, hi = (metrics[f'{metric}_{edge}_bound'] for edge in ('lower', 'upper'))
                if any(v is None or not math.isfinite(v) for v in (lo, hi)):
                    raise ValueError('Undefined bounds cannot be silently plotted as zero')
                if lo < -1e-12 or hi < lo - 1e-12 or (metric == 'pcmd' and hi > 1+1e-12):
                    raise ValueError('Invalid bound order or range')
                for edge, value in (('lower', lo), ('upper', hi)):
                    task_mean = statistics.mean(t[f'{metric}_{edge}_bound'] for t in tasks)
                    if not math.isclose(value, task_mean, abs_tol=1e-12, rel_tol=1e-12):
                        raise ValueError('Cohort bound does not match its fixed-task mean')
                plot[cohort][checkpoint][metric] = {'lower_bound': lo, 'upper_bound': hi}
            plot[cohort][checkpoint]['mean_unresolved_probability'] = metrics['unresolved_probability']
    if digest(path) != before:
        raise ValueError('Summary changed while loading')
    return data, plot, before


def render(summary: Path, output: Path):
    targets = {ext: output / f'qa_checkpoint_trajectory.{ext}' for ext in ('png', 'pdf')}
    receipt_path = output / 'qa_checkpoint_trajectory_receipt.json'
    for path in [*targets.values(), receipt_path]:
        if path.exists():
            raise FileExistsError(path)
    data, plot, summary_sha = load_plot_data(summary)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
        'axes.axisbelow': True,
    })
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.8), sharex=True)
    fig.subplots_adjust(left=.12, right=.98, top=.79, bottom=.18, hspace=.34, wspace=.28)
    colors = {'maxrl': '#2864A5', 'remax': '#BE541C'}
    steps = [0, 32, 64, 128]
    for row, (cohort, label) in enumerate((('train', 'Trained questions (8)'), ('dev', 'Development questions (32)'))):
        for col, (metric, title, ylabel) in enumerate((
            ('pcmd', 'Population PCMD', 'Conditional mode diversity'),
            ('ed32', 'Expected distinct correct modes @32', 'Distinct correct modes'),
        )):
            ax = axes[row, col]
            values = []
            for arm in ('maxrl', 'remax'):
                names = ['base'] + [f'{arm}_{step}' for step in steps[1:]]
                lower = [plot[cohort][name][metric]['lower_bound'] for name in names]
                upper = [plot[cohort][name][metric]['upper_bound'] for name in names]
                values.extend(lower+upper)
                ax.fill_between(steps, lower, upper, color=colors[arm], alpha=.20, linewidth=0, zorder=2)
                # Draw both interval boundaries; no midpoint is presented as an estimate.
                ax.plot(steps, lower, color=colors[arm], lw=1.5, marker='o', markersize=3.5, zorder=3)
                ax.plot(steps, upper, color=colors[arm], lw=1.15, ls='--', zorder=3)
            base = plot[cohort]['base'][metric]
            ax.vlines(0, base['lower_bound'], base['upper_bound'], color='#282828', lw=2, zorder=5)
            ax.plot(0, base['lower_bound'], marker='D', ms=4.7, color='#282828', zorder=6)
            ax.plot(0, base['upper_bound'], marker='D', ms=4.7, color='#282828', zorder=6)
            lo, hi = min(values), max(values)
            span = max(hi-lo, .035 if metric == 'pcmd' else .12)
            center = (lo+hi)/2
            ymin, ymax = max(0, center-.68*span), center+.68*span
            if metric == 'pcmd': ymax = min(1, ymax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(-4, 133)
            ax.set_xticks(steps, ['0', '32', '64', '128'])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.grid(axis='y', color='#e3e5e7', lw=.7)
            ax.axvline(128, color='#a5a8ac', lw=.8, ls=':', zorder=1)
            if row == 0: ax.set_title(title, pad=13, fontweight='medium')
            if row == 1: ax.set_xlabel('Training updates')
            ax.set_ylabel(ylabel)
            ax.text(.015, .98, label, transform=ax.transAxes, ha='left', va='top', fontsize=9,
                    bbox={'facecolor':'white', 'alpha':.85, 'edgecolor':'none', 'pad':2})
    fig.suptitle('Multi-answer QA checkpoint trajectory · one paired seed', fontsize=15, y=.965, fontweight='medium')
    legend = [Patch(facecolor=colors['maxrl'], edgecolor=colors['maxrl'], alpha=.65, label='MaxRL bounds'),
              Patch(facecolor=colors['remax'], edgecolor=colors['remax'], alpha=.65, label='Re:Max bounds'),
              Line2D([0], [0], marker='D', color='#282828', lw=0, ms=5, label='Shared initial policy')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.55, .916), ncol=3, frameon=False,
               handlelength=1.4, columnspacing=2.2)
    fig.text(.12, .083, 'Bands bound unenumerated output probability; they are not confidence intervals across seeds.',
             ha='left', fontsize=9, color='#303338')
    fig.text(.12, .053, 'HF teacher-forced probabilities; fixed-task means. Lines connect checkpoints; y axes are zoomed.',
             ha='left', fontsize=9, color='#303338')
    fig.text(.12, .023, 'Step 128 is the predefined primary endpoint. All checkpoints shown; no checkpoint selection.',
             ha='left', fontsize=9, color='#303338')
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(targets['png'], dpi=220)
    fig.savefig(targets['pdf'], metadata={'Title':'QA checkpoint population bounds; one paired seed',
                                        'Subject':f'Source summary SHA256 {summary_sha}',
                                        'Creator':Path(__file__).name})
    plt.close(fig)
    if digest(summary) != summary_sha:
        raise ValueError('Summary changed during rendering')
    receipt = {
        'schema':'real-domains-qa-checkpoint-figure-20260921-v1', 'status':'complete',
        'summary_path':str(summary.resolve()), 'summary_sha256':summary_sha,
        'summary_sources':data['sources'], 'reducer_sha256':data['reducer_sha256'],
        'plot_source_path':str(Path(__file__).resolve()), 'plot_source_sha256':digest(Path(__file__)),
        'outputs':{ext:{'path':str(p.resolve()),'sha256':digest(p)} for ext,p in targets.items()},
        'matplotlib_version':matplotlib.__version__, 'python_version':platform.python_version(),
        'plotted_values':plot, 'primary_checkpoint':128, 'fixed_steps':steps,
        'paired_seeds':1, 'bands':'sharp PCMD / conservative ED32 bounds from unenumerated output mass; not seed confidence intervals',
        'lines':'lower and upper bound boundaries, never an estimated midpoint',
        'y_axes':'zoomed per panel, explicitly disclosed on figure',
        'published_to_paper':False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'receipt':str(receipt_path), 'summary_sha256':summary_sha, 'outputs':receipt['outputs']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.output_dir)
