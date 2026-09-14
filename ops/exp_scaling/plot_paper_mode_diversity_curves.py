#!/usr/bin/env python3
"""Verified-mode diversity throughout training, on the success-conditional axis.

The companion to the ``pass@8`` and ``distinct@8`` factorial curves. Because
pairwise modal diversity conditions on the verified draws, a line that falls
here is reporting concentration of successes rather than loss of them, which is
the distinction the accuracy curves cannot make on their own.

A step contributes to a seed's line only when at least 30 of its prompts return
two verified responses. Segments are never joined across a suppressed step: a
policy whose successes became too rare to measure has not been observed to have
any particular breadth, and drawing a line through that would assert one.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops', ROOT / 'ops' / 'exp_scaling'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paper_style as style
from plot_paper_training_curves import DOMAINS, DOMAIN_LABELS, METHODS, SCALE_LABELS, SCALES

PAYLOAD = ROOT / 'paper/results/mode_diversity_curves.json'
OUT = ROOT / 'paper/figures/factorial_training_curves_pmd'
SCRIPT = Path(__file__).resolve()
SCHEMA = 'paper-mode-diversity-curves-figure-v1'


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def series(payload: dict, level: str = 'level1'):
    """(scale, domain, method) -> step -> list of per-seed PMD values."""
    table: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for curve in payload['curves']:
        if curve['level'] != level:
            continue
        key = (curve['scale'], curve['domain'], curve['method'])
        for point in curve['points']:
            if point['reportable'] and point['pmd'] is not None:
                table[key][point['step']].append(point['pmd'])
    return table


def _segments(steps, values):
    """Split into runs of consecutive measured steps so gaps stay open."""
    runs, current = [], []
    for step, value in zip(steps, values):
        if value is None:
            if current:
                runs.append(current)
                current = []
        else:
            current.append((step, value))
    if current:
        runs.append(current)
    return runs


def build_figure(payload: dict, level: str = 'level1'):
    table = series(payload, level)
    all_steps = sorted({s for v in table.values() for s in v})
    style.apply_rcparams()
    # Only Qwen2.5-0.5B is trained at Level 2, so the row set follows the data
    # rather than assuming the three-scale Level-1 layout.
    scales = [scale for scale in SCALES if any(k[0] == scale for k in table)]
    rows, cols = len(scales), len(DOMAINS)
    # The legend and axis labels need a fixed strip of inches, not a fixed
    # fraction: at one row a fraction tuned for three rows puts the legend on
    # top of the tick labels.
    chrome = .78
    height = style.panel_height(rows) + chrome
    figure, axes = plt.subplots(rows, cols, figsize=(style.WIDTH, height),
                                sharex=True, squeeze=False)
    bottom = chrome / height
    figure.subplots_adjust(left=.085, right=.99, bottom=bottom, top=1 - .06 / height * 3,
                           wspace=.34, hspace=.26)
    # One vertical scale per domain, shared down the model rows, so a reader
    # compares scales within a domain without rescaling between panels.
    column_max = {}
    for domain in DOMAINS:
        values = [v for (sc, dm, me), by_step in table.items() if dm == domain
                  for vals in by_step.values() for v in vals]
        column_max[domain] = max(values) if values else 1.0
    for row, scale in enumerate(scales):
        for col, (domain, label) in enumerate(zip(DOMAINS, DOMAIN_LABELS)):
            axis = axes[row][col]
            for method, spec in METHODS.items():
                by_step = table.get((scale, domain, method), {})
                if not by_step:
                    continue
                means = [statistics.fmean(by_step[s]) if s in by_step else None for s in all_steps]
                for run in _segments(all_steps, means):
                    axis.plot([s for s, _ in run], [v for _, v in run], color=spec['color'],
                              linestyle=spec['dash'], linewidth=style.MEAN_LW, zorder=3)
                lows = [min(by_step[s]) if s in by_step and len(by_step[s]) > 1 else None for s in all_steps]
                highs = [max(by_step[s]) if s in by_step and len(by_step[s]) > 1 else None for s in all_steps]
                for run_lo, run_hi in zip(_segments(all_steps, lows), _segments(all_steps, highs)):
                    axis.fill_between([s for s, _ in run_lo], [v for _, v in run_lo],
                                      [v for _, v in run_hi], color=spec['color'],
                                      alpha=style.BAND_ALPHA, linewidth=0, zorder=2)
            style.style_axis(axis, title=label if row == 0 else None)
            axis.set_ylim(0, column_max[domain] * 1.08)
            axis.margins(x=.02)
            if col == 0:
                axis.set_ylabel(SCALE_LABELS[scale], fontsize=style.LABEL_FONT)
            if row == rows - 1:
                axis.set_xlabel('Training step', fontsize=style.LABEL_FONT)
    figure.text(.012, (bottom + 1) / 2, 'Pairwise modal diversity', rotation=90,
                va='center', ha='center', fontsize=style.TITLE_FONT)
    handles = [Line2D([], [], color=spec['color'], linestyle=spec['dash'],
                      linewidth=style.MEAN_LW, label=spec['label'])
               for spec in METHODS.values()]
    figure.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
                  fontsize=style.SMALL_FONT, bbox_to_anchor=(.54, .004),
                  handletextpad=.45, handlelength=1.8, columnspacing=2.0)
    return figure, table, all_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=PAYLOAD)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--level', default='level1')
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    if payload.get('schema') != 'paper-mode-diversity-curves-v1':
        raise ValueError('unexpected curve payload schema')
    figure, table, steps = build_figure(payload, args.level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix('.pdf'))
    plt.close(figure)
    record = {
        'schema': SCHEMA, 'level': args.level,
        'source': {'path': str(args.payload.relative_to(ROOT)), 'sha256': digest(args.payload)},
        'display': {'y': 'pairwise modal diversity', 'x': 'training step',
                    'min_defined_prompts': payload['definition']['min_defined_prompts'],
                    'gaps_are_not_joined': True,
                    'bands': 'seed range, not a confidence interval'},
        'scope': {'scales': sorted({k[0] for k in table}), 'domains': list(DOMAINS),
                  'methods': list(METHODS), 'steps': steps,
                  'series': {'|'.join(k): {'steps': len(v),
                                           'seeds': max((len(x) for x in v.values()), default=0)}
                             for k, v in sorted(table.items())}},
        'renderer': {'path': str(SCRIPT), 'sha256': digest(SCRIPT),
                     'style_path': str(Path(style.__file__).resolve()),
                     'style_sha256': digest(Path(style.__file__))},
        'outputs': {str(args.output.with_suffix('.pdf')): digest(args.output.with_suffix('.pdf'))},
    }
    args.output.with_suffix('.json').write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'event': 'published', 'output': str(args.output.with_suffix('.pdf')),
                      'series': len(table)}))


if __name__ == '__main__':
    main()
