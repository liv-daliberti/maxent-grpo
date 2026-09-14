#!/usr/bin/env python3
"""Frozen base-model success against success-conditional breadth.

Replaces the ``pass@8`` versus ``distinct@8`` reading of the base-model grid.
The vertical axis is pairwise modal diversity, which is estimated from the
verified draws alone, so a point's height no longer moves with its horizontal
position. Cells whose frozen success is too rare to define the metric on enough
prompts are not given a height: they appear as ticks in a separate strip below
the axis, because a missing measurement and a measured zero are different
claims and must not share a coordinate.

Two figures come from one payload: the four-scale appendix grid and the
single-scale main-text panel.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

import paper_style as style
import plot_paper_modebench_base_grid as base_grid
import plot_paper_modebench_examples as domain_examples
from plot_paper_modebench_base_levels_appendix import SCALE_RAMP, scale_colors

PAYLOAD = ROOT / 'paper/results/mode_diversity_base_grid.json'
OUT_APPENDIX = ROOT / 'paper/figures/mode_diversity_levels_appendix'
OUT_MAIN = ROOT / 'paper/figures/mode_diversity_level_construction'
SCRIPT = Path(__file__).resolve()
SCHEMA = 'paper-mode-diversity-levels-v1'

DOMAINS = base_grid.DOMAINS
MODELS = base_grid.MODELS
LEVELS = ('level1', 'level2', 'level3')
TITLES = dict(zip(DOMAINS, ('Graph', 'Countdown', 'Python', 'MathIR', 'Pantry')))
COLORS = scale_colors()
LEVEL_MARKERS = dict(zip(LEVELS, ('o', 's', '^')))
MARKER_AREA = 22
DOMAIN_BACKGROUNDS = {domain: domain_examples.DOMAIN_PANEL[letter]
                      for domain, letter in zip(DOMAINS, 'ABCDE', strict=True)}

X_LIMITS = (0.0, 1.0)
Y_LIMITS = (0.0, 0.62)
# The unmeasurable strip sits below the data axis in its own band so that a
# tick there cannot be read as a value near zero.
STRIP_TOP = -0.035
STRIP_BOTTOM = -0.10


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(payload_path: Path = PAYLOAD) -> dict:
    payload = json.loads(payload_path.read_text())
    if payload.get('schema') != 'paper-mode-diversity-base-grid-v1':
        raise ValueError('unexpected mode diversity payload schema')
    return payload


def _panel(axis, cells, models, domain):
    axis.set_facecolor(DOMAIN_BACKGROUNDS[domain])
    axis.axhspan(STRIP_BOTTOM, STRIP_TOP, color=style.GRID, alpha=.45, zorder=0, lw=0)
    axis.axhline(0.0, color=style.MUTED, lw=.5, zorder=1)
    for order, model in enumerate(reversed(models)):
        for level in LEVELS:
            cell = next((c for c in cells if c['domain'] == domain
                         and c['model_label'] == model and c['level'] == level), None)
            if cell is None:
                continue
            if cell['reportable']:
                axis.scatter(cell['pass8'], cell['pmd'], s=MARKER_AREA,
                             marker=LEVEL_MARKERS[level], facecolors=COLORS[model],
                             edgecolors=style.INK, alpha=.85, linewidths=.35,
                             zorder=3 + order, clip_on=False)
            else:
                # No height is claimed; the tick records only that the cell exists.
                axis.scatter(cell['pass8'], (STRIP_TOP + STRIP_BOTTOM) / 2,
                             s=9, marker='|', color=style.MUTED, alpha=.85,
                             linewidths=.8, zorder=2, clip_on=False)
    axis.set_title(TITLES[domain], fontsize=8.5, pad=4)
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(STRIP_BOTTOM, Y_LIMITS[1])
    axis.set_xticks([0, .5, 1], ['0', '.5', '1'], fontsize=7.4)
    axis.set_yticks([0, .2, .4, .6], ['0', '.2', '.4', '.6'], fontsize=7.4)
    axis.grid(color=style.GRID, linewidth=.55, zorder=0)
    axis.spines[['top', 'right']].set_visible(False)
    for side in ('bottom', 'left'):
        axis.spines[side].set_color(style.MUTED)
        axis.spines[side].set_linewidth(.6)
    axis.tick_params(length=2, width=.6, pad=2)


def build_figure(payload: dict, models=MODELS, *, figsize=(7.35, 2.35), legends=True):
    cells = payload['cells']
    style.apply_rcparams()
    figure, axes = plt.subplots(1, 5, figsize=figsize, sharex=True, sharey=True)
    bottom = .40 if legends else .22
    figure.subplots_adjust(left=.075, right=.99, bottom=bottom, top=.88, wspace=.20)
    for axis, domain in zip(axes, DOMAINS):
        _panel(axis, cells, models, domain)
    # Centre the label on the plotting band, not the whole canvas, or a short
    # figure pushes the ascender past the top edge.
    figure.text(.016, (bottom + .88) / 2, 'Pairwise modal diversity', rotation=90,
                va='center', ha='center', fontsize=8.5)
    figure.text(.53, bottom - .115, 'pass@8', va='center', ha='center', fontsize=8.5)
    if legends:
        levels = [Line2D([], [], marker=LEVEL_MARKERS[level], linestyle='none', markersize=4.7,
                         markerfacecolor=style.MUTED, markeredgecolor=style.INK,
                         markeredgewidth=.35, label=f'Level {number}')
                  for number, level in enumerate(LEVELS, 1)]
        levels.append(Line2D([], [], marker='|', linestyle='none', markersize=5,
                             color=style.MUTED, markeredgewidth=.9,
                             label='not measurable'))
        scales = [Line2D([], [], marker='o', linestyle='none', markersize=4.7,
                         markerfacecolor=COLORS[model], markeredgecolor=style.INK,
                         markeredgewidth=.35, label=base_grid.MODEL_NAMES[model])
                  for model in models]
        figure.legend(handles=levels, loc='lower center', bbox_to_anchor=(.52, .15),
                      ncol=4, frameon=False, fontsize=7.4, handletextpad=.35,
                      handlelength=.8, columnspacing=1.6, borderaxespad=0)
        figure.legend(handles=scales, loc='lower center', bbox_to_anchor=(.52, .055),
                      ncol=4, frameon=False, fontsize=7.4, handletextpad=.35,
                      handlelength=.8, columnspacing=1.8, borderaxespad=0)
    return figure


def record_for(payload: dict, models, output: Path) -> dict:
    cells = [c for c in payload['cells'] if c['model_label'] in models]
    return {
        'schema': SCHEMA, 'status': 'complete',
        'source': {'path': str(PAYLOAD.relative_to(ROOT)), 'sha256': digest(PAYLOAD)},
        'scope': {'models': list(models), 'levels': list(LEVELS), 'domains': list(DOMAINS),
                  'cells': len(cells),
                  'reportable_cells': sum(1 for c in cells if c['reportable']),
                  'gap_cells': sum(1 for c in cells if not c['reportable'])},
        'display': {'x': 'pass@8', 'y': 'pairwise modal diversity',
                    'x_limits': list(X_LIMITS), 'y_limits': list(Y_LIMITS),
                    'unmeasurable_strip': [STRIP_BOTTOM, STRIP_TOP],
                    'unmeasurable_cells_have_no_height': True,
                    'min_defined_prompts': payload['definition']['min_defined_prompts'],
                    'model_colors': deepcopy({m: COLORS[m] for m in models}),
                    'level_markers': deepcopy(LEVEL_MARKERS)},
        'points': [{k: c[k] for k in ('model_label', 'level', 'domain', 'pass8', 'pmd',
                                      'defined_prompts', 'support', 'reportable')}
                   for c in cells],
        'renderer': {'path': str(SCRIPT), 'sha256': digest(SCRIPT),
                     'style_path': str(Path(style.__file__).resolve()),
                     'style_sha256': digest(Path(style.__file__))},
        'output': str(output.relative_to(ROOT)),
    }


def publish(payload: dict, models, output: Path, *, figsize, legends) -> dict:
    figure = build_figure(payload, models, figsize=figsize, legends=legends)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix('.pdf'))
    plt.close(figure)
    record = record_for(payload, models, output)
    record['outputs'] = {str(output.with_suffix('.pdf')): digest(output.with_suffix('.pdf'))}
    output.with_suffix('.json').write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=PAYLOAD)
    args = parser.parse_args()
    payload = load(args.payload)
    appendix = publish(payload, MODELS, OUT_APPENDIX, figsize=(7.35, 2.35), legends=True)
    main_panel = publish(payload, ('7b',), OUT_MAIN, figsize=(7.35, 2.35), legends=True)
    print(json.dumps({'event': 'published',
                      'appendix': appendix['scope'], 'main': main_panel['scope']}))


if __name__ == '__main__':
    main()
