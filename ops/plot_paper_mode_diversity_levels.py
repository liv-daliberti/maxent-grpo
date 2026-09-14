#!/usr/bin/env python3
"""Frozen base-model success against success-conditional breadth.

Replaces the ``pass@8`` versus ``distinct@8`` reading of the base-model grid.
The vertical axis is pairwise modal diversity, which is estimated from the
verified draws alone, so a point's height no longer moves with its horizontal
position. Cells whose frozen success is too rare to define the metric on enough
prompts are not given a height: they appear as ticks in a separate strip below
the axis, because a missing measurement and a measured zero are different
claims and must not share a coordinate.

Two figures come from one payload: the all-scale appendix grid, which overlays
every scale in five domain panels, and the per-scale appendix grid, which gives
each scale its own row so one scale can be read without the others on top of it.
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
from plot_paper_modebench_base_levels_appendix import SCALE_RAMP, scale_areas, scale_colors, scale_areas

PAYLOAD = ROOT / 'paper/results/mode_diversity_base_grid.json'
OUT_APPENDIX = ROOT / 'paper/figures/mode_diversity_levels_appendix'
OUT_MAIN = ROOT / 'paper/figures/mode_diversity_level_construction'
SCRIPT = Path(__file__).resolve()
SCHEMA = 'paper-mode-diversity-levels-v1'

DOMAINS = base_grid.DOMAINS
MODELS = base_grid.MODELS


def models_in(payload: dict) -> tuple[str, ...]:
    """Every scale the payload actually contains, smallest first."""
    import evaluate_modebench_base_grid as grid
    present = {c['model_label'] for c in payload['cells']}
    known = [m for m in grid.MODEL_PARAMS if m in present]
    return tuple(sorted(known, key=lambda m: grid.MODEL_PARAMS[m]))
LEVELS = ('level1', 'level2', 'level3', 'level4', 'level5')
TITLES = dict(zip(DOMAINS, ('Graph', 'Countdown', 'Python', 'MathIR', 'Pantry')))
COLORS = scale_colors()
LEVEL_MARKERS = dict(zip(LEVELS, ('o', 's', '^', 'D', 'v')))


def levels_in(cells):
    """Every construction level the payload actually contains, easiest first.

    The grid grows a level at a time, so a hard-coded triple would silently
    drop Level 4 cells instead of plotting them.
    """
    present = {c['level'] for c in cells}
    return tuple(level for level in LEVELS if level in present)
MARKER_AREA = 22
DOMAIN_BACKGROUNDS = {domain: domain_examples.DOMAIN_PANEL[letter]
                      for domain, letter in zip(DOMAINS, 'ABCDE', strict=True)}

X_LIMITS = (0.0, 1.0)
Y_LIMITS = (0.0, 0.80)
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


FRONTIER = ROOT / 'paper/results/mode_diversity_frontier_points.json'
# Hosted deployments have no parameter count, so they sit off the scale ramp:
# one neutral ink-grey fill, with the level shapes they share with the grid.
FRONTIER_FACE = '#6B7280'


def load_frontier(path: Path = FRONTIER):
    if not Path(path).is_file():
        return []
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') != 'mode-diversity-frontier-points-v1':
        raise ValueError('unexpected frontier point schema')
    return payload['cells']


def _panel(axis, cells, models, domain, frontier=(), colors=None):
    colors = COLORS if colors is None else colors
    axis.set_facecolor(DOMAIN_BACKGROUNDS[domain])
    axis.axhspan(STRIP_BOTTOM, STRIP_TOP, color=style.GRID, alpha=.45, zorder=0, lw=0)
    axis.axhline(0.0, color=style.MUTED, lw=.5, zorder=1)
    for order, model in enumerate(reversed(models)):
        for level in levels_in(cells):
            cell = next((c for c in cells if c['domain'] == domain
                         and c['model_label'] == model and c['level'] == level), None)
            if cell is None:
                continue
            if cell['reportable']:
                axis.scatter(cell['pass8'], cell['pmd'], s=MARKER_AREA,
                             marker=LEVEL_MARKERS[level], facecolors=colors[model],
                             edgecolors=style.INK, alpha=.85, linewidths=.35,
                             zorder=3 + order, clip_on=False)
            else:
                # No height is claimed; the tick records only that the cell exists.
                axis.scatter(cell['pass8'], (STRIP_TOP + STRIP_BOTTOM) / 2,
                             s=9, marker='|', color=style.MUTED, alpha=.85,
                             linewidths=.8, zorder=2, clip_on=False)
    for cell in frontier:
        if cell['domain'] != domain or not cell.get('reportable'):
            continue
        axis.scatter(cell['pass8'], cell['pmd'], s=MARKER_AREA * 1.15,
                     marker=LEVEL_MARKERS[cell['level']], facecolors=FRONTIER_FACE,
                     edgecolors='white', alpha=.95, linewidths=.55,
                     zorder=9, clip_on=False)
    axis.set_title(TITLES[domain], fontsize=8.5, pad=4)
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(STRIP_BOTTOM, Y_LIMITS[1])
    axis.set_xticks([0, .5, 1], ['0', '.5', '1'], fontsize=7.4)
    axis.set_yticks([0, .2, .4, .6, .8], ['0', '.2', '.4', '.6', '.8'], fontsize=7.4)
    axis.grid(color=style.GRID, linewidth=.55, zorder=0)
    axis.spines[['top', 'right']].set_visible(False)
    for side in ('bottom', 'left'):
        axis.spines[side].set_color(style.MUTED)
        axis.spines[side].set_linewidth(.6)
    axis.tick_params(length=2, width=.6, pad=2)


def build_figure(payload: dict, models=MODELS, *, figsize=(6.4, 2.10),
                 legends=True, frontier=None):
    cells = payload['cells']
    frontier = load_frontier() if frontier is None else frontier
    colors = scale_colors(tuple(models))
    style.apply_rcparams()
    figure, axes = plt.subplots(1, 5, figsize=figsize, sharex=True, sharey=True)
    bottom = .40 if legends else .22
    figure.subplots_adjust(left=.068, right=.995, bottom=bottom, top=.88, wspace=.16)
    for axis, domain in zip(axes, DOMAINS):
        _panel(axis, cells, models, domain, frontier, colors)
    # Centre the label on the plotting band, not the whole canvas, or a short
    # figure pushes the ascender past the top edge.
    figure.text(.016, (bottom + .88) / 2, 'Pairwise modal diversity', rotation=90,
                va='center', ha='center', fontsize=8.5)
    figure.text(.53, bottom - .115, 'pass@8', va='center', ha='center', fontsize=8.5)
    if legends:
        levels = [Line2D([], [], marker=LEVEL_MARKERS[level], linestyle='none', markersize=4.7,
                         markerfacecolor=style.MUTED, markeredgecolor=style.INK,
                         markeredgewidth=.35, label='Level ' + level[len('level'):])
                  for level in levels_in(cells)]
        levels.append(Line2D([], [], marker='|', linestyle='none', markersize=5,
                             color=style.MUTED, markeredgewidth=.9,
                             label='not measurable'))
        scales = [Line2D([], [], marker='o', linestyle='none', markersize=4.7,
                         markerfacecolor=colors[model], markeredgecolor=style.INK,
                         markeredgewidth=.35, label=base_grid.MODEL_NAMES[model])
                  for model in models]
        if frontier:
            scales.append(Line2D([], [], marker='o', linestyle='none', markersize=5.0,
                                 markerfacecolor=FRONTIER_FACE, markeredgecolor='white',
                                 markeredgewidth=.55, label='frontier'))
        figure.legend(handles=levels, loc='lower center', bbox_to_anchor=(.53, .155),
                      ncol=len(levels), frameon=False, fontsize=7.2, handletextpad=.3,
                      handlelength=.8, columnspacing=1.3, borderaxespad=0)
        figure.legend(handles=scales, loc='lower center', bbox_to_anchor=(.53, .015),
                      ncol=min(9, len(scales)), frameon=False, fontsize=7.2,
                      handletextpad=.3, handlelength=.8, columnspacing=1.0,
                      borderaxespad=0)
    return figure


def _row_panel(axis, cells, model, domain, frontier, colors, *, top, left):
    """One scale, one domain. Same encoding as _panel, but labelled for a grid."""
    _panel(axis, cells, (model,), domain, frontier, colors)
    axis.set_title(TITLES[domain] if top else '', fontsize=8.0, pad=3)
    if left:
        axis.set_ylabel(base_grid.MODEL_NAMES[model], fontsize=7.6, labelpad=3)
    axis.tick_params(labelbottom=False, labelleft=left, labelsize=6.6)


def build_scale_grid(payload: dict, models, *, frontier=None, figsize=None):
    """Every scale on its own row, so no scale is hidden behind another.

    The overlay in the all-scale appendix figure answers how the scales compare;
    it cannot show where a single scale sits when its points fall under a darker
    one. One row per scale answers that, at the cost of a taller figure.
    """
    cells = payload['cells']
    # The hosted frontier deployments carry no parameter count, so they belong to no
    # row. Drawing them in all eleven would repeat the same points eleven times and
    # stop any row from being that scale alone, which is the whole point here.
    frontier = () if frontier is None else frontier
    colors = scale_colors(tuple(models))
    style.apply_rcparams()
    rows = len(models)
    figsize = (6.4, 0.60 * rows + 1.00) if figsize is None else figsize
    figure, axes = plt.subplots(rows, len(DOMAINS), figsize=figsize,
                                sharex=True, sharey=True, squeeze=False)
    bottom = 0.95 / figsize[1]
    top = 1 - 0.34 / figsize[1]
    figure.subplots_adjust(left=.093, right=.995, bottom=bottom, top=top,
                           wspace=.14, hspace=.22)
    for row, model in enumerate(models):
        for column, domain in enumerate(DOMAINS):
            _row_panel(axes[row][column], cells, model, domain, frontier, colors,
                       top=row == 0, left=column == 0)
    for column in range(len(DOMAINS)):
        axes[-1][column].tick_params(labelbottom=True, labelsize=6.6)
    figure.text(.012, (bottom + top) / 2, 'Pairwise modal diversity', rotation=90,
                va='center', ha='center', fontsize=8.5)
    figure.text(.54, bottom - 0.62 / figsize[1], 'pass@8',
                va='center', ha='center', fontsize=8.5)
    levels = [Line2D([], [], marker=LEVEL_MARKERS[level], linestyle='none', markersize=4.7,
                     markerfacecolor=style.MUTED, markeredgecolor=style.INK,
                     markeredgewidth=.35, label='Level ' + level[len('level'):])
              for level in levels_in(cells)]
    levels.append(Line2D([], [], marker='|', linestyle='none', markersize=5,
                         color=style.MUTED, markeredgewidth=.9, label='not measurable'))
    if frontier:
        levels.append(Line2D([], [], marker='o', linestyle='none', markersize=5.0,
                             markerfacecolor=FRONTIER_FACE, markeredgecolor='white',
                             markeredgewidth=.55, label='frontier'))
    figure.legend(handles=levels, loc='lower center',
                  bbox_to_anchor=(.54, 0.10 / figsize[1]),
                  ncol=len(levels), frameon=False, fontsize=7.2, handletextpad=.3,
                  handlelength=.8, columnspacing=1.3, borderaxespad=0)
    return figure


def record_for(payload: dict, models, output: Path) -> dict:
    cells = [c for c in payload['cells'] if c['model_label'] in models]
    return {
        'schema': SCHEMA, 'status': 'complete',
        'source': {'path': str(PAYLOAD.relative_to(ROOT)), 'sha256': digest(PAYLOAD)},
        'scope': {'models': list(models), 'levels': list(levels_in(cells)), 'domains': list(DOMAINS),
                  'cells': len(cells),
                  'reportable_cells': sum(1 for c in cells if c['reportable']),
                  'gap_cells': sum(1 for c in cells if not c['reportable'])},
        'display': {'x': 'pass@8', 'y': 'pairwise modal diversity',
                    'x_limits': list(X_LIMITS), 'y_limits': list(Y_LIMITS),
                    'unmeasurable_strip': [STRIP_BOTTOM, STRIP_TOP],
                    'unmeasurable_cells_have_no_height': True,
                    'min_defined_prompts': payload['definition']['min_defined_prompts'],
                    'model_colors': deepcopy({m: scale_colors(tuple(models))[m] for m in models}),
                    'marker_area': MARKER_AREA,
                    'encoding': 'scale -> colour ramp; level -> marker shape',
                    'level_markers': deepcopy(LEVEL_MARKERS)},
        'points': [{k: c[k] for k in ('model_label', 'level', 'domain', 'pass8', 'pmd',
                                      'defined_prompts', 'support', 'reportable')}
                   for c in cells],
        'renderer': {'path': str(SCRIPT), 'sha256': digest(SCRIPT),
                     'style_path': str(Path(style.__file__).resolve()),
                     'style_sha256': digest(Path(style.__file__))},
        'output': str(output.relative_to(ROOT)),
    }


def publish(payload: dict, models, output: Path, *, figsize, legends,
            builder=None) -> dict:
    figure = (build_scale_grid(payload, models, figsize=figsize) if builder == 'scale_grid'
              else build_figure(payload, models, figsize=figsize, legends=legends))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix('.pdf'))
    plt.close(figure)
    record = record_for(payload, models, output)
    record['display']['layout'] = ('one row per scale, five domain columns'
                                   if builder == 'scale_grid'
                                   else 'five domain panels, scales overlaid')
    record['outputs'] = {str(output.with_suffix('.pdf')): digest(output.with_suffix('.pdf'))}
    output.with_suffix('.json').write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=PAYLOAD)
    args = parser.parse_args()
    payload = load(args.payload)
    present = models_in(payload)
    appendix = publish(payload, present, OUT_APPENDIX, figsize=(6.4, 2.10), legends=True)
    # The construction figure used to carry 7B alone. Every scale now gets its own
    # row, so a reader can look up any single scale rather than only that one.
    construction = publish(payload, present, OUT_MAIN, figsize=None, legends=True,
                           builder='scale_grid')
    print(json.dumps({'event': 'published',
                      'appendix': appendix['scope'],
                      'construction': construction['scope']}))


if __name__ == '__main__':
    main()
