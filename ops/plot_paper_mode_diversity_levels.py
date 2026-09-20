#!/usr/bin/env python3
"""Frozen base-model success against success-conditional breadth.

Replaces the ``pass@8`` versus ``distinct@8`` reading of the base-model grid.
The vertical axis is pairwise correct-mode diversity, which is estimated from the
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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

import paper_style as style
import plot_paper_modebench_base_grid as base_grid
import plot_paper_modebench_examples as domain_examples
from plot_paper_modebench_base_levels_appendix import SCALE_RAMP, scale_areas, scale_colors, scale_areas

PAYLOAD = ROOT / 'paper/results/mode_diversity_base_grid.json'
# Stem names predate their current placement and are pinned by the paper
# contract, so they are kept: OUT_APPENDIX is the main-body scatter (Qwen2.5
# scales alone), OUT_FAMILIES is its cross-family counterpart in the appendix,
# and OUT_MAIN is the appendix grid of one row per scale.
OUT_APPENDIX = ROOT / 'paper/figures/mode_diversity_levels_appendix'
OUT_FAMILIES = ROOT / 'paper/figures/mode_diversity_families_appendix'
OUT_MAIN = ROOT / 'paper/figures/mode_diversity_level_construction'
# The main body carries one hosted deployment so the scale story stays readable;
# the full seven-deployment cohort is the appendix figure's job.
MAIN_FRONTIER = 'GPT-5.6 Sol'
MAIN_FAMILY = 'Qwen2.5'
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


def levels_in(cells, *extra):
    """Every construction level the plotted data actually contains, easiest first.

    The grid grows a level at a time, so a hard-coded triple would silently
    drop Level 4 cells instead of plotting them. Frontier points are passed in
    as well, because a level can arrive in the hosted overlay before the local
    grid finishes it, and a marker drawn with no legend entry is worse than one
    that is simply absent.
    """
    present = {c['level'] for c in cells}
    for group in extra:
        present |= {c['level'] for c in group or ()}
    if present == {'levels_mean'}:
        return ('levels_mean',)
    return tuple(level for level in LEVELS if level in present)
MARKER_AREA = 22
# Below the paper's support bar PCMD is still estimated, just less precisely: at
# 10-29 defined prompts its standard error runs about 1.5x that of a cell above
# the bar, not off the scale. Draw those hollow with their error bar rather than
# discarding a measurement the grid actually made. The band that cannot be
# placed is 1-4 prompts, where the median standard error is .28; the threshold
# now sits at the edge of that band rather than well above it, which places 24
# further cells hollow. It is not lowered past 5: under that, a standard error
# of .000 records two or three prompts agreeing, not a precise estimate, so a
# mark would read as a measurement the cell cannot support.
PROVISIONAL_MIN_DEFINED = 5
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
# Black, not grey: the hosted deployments are a small set of points that carry
# their own claim, and a grey fill lost them against the domain panel tints.
FRONTIER_FACE = '#000000'
ICONS = ROOT / 'paper/icons'
# A provider mark may only stand beside a single provider's points. The
# families and construction plates carry seven deployments at once, so they get
# the generic label and no logo rather than one vendor's mark over all of them.
PROVIDER_ICONS = {'GPT-5.6 Sol': 'openai.png', 'GPT-5.4': 'openai.png',
                  'Claude Opus 5': 'claude.png', 'Claude Opus 4.8': 'claude.png',
                  'DeepSeek V4 Pro': 'deepseek.png', 'Kimi K3': 'kimi.png',
                  'Grok 4.3': 'grok_official_docs_print.png'}
# Provider marks sit beside the names they belong to. Drawn after layout, in
# figure coordinates, so a logo tracks its legend block rather than guessing at
# a position that shifts whenever an entry is added.
LOGO_HEIGHT_IN = 0.105


def _family(model: str) -> str:
    import evaluate_modebench_base_grid as grid
    return grid.MODEL_FAMILY.get(model, model)


def _logo(figure, legend, name, *, pad=0.006):
    path = ICONS / name
    if not path.is_file():
        return
    image = plt.imread(str(path))
    figure.canvas.draw()
    box = legend.get_window_extent().transformed(figure.transFigure.inverted())
    zoom = LOGO_HEIGHT_IN * figure.dpi / image.shape[0]
    figure.add_artist(AnnotationBbox(
        OffsetImage(image, zoom=zoom), (box.x0 - pad, (box.y0 + box.y1) / 2),
        xycoords='figure fraction', frameon=False, box_alignment=(1.0, 0.5),
        annotation_clip=False))


def load_frontier(path: Path = FRONTIER):
    if not Path(path).is_file():
        return []
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') != 'mode-diversity-frontier-points-v1':
        raise ValueError('unexpected frontier point schema')
    return payload['cells']


def _trail(axis, points, colour, order, *, width=.65, alpha=.45, zorder=2):
    """Join consecutive levels of one series into a difficulty trajectory.

    Only adjacent levels are joined. Bridging a level whose success was too
    rare to estimate PCMD would draw a step the grid never measured, so a gap
    breaks the line rather than being spanned.
    """
    run = []
    for level in order:
        cell = points.get(level)
        if cell is not None and cell.get('reportable'):
            run.append((cell['pass8'], cell['pmd']))
            continue
        if len(run) > 1:
            axis.plot([x for x, _ in run], [y for _, y in run], color=colour,
                      lw=width, alpha=alpha, zorder=zorder, solid_capstyle='round')
        run = []
    if len(run) > 1:
        axis.plot([x for x, _ in run], [y for _, y in run], color=colour,
                  lw=width, alpha=alpha, zorder=zorder, solid_capstyle='round')


def collapse_cells(cells, key='model_label'):
    """Average each series over the levels it was measured at.

    The main-body plate answers a question about scale and about the separation
    of success from breadth; it does not answer one about difficulty, because
    difficulty does not move PCMD (Level 1 to Level 5 marginals are flat in four
    of five domains). Spending the marker-shape channel on a factor with no
    effect cost the plate its legibility, so the levels are averaged into one
    point per series and the per-level reading moves to the construction plate.

    The three-way support rule is preserved rather than flattened: a series is
    averaged over its reportable levels when it has any, else over its
    provisional levels, else it keeps no height at all. So a hollow mark here
    means no level of any difficulty cleared the support bar for that series,
    which is a stronger and cleaner statement than the per-level hollow.

    Every level the grid measured contributes, Level 4 MathIR included.
    """
    groups = {}
    for cell in cells:
        groups.setdefault((cell['domain'], cell[key]), []).append(cell)
    out = []
    for (domain, name), group in groups.items():
        sized = [c for c in group if c.get('pmd') is not None and c.get('pass8') is not None]
        if not sized:
            continue
        reportable = [c for c in sized if c.get('reportable')]
        provisional = [c for c in sized if not c.get('reportable')
                       and c.get('defined_prompts', 0) >= PROVISIONAL_MIN_DEFINED]
        used = reportable or provisional or sized
        out.append({
            'domain': domain, key: name, 'level': 'levels_mean',
            'pass8': sum(c['pass8'] for c in used) / len(used),
            'pmd': sum(c['pmd'] for c in used) / len(used),
            'reportable': bool(reportable),
            'defined_prompts': max(c.get('defined_prompts', 0) for c in used),
            'support': sum(c.get('support', 0) or 0 for c in used) / len(used),
            'levels_averaged': sorted(c['level'] for c in used),
            'levels_measured': sorted(c['level'] for c in sized),
        })
    return out


def _panel(axis, cells, models, domain, frontier=(), colors=None, connect=False):
    colors = COLORS if colors is None else colors
    axis.set_facecolor(DOMAIN_BACKGROUNDS[domain])
    axis.axhspan(STRIP_BOTTOM, STRIP_TOP, color=style.GRID, alpha=.45, zorder=0, lw=0)
    axis.axhline(0.0, color=style.MUTED, lw=.5, zorder=1)
    ladder = levels_in(cells)
    collapsed = ladder == ('levels_mean',)
    marker_for = (lambda level: 'o') if collapsed else LEVEL_MARKERS.__getitem__
    for order, model in enumerate(reversed(models)):
        if connect:
            _trail(axis, {c['level']: c for c in cells if c['domain'] == domain
                          and c['model_label'] == model},
                   colors[model], ladder, zorder=2)
        for level in ladder:
            cell = next((c for c in cells if c['domain'] == domain
                         and c['model_label'] == model and c['level'] == level), None)
            if cell is None:
                continue
            if cell['reportable']:
                axis.scatter(cell['pass8'], cell['pmd'], s=MARKER_AREA,
                             marker=marker_for(level), facecolors=colors[model],
                             edgecolors=style.INK, alpha=.85, linewidths=.35,
                             zorder=3 + order, clip_on=False)
            elif cell['defined_prompts'] >= PROVISIONAL_MIN_DEFINED:
                error = cell.get('pmd_standard_error')
                if error:
                    axis.errorbar(cell['pass8'], cell['pmd'], yerr=error, fmt='none',
                                  ecolor=colors[model], elinewidth=.55, capsize=1.1,
                                  capthick=.55, alpha=.75, zorder=2 + order,
                                  clip_on=False)
                axis.scatter(cell['pass8'], cell['pmd'], s=MARKER_AREA,
                             marker=marker_for(level), facecolors='none',
                             edgecolors=colors[model], alpha=.95, linewidths=.7,
                             zorder=3 + order, clip_on=False)
            else:
                # No height is claimed; the tick records only that the cell exists.
                axis.scatter(cell['pass8'], (STRIP_TOP + STRIP_BOTTOM) / 2,
                             s=9, marker='|', color=style.MUTED, alpha=.85,
                             linewidths=.8, zorder=2, clip_on=False)
    if connect:
        for name in sorted({c['model'] for c in frontier if c['domain'] == domain}):
            _trail(axis, {c['level']: c for c in frontier
                          if c['domain'] == domain and c['model'] == name},
                   FRONTIER_FACE, ladder, width=.6, alpha=.55, zorder=11)
    for cell in frontier:
        if cell['domain'] != domain or not cell.get('reportable'):
            continue
        axis.scatter(cell['pass8'], cell['pmd'], s=MARKER_AREA * 1.15,
                     marker=('o' if collapsed else LEVEL_MARKERS[cell['level']]), facecolors=FRONTIER_FACE,
                     edgecolors='white', alpha=1.0, linewidths=.6,
                     zorder=12, clip_on=False)
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
                 legends=True, frontier=None, connect=False, frontier_models=None,
                 collapse=False):
    cells = payload['cells']
    frontier = load_frontier() if frontier is None else frontier
    if frontier_models is not None:
        frontier = [c for c in frontier if c['model'] in frontier_models]
    if collapse:
        cells = collapse_cells([c for c in cells if c['model_label'] in models])
        frontier = collapse_cells(frontier, key='model')
    colors = scale_colors(tuple(models))
    style.apply_rcparams()
    figure, axes = plt.subplots(1, 5, figsize=figsize, sharex=True, sharey=True)
    bottom = .40 if legends else .22
    figure.subplots_adjust(left=.068, right=.995, bottom=bottom, top=.88, wspace=.16)
    for axis, domain in zip(axes, DOMAINS):
        _panel(axis, cells, models, domain, frontier, colors, connect=connect)
    # Centre the label on the plotting band, not the whole canvas, or a short
    # figure pushes the ascender past the top edge.
    figure.text(.026, (bottom + .88) / 2, 'Pairwise correct-mode\ndiversity', rotation=90,
                linespacing=.95,
                va='center', ha='center', fontsize=8.5)
    figure.text(.53, bottom - .115, 'pass@8', va='center', ha='center', fontsize=8.5)
    if legends:
        if collapse:
            levels = [Line2D([], [], marker='o', linestyle='none', markersize=4.7,
                             markerfacecolor=style.MUTED, markeredgecolor=style.INK,
                             markeredgewidth=.35, label='mean over Levels 1\u20135')]
        else:
            levels = [Line2D([], [], marker=LEVEL_MARKERS[level], linestyle='none', markersize=4.7,
                             markerfacecolor=style.MUTED, markeredgecolor=style.INK,
                             markeredgewidth=.35, label='Level ' + level[len('level'):])
                      for level in levels_in(cells, frontier)]
        levels.append(Line2D([], [], marker='o', linestyle='none', markersize=4.7,
                             markerfacecolor='none', markeredgecolor=style.MUTED,
                             markeredgewidth=.7,
                             label='below support at every level' if collapse else 'provisional'))
        # The tick strip survives collapsing: a series whose every level stayed
        # under the estimation floor still gets no height, so it still needs its
        # legend entry. Only drop the entry when no tick is actually drawn.
        if not collapse or any(c.get('defined_prompts', 0) < PROVISIONAL_MIN_DEFINED
                               for c in cells):
            levels.append(Line2D([], [], marker='|', linestyle='none', markersize=5,
                                 color=style.MUTED, markeredgewidth=.9,
                                 label='too rare to estimate'))
        scales = [Line2D([], [], marker='o', linestyle='none', markersize=4.7,
                         markerfacecolor=colors[model], markeredgecolor=style.INK,
                         markeredgewidth=.35, label=base_grid.MODEL_NAMES[model])
                  for model in models]
        hosted = []
        if frontier:
            names = sorted({c['model'] for c in frontier})
            hosted.append(Line2D([], [], marker='o', linestyle='none', markersize=5.0,
                                 markerfacecolor=FRONTIER_FACE, markeredgecolor='white',
                                 markeredgewidth=.55,
                                 label=names[0] if len(names) == 1 else 'frontier'))
        figure.legend(handles=levels, loc='lower center', bbox_to_anchor=(.53, .155),
                      ncol=len(levels), frameon=False, fontsize=7.2, handletextpad=.3,
                      handlelength=.8, columnspacing=1.3, borderaxespad=0)
        scale_legend = figure.legend(
            handles=scales, loc='lower center',
            bbox_to_anchor=(.42 if hosted else .53, .015),
            ncol=min(9, len(scales)), frameon=False, fontsize=7.2,
            handletextpad=.3, handlelength=.8, columnspacing=1.0, borderaxespad=0)
        # The hosted mark below is drawn only for a single deployment; the
        # scale mark needs the same guard. This plate is built twice: once over
        # Qwen2.5 alone for the main body, where the logo names the family, and
        # once over SmolLM2, Qwen2.5, Falcon3 and OLMo-2 for the appendix, where
        # a Qwen mark beside a four-family legend claims the wrong maker.
        if {_family(model) for model in models} == {'Qwen2.5'}:
            _logo(figure, scale_legend, 'qwen.png')
        if hosted:
            hosted_legend = figure.legend(
                handles=hosted, loc='lower center', bbox_to_anchor=(.88, .015),
                ncol=1, frameon=False, fontsize=7.2, handletextpad=.3,
                handlelength=.8, borderaxespad=0)
            figure.add_artist(scale_legend)
            if len(names) == 1 and names[0] in PROVIDER_ICONS:
                _logo(figure, hosted_legend, PROVIDER_ICONS[names[0]])
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
    # Per-row height trimmed from .60: the plate reads the same and gives the
    # page back a little vertical room.
    figsize = (6.4, 0.44 * rows + 0.88) if figsize is None else figsize
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
    figure.text(.022, (bottom + top) / 2, 'Pairwise correct-mode\ndiversity', rotation=90,
                linespacing=.95,
                va='center', ha='center', fontsize=8.5)
    figure.text(.54, bottom - 0.62 / figsize[1], 'pass@8',
                va='center', ha='center', fontsize=8.5)
    levels = [Line2D([], [], marker=LEVEL_MARKERS[level], linestyle='none', markersize=4.7,
                     markerfacecolor=style.MUTED, markeredgecolor=style.INK,
                     markeredgewidth=.35, label='Level ' + level[len('level'):])
              for level in levels_in(cells)]
    levels.append(Line2D([], [], marker='o', linestyle='none', markersize=4.7,
                         markerfacecolor='none', markeredgecolor=style.MUTED,
                         markeredgewidth=.7, label='provisional'))
    levels.append(Line2D([], [], marker='|', linestyle='none', markersize=5,
                         color=style.MUTED, markeredgewidth=.9, label='too rare to estimate'))
    if frontier:
        levels.append(Line2D([], [], marker='o', linestyle='none', markersize=5.0,
                             markerfacecolor=FRONTIER_FACE, markeredgecolor='white',
                             markeredgewidth=.55, label='frontier'))
    figure.legend(handles=levels, loc='lower center',
                  bbox_to_anchor=(.54, 0.10 / figsize[1]),
                  ncol=len(levels), frameon=False, fontsize=7.2, handletextpad=.3,
                  handlelength=.8, columnspacing=1.3, borderaxespad=0)
    return figure


def record_for(payload: dict, models, output: Path, frontier_models=None,
               collapse=False) -> dict:
    cells = [c for c in payload['cells'] if c['model_label'] in models]
    drawn_cells = collapse_cells(cells) if collapse else cells
    # Which hosted deployments this figure actually draws. The main body carries
    # one and the appendix carries the cohort, so a record that omitted this
    # would not distinguish them.
    drawn = sorted({c['model'] for c in load_frontier()
                    if frontier_models is None or c['model'] in frontier_models})
    return {
        'schema': SCHEMA, 'status': 'complete',
        'source': {'path': str(PAYLOAD.relative_to(ROOT)), 'sha256': digest(PAYLOAD)},
        'scope': {'models': list(models), 'levels': list(levels_in(cells)), 'domains': list(DOMAINS),
                  'frontier_models': drawn,
                  'cells': len(cells),
                  'reportable_cells': sum(1 for c in cells if c['reportable']),
                  'gap_cells': sum(1 for c in cells if not c['reportable'])},
        'display': {'x': 'pass@8', 'y': 'pairwise correct-mode diversity',
                    'x_limits': list(X_LIMITS), 'y_limits': list(Y_LIMITS),
                    'unmeasurable_strip': [STRIP_BOTTOM, STRIP_TOP],
                    'unmeasurable_cells_have_no_height': True,
                    'provisional_min_defined_prompts': PROVISIONAL_MIN_DEFINED,
                    'provisional_cells': sum(
                        1 for c in cells if not c['reportable']
                        and c['defined_prompts'] >= PROVISIONAL_MIN_DEFINED),
                    'provisional_encoding': 'hollow marker with PCMD standard error',
                    'min_defined_prompts': payload['definition']['min_defined_prompts'],
                    'model_colors': deepcopy({m: scale_colors(tuple(models))[m] for m in models}),
                    'marker_area': MARKER_AREA,
                    'encoding': ('scale -> colour ramp; levels averaged into one mark'
                                 if collapse else 'scale -> colour ramp; level -> marker shape'),
                    'levels_collapsed': collapse,
                    'marks_drawn': len(drawn_cells),
                    'level_markers': None if collapse else deepcopy(LEVEL_MARKERS)},
        'points': [{k: c[k] for k in ('model_label', 'level', 'domain', 'pass8', 'pmd',
                                      'defined_prompts', 'support', 'reportable')}
                   for c in cells],
        'renderer': {'path': str(SCRIPT), 'sha256': digest(SCRIPT),
                     'style_path': str(Path(style.__file__).resolve()),
                     'style_sha256': digest(Path(style.__file__))},
        'output': str(output.relative_to(ROOT)),
    }


def publish(payload: dict, models, output: Path, *, figsize, legends,
            builder=None, connect=False, frontier_models=None, collapse=False) -> dict:
    figure = (build_scale_grid(payload, models, figsize=figsize) if builder == 'scale_grid'
              else build_figure(payload, models, figsize=figsize, legends=legends,
                                connect=connect, frontier_models=frontier_models,
                                collapse=collapse))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix('.pdf'))
    plt.close(figure)
    record = record_for(payload, models, output, frontier_models, collapse=collapse)
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
    import evaluate_modebench_base_grid as grid
    family = tuple(m for m in present if grid.MODEL_FAMILY.get(m) == MAIN_FAMILY)
    if not family:
        raise ValueError(f'payload carries no {MAIN_FAMILY} scale to plot in the main body')
    # The main body reads the scale axis within one family, where the training
    # recipe is held fixed; the cross-family comparison moves to the appendix.
    # The main-body plate averages the levels into one mark per scale; the
    # per-level reading is Fig. level-construction in the appendix.
    main_body = publish(payload, family, OUT_APPENDIX, figsize=(6.4, 2.10), legends=True,
                        frontier_models={MAIN_FRONTIER}, collapse=True)
    families = publish(payload, present, OUT_FAMILIES, figsize=(6.4, 2.10), legends=True)
    # The construction figure used to carry 7B alone. Every scale now gets its own
    # row, so a reader can look up any single scale rather than only that one.
    construction = publish(payload, present, OUT_MAIN, figsize=None, legends=True,
                           builder='scale_grid')
    print(json.dumps({'event': 'published',
                      'main_body': main_body['scope'],
                      'families': families['scope'],
                      'construction': construction['scope']}))


if __name__ == '__main__':
    main()
