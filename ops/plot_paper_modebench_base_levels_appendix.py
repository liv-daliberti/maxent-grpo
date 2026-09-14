#!/usr/bin/env python3
"""Publish the complete frozen-Qwen comparison at admitted Levels 1--3.

The appendix scope is 60 model/level/domain cells, all independently measured.
Levels 4--5 remain outside this scope until their datasets are admitted; they
are never represented by zero-valued coordinates. The separate full 100-cell
publication gate in plot_paper_modebench_base_grid is unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paper_style as style
import plot_paper_modebench_base_grid as base_grid
import plot_paper_modebench_examples as domain_examples

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/modebench_base_level_grid_20260911/status_updates/20260912T152423Z/figure_source.json'
OUT = ROOT / 'paper/figures/modebench_base_levels_appendix'
SCRIPT = Path(__file__).resolve()
SCHEMA = 'modebench-base-levels-appendix-v1'
DOMAINS = base_grid.DOMAINS
MODELS = base_grid.MODELS
LEVELS = ('level1', 'level2', 'level3')
TITLES = dict(zip(DOMAINS, ('Graph', 'Countdown', 'Python', 'MathIR', 'Pantry')))
# Scale is the ordered variable, so it gets the ordered channel: a sequential
# ramp sampled over however many scales are present, dark enough to stay legible
# on the tinted domain panels. Levels are unordered for the reader's purposes
# and take marker shape. Adding a scale needs no new colour constant.
# Green -> amber -> red with model scale, so the largest models read hottest.
# Marker area rises with scale as well: the ordering is encoded twice, which
# keeps it legible to a red-green colourblind reader.
SCALE_RAMP = ('#1A9850', '#66BD63', '#FDAE61', '#F46D43',
              '#D73027', '#A50026', '#7A0018')
SCALE_AREAS = (14, 20, 27, 35, 44, 54, 66)


def _spread(values, models):
    if len(models) > len(values):
        raise ValueError('extend the scale ramp before adding more model scales')
    if len(models) == 1:
        return {models[0]: values[-1]}
    step = (len(values) - 1) / (len(models) - 1)
    return {model: values[round(index * step)] for index, model in enumerate(models)}


def scale_colors(models=MODELS):
    """Map each scale onto the ramp by log parameter count.

    Position on the ramp then means size, not list order, so two families at
    the same parameter count get the same colour and a gap in the sizes shows
    as a gap in the colour.
    """
    import math
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    import evaluate_modebench_base_grid as grid
    params = {m: grid.MODEL_PARAMS[m] for m in models if m in grid.MODEL_PARAMS}
    if len(params) < len(models):          # unregistered scale: fall back to order
        return _spread(SCALE_RAMP, tuple(models))
    ramp = LinearSegmentedColormap.from_list('scale', SCALE_RAMP)
    lo, hi = math.log(min(params.values())), math.log(max(params.values()))
    if hi <= lo:
        return {m: SCALE_RAMP[-1] for m in models}
    return {m: to_hex(ramp((math.log(params[m]) - lo) / (hi - lo))) for m in models}


def scale_areas(models=MODELS):
    """Uniform marker area; scale is carried by colour alone."""
    return {model: 22 for model in models}


COLORS = scale_colors()
LEVEL_MARKERS = dict(zip(LEVELS, ('o', 's', '^')))
MARKER_AREA = 22
DOMAIN_BACKGROUNDS = {domain: domain_examples.DOMAIN_PANEL[letter]
                      for domain, letter in zip(DOMAINS, 'ABCDE', strict=True)}
EXPECTED = {(model, level, domain) for model in MODELS for level in LEVELS for domain in DOMAINS}
OMITTED = base_grid.EXPECTED - EXPECTED
FIGSIZE = (7.35, 2.35)
X_LIMITS = (0, 1)
Y_LIMITS = (0, 1.2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(source: Path = SOURCE) -> dict:
    """Reconstruct every coordinate with the original collection interpreter."""
    source = Path(source).resolve()
    native = base_grid.build_record(source, allow_partial=True)
    points = deepcopy(native['points'])
    keys = {base_grid.key(point) for point in points}
    if len(points) != len(EXPECTED) or keys != EXPECTED:
        raise ValueError('appendix requires exactly all 60 model/level/domain cells at Levels 1--3')
    if any(not X_LIMITS[0] <= point['metrics']['pass8'] <= X_LIMITS[1]
           or not Y_LIMITS[0] <= point['metrics']['distinct8'] <= Y_LIMITS[1]
           for point in points):
        raise ValueError('appendix coordinates exceed its registered common display limits')
    omitted = deepcopy(native['missing_cells'])
    if len(omitted) != len(OMITTED) or {base_grid.key(cell) for cell in omitted} != OMITTED:
        raise ValueError('appendix omitted cells must be the 40 unmeasured Level 4--5 cells')
    if any(cell.get('reason') != 'awaiting_dataset_admission' for cell in omitted):
        raise ValueError('appendix Level 4--5 admission status differs from its frozen source')
    sampling = {key: deepcopy(value) for key, value in native['sampling'].items()
                if key != 'complete_grid_samples'}
    sampling.update(required_scope_samples=60 * 4096, observed_samples=60 * 4096)
    return {
        'schema': SCHEMA, 'status': 'complete',
        'scope': 'All four frozen Qwen2.5-Instruct scales at Levels 1--3 in all five domains',
        'population': 'held-out frozen upstream base-model evaluation, not historical development admission',
        'source': base_grid.binding(source), 'registry': deepcopy(native['registry']),
        'registry_extensions': deepcopy(native['registry_extensions']),
        'validator': deepcopy(native['validator']),
        'validator_runtime': deepcopy(native['validator_runtime']),
        'native_reconstruction': base_grid.binding(Path(base_grid.__file__)),
        'domain_palette_source': base_grid.binding(Path(domain_examples.__file__)),
        'points': points, 'missing_cells': [], 'omitted_cells': omitted,
        'complete_cells': 60, 'required_cells': 60,
        'coverage_by_domain': {domain: {'complete': 12, 'required': 12} for domain in DOMAINS},
        'coverage_by_level': {level: {'complete': 20, 'required': 20} for level in LEVELS},
        'coverage_by_model': {model: {'complete': 15, 'required': 15} for model in MODELS},
        'models': deepcopy(native['models']), 'datasets': deepcopy(native['datasets']),
        'sampling': sampling, 'metrics': deepcopy(native['metrics']),
        'display': {
            'domains': list(DOMAINS), 'levels': list(LEVELS), 'model_labels': list(MODELS),
            'x': 'pass@8', 'y': 'mean distinct@8',
            'x_limits': list(X_LIMITS), 'y_limits': list(Y_LIMITS),
            'figure_inches': list(FIGSIZE), 'model_colors': deepcopy(COLORS),
            'domain_backgrounds': deepcopy(DOMAIN_BACKGROUNDS),
            'level_markers': deepcopy(LEVEL_MARKERS), 'marker_area': MARKER_AREA,
            'encoding': 'scale -> sequential colour ramp; level -> marker shape',
            'marker_fill': 'filled by model scale', 'jitter': False, 'connections': False,
            'overlap': 'filled translucent markers, larger scales drawn first',
            'omitted_cells_plotted': False,
            'omitted_levels_note': 'Levels 4–5 await dataset admission and are not shown.',
        },
        'validation': {
            **deepcopy(native['validation']),
            'complete_60_cell_scope': True, 'complete_100_cell_grid': False,
            'omitted_levels_do_not_have_coordinates': True,
        },
        'renderer': {'path': str(SCRIPT), 'sha256': digest(SCRIPT),
                     'style_path': str(Path(style.__file__).resolve()),
                     'style_sha256': digest(Path(style.__file__))},
    }


def build_figure(record: dict):
    if record.get('schema') != SCHEMA:
        raise ValueError('unexpected appendix figure record')
    if (record.get('status') != 'complete' or len(record.get('points', [])) != len(EXPECTED)
            or {base_grid.key(point) for point in record['points']} != EXPECTED):
        raise ValueError('appendix figure requires the complete 60-cell Level 1--3 scope')
    style.apply_rcparams()
    figure, axes = plt.subplots(1, 5, figsize=FIGSIZE, sharex=True, sharey=True)
    figure.subplots_adjust(left=.065, right=.99, bottom=.40, top=.88, wspace=.20)
    for axis, domain in zip(axes, DOMAINS):
        axis.set_facecolor(record['display']['domain_backgrounds'][domain])
        for order, model in enumerate(reversed(MODELS)):
            for level in LEVELS:
                point = next(point for point in record['points']
                             if point['domain'] == domain and point['model_label'] == model
                             and point['level'] == level)
                axis.scatter(point['metrics']['pass8'], point['metrics']['distinct8'],
                             s=MARKER_AREA, marker=LEVEL_MARKERS[level],
                             facecolors=COLORS[model], edgecolors=style.INK,
                             alpha=.85, linewidths=.35,
                             zorder=3 + order, clip_on=False)
        axis.set_title(TITLES[domain], fontsize=8.5, pad=4)
        axis.set_xlim(*X_LIMITS); axis.set_ylim(*Y_LIMITS)
        axis.set_xticks([0, .5, 1], ['0', '.5', '1'], fontsize=7.4)
        axis.set_yticks([0, .4, .8, 1.2], ['0', '.4', '.8', '1.2'], fontsize=7.4)
        axis.grid(color=style.GRID, linewidth=.55, zorder=0)
        axis.spines[['top', 'right']].set_visible(False)
        for side in ('bottom', 'left'):
            axis.spines[side].set_color(style.MUTED)
            axis.spines[side].set_linewidth(.6)
        axis.tick_params(length=2, width=.6, pad=2)
    figure.text(.018, .64, 'Mean distinct@8', rotation=90, va='center', ha='center', fontsize=8.5)
    figure.text(.53, .285, 'pass@8', va='center', ha='center', fontsize=8.5)
    levels = [Line2D([], [], marker=LEVEL_MARKERS[level], linestyle='none', markersize=4.7,
                     markerfacecolor=style.MUTED, markeredgecolor=style.INK,
                     markeredgewidth=.35, label=f'Level {number}')
              for number, level in enumerate(LEVELS, 1)]
    models = [Line2D([], [], marker='o', linestyle='none', markersize=4.7,
                     markerfacecolor=COLORS[model], markeredgecolor=style.INK,
                     markeredgewidth=.35, label=base_grid.MODEL_NAMES[model])
              for model in MODELS]
    figure.legend(handles=levels, loc='lower center', bbox_to_anchor=(.52, .15),
                  ncol=3, frameon=False, fontsize=7.4, handletextpad=.35,
                  handlelength=.8, columnspacing=1.6, borderaxespad=0)
    figure.legend(handles=models, loc='lower center', bbox_to_anchor=(.52, .055),
                  ncol=4, frameon=False, fontsize=7.4, handletextpad=.35,
                  handlelength=.8, columnspacing=1.8, borderaxespad=0)
    return figure


def render(record: dict, output: Path = OUT) -> None:
    figure = build_figure(record)
    style.save(figure, Path(output), png=True, dpi=240)
    plt.close(figure)


def validate_record(record: dict, *, source: Path = SOURCE, output: Path = OUT) -> None:
    if record.get('source') != base_grid.binding(Path(source).resolve()):
        raise ValueError('appendix figure references an unexpected frozen source')
    expected_outputs = {str(Path(output).with_suffix(suffix).resolve()) for suffix in ('.pdf', '.png')}
    if set(record.get('outputs', {})) != expected_outputs:
        raise ValueError('appendix figure requires its published PDF and PNG bindings')
    for path, sha in record['outputs'].items():
        if digest(Path(path)) != sha:
            raise ValueError(f'appendix figure output changed: {path}')
    expected = build_record(source)
    if {key: value for key, value in record.items() if key != 'outputs'} != expected:
        raise ValueError('appendix figure record differs from native frozen-base evidence')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    record = build_record(args.source)
    render(record, args.output)
    record['outputs'] = {str(args.output.with_suffix(suffix).resolve()):
                         digest(args.output.with_suffix(suffix)) for suffix in ('.pdf', '.png')}
    validate_record(record, source=args.source, output=args.output)
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
