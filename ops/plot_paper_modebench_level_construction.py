#!/usr/bin/env python3
"""Show frozen Qwen-7B correctness and verified breadth at Levels 1--3.

The plotted scope is 15 complete domain/level cells for one model. Native evidence
uses its existing frozen collection validator; Levels 4--5 are outside this scope.
This display does not change the separate 100-cell all-model publication gate.
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
SOURCE = ROOT / 'artifacts/modebench_base_level_grid_20260911/qwen7b_levels123_20260912/figure_source_v1.json'
OUT = ROOT / 'paper/figures/modebench_level_construction'
SCRIPT = Path(__file__).resolve()
SCHEMA = 'modebench-qwen7b-levels123-construction-v1'
DOMAINS = base_grid.DOMAINS
LEVELS = base_grid.LEVELS[:3]
TITLES = dict(zip(DOMAINS, ('Graph', 'Countdown', 'Python', 'MathIR', 'Pantry')))
COLORS = {level: base_grid.COLORS[level] for level in LEVELS}
DOMAIN_BACKGROUNDS = {domain: domain_examples.DOMAIN_PANEL[letter]
                      for domain, letter in zip(DOMAINS, 'ABCDE', strict=True)}
EXPECTED = {('7b', level, domain) for level in LEVELS for domain in DOMAINS}
FIGSIZE = (7.35, 1.55)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(source: Path = SOURCE) -> dict:
    source = Path(source).resolve()
    manifest = json.loads(source.read_text())
    entries = manifest.get('receipts', [])
    if not entries or any(entry.get('model_label') != '7b' for entry in entries):
        raise ValueError('construction source requires only frozen Qwen-7B receipts')
    native = base_grid.build_record(source, allow_partial=True)
    points = deepcopy(native['points'])
    keys = {base_grid.key(point) for point in points}
    if len(keys) != len(points) or keys != EXPECTED:
        raise ValueError('construction requires all 15 Qwen-7B Level-1--3 cells')
    if any(not (0 <= point['metrics']['pass8'] <= 1
                and 0 <= point['metrics']['distinct8'] <= 1.2) for point in points):
        raise ValueError('new measured coordinates exceed the registered display limits')
    missing = []
    coverage = {
        domain: {'complete': sum(point['domain'] == domain for point in points),
                 'required': 3,
                 'pending': [cell for cell in missing if cell['domain'] == domain]}
        for domain in DOMAINS
    }
    level_coverage = {
        level: {'complete': sum(point['level'] == level for point in points), 'required': 5,
                'pending': [cell for cell in missing if cell['level'] == level]}
        for level in LEVELS
    }
    sampling = {key: deepcopy(value) for key, value in native['sampling'].items()
                if key != 'complete_grid_samples'}
    sampling.update(required_scope_samples=15 * 4096, observed_samples=len(points) * 4096)
    return {
        'schema': SCHEMA,
        'status': 'complete',
        'model': 'Qwen2.5-7B-Instruct', 'model_label': '7b',
        'model_identity': deepcopy(native['models']['7b']),
        'population': 'held-out frozen upstream base-model evaluation, not historical development admission',
        'source': base_grid.binding(source), 'registry': deepcopy(native['registry']),
        'registry_extensions': deepcopy(native['registry_extensions']),
        'validator': deepcopy(native['validator']),
        'validator_runtime': deepcopy(native['validator_runtime']),
        'native_reconstruction': base_grid.binding(Path(base_grid.__file__)),
        'domain_palette_source': base_grid.binding(Path(domain_examples.__file__)),
        'points': points, 'missing_cells': missing,
        'complete_cells': len(points), 'required_cells': 15,
        'excluded_levels': ['level4', 'level5'],
        'coverage_by_domain': coverage, 'coverage_by_level': level_coverage,
        'sampling': sampling, 'metrics': deepcopy(native['metrics']),
        'display': {'domains': list(DOMAINS), 'levels': list(LEVELS),
                    'x': 'pass@8', 'y': 'mean distinct@8',
                    'x_limits': [0, 1], 'y_limits': [0, 1.2],
                    'figure_inches': list(FIGSIZE), 'level_colors': COLORS,
                    'domain_backgrounds': deepcopy(DOMAIN_BACKGROUNDS),
                    'marker': 'filled circle', 'jitter': False, 'connections': False,
                    'pending_cells_plotted': False},
        'validation': {
            'all_present_receipts_authenticated': native['validation']['all_present_receipts_authenticated'],
            'frozen_validator_used_in_isolated_interpreter': native['validation']['frozen_validator_used_in_isolated_interpreter'],
            'collection_interpreter_binary_version_and_stdlib_authenticated': native['validation']['collection_interpreter_binary_version_and_stdlib_authenticated'],
            'all_metrics_reconstructed_from_canonical_sets': native['validation']['all_metrics_reconstructed_from_canonical_sets'],
            'only_frozen_qwen7b': True, 'complete_15_cell_scope': True},
        'renderer': {'path': str(SCRIPT), 'sha256': digest(SCRIPT),
                     'style_path': str(Path(style.__file__).resolve()),
                     'style_sha256': digest(Path(style.__file__))},
    }


def pending_note(cells: list[dict]) -> str:
    numbers = sorted(int(cell['level'].removeprefix('level')) for cell in cells)
    if not numbers:
        return ''
    label = (f'L{numbers[0]}–{numbers[-1]}' if numbers == list(range(numbers[0], numbers[-1] + 1))
             else ', '.join(f'L{number}' for number in numbers))
    return label + ' pending'


def build_figure(record: dict):
    if record.get('schema') != SCHEMA:
        raise ValueError('unexpected construction figure record')
    style.apply_rcparams()
    figure, axes = plt.subplots(1, 5, figsize=FIGSIZE, sharex=True, sharey=True)
    figure.subplots_adjust(left=.065, right=.99, bottom=.38, top=.83, wspace=.20)
    for axis, domain in zip(axes, DOMAINS):
        axis.set_facecolor(record['display']['domain_backgrounds'][domain])
        for level in LEVELS:
            for point in record['points']:
                if point['domain'] == domain and point['level'] == level:
                    axis.scatter(point['metrics']['pass8'], point['metrics']['distinct8'],
                                 s=29, marker='o', facecolors=COLORS[level],
                                 edgecolors='white', linewidths=.35, zorder=3, clip_on=False)
        axis.set_title(TITLES[domain], fontsize=8.5, pad=4)
        axis.set_xlim(*record['display']['x_limits'])
        axis.set_ylim(*record['display']['y_limits'])
        axis.set_xticks([0, .5, 1], ['0', '.5', '1'], fontsize=7.4)
        axis.set_yticks([0, .5, 1], ['0', '.5', '1'], fontsize=7.4)
        axis.grid(color=style.GRID, linewidth=.55, zorder=0)
        axis.spines[['top', 'right']].set_visible(False)
        for side in ('bottom', 'left'):
            axis.spines[side].set_color(style.MUTED)
            axis.spines[side].set_linewidth(.6)
        axis.tick_params(length=2, width=.6, pad=2)
        note = pending_note(record['coverage_by_domain'][domain]['pending'])
        if note:
            axis.text(.025, .98, note, transform=axis.transAxes, ha='left', va='top',
                      color=style.MUTED, fontsize=6.8)
    figure.text(.018, .60, 'Mean distinct@8', rotation=90, va='center', ha='center', fontsize=8.5)
    figure.text(.53, .235, 'pass@8', va='center', ha='center', fontsize=8.5)
    handles = []
    for number, level in enumerate(LEVELS, 1):
        count = record['coverage_by_level'][level]['complete']
        label = f'Level {number}' + (' (pending)' if count == 0 else ' (partial)' if count < 5 else '')
        handles.append(Line2D([], [], marker='o', linestyle='none', markersize=5,
                              markerfacecolor=COLORS[level], markeredgecolor='white',
                              markeredgewidth=.35, label=label))
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .005),
                  ncol=3, frameon=False, fontsize=7.4, handletextpad=.35,
                  handlelength=.8, columnspacing=1.15, borderaxespad=0)
    return figure


def render(record: dict, output: Path = OUT) -> None:
    figure = build_figure(record)
    style.save(figure, Path(output), png=True, dpi=240)
    plt.close(figure)


def validate_record(record: dict, *, output: Path = OUT, source: Path = SOURCE) -> None:
    expected_source = base_grid.binding(Path(source).resolve())
    if record.get('source') != expected_source:
        raise ValueError('construction figure references an unexpected frozen source')
    expected_outputs = {str(Path(output).with_suffix(suffix).resolve()) for suffix in ('.pdf', '.png')}
    if set(record.get('outputs', {})) != expected_outputs:
        raise ValueError('construction figure requires its published PDF and PNG bindings')
    for path, sha in record['outputs'].items():
        if digest(Path(path)) != sha:
            raise ValueError(f'construction figure output changed: {path}')
    expected = build_record(source)
    if {key: value for key, value in record.items() if key != 'outputs'} != expected:
        raise ValueError('construction figure record differs from native frozen-base evidence')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    record = build_record(args.source)
    render(record, args.output)
    record['outputs'] = {str(args.output.with_suffix(suffix).resolve()):
                         digest(args.output.with_suffix(suffix)) for suffix in ('.pdf', '.png')}
    validate_record(record, output=args.output, source=args.source)
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
