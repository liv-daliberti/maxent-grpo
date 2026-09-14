#!/usr/bin/env python3
"""Render a separately authenticated, explicitly interim 128-draw all-level study.

This entry point cannot write a final 512-draw record. It requires complete
128-slot pools for all 240 fixed problems and uses the same numerical validation
and domain-background palette as the final renderer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

try:
    from . import plot_paper_gpt56_all_levels_sampling as core
except ImportError:
    _path = Path(__file__).with_name('plot_paper_gpt56_all_levels_sampling.py')
    _spec = importlib.util.spec_from_file_location('_all_levels_sampling_core', _path)
    core = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(core)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/modebench_discovery_all_levels_sol512_20260912/interim128/analysis.json'
OUTPUT = ROOT / 'paper/figures/gpt56_all_levels_sampling_interim128'
SCHEMA = 'gpt56-all-levels-interim-v1'
DRAWS = 128
X_TICKS = (1, 4, 16, 128)
TITLE = 'GPT-5.6 Sol · Interim · 128 draws / prompt'
require, binding, read_json = core.require, core.binding, core.read_json


def validate_report(report):
    require(report.get('schema') == SCHEMA and report.get('status') == 'complete'
            and report.get('model') == 'gpt-5.6-sol'
            and report.get('grading') == 'normalized_secondary'
            and report.get('prompt_arm') == 'original',
            'Use only the authenticated complete 128-draw interim report')
    require(report.get('levels') == [1, 2, 3] and report.get('prompts') == 240
            and report.get('responses') == 30720,
            'The interim display requires all fifteen cells, 240 problems, and 30,720 responses')
    core.validate_controls(report)
    core.validate_pool_cells(report['cells'], DRAWS)
    core.validate_pool_cells(report['strict_cells'], DRAWS)
    core.validate_grading_conventions(report)
    return report


def build_record(source=SOURCE):
    source = Path(source)
    report = validate_report(read_json(source))
    cells = deepcopy(report['cells'])
    cells.sort(key=lambda c: (core.DOMAINS.index(c['domain']), c['level']))
    return {
        'schema': 'paper-gpt56-all-levels-sampling-interim128-v1', 'status': 'interim_complete',
        'source': binding(source), 'renderer': binding(__file__),
        'shared_renderer': binding(core.__file__), 'model': report['model'],
        'grading': report['grading'], 'prompt_arm': 'original', 'cells': cells,
        'analysis_provenance': {key: deepcopy(value) for key, value in report.items()
                                if key not in ('cells', 'strict_cells')},
        'display': {
            'title': TITLE, 'figure_inches': list(core.FIGSIZE), 'domains': list(core.DOMAINS),
            'levels': list(core.LEVELS), 'x': 'Samples per prompt, k',
            'x_scale': 'log2; shared across all five domains', 'x_range': [1, DRAWS],
            'x_ticks': list(X_TICKS), 'y': 'Mean distinct verified modes',
            'y_scale': 'linear; separate per domain; includes observed intervals and all support references',
            'domain_backgrounds': deepcopy(core.DOMAIN_BACKGROUNDS),
            'domain_palette_source': 'Figure 2 DOMAIN_PANEL; identical to Figure 3',
            'curves': {str(level): {'label': f'Level {level}', **style}
                       for level, style in core.STYLES.items()},
            'intervals': 'Pointwise 95% whole-problem bootstrap intervals from the bound interim analysis.',
            'support': 'Grey reference means are listed in Level 1 / Level 2 / Level 3 order. Graph exact; all other references certified lower bounds.',
            'endpoints': 'Every curve stops at 128 observed draws per prompt; the 512-draw study is separate and no value is extrapolated.',
        },
        'validation': {'all_five_domains_and_three_levels': True, 'all_240_prompts_retained': True,
                       'all_30720_interim_responses_retained': True,
                       'both_grading_pools_reconstructed': True,
                       'all_point_and_tail_means_reconstructed': True,
                       'support_means_and_ranges_reconstructed': True,
                       'native_authentication_and_bootstrap': 'Responsibility of the bound interim analysis.',
                       'api_calls': 0},
    }


def build_figure(record):
    return core.build_budget_figure(record, draws=DRAWS, x_ticks=X_TICKS, title=TITLE)


def render(record, output=OUTPUT):
    import matplotlib.pyplot as plt
    output = Path(output)
    require(output.with_suffix('').resolve() != core.OUTPUT.with_suffix('').resolve(), 'Interim output cannot replace the final 512-draw figure')
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(record)
    result = deepcopy(record)
    result['outputs'] = {}
    for extension in ('pdf', 'png'):
        path = output.with_suffix('.' + extension)
        kwargs = {'dpi': 300} if extension == 'png' else {'metadata': {'CreationDate': None, 'ModDate': None}}
        figure.savefig(path, facecolor='white', **kwargs)
        result['outputs'][extension] = binding(path)
    plt.close(figure)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def check(output=OUTPUT, source=SOURCE):
    output = Path(output)
    expected = build_record(source)
    retained = read_json(output.with_suffix('.json'))
    outputs = retained.pop('outputs')
    require(retained == expected, 'Retained interim figure differs from its source or rendering helpers')
    require(set(outputs) == {'pdf', 'png'}, 'Incomplete interim figure output')
    for extension, item in outputs.items():
        require(item == binding(output.with_suffix('.' + extension)), 'Changed interim figure rendering')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    record = check(args.output, args.source) if args.check else render(build_record(args.source), args.output)
    print(json.dumps({'status': 'pass', 'mode': 'check' if args.check else 'render',
                      'output': core.relative(args.output), 'cells': len(record['cells']),
                      'draws_per_prompt': DRAWS, 'interim': True}))


if __name__ == '__main__':
    main()
