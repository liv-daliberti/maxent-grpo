#!/usr/bin/env python3
"""Build source-bound appendix evidence for all fifteen Sol cells at 512 draws."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

try:
    from . import plot_paper_gpt56_all_levels32_sampling as validation
except ImportError:
    _validation_path = Path(__file__).with_name('plot_paper_gpt56_all_levels32_sampling.py')
    _spec = importlib.util.spec_from_file_location('_all_levels_sampling_validation', _validation_path)
    validation = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(validation)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/modebench_discovery_all_levels_sol32x512_20260913/analysis/analysis.json'
OUTPUT_DIR = ROOT / 'paper/results'
STEM = 'gpt56_all_levels32_sampling_20260913'
DOMAINS = validation.DOMAINS
LEVELS = validation.LEVELS
LABELS = {'graph_coloring': 'Graph', 'countdown': 'Countdown', 'python_factors': 'Python',
          'mathir': 'MathIR', 'pantry_plan': 'PantryPlan'}
DRAWS = validation.DRAWS
TOTAL_PROMPTS = validation.TOTAL_PROMPTS
require = validation.require
binding = validation.binding
read_json = validation.read_json


def prefix_successes(cell):
    return {str(k): int(sum(prompt['prefix'][str(k)]['pass'] for prompt in cell['prompts']))
            for k in validation.K_GRID}


def build_record(source=SOURCE):
    source = Path(source)
    report = validation.validate_report(read_json(source))
    require(report.get('served_snapshot_header') == 'gpt-5.6-sol-2026-07-09',
            'Expected authenticated provider-reported snapshot header')
    normalized = {(cell['domain'], cell['level']): (index, cell)
                  for index, cell in enumerate(report['cells'])}
    strict = {(cell['domain'], cell['level']): (index, cell)
              for index, cell in enumerate(report['strict_cells'])}
    rows = []
    for domain in DOMAINS:
        for level in LEVELS:
            nindex, normal = normalized[domain, level]
            sindex, raw = strict[domain, level]
            rows.append({
                'domain': domain, 'label': LABELS[domain], 'level': level,
                'n_prompts': validation.PROMPTS_PER_CELL, 'max_draws': DRAWS,
                'support': deepcopy(normal['support']),
                'support_counts': [p['support']['support_count'] for p in normal['prompts']],
                'prompts': [{'row_index': p['row_index'], 'row_sha256': p['row_sha256']}
                            for p in normal['prompts']],
                'normalized': {metric: deepcopy(normal['points'][-1][metric])
                               for metric in ('distinct', 'pass')},
                'strict': {metric: deepcopy(raw['points'][-1][metric]) for metric in ('distinct', 'pass')},
                'tails': {'normalized': deepcopy(normal['tail']), 'strict': deepcopy(raw['tail'])},
                'curves': {'normalized': deepcopy(normal['points']), 'strict': deepcopy(raw['points'])},
                'prefix_successes': {'normalized': prefix_successes(normal), 'strict': prefix_successes(raw)},
                'source_cells': {'normalized': f'/cells/{nindex}', 'strict': f'/strict_cells/{sindex}'},
            })
    return {
        'schema': 'paper-gpt56-all-levels-discovery-v1', 'status': 'complete',
        'source': binding(source), 'builder': binding(__file__),
        'validation_dependency': binding(validation.__file__),
        'model': report['model'], 'served_snapshot_header': report['served_snapshot_header'],
        'prompt_arm': report['prompt_arm'], 'levels': list(LEVELS), 'rows': rows,
        'prompts': TOTAL_PROMPTS, 'responses': TOTAL_PROMPTS * DRAWS, 'draws_per_prompt': DRAWS,
        'protocol': deepcopy(report['protocol']),
        'prefix_successes': {
            grading: {str(k): sum(row['prefix_successes'][grading][str(k)] for row in rows)
                      for k in validation.K_GRID} for grading in ('normalized', 'strict')},
        'interpretation': {
            'cohort': 'All five domains, all three levels, all thirty-two fixed prompts per cell, all 512 response slots.',
            'main_display': 'Frozen formatting normalization; the same response pools are also reported under the frozen strict verifier.',
            'last_doubling': 'Paired D_512(512)-D_512(256) rarefaction contrast within each final pool, not chronological new-key counts.',
            'equal_budgets': 'All fifteen curves and endpoints use the same 512-draw budget.',
            'support': 'Graph counts are exhaustive. Other references are certified lower bounds, may be exceeded, and cannot establish exhaustive coverage.',
            'extension': 'The expansion from 16 to 32 fixed problems per cell followed earlier observations, retaining every original problem and selecting ranks17–32 under the unchanged hash ranking; all pools have512draws. Small late gains do not establish asymptotic saturation.',
        },
        'analysis_provenance': {k: deepcopy(v) for k, v in report.items()
                                if k not in ('cells', 'strict_cells')},
    }


def support_tex(support):
    lower = support['kind'] == 'certified_lower_bound'
    number = math.floor(support['mean'] * 100) / 100 if lower else support['mean']
    prefix = r'$\geq ' if lower else '$'
    lo, hi = support['range']
    interval = str(lo) if lo == hi else f'{lo}--{hi}'
    return prefix + f'{number:.2f}' + '$ (' + interval + ')'


def format_endpoint(value):
    return f"{value['distinct']['estimate']:.3f}/{100 * value['pass']['estimate']:.1f}"


def format_interval(value):
    return f"{value['estimate']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]"


def render_tex(record):
    rows = record['rows']
    prefix = record['prefix_successes']
    bootstrap = record['protocol']['bootstrap']
    lines = [
        r'% Generated by ops/build_paper_gpt56_all_levels32_discovery.py; do not edit.',
        r'\subsection{All five domains and all three levels at 512 draws}',
        r'\label{app:gpt56-all-levels-discovery}', '',
        'This original-wording follow-up retains thirty-two fixed problems in each',
        'of five domains at Levels~1, 2, and~3. Every problem has 512 responses:',
        f'{record["prompts"]} problems and {record["responses"]:,} responses in total.',
        'Every request uses GPT-5.6 Sol with medium reasoning, an 8,192-token',
        'output limit, and omitted temperature and top-$p$. Native receipt',
        r'headers identify the provider-reported snapshot \texttt{gpt-5.6-sol-2026-07-09}.',
        'Collection occurred in stages; common settings and this header do not',
        'establish identical stochastic behavior across collection times.', '',
        'The main figure applies the frozen formatting normalizer. The strict',
        'frozen verifier remains primary and grades the identical complete pools.',
        f'In fixed draw-index order, the first 64 responses solve {prefix["normalized"]["64"]}',
        f'of {record["prompts"]} problems after normalization and {prefix["strict"]["64"]} under strict grading.',
        f'At 512 responses these counts are {prefix["normalized"]["512"]} and {prefix["strict"]["512"]}, respectively.',
        'These ordered prefixes differ from rarefaction at the same budget inside',
        'the full pool. Invalid, refused, and truncated answers remain in every',
        'sampling denominator; transport retries do not count as extra samples.', '',
        'For each complete pool of size $N=512$, with $c$ correct responses and',
        'verified-mode multiplicities $n_j$, we compute',
        r'\[',
        r' P_N(k)=1-\frac{\binom{N-c}{k}}{\binom Nk},\qquad',
        r' D_N(k)=\sum_j\left[1-\frac{\binom{N-n_j}{k}}{\binom Nk}\right],',
        r'\]',
        'then average all thirty-two problems equally within each domain and level.',
        'All curves use the same complete 512-response pools and budgets',
        r'$k\in\{1,2,4,8,16,32,64,128,256,512\}$. An earlier 64-response endpoint',
        'need not equal the rarefied $k=64$ estimate from this expanded pool.',
        r'The last-doubling contrast is $D_{512}(512)-D_{512}(256)$.',
        rf'Pointwise 95\% intervals use {bootstrap["replicates"]:,} whole-problem bootstrap replicates',
        f'within domain and level (seed {bootstrap["seed"]}); the same resampled problems',
        'form both sides of each contrast. Curves and ordered-prefix sensitivities',
        'under both grading conventions remain in the accompanying source record.', '',
        # The endpoint and final-doubling tables ran over the same fifteen cells
        # in the same order, so the second repeated every row label to add two
        # columns. They are one table: where each cell ends, and what the last
        # doubling of the budget added to it.
        r'\begin{table}[!htbp]', r'  \centering', r'  \footnotesize',
        r'  \setlength{\tabcolsep}{4pt}', r'  \begin{tabular}{llrrrrr}', r'    \toprule',
        r'    & & & \multicolumn{2}{c}{$D/P$ at 512}'
        r' & \multicolumn{2}{c}{$\Delta D$, 256 to 512 [95\% CI]} \\',
        r'    \cmidrule(lr){4-5}\cmidrule(l){6-7}',
        r'    Domain & Level & Known support (range) & Normalized & Strict'
        r' & Normalized & Strict \\',
        r'    \midrule',
    ]
    for index, row in enumerate(rows):
        if index and row['level'] == 1:
            lines.append(r'    \addlinespace[2pt]')
        lines.append(f"    {row['label']} & {row['level']} & {support_tex(row['support'])} & "
                     + format_endpoint(row['normalized']) + ' & ' + format_endpoint(row['strict'])
                     + ' & ' + format_interval(row['tails']['normalized']['distinct'])
                     + ' & ' + format_interval(row['tails']['strict']['distinct']) + r' \\')
    lines += [
        r'    \bottomrule', r'  \end{tabular}',
        r'  \caption{\textbf{All fifteen domain--level cells at 512 responses per problem,',
        r'  and what the final doubling added.}',
        r'  $D/P$ gives mean distinct verified modes and the percentage of problems',
        r'  solved at least once in the complete 512-response pool.',
        r"  Support gives each cell's mean and range",
        r'  across thirty-two problems: exact for Graph, a certified lower bound elsewhere.',
        r'  Both grading conventions retain all 480 problems and 245,760 responses.',
        r'  The paired contrast $\Delta D=D_{512}(512)-D_{512}(256)$ compares rarefaction',
        r'  means within the final pool, rather than novel keys in the last chronological',
        r'  block. Intervals are pointwise; they neither provide simultaneous coverage',
        r'  across budgets and cells nor establish asymptotic saturation.}',
        r'  \label{tab:gpt56-all-levels-endpoints}', r'\end{table}', '',
        'Graph support exhausts all fixed-color-compatible assignments.',
        'Countdown references enumerate binary expression trees; the validator',
        'also accepts unary negation and retains it in canonical keys, so binary',
        'counts are lower bounds. Python, MathIR, and Pantry references retain',
        'their certified interpretation. Discovered modes may exceed a lower',
        'bound without establishing exhaustive coverage of the available modes.', '',
        'The expansion from sixteen to thirty-two problems per cell followed earlier',
        'observations. It retains all 240 original problems and adds the next sixteen',
        'per cell in the same outcome-independent SHA256 ranking (seed 20260911).',
        'The 122,880 new responses use 512 fresh draws per added problem; earlier',
        'eight-draw evaluations of those problems are excluded. Each launched cell',
        'retains its complete fixed',
        'cohort and all response slots. These results describe finite-budget',
        'discovery: small or zero late gains do not rule out rare unseen modes.',
        'Source bindings, full curves, strict grades, support references, and',
        'ordered-prefix sensitivities remain in the accompanying JSON.', '',
    ]
    return '\n'.join(lines)


def write(record, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / (STEM + '.json')).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output_dir / (STEM + '.tex')).write_text(render_tex(record))


def check(source=SOURCE, output_dir=OUTPUT_DIR):
    expected = build_record(source)
    output_dir = Path(output_dir)
    require(read_json(output_dir / (STEM + '.json')) == expected,
            'Publication JSON differs from the completed source, builder, or validation dependency')
    require((output_dir / (STEM + '.tex')).read_text() == render_tex(expected),
            'Publication TeX differs from the completed three-level evidence')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    record = check(args.source, args.output_dir) if args.check else build_record(args.source)
    if not args.check:
        write(record, args.output_dir)
    print(json.dumps({'status': 'pass', 'rows': len(record['rows']), 'responses': record['responses'],
                      'output_dir': str(args.output_dir), 'mode': 'check' if args.check else 'write'}))


if __name__ == '__main__':
    main()
