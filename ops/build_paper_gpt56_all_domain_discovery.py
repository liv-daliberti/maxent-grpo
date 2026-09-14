#!/usr/bin/env python3
"""Render source-bound appendix evidence for completed five-domain Sol discovery."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/modebench_discovery_five_domains_sol_20260912/analysis/analysis.json'
OUTPUT_DIR = ROOT / 'paper/results'
STEM = 'gpt56_all_domain_sampling_20260912'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LABELS = {'graph_coloring': 'Graph', 'countdown': 'Countdown', 'python_factors': 'Python',
          'mathir': 'MathIR', 'pantry_plan': 'PantryPlan'}
LEVELS = (2, 3)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text())


def by_cell(cells):
    require(isinstance(cells, list) and len(cells) == 10, 'Require all ten domain-level cells')
    result = {(cell['domain'], cell['level']): cell for cell in cells}
    require(set(result) == {(d, l) for d in DOMAINS for l in LEVELS}, 'Invalid complete cell inventory')
    return result


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def estimate(value):
    require(isinstance(value, dict) and finite(value['estimate']), 'Invalid estimate')
    ci = value['ci95']
    require(isinstance(ci, list) and len(ci) == 2 and all(finite(v) for v in ci)
            and ci[0] <= value['estimate'] + 1e-10 and value['estimate'] <= ci[1] + 1e-10,
            'Invalid pointwise uncertainty interval')
    return deepcopy(value)


def first64_successes(cell):
    prompts = cell['prompts']
    require(len(prompts) == 16 and len({p['row_index'] for p in prompts}) == 16,
            'Expected every distinct fixed prompt')
    values = [p['prefix']['64']['pass'] for p in prompts]
    require(all(v in (0, 1) for v in values), 'Invalid ordered-prefix success')
    return sum(values)


def build_record(source=SOURCE):
    source = Path(source)
    report = read_json(source)
    require(report.get('schema') == 'gpt56-all-domain-discovery-v1'
            and report.get('status') == 'complete' and report.get('model') == 'gpt-5.6-sol'
            and report.get('prompt_arm') == 'original' and report.get('grading') == 'normalized_secondary',
            'Use the completed original-wording five-domain report')
    require(report.get('served_snapshot_header') == 'gpt-5.6-sol-2026-07-09',
            'Expected authenticated provider snapshot header')
    normalized, strict = by_cell(report['cells']), by_cell(report['strict_cells'])
    rows = []
    for domain in DOMAINS:
        for level in LEVELS:
            ncell, scell = normalized[domain, level], strict[domain, level]
            budget = ncell['max_draws']
            require(type(budget) is int and budget >= 64 and budget & (budget - 1) == 0
                    and scell['max_draws'] == budget and ncell['n_prompts'] == scell['n_prompts'] == 16,
                    'Both grading conventions need the identical complete prompt budget')
            require({p['row_index'] for p in ncell['prompts']} == {p['row_index'] for p in scell['prompts']},
                    'Grading conventions differ in included problems')
            support = ncell['support']
            require(support == scell['support'] and finite(support['mean']) and support['mean'] > 0,
                    'Support reference differs between grading conventions')
            expected_kind = 'exact' if domain == 'graph_coloring' else 'certified_lower_bound'
            require(support['kind'] == expected_kind, 'Changed exact/lower-bound support semantics')
            endpoints = {}
            for label, cell in [('normalized', ncell), ('strict', scell)]:
                end = cell['points'][-1]
                require(end['k'] == budget, 'Endpoint must use the full measured pool')
                endpoints[label] = {metric: estimate(end[metric]) for metric in ('distinct', 'pass')}
                require(0 <= end['pass']['estimate'] <= 1 and 0 <= end['distinct']['estimate'] <= budget,
                        'Impossible endpoint estimate')
            tail = ncell['tail']
            require(tail['from_k'] == budget // 2 and tail['to_k'] == budget,
                    'Tail must compare the final complete doubling')
            rows.append({'domain': domain, 'label': LABELS[domain], 'level': level, 'n_prompts': 16,
                         'max_draws': budget, 'support': deepcopy(support), **endpoints,
                         'normalized_last_doubling': {'from_k': tail['from_k'], 'to_k': tail['to_k'],
                                                     'distinct': estimate(tail['distinct'])},
                         'first64_prefix_successes': {'normalized': first64_successes(ncell),
                                                      'strict': first64_successes(scell)}})
    responses = sum(row['n_prompts'] * row['max_draws'] for row in rows)
    require(report['responses'] == responses and report['prompts'] == 160, 'Incomplete response accounting')
    controls = report['protocol']
    require(controls['reasoning'] == 'medium' and controls['max_output_tokens'] == 8192
            and controls['temperature_and_top_p'] == 'omitted', 'Changed collection controls')
    bootstrap = controls['bootstrap']
    require(bootstrap['unit'] == 'whole prompt' and bootstrap['pointwise'] is True,
            'Expected pointwise whole-prompt uncertainty')
    return {'schema': 'paper-gpt56-five-domain-discovery-v1', 'status': 'complete',
            'source': binding(source), 'builder': binding(__file__), 'model': report['model'],
            'served_snapshot_header': report['served_snapshot_header'], 'rows': rows,
            'prompts': 160, 'responses': responses, 'protocol': deepcopy(controls),
            'first64_prefix_successes': {grading: sum(r['first64_prefix_successes'][grading] for r in rows)
                                         for grading in ('normalized', 'strict')},
            'interpretation': {'main_display': 'Frozen formatting normalization, a separately labeled sensitivity.',
                               'strict': 'Frozen verifier primary; identical response pools and failures retained.',
                               'last_doubling': 'Paired rarefaction difference within each final full pool; not the literal new-key count in the last chronological block.',
                               'varying_budgets': 'Compare domains at a common k; domain endpoints have different measured budgets.',
                               'extension': 'Exploratory staged collection informed by earlier results; no asymptotic saturation claim.'},
            'analysis_provenance': {k: deepcopy(report[k]) for k in ('sources', 'prior_analysis', 'new_support', 'support_certificate', 'analyzer')}}


def support_tex(support):
    lower = support['kind'] == 'certified_lower_bound'
    # Rounding downward preserves a numerical lower-bound statement.
    number = math.floor(support['mean'] * 100) / 100 if lower else support['mean']
    return ('$\\geq ' if lower else '$') + f'{number:.2f}' + '$'


def render_tex(record):
    rows = record['rows']
    budget_parts = []
    for domain in DOMAINS:
        budgets = {r['max_draws'] for r in rows if r['domain'] == domain}
        require(len(budgets) == 1, 'The publication budget sentence expects matched levels within each domain')
        budget_parts.append(f"{LABELS[domain]} {budgets.pop()}")
    prefix = record['first64_prefix_successes']
    replicates = record['protocol']['bootstrap']['replicates']
    seed = record['protocol']['bootstrap']['seed']
    lines = [r'% Generated by ops/build_paper_gpt56_all_domain_discovery.py; do not edit.',
             r'\subsection{Five-domain hosted sampling-budget follow-up}',
             r'\label{app:gpt56-five-domain-discovery}', '',
             'This original-wording follow-up retains sixteen fixed problems in each of',
             'five domains at Levels~2 and~3. The complete per-problem budgets are',
             ', '.join(budget_parts) + f' draws, totaling {record["responses"]:,} responses.',
             'Every request uses GPT-5.6 Sol with medium reasoning, an 8,192-token',
             'output limit, and omitted temperature and top-$p$. Native receipt',
             r'headers identify the same provider-reported snapshot, \texttt{gpt-5.6-sol-2026-07-09}.',
             'The stages were collected at different times; shared controls and this',
             'header do not establish identical stochastic behavior across stages.', '',
             'Frozen formatting normalization is the labeled main-figure sensitivity;',
             'the frozen strict verifier remains primary and uses the identical pools.',
             f'The first 64 responses in the fixed draw-index order solve {prefix["normalized"]}',
             f'of {record["prompts"]} problems after normalization and {prefix["strict"]} under strict grading.',
             'This prefix result is distinct from rarefaction at $k=64$ within an',
             'expanded pool. Invalid, refused, and truncated responses remain in every',
             'sampling denominator; transport retries do not become extra samples.', '',
             'For each complete pool of size $N$, with $c$ correct draws and mode',
             'multiplicities $n_j$, we compute',
             r'\[',
             r' P_N(k)=1-\frac{\binom{N-c}{k}}{\binom Nk},\qquad',
             r' D_N(k)=\sum_j\left[1-\frac{\binom{N-n_j}{k}}{\binom Nk}\right],',
             r'\]',
             'then average all sixteen problems equally. Every curve is recalculated',
             'from its final complete pool, so an expanded-pool estimate at $k=64$',
             'need not equal the earlier 64-draw endpoint. The final-doubling contrast',
             r'is $D_N(N)-D_N(N/2)$ within that same pool.',
             rf'Pointwise 95\% intervals use {replicates:,} whole-problem bootstrap replicates',
             f'within domain and level (seed {seed}); the same resampled problems form',
             'both sides of each contrast. The retained analysis also reports every',
             'ordered-prefix sensitivity. Domain endpoints use different budgets;',
             'equal-budget comparisons therefore use the same $k$.', '',
             r'\begin{table}[!htbp]', r'  \centering', r'  \footnotesize',
             r'  \setlength{\tabcolsep}{4pt}', r'  \begin{tabular}{llrrrrr}', r'    \toprule',
             r'    Domain & Level & $N$ & Support & Norm. $D/P$ & Strict $D/P$ & Last $\Delta D$ [95\% CI] \\',
             r'    \midrule']
    for row in rows:
        n, s = row['normalized'], row['strict']
        gain = row['normalized_last_doubling']['distinct']
        endpoints = [f"{value['distinct']['estimate']:.3f}/{100 * value['pass']['estimate']:.1f}"
                     for value in (n, s)]
        lines.append(f"    {row['label']} & {row['level']} & {row['max_draws']} & {support_tex(row['support'])} & "
                     + ' & '.join(endpoints)
                     + f" & {gain['estimate']:.3f} [{gain['ci95'][0]:.3f}, {gain['ci95'][1]:.3f}] " + r'\\')
    lines += [r'    \bottomrule', r'  \end{tabular}',
              r'  \caption{\textbf{All five domains at their measured sampling budgets.}',
              r'  $D/P$ gives mean distinct verified modes and empirical pass probability',
              r'  (percent) at the full $N$-draw endpoint. Support is the mean available-mode',
              r'  reference: exact for Graph and a certified lower bound elsewhere.',
              r'  The normalized last-doubling gain is a paired within-pool rarefaction',
              r'  contrast. Intervals are pointwise and exploratory.}',
              r'  \label{tab:gpt56-five-domain-discovery}', r'\end{table}', '',
              'Graph references exhaust all fixed-color-compatible assignments.',
              'Countdown exhausts binary expression trees only: the hosted validator',
              'also accepts unary negation and preserves it in canonical keys, so its',
              'binary count is a lower bound. Python, MathIR, and Pantry retain their',
              'prior certified lower bounds. A ratio to one of these bounds is not',
              'exhaustive coverage and may exceed one.', '',
              'Sampling extensions were chosen after examining earlier complete stages;',
              'once launched, each stage retained every fixed problem and response slot.',
              'These exploratory observations measure finite-budget discovery.',
              'Small or zero late gains do not establish asymptotic saturation or the',
              'absence of rare unseen modes. Source bindings, strict results, complete',
              'curves, and prefix counts are retained in the accompanying JSON.', '']
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
            'Publication JSON differs from the completed source or builder')
    require((output_dir / (STEM + '.tex')).read_text() == render_tex(expected),
            'Publication TeX differs from the completed source')
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
