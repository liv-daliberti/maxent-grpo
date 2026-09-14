#!/usr/bin/env python3
"""Success-conditional modal diversity for the hosted GPT-5.6 Sol evaluation.

The hosted summary already reports ``correct_pair_collision``, which is one
minus pairwise modal diversity pooled over pairs. This recomputes the same
quantity from the audited per-sample records using the prompt-unweighted mean
the rest of the manuscript uses, so the hosted numbers sit on the same axis as
the frozen and trained ones, and reports the pooled reading beside it.

Prompt-unweighted is the right default here: weighting by pair count lets the
prompts a model happens to solve most often dominate the breadth estimate,
which is the accuracy coupling the metric exists to avoid.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

from followup_metrics import atomic_new, file_sha  # noqa: E402
from mode_diversity import DEFAULT_MIN_DEFINED_PROMPTS, effective_modes, mode_diversity  # noqa: E402

SCHEMA = 'paper-mode-diversity-hosted-v1'
SAMPLES = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911/audited_primary_samples.jsonl'
SUMMARY = ROOT / 'paper/results/frontier_hosted_20260911.json'
COMPARISON = ROOT / 'paper/results/frontier_comparison_20260911.json'
OUT = ROOT / 'paper/results/mode_diversity_hosted.json'


def build(samples: Path = SAMPLES, summary: Path = SUMMARY,
          min_defined: int = DEFAULT_MIN_DEFINED_PROMPTS) -> dict:
    prompts: dict[tuple, Counter] = defaultdict(Counter)
    draws: Counter = Counter()
    with samples.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row['level']), row['domain'], int(row['row_index']))
            draws[key] += 1
            if row.get('verified') and row.get('canonical_key') is not None:
                prompts[key][row['canonical_key']] += 1

    cells: dict[tuple, list] = defaultdict(list)
    for (level, domain, _), counts in prompts.items():
        cells[(level, domain)].append(counts)
    # Prompts with no verified draw never enter ``prompts`` above; recover them
    # from the draw index so support fractions have the right denominator.
    totals: Counter = Counter()
    for (level, domain, _) in draws:
        totals[(level, domain)] += 1

    published = json.loads(summary.read_text())['cells']
    rows = []
    for (level, domain), counters in sorted(cells.items()):
        values = [v for v in (mode_diversity(c) for c in counters) if v is not None]
        colliding = sum(sum(n * (n - 1) for n in c.values()) for c in counters)
        pairs = sum(sum(c.values()) * (sum(c.values()) - 1) for c in counters)
        reference = published.get(f'level{level}/{domain}', {})
        rows.append({
            'level': level, 'domain': domain,
            'prompts': totals[(level, domain)],
            'defined_prompts': len(values),
            'support': len(values) / totals[(level, domain)] if totals[(level, domain)] else 0.0,
            'reportable': len(values) >= min_defined,
            'pmd': statistics.fmean(values) if values else None,
            'pmd_standard_error': (statistics.stdev(values) / len(values) ** 0.5
                                   if len(values) > 1 else None),
            'effective_modes': effective_modes(statistics.fmean(values) if values else None),
            'pmd_pair_pooled': 1.0 - colliding / pairs if pairs else None,
            'published_correct_pair_collision':
                reference.get('metrics', {}).get('correct_pair_collision', {}).get('estimate'),
        })

    for row in rows:
        published_collision = row['published_correct_pair_collision']
        if published_collision is not None and row['pmd_pair_pooled'] is not None:
            if abs((1.0 - published_collision) - row['pmd_pair_pooled']) > 1e-9:
                raise ValueError(f'pooled PMD disagrees with the published collision for {row}')

    return {
        'schema': SCHEMA,
        'model': 'gpt-5.6-sol',
        'samples': {'path': str(samples.relative_to(ROOT)), 'sha256': file_sha(samples)},
        'summary': {'path': str(summary.relative_to(ROOT)), 'sha256': file_sha(summary)},
        'builder': {'path': 'ops/build_mode_diversity_hosted.py',
                    'sha256': file_sha(Path(__file__).resolve())},
        'definition': {
            'metric': 'pairwise modal diversity (PMD)',
            'aggregation': 'unweighted mean over prompts with at least two verified draws',
            'pair_pooled_check': 'pmd_pair_pooled equals 1 - published correct_pair_collision',
            'min_defined_prompts': min_defined,
        },
        'cells': rows,
        'coverage': {'cells': len(rows),
                     'reportable_cells': sum(1 for r in rows if r['reportable'])},
    }


def build_cohort(comparison: Path = COMPARISON,
                 min_defined: int = DEFAULT_MIN_DEFINED_PROMPTS) -> dict:
    """PMD for every completed deployment in the multi-model comparison.

    Only the reference deployment has a published collision series to check
    against, so the other models are reported without that cross-check and
    carry their own support counts instead.
    """
    manifest = json.loads(comparison.read_text())
    models = []
    for entry in manifest['models']:
        samples = ROOT / entry['primary_samples_path']
        prompts: dict[tuple, Counter] = defaultdict(Counter)
        seen: set[tuple] = set()
        with samples.open() as handle:
            for line in handle:
                row = json.loads(line)
                key = (int(row['level']), row['domain'], int(row['row_index']))
                seen.add(key)
                if row.get('verified') and row.get('canonical_key') is not None:
                    prompts[key][row['canonical_key']] += 1
        cells: dict[tuple, list] = defaultdict(list)
        totals: Counter = Counter()
        for key in seen:
            cells[(key[0], key[1])].append(prompts.get(key, Counter()))
            totals[(key[0], key[1])] += 1
        rows = []
        for (level, domain), counters in sorted(cells.items()):
            values = [v for v in (mode_diversity(c) for c in counters) if v is not None]
            rows.append({
                'level': level, 'domain': domain,
                'prompts': totals[(level, domain)], 'defined_prompts': len(values),
                'support': len(values) / totals[(level, domain)] if totals[(level, domain)] else 0.0,
                'reportable': len(values) >= min_defined,
                'pmd': statistics.fmean(values) if values else None,
                'effective_modes': effective_modes(statistics.fmean(values) if values else None),
            })
        reportable = [r for r in rows if r['reportable']]
        models.append({
            'model': entry['model'], 'label': entry['label'],
            'samples': {'path': entry['primary_samples_path'],
                        'sha256': file_sha(samples)},
            'cells': rows,
            'macro_pmd': statistics.fmean(r['pmd'] for r in reportable) if reportable else None,
            'reportable_cells': len(reportable), 'cells_total': len(rows),
        })
    return {
        'schema': 'paper-mode-diversity-hosted-cohort-v1',
        'comparison': {'path': str(comparison.relative_to(ROOT)), 'sha256': file_sha(comparison)},
        'builder': {'path': 'ops/build_mode_diversity_hosted.py',
                    'sha256': file_sha(Path(__file__).resolve())},
        'definition': {'metric': 'pairwise modal diversity (PMD)',
                       'aggregation': 'unweighted mean over prompts with at least two verified draws',
                       'min_defined_prompts': min_defined},
        'models': models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=SAMPLES)
    parser.add_argument('--summary', type=Path, default=SUMMARY)
    parser.add_argument('--comparison', type=Path, default=COMPARISON)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--cohort-output', type=Path,
                        default=ROOT / 'paper/results/mode_diversity_hosted_cohort.json')
    args = parser.parse_args()
    payload = build(args.samples, args.summary)
    atomic_new(args.output, payload)
    cohort = build_cohort(args.comparison)
    atomic_new(args.cohort_output, cohort)
    print(json.dumps({'event': 'built', 'output': str(args.output), **payload['coverage'],
                      'cohort_output': str(args.cohort_output),
                      'cohort_models': len(cohort['models'])}))


if __name__ == '__main__':
    main()
