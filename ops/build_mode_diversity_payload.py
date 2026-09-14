#!/usr/bin/env python3
"""Recompute success-conditional modal diversity for the frozen base-model grid.

Reads the same 60 receipts that back the base-level figures and emits the
pairwise modal diversity (PMD) of each model--domain--level cell alongside the
``pass@8`` and ``distinct@8`` it is reported with. Every attempt's canonical key
is already stored in those receipts, so nothing is resampled or re-graded here:
this is a re-analysis of saved responses, not a new evaluation.

Cells whose defined-prompt count falls below the support bar carry
``reportable: false`` and must be rendered as explicit gaps rather than as
numbers, because a cell where the model almost never succeeds cannot support a
statement about how its successes are distributed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'ops', ROOT / 'ops' / 'exp_scaling'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from followup_metrics import atomic_new, file_sha, sha  # noqa: E402
from mode_diversity import (  # noqa: E402
    DEFAULT_MIN_DEFINED_PROMPTS, POOLED, PER_GROUP, cell_summary,
    effective_modes, prompt_mode_diversity, rarefied_distinct,
    verified_mode_counts,
)

SCHEMA = 'paper-mode-diversity-base-grid-v1'
SOURCE = ROOT / 'artifacts/modebench_base_level_grid_20260911/status_updates/20260912T152423Z/figure_source.json'
OUT = ROOT / 'paper/results/mode_diversity_base_grid.json'


def _pooled_counts(prompt_result):
    from collections import Counter
    pooled: Counter = Counter()
    for draw in prompt_result['draws']:
        pooled.update(verified_mode_counts(draw['attempts']))
    return pooled


def build(source: Path = SOURCE, min_defined: int = DEFAULT_MIN_DEFINED_PROMPTS) -> dict:
    manifest = json.loads(source.read_text())
    cells = []
    for meta in manifest['receipts']:
        path = ROOT / meta['path']
        receipt = json.loads(path.read_text())
        if file_sha(path) != meta['sha256']:
            raise ValueError(f'receipt changed since the figure manifest: {meta["path"]}')
        results = receipt['prompt_results']
        summary = cell_summary(results, POOLED, min_defined_prompts=min_defined)
        per_group = cell_summary(results, PER_GROUP, min_defined_prompts=min_defined)
        pooled = [_pooled_counts(result) for result in results]
        depth3 = [value for value in (rarefied_distinct(c, 3) for c in pooled) if value is not None]
        cells.append({
            'model_label': meta['model_label'], 'level': meta['level'], 'domain': meta['domain'],
            'receipt': meta['path'], 'receipt_sha256': meta['sha256'],
            'prompts': summary['prompts'],
            # Registered endpoints, retained for comparison with the new metric.
            'pass8': statistics.fmean(r['pass8'] for r in results),
            'distinct8': statistics.fmean(r['distinct8'] for r in results),
            # Success-conditional breadth.
            'pmd': summary['d_mode'],
            'pmd_standard_error': summary['standard_error'],
            'rarefied_distinct_at_2': summary['rarefied_distinct_at_2'],
            'effective_modes': summary['effective_modes'],
            'defined_prompts': summary['defined_prompts'],
            'support': summary['support'],
            'reportable': summary['reportable'],
            # Robustness: a deeper rarefaction and the eight-draw group convention.
            'rarefied_distinct_at_3': statistics.fmean(depth3) if depth3 else None,
            'defined_prompts_at_3': len(depth3),
            'pmd_per_group': per_group['d_mode'],
            'defined_prompts_per_group': per_group['defined_prompts'],
        })
    cells.sort(key=lambda c: (c['model_label'], c['level'], c['domain']))
    reportable = [c for c in cells if c['reportable']]
    return {
        'schema': SCHEMA,
        'source': {'path': str(source.relative_to(ROOT)), 'sha256': file_sha(source)},
        'builder': {'path': 'ops/build_mode_diversity_payload.py',
                    'sha256': file_sha(Path(__file__).resolve())},
        'definition': {
            'metric': 'pairwise modal diversity (PMD)',
            'estimator': 'PMD = 1 - sum_m n_m (n_m - 1) / (K (K - 1)), K verified draws per prompt',
            'estimand': 'P(two independent verified responses occupy different modes) = 1 - sum_m q_m^2',
            'aggregation': 'pooled over a prompt\'s four groups, unweighted mean over defined prompts',
            'equals': 'one minus the registered conditional-concentration collision U-statistic',
            'undefined_when': 'fewer than two verified responses for the prompt',
            'min_defined_prompts': min_defined,
        },
        'cells': cells,
        'coverage': {
            'cells': len(cells),
            'reportable_cells': len(reportable),
            'gap_cells': len(cells) - len(reportable),
            'defined_prompts': sum(c['defined_prompts'] for c in cells),
            'total_prompts': sum(c['prompts'] for c in cells),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--min-defined', type=int, default=DEFAULT_MIN_DEFINED_PROMPTS)
    args = parser.parse_args()
    payload = build(args.source, args.min_defined)
    atomic_new(args.output, payload)
    coverage = payload['coverage']
    print(json.dumps({'event': 'built', 'output': str(args.output), **coverage}))


if __name__ == '__main__':
    main()
