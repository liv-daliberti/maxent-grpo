#!/usr/bin/env python3
"""Retrospective conditional concentration from source-admitted saved keys.

Collection verifies raw source prefixes once and retains compact sufficient
samples. Analysis never rereads changing experiment directories. The protocol
is bound before concentration results are calculated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ops'))
AUDIT = ROOT / 'paper/audits/conditional_concentration_20260911'
DEFAULT_CACHE = AUDIT / 'verified_samples.jsonl.gz'
DEFAULT_RESULT = ROOT / 'paper/results/conditional_concentration_20260911.json'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
SCALES = ('qwen05b', 'falcon1b', 'qwen3b')
T95_DF4 = 2.7764451051977987
METRICS = ('collision', 'mean8', 'pass8', 'distinct8', 'extra8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def collision(counts):
    """Unbiased collision U-statistic for fixed-policy iid correct labels."""
    values = list(counts.values())
    if any(type(n) is not int or n < 0 for n in values):
        raise ValueError('key counts must be nonnegative integers')
    n = sum(values)
    return sum(v * (v - 1) for v in values) / (n * (n - 1)) if n >= 2 else None


def prompt_metrics(draws):
    if not draws or any(len(d) != 8 for d in draws):
        raise ValueError('metrics require nonempty intact K8 draw groups')
    if any(k is not None and (not isinstance(k, str) or not k) for d in draws for k in d):
        raise ValueError('verified keys must be nonempty strings or null')
    counts = Counter(k for d in draws for k in d if k is not None)
    correct = sum(counts.values())
    pass8 = statistics.mean(any(k is not None for k in d) for d in draws)
    distinct8 = statistics.mean(len({k for k in d if k is not None}) for d in draws)
    return {'collision': collision(counts), 'correct': correct, 'total': 8 * len(draws),
            'mean8': correct / (8 * len(draws)), 'pass8': pass8,
            'distinct8': distinct8, 'extra8': distinct8 - pass8,
            'correct_pairs': correct * (correct - 1) // 2,
            'colliding_pairs': sum(n * (n - 1) // 2 for n in counts.values())}


def means(records):
    return {k: statistics.mean(r[k] for r in records) if records else None for k in METRICS}


def pooled(records):
    denominator = sum(r['correct_pairs'] for r in records)
    return sum(r['colliding_pairs'] for r in records) / denominator if denominator else None


def pair_seed(a, b, *, draws_a=(0, 1, 2, 3), draws_b=(0, 1, 2, 3), prompt_ids=None):
    """Equal-prompt B-minus-A contrast on explicitly shared eligibility.

    Inputs map exact prompt IDs to four intact verified-key draw groups.
    Common-seed sample selection is descriptive unless independence assumptions
    for count/label conditioning hold. No missing prompts are silently joined.
    """
    if set(a) != set(b):
        raise ValueError('paired conditions have different exact prompt populations')
    if not draws_a or not draws_b or any(i not in range(4) for i in (*draws_a, *draws_b)):
        raise ValueError('invalid selected draw groups')
    if len(set(draws_a)) != len(draws_a) or len(set(draws_b)) != len(draws_b):
        raise ValueError('draw indices must be unique within each condition')
    ids = set(a) if prompt_ids is None else set(prompt_ids)
    if not ids <= set(a):
        raise ValueError('requested sensitivity population is not in both conditions')
    ma = {x: prompt_metrics([a[x][i] for i in draws_a]) for x in sorted(ids)}
    mb = {x: prompt_metrics([b[x][i] for i in draws_b]) for x in sorted(ids)}
    eligible = [x for x in sorted(ids) if ma[x]['collision'] is not None and mb[x]['collision'] is not None]
    ar, br = [ma[x] for x in eligible], [mb[x] for x in eligible]
    am, bm = means(ar), means(br)
    full_a = {k: statistics.mean(r[k] for r in ma.values()) if ma else None for k in METRICS if k != 'collision'}
    full_b = {k: statistics.mean(r[k] for r in mb.values()) if mb else None for k in METRICS if k != 'collision'}
    return {'n_total': len(ids), 'n_eligible': len(eligible), 'eligible_ids': eligible,
            'coverage': len(eligible) / len(ids) if ids else None,
            'a': am, 'b': bm,
            'delta': {k: bm[k] - am[k] if eligible else None for k in METRICS},
            'full_a': full_a, 'full_b': full_b,
            'pooled_collision_a': pooled(ar), 'pooled_collision_b': pooled(br),
            'pooled_all_eligible_a': pooled([r for r in ma.values() if r['collision'] is not None]),
            'pooled_all_eligible_b': pooled([r for r in mb.values() if r['collision'] is not None]),
            'eligibility': {'both': len(eligible),
                'a_only': sum(ma[x]['collision'] is not None and mb[x]['collision'] is None for x in ids),
                'b_only': sum(ma[x]['collision'] is None and mb[x]['collision'] is not None for x in ids),
                'neither': sum(ma[x]['collision'] is None and mb[x]['collision'] is None for x in ids)},
            'draws_a': list(draws_a), 'draws_b': list(draws_b)}


def summarize_seeds(values, registered_n=5):
    if registered_n != 5:
        raise ValueError('the frozen analysis uses five registered seeds per complete block')
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values.values()):
        raise ValueError('seed estimates must be finite; undefined seeds must be explicit upstream')
    vals = list(values.values())
    avg = statistics.mean(vals) if vals else None
    interval = None
    if len(vals) == 5:
        half = T95_DF4 * statistics.stdev(vals) / math.sqrt(5)
        interval = [avg - half, avg + half]
    return {'n': len(vals), 'registered_n': 5, 'mean': avg, 'ci95': interval,
            'range': [min(vals), max(vals)] if vals else None,
            'values': {str(k): v for k, v in sorted(values.items())},
            'uncertainty': 'nominal paired Student-t over five measured seed estimates' if interval is not None else 'descriptive; incomplete five-seed block'}


def compact_cell(cell):
    result = {k: v for k, v in cell.items() if k != 'checkpoints'}
    result['checkpoints'] = {}
    for step, cp in cell['checkpoints'].items():
        if cp is None:
            result['checkpoints'][str(step)] = None
            continue
        try:
            prompts = {}
            for draw in cp['draws']:
                rows = []
                for p in draw['prompts']:
                    pm = prompt_metrics([p['verified_keys']])
                    rows.append(pm)
                    item = prompts.setdefault(p['prompt_id'], {'prompt_index': p['prompt_index'],
                        'draws': [], 'request_seeds_by_draw': [], 'option_ids_by_draw': []})
                    item['draws'].append(p['verified_keys'])
                    item['request_seeds_by_draw'].append(p['request_seeds_by_option'])
                    item['option_ids_by_draw'].append(p['option_ids'])
                for metric, original in [('pass8', 'any_correct_at_k'), ('distinct8', 'distinct_correct_modes_at_k'), ('mean8', 'mean_at_k')]:
                    measured = statistics.mean(row[metric] for row in rows)
                    if abs(measured - draw['observed_metrics'][original]) > 1e-12:
                        raise ValueError(f'{metric} reconstructed from verified keys differs from frozen draw')
            result['checkpoints'][str(step)] = {'prompts': prompts,
                'sampling_certificate': cp['sampling_certificate'],
                'origins': [{'draw_index': d['draw_index'], 'origins': d['origins'],
                             'raw_payload_sha256': d['raw_payload_sha256']} for d in cp['draws']],
                'primary_metrics_reconstructed': True}
        except (ValueError, KeyError, TypeError) as exc:
            result['checkpoints'][str(step)] = None
            result['sample_issues'].append({'kind': 'metric_reconstruction_failure', 'step': int(step), 'reason': str(exc)})
    result['before_after_available'] = all(result['checkpoints'].get(s) is not None for s in ('0', '3072'))
    return result


def collect(output, workers=4):
    from exp_scaling.load_paper_collision_samples import load_primary_manifest, iter_primary_sample_cells
    from exp_scaling.load_paper_grpo_collision_sources import load_grpo_manifest, iter_grpo_sample_cells
    if output.exists():
        raise ValueError('refusing to overwrite a frozen collection')
    binding = json.loads((AUDIT / 'analysis_plan_binding.json').read_text())
    if sha(AUDIT / 'analysis_plan.md') != binding['analysis_plan_sha256']:
        raise ValueError('frozen analysis protocol changed')
    primary = load_primary_manifest()
    grpo = load_grpo_manifest()
    metadata = {'record_kind': 'manifest', 'schema': 'paper-conditional-collision-samples-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': binding,
        'primary': {k: v for k, v in primary.items() if k != 'snapshot'},
        'grpo': {k: v for k, v in grpo.items() if k not in ('cells', 'snapshot')},
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'ops/exp_scaling/load_paper_collision_samples.py', ROOT/'ops/exp_scaling/load_paper_grpo_collision_sources.py']}}
    temp = output.with_suffix(output.suffix + '.partial')
    counts, available, issue_count = Counter(), Counter(), 0
    with gzip.open(temp, 'wt', encoding='utf-8') as f:
        f.write(dump(metadata) + '\n')
        streams = (iter_primary_sample_cells(primary, workers=workers), iter_grpo_sample_cells(grpo, workers=workers))
        for stream in streams:
            for raw in stream:
                cell = compact_cell(raw)
                f.write(dump({'record_kind': 'cell', **cell}) + '\n')
                counts[cell['method']] += 1
                for step, cp in cell['checkpoints'].items():
                    if cp is not None:
                        available[f"{cell['level']}/{cell['method']}/{step}"] += 1
                issue_count += len(cell['sample_issues'])
                if sum(counts.values()) % 25 == 0:
                    print(f'Collected {sum(counts.values())} cells; source/sample issues={issue_count}', flush=True)
    temp.replace(output)
    summary = {'status': 'collected', 'path': str(output.relative_to(ROOT)), 'sha256': sha(output),
        'cells_by_method': dict(counts), 'available_checkpoints': dict(available), 'sample_issue_count': issue_count}
    (AUDIT / 'collection_receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'analyze'])
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output', type=Path, default=DEFAULT_RESULT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'collect':
        collect(args.cache, args.workers)
    else:
        analyze(args.cache, args.output)


if __name__ == '__main__':
    main()
