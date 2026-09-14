#!/usr/bin/env python3
"""Authenticate and summarize the exploratory all-domain Sol discovery follow-up.

Combines disjoint fresh response slots, never repeated provider attempts. The
original three-domain 64-draw analysis remains untouched. Formatting-normalized
results are a labeled display sensitivity; strict curves and ordered prefixes
are retained alongside them. Each point is finite-pool rarefaction, not an
extrapolation to unobserved modes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_five_domains_sol_20260912'
OLD = ROOT / 'paper/results/modebench_discovery_curves_20260911.json'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LEVELS = (2, 3)
REPLICATES = 20000
SEED = 20260911


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), 'sha256': sha(path)}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def identity(row):
    return row['level'], row['domain'], row['row_index']


def sample_identity(row):
    return *identity(row), row['sample_index']


def authenticate_manifest(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    for name, expected in manifest['artifact_sha256'].items():
        require(sha(directory / name) == expected, 'Changed immutable input: ' + str(directory / name))
    for name, expected in manifest['code_sha256'].items():
        require(sha(directory / 'code' / name) == expected, 'Changed frozen code: ' + name)
    require(manifest['model'] == 'gpt-5.6-sol', 'Different requested model')
    return manifest


def load_native_auditor(directory):
    path = Path(directory) / 'code/ops/evaluate_native_prompt_ablation.py'
    spec = importlib.util.spec_from_file_location('_native_discovery_' + sha(path)[:16], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_grading(directory):
    directory = Path(directory)
    new = directory / 'all_domain_grading_audit.json'
    if new.exists():
        audit = read_json(new)
        require(audit['status'] == 'complete', 'Incomplete grading audit')
        for key, name in [('samples', 'samples.jsonl'), ('grades', 'discovery_hosted_grades.jsonl')]:
            require(sha(directory / name) == audit[key]['sha256'], 'Grading audit differs from ' + name)
    else:
        audit = read_json(directory / 'discovery_hosted_grading_audit.json')
        require(audit['status'] == 'complete'
                and sha(directory / 'samples.jsonl') == audit['raw_samples_sha256']
                and sha(directory / 'discovery_hosted_grades.jsonl') == audit['cache_sha256'],
                'Legacy grading audit differs from raw samples or grade cache')
    return binding(new if new.exists() else directory / 'discovery_hosted_grading_audit.json')


def grade_run(directory):
    """Run once in an isolated process, using this run's frozen grader/normalizer."""
    directory = Path(directory).resolve()
    authenticate_manifest(directory)
    sys.path[:0] = [str(directory / 'code/ops'), str(directory / 'code/src')]
    from frontier_modebench_contract import grade_response
    from frontier_modebench_normalization import normalize_and_grade
    rows = {identity(row): row for row in read_jsonl(directory / 'rows.jsonl')}
    samples = read_jsonl(directory / 'samples.jsonl')
    grades = []
    for sample in samples:
        row = rows[identity(sample)]
        strict = grade_response(row['level'], row['domain'], row, sample['text'])
        require(all(strict[key] == sample[key] for key in ('verified', 'canonical_key')),
                'Frozen serial grade differs from retained grade: ' + sample['sample_id'])
        normalized = normalize_and_grade(row, sample['text'], strict_grade=strict, grader=grade_response)
        grades.append({**{key: sample[key] for key in ('domain', 'level', 'row_index', 'sample_index')},
                       'strict': strict, 'normalization': normalized, 'raw_sample_sha256': object_sha(sample)})
    path = directory / 'discovery_hosted_grades.jsonl'
    content = ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in grades)
    if path.exists():
        require(path.read_text() == content, 'Preserve existing differing grading cache')
    else:
        path.write_text(content)
    write_json(directory / 'all_domain_grading_audit.json', {
        'status': 'complete', 'samples': binding(directory / 'samples.jsonl'),
        'grades': binding(path), 'grader': binding(directory / 'code/ops/frontier_modebench_contract.py'),
        'normalizer': binding(directory / 'code/ops/frontier_modebench_normalization.py'),
        'responses': len(grades), 'strict_verified': sum(row['strict']['verified'] for row in grades),
        'normalized_verified': sum(row['normalization']['verified'] for row in grades), 'api_calls': 0})
    return len(grades)


def rarefaction(counts, total, k):
    require(type(total) is int and total > 0 and type(k) is int and 0 <= k <= total,
            'Invalid finite-pool budget')
    require(all(type(n) is int and n > 0 for n in counts) and sum(counts) <= total,
            'Invalid verified mode counts')
    denominator = math.comb(total, k)
    seen = lambda n: 1 - math.comb(total - n, k) / denominator
    p = seen(sum(counts))
    d = math.fsum(seen(n) for n in counts)
    return {'distinct': d, 'pass': p, 'breadth': d - p}


def estimate(values, indices):
    values = np.asarray(values, dtype=float)
    means = values[indices].mean(axis=1)
    return {'estimate': float(values.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist()}


def summarize_cell(prompts, level, domain):
    require(len(prompts) == 16, 'Expected all 16 fixed prompts: ' + str((level, domain)))
    sizes = {len(p['keys']) for p in prompts}
    require(len(sizes) == 1 and next(iter(sizes)) >= 64, 'Incomplete or unequal cell budgets')
    total = sizes.pop()
    require(total & (total - 1) == 0, 'Expected a complete fixed power-of-two budget')
    grid = [2**i for i in range(total.bit_length())]
    seed = int(hashlib.sha256(f'{SEED}:{level}:{domain}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 16, size=(REPLICATES, 16))
    support_values = [p['support']['support_count'] for p in prompts]
    kinds = {p['support']['support_kind'] for p in prompts}
    points, vectors = [], {}
    for k in grid:
        results = [rarefaction(list(Counter(key for key in p['keys'] if key is not None).values()), total, k)
                   for p in prompts]
        vectors[k] = {metric: [result[metric] for result in results] for metric in ('distinct', 'pass', 'breadth')}
        points.append({'k': k, **{metric: estimate(values, indices) for metric, values in vectors[k].items()}})
    tail = {'from_k': total // 2, 'to_k': total}
    for metric in ('distinct', 'pass', 'breadth'):
        tail[metric] = estimate(np.array(vectors[total][metric]) - np.array(vectors[total // 2][metric]), indices)
    observed = [{**{key: p[key] for key in ('level', 'domain', 'row_index', 'row_sha256', 'support')},
                 'responses': total, 'correct_draws': sum(key is not None for key in p['keys']),
                 'mode_counts': sorted(Counter(key for key in p['keys'] if key is not None).values(), reverse=True),
                 'prefix': {str(k): {'distinct': len({key for key in p['keys'][:k] if key is not None}),
                                    'pass': int(any(key is not None for key in p['keys'][:k]))} for k in grid}}
                for p in prompts]
    return {'domain': domain, 'level': level, 'n_prompts': len(prompts), 'max_draws': total,
            'support': {'mean': float(np.mean(support_values)), 'range': [min(support_values), max(support_values)],
                        'kind': 'exact' if kinds == {'exact'} else 'certified_lower_bound'},
            'points': points, 'tail': tail, 'prompts': observed,
            'bootstrap': {'seed': seed, 'replicates': REPLICATES, 'unit': 'whole prompt', 'pointwise': True}}


def load_run(directory, pools, providers):
    directory = Path(directory).resolve()
    manifest = authenticate_manifest(directory)
    grading_audit = authenticate_grading(directory)
    native = load_native_auditor(directory)
    groups = {row['group_id']: row for row in read_jsonl(directory / 'http_requests.jsonl')}
    rows = {identity(row): row for row in read_jsonl(directory / 'rows.jsonl')}
    samples = read_jsonl(directory / 'samples.jsonl')
    requests = {sample_identity(row): row for row in read_jsonl(directory / 'requests.jsonl')}
    grades = {sample_identity(row): row for row in read_jsonl(directory / 'discovery_hosted_grades.jsonl')}
    require(len(samples) == len(requests) == len(grades) == manifest['request_count'], 'Incomplete run: ' + str(directory))
    seen = set()
    request_controls = set()
    served = set()
    snapshots = set()
    for sample in samples:
        slot = sample_identity(sample)
        require(slot not in seen and slot in requests and slot in grades, 'Duplicate or unknown response slot')
        seen.add(slot)
        item, grade, row = requests[slot], grades[slot], rows[identity(sample)]
        require(grade['raw_sample_sha256'] == object_sha(sample), 'Grade cache refers to different raw sample')
        require(sample['request_sha256'] == item['request_sha256'] == object_sha(item['request']), 'Request payload mismatch')
        require(sample['row_sha256'] == item['row_sha256'] == object_sha(row), 'Changed problem identity')
        request = item['request']
        controls = {key: value for key, value in request.items() if key not in ('input', 'messages')}
        require(controls.get('model') == 'gpt-5.6-sol' and controls.get('reasoning') == {'effort': 'medium'}
                and controls.get('max_output_tokens') == 8192 and 'temperature' not in controls and 'top_p' not in controls,
                'Different deployment controls')
        request_controls.add(object_sha(controls))
        native.validate_completed(directory, item, groups[item['group_id']], sample, {})
        raw = directory / sample['raw_receipt']
        receipt = read_json(raw)
        require(object_sha(receipt) == sample['raw_receipt_sha256'], 'Changed native response receipt')
        headers = {str(key).lower(): value for key, value in receipt['headers'].items()}
        require(headers.get('x-ms-served-model') == 'gpt-5.6-sol-2026-07-09', 'Unexpected served-model snapshot header')
        snapshots.add(headers['x-ms-served-model'])
        body = receipt.get('body', receipt.get('response', {}))
        require(receipt['request_sha256'] == item['request_sha256'] and sample['sample_id'] in receipt['sample_ids'],
                'Native receipt is for a different request')
        require(body.get('id') == sample['provider_sample_identity'][0], 'Provider receipt identity mismatch')
        if isinstance(body, dict) and body.get('model'):
            served.add(body['model'])
        provider = tuple(sample['provider_sample_identity'])
        require(provider not in providers, 'Repeated provider response counted twice')
        providers.add(provider)
        key = identity(sample)
        pool = pools.setdefault(key, {'level': key[0], 'domain': key[1], 'row_index': key[2],
                                     'row_sha256': object_sha(row), 'messages_sha256': object_sha(request.get('input', request.get('messages'))),
                                     'strict': {}, 'normalization': {}})
        require(pool['row_sha256'] == object_sha(row)
                and pool['messages_sha256'] == object_sha(request.get('input', request.get('messages'))),
                'Cannot combine different prompts or problem revisions')
        require(slot[-1] not in pool['strict'], 'Overlapping sample indices across pools')
        for grading in ('strict', 'normalization'):
            result = grade[grading]
            require(bool(result['verified']) == (result['canonical_key'] is not None), 'Malformed correctness/mode grade')
            pool[grading][slot[-1]] = result['canonical_key'] if result['verified'] else None
    require(len(request_controls) == 1, 'Mixed request controls within run')
    return {'directory': str(directory), 'manifest': binding(directory / 'manifest.json'),
            'samples': binding(directory / 'samples.jsonl'), 'grades': binding(directory / 'discovery_hosted_grades.jsonl'),
            'request_controls_sha256': request_controls.pop(), 'served_models': sorted(served),
            'served_snapshots': sorted(snapshots), 'grading_audit': grading_audit,
            'responses': len(samples), 'native_receipts_authenticated': True}


def build_report(run_dirs, support_path):
    original = read_json(OLD)
    model = next(m for m in original['models'] if m['model_id'] == 'gpt56sol')
    legacy_support = {identity(p): p['support_reference'] for p in model['analyses']['strict']['prompts']['original']}
    support_document = read_json(support_path)
    for record in support_document['sources'].values():
        require(sha(record['path']) == record['sha256'], 'Changed source of support certificate')
    certificate_path = Path(support_path).with_name('support_certificate_manifest.json')
    certificate = read_json(certificate_path)
    require(sha(support_path) == certificate['output']['sha256'], 'Changed certified support output')
    new_rows = {identity(row): row for row in read_jsonl(support_document['sources']['new_rows']['path'])}
    support = {}
    for reference in support_document['references'].values():
        key = identity(reference)
        require(key not in support, 'Duplicate support reference')
        if key in legacy_support:
            require(reference == legacy_support[key], 'Changed legacy support reference')
        else:
            require(key in new_rows, 'New support lacks an authenticated problem row')
            reference = {**reference, 'row_sha256': object_sha(new_rows[key])}
        support[key] = reference
    pools, providers = {}, set()
    sources = [load_run(directory, pools, providers) for directory in run_dirs]
    require(len({s['request_controls_sha256'] for s in sources}) == 1, 'Deployment controls changed across collection runs')
    served = {name for source in sources for name in source['served_models']}
    require(len(served) == 1 and served == {'gpt-5.6-sol'}, 'Mixed or unexpected reported model alias')
    snapshots = {name for source in sources for name in source['served_snapshots']}
    require(snapshots == {'gpt-5.6-sol-2026-07-09'}, 'Inconsistent provider-reported model snapshot headers')
    require(len(pools) == len(support) == 160 and set(pools) == set(support), 'Expected identical full five-domain support and sample inventories')
    analyses = {}
    for grading in ('strict', 'normalization'):
        prompts = []
        for key, pool in sorted(pools.items()):
            slots = pool[grading]
            require(set(slots) == set(range(len(slots))), 'Non-contiguous or missing response slots')
            require(pool['row_sha256'] == support[key]['row_sha256'], 'Support is for a different problem')
            prompts.append({**pool, 'keys': [slots[index] for index in range(len(slots))], 'support': support[key]})
        analyses[grading] = [summarize_cell([p for p in prompts if (p['level'], p['domain']) == (level, domain)], level, domain)
                             for level in LEVELS for domain in DOMAINS]
    return {'schema': 'gpt56-all-domain-discovery-v1', 'status': 'complete', 'model': 'gpt-5.6-sol',
            'served_model': served.pop(), 'served_snapshot_header': snapshots.pop(), 'grading': 'normalized_secondary', 'wording': 'original', 'prompt_arm': 'original',
            'cells': analyses['normalization'], 'strict_cells': analyses['strict'],
            'sources': sources, 'prior_analysis': binding(OLD), 'new_support': binding(support_path), 'support_certificate': binding(certificate_path), 'analyzer': binding(__file__),
            'responses': sum(source['responses'] for source in sources), 'prompts': 160,
            'protocol': {'reasoning': 'medium', 'max_output_tokens': 8192, 'temperature_and_top_p': 'omitted',
                         'selection': 'Fixed outcome-independent selection; retain all 16 problems per domain and level.',
                         'estimator': 'Exact rarefaction within each complete observed response pool; average all prompts equally.',
                         'bootstrap': {'replicates': REPLICATES, 'seed': SEED, 'unit': 'whole prompt', 'strata': 'domain and level', 'pointwise': True},
                         'extension': 'Exploratory additional collection after examining the original 64-draw results; no population or failure exclusions.'},
            'limitations': ['Finite-pool curves do not establish asymptotic saturation or absence of unseen modes.',
                            'Certified support lower bounds are not exhaustive totals.',
                            'New Graph/Countdown samples were collected after the original three-domain experiment.',
                            'Formatting-normalized main display is a sensitivity; strict verification results are retained.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grade-run', type=Path)
    parser.add_argument('--run', type=Path, action='append', default=[])
    parser.add_argument('--support', type=Path, default=BASE / 'support_reference.json')
    parser.add_argument('--output', type=Path, default=BASE / 'analysis/analysis.json')
    args = parser.parse_args()
    if args.grade_run:
        print(json.dumps({'graded': grade_run(args.grade_run), 'api_calls': 0}))
        return
    require(bool(args.run), 'Specify every complete response run')
    report = build_report(args.run, args.support)
    if args.output.exists():
        require(read_json(args.output) == report, 'Existing analysis differs; use a new output path')
    else:
        write_json(args.output, report)
    print(json.dumps({'status': 'complete', 'output': str(args.output), 'responses': report['responses'], 'prompts': report['prompts']}))


if __name__ == '__main__':
    main()
