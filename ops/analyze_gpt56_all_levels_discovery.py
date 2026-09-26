#!/usr/bin/env python3
"""Authenticate all five domains at Levels 1--3 with 512 draws per problem.

The numerical and native-receipt core is the immutable, tested source from the
previous completed study. Parallel authentication changes execution only; each
run and every provider identity is checked before complete pools are merged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_all_levels_sol512_20260912'
PRIOR = ROOT / 'artifacts/modebench_discovery_five_domains_sol_20260912'
CORE = PRIOR / 'final_analysis_code/ops/analyze_gpt56_all_domain_discovery.py'
CORE_SHA256 = 'bdcc074304ac1bfa74d802a5a39fac4a20675e95639953a1195da47b12de4f0f'
PRIOR_SHA256 = '4af29a6ca79428290a1bf0ea3d058a97fc57c65cb7201c378145e52eb15236a0'
assert hashlib.sha256(CORE.read_bytes()).hexdigest() == CORE_SHA256, 'Changed immutable analysis core'
_spec = importlib.util.spec_from_file_location('_frozen_sol_discovery_core', CORE)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
# Frozen copies retain the original relative layout; bind actual repository
# artifacts to this workspace without changing any numerical or audit setting.
core.ROOT = ROOT
require, sha, binding = core.require, core.sha, core.binding
read_json, read_jsonl, write_json = core.read_json, core.read_jsonl, core.write_json
identity, summarize_cell = core.identity, core.summarize_cell
DOMAINS = core.DOMAINS
LEVELS = (1, 2, 3)
PROMPTS_PER_CELL = 16
DRAWS = 512



def grade_samples(rows, samples, grader, normalizer):
    """Execute the same frozen verifier once per identical problem/text pair.

    Every retained sample is still compared to its independently collected
    strict grade and receives its own raw-sample hash. Different texts, problem
    rows, and formatting are never merged by this execution cache.
    """
    cache, grades = {}, []
    for sample in samples:
        row = rows[identity(sample)]
        key = (identity(sample), sample['text'])
        if key not in cache:
            strict = grader(row['level'], row['domain'], row, sample['text'])
            normalized = normalizer(row, sample['text'], strict_grade=deepcopy(strict), grader=grader)
            cache[key] = strict, normalized
        strict, normalized = cache[key]
        require(all(strict[k] == sample[k] for k in ('verified', 'canonical_key')),
                'Frozen regrade differs from retained strict grade: ' + sample['sample_id'])
        grades.append({**{k: sample[k] for k in ('domain', 'level', 'row_index', 'sample_index')},
                       'strict': deepcopy(strict), 'normalization': deepcopy(normalized),
                       'raw_sample_sha256': core.object_sha(sample)})
    return grades, len(cache)


def grade_run(directory):
    directory = Path(directory).resolve()
    core.authenticate_manifest(directory)
    sys.path[:0] = [str(directory / 'code/ops'), str(directory / 'code/src')]
    from frontier_modebench_contract import grade_response
    from frontier_modebench_normalization import normalize_and_grade
    rows = {identity(row): row for row in read_jsonl(directory / 'rows.jsonl')}
    samples = read_jsonl(directory / 'samples.jsonl')
    grades, unique = grade_samples(rows, samples, grade_response, normalize_and_grade)
    path = directory / 'discovery_hosted_grades.jsonl'
    content = ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in grades)
    if path.exists():
        require(path.read_text() == content, 'Preserve an existing differing grading cache')
    else:
        path.write_text(content)
    write_json(directory / 'all_domain_grading_audit.json', {
        'status': 'complete', 'samples': binding(directory / 'samples.jsonl'), 'grades': binding(path),
        'grader': binding(directory / 'code/ops/frontier_modebench_contract.py'),
        'normalizer': binding(directory / 'code/ops/frontier_modebench_normalization.py'),
        'grader_driver': binding(__file__), 'responses': len(grades),
        'strict_verified': sum(row['strict']['verified'] for row in grades),
        'normalized_verified': sum(row['normalization']['verified'] for row in grades),
        'exact_input_cache': {'key': 'same frozen problem identity and exact response text',
                              'unique_inputs': unique, 'all_retained_strict_grades_compared': True},
        'api_calls': 0})
    return len(grades)


def authenticate_one(directory):
    pools, providers = {}, set()
    source = core.load_run(Path(directory), pools, providers)
    return source, pools, providers


def merge_authenticated(results):
    """Merge disjoint authenticated stages while preserving prompt identity."""
    pools, providers, sources = {}, set(), []
    directories = set()
    for source, incoming, ids in results:
        directory = str(Path(source['directory']).resolve())
        require(directory not in directories, 'Duplicate source run')
        directories.add(directory)
        require(not providers.intersection(ids), 'Repeated provider response across runs')
        providers.update(ids)
        sources.append(source)
        for key, pool in incoming.items():
            if key not in pools:
                pools[key] = {**pool, 'strict': dict(pool['strict']), 'normalization': dict(pool['normalization'])}
                continue
            target = pools[key]
            require(target['row_sha256'] == pool['row_sha256']
                    and target['messages_sha256'] == pool['messages_sha256'],
                    'Cannot merge changed problem or prompt bytes')
            for grading in ('strict', 'normalization'):
                require(not set(target[grading]).intersection(pool[grading]),
                        'Overlapping global sample indices')
                target[grading].update(pool[grading])
    require(len({s['request_controls_sha256'] for s in sources}) == 1,
            'Deployment controls changed across collection stages')
    require({name for s in sources for name in s['served_models']} == {'gpt-5.6-sol'},
            'Mixed provider-reported model aliases')
    require({name for s in sources for name in s['served_snapshots']} == {'gpt-5.6-sol-2026-07-09'},
            'Mixed provider-reported model snapshots')
    require(sum(s['responses'] for s in sources) == len(providers), 'Provider response accounting differs')
    return sources, pools


def load_support(support_path, prior):
    document = read_json(support_path)
    require(document['schema'] == 'gpt56-all-levels-discovery-support-v1', 'Unexpected all-level support schema')
    for item in document['sources'].values():
        require(sha(ROOT / item['path']) == item['sha256'], 'Changed support-certificate source')
    certificate_path = Path(support_path).with_name('support_certificate_manifest.json')
    certificate = read_json(certificate_path)
    require(certificate['schema'] == 'gpt56-all-levels-support-certificate-manifest-v1'
            and sha(support_path) == certificate['certificate']['sha256'], 'Changed support certificate output')
    require(certificate['sources'] == document['sources'], 'Support certificate source bindings differ')
    for item in certificate['code_copies']:
        require(sha(ROOT / item['path']) == item['sha256'], 'Changed frozen support code copy')
    old = {identity(p): p['support'] for cell in prior['cells'] for p in cell['prompts']}
    references = {}
    for reference in document['references'].values():
        key = identity(reference)
        require(key not in references, 'Duplicate support reference')
        if key in old:
            # Earlier Graph/Countdown certificate entries acquired the row hash
            # during their authenticated analysis; preserve all other fields.
            expected = old[key]
            reference = {**reference, 'row_sha256': reference.get('row_sha256', expected['row_sha256'])}
            require(reference == expected, 'Changed prior Level-2/3 support reference')
        else:
            require(key[0] == 1 and bool(reference.get('row_sha256')),
                    'New support must bind an authenticated Level-1 row')
        require(reference['support_count'] > 0, 'Empty known support certificate')
        require(reference['support_kind'] == ('exact' if key[1] == 'graph_coloring' else 'certified_lower_bound'),
                'Changed exact versus lower-bound support interpretation')
        references[key] = reference
    require(len(references) == 240 and set(old).issubset(references),
            'Expected all 80 new and 160 preserved support references')
    return references, certificate_path


def validate_complete_pools(pools, support):
    require(len(pools) == len(support) == 240 and set(pools) == set(support),
            'Expected the identical complete 240-problem sample/support inventory')
    expected_cells = {(level, domain) for level in LEVELS for domain in DOMAINS}
    counts = {cell: 0 for cell in expected_cells}
    for key, pool in pools.items():
        require(key[:2] in counts, 'Unexpected level or domain')
        counts[key[:2]] += 1
        require(pool['row_sha256'] == support[key]['row_sha256'], 'Support binds a different problem revision')
        for grading in ('strict', 'normalization'):
            require(set(pool[grading]) == set(range(DRAWS)),
                    'Each problem requires all 512 disjoint, gap-free global draw slots')
    require(set(counts.values()) == {PROMPTS_PER_CELL}, 'Expected exactly sixteen fixed problems per cell')


def build_report(run_dirs, support_path=BASE / 'support_reference.json', workers=4):
    prior_path = PRIOR / 'analysis/analysis.json'
    require(sha(prior_path) == PRIOR_SHA256, 'Changed frozen prior analysis')
    prior = read_json(prior_path)
    support, certificate_path = load_support(support_path, prior)
    require(type(workers) is int and 1 <= workers <= 8, 'Use one to eight authentication workers')
    directories = [str(Path(path).resolve()) for path in run_dirs]
    if workers == 1:
        sources, pools = merge_authenticated(map(authenticate_one, directories))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            sources, pools = merge_authenticated(executor.map(authenticate_one, directories))
    by_directory = {s['directory']: s for s in sources}
    for old in prior['sources']:
        require(old['directory'] in by_directory, 'The complete prior response pools must be retained')
        current = by_directory[old['directory']]
        require(all(current[k] == old[k] for k in ('manifest', 'samples', 'grades', 'responses')),
                'Changed prior immutable response source')
    validate_complete_pools(pools, support)
    analyses = {}
    for grading in ('strict', 'normalization'):
        prompts = [{**pool, 'keys': [pool[grading][i] for i in range(DRAWS)], 'support': support[key]}
                   for key, pool in sorted(pools.items())]
        analyses[grading] = [summarize_cell([p for p in prompts if (p['level'], p['domain']) == (level, domain)], level, domain)
                             for level in LEVELS for domain in DOMAINS]
    require(sum(s['responses'] for s in sources) == 122880, 'Expected all 122,880 unique responses')
    return {'schema': 'gpt56-all-levels-discovery-v1', 'status': 'complete', 'model': 'gpt-5.6-sol',
            'served_model': 'gpt-5.6-sol', 'served_snapshot_header': 'gpt-5.6-sol-2026-07-09',
            'grading': 'normalized_secondary', 'wording': 'original', 'prompt_arm': 'original',
            'levels': list(LEVELS), 'cells': analyses['normalization'], 'strict_cells': analyses['strict'],
            'sources': sources, 'prior_analysis': binding(prior_path), 'new_support': binding(support_path),
            'support_certificate': binding(certificate_path), 'analyzer': binding(__file__),
            'shared_analyzer_core': binding(CORE), 'responses': 122880, 'prompts': 240,
            'protocol': {'reasoning': 'medium', 'max_output_tokens': 8192, 'temperature_and_top_p': 'omitted',
                         'selection': 'First sixteen problems per domain/level by outcome-independent SHA256 ranking with seed20260911; preserve every prior selected Level-2/3 problem.',
                         'estimator': 'Exact rarefaction within each complete 512-response pool; average all sixteen problems equally.',
                         'bootstrap': {'replicates': core.REPLICATES, 'seed': core.SEED, 'unit': 'whole prompt',
                                       'strata': 'domain and level', 'pointwise': True},
                         'extension': 'User-requested exploratory fixed extension to512 in all five domains and all three levels, after previous completed stages; no outcome exclusions or early stopping within a stage.'},
            'limitations': ['Finite-pool curves cannot establish asymptotic saturation or absence of unseen modes.',
                            'Graph support is exact; other available-mode references are certified lower bounds.',
                            'Level1 and additional Level2/3 draws were collected after the previous study; deployment controls and provider-reported snapshot match but times differ.',
                            'Formatting normalization is the labeled main-figure sensitivity; strict results and all failures are retained.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grade-run', type=Path)
    parser.add_argument('--run', type=Path, action='append', default=[])
    parser.add_argument('--support', type=Path, default=BASE / 'support_reference.json')
    parser.add_argument('--output', type=Path, default=BASE / 'analysis/analysis.json')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.grade_run:
        print(json.dumps({'graded': grade_run(args.grade_run), 'api_calls': 0}))
        return
    require(args.run, 'Specify all prior and new completed runs')
    report = build_report(args.run, args.support, args.workers)
    if args.output.exists():
        require(read_json(args.output) == report, 'Preserve differing prior output; use a new path')
    else:
        write_json(args.output, report)
    print(json.dumps({'status': 'complete', 'output': str(args.output), 'responses': report['responses'], 'prompts': report['prompts']}))


if __name__ == '__main__':
    main()
