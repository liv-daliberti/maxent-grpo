#!/usr/bin/env python3
"""Authenticate 32 fixed problems per cell at 512 draws without new model calls.

Preserve every previous 16-problem response pool. The only numerical change to
the authenticated earlier core is the whole-problem bootstrap's cohort size.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_all_levels_sol32x512_20260913'
PRIOR = ROOT / 'artifacts/modebench_discovery_all_levels_sol512_20260912'
DRIVER = ROOT / 'ops/analyze_gpt56_all_levels_discovery.py'
DRIVER_SHA256 = '9415b36227a5515f4151b99421ee6289b1a02628b09344087ee218f3034cd9d0'
PRIOR_SHA256 = '54eac7ed94822927abb926f267b35b98a24f581088b8ea102dda41196ba58e62'
AMENDMENT = ROOT / 'paper/preregistration/gpt56sol_generalization32x512_20260913.md'
AMENDMENT_SHA256 = '52ef0bf7a919db4271721e7a493d97a08e9d30b17605bf027ae057f96c04d76d'
SUPPORT_SHA256 = '4ea33166250d056b466785457d98d1462164ef416ce69fd462995992bfdf4674'
if hashlib.sha256(DRIVER.read_bytes()).hexdigest() != DRIVER_SHA256:
    raise ValueError('Changed immutable prior analysis driver')
_spec = importlib.util.spec_from_file_location('_sol32_authenticated_driver', DRIVER)
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
core = previous.core
require, sha, binding = core.require, core.sha, core.binding
read_json, read_jsonl, write_json = core.read_json, core.read_jsonl, core.write_json
identity = core.identity
DOMAINS, LEVELS = core.DOMAINS, (1, 2, 3)
PROMPTS_PER_CELL, DRAWS, PROMPTS, RESPONSES = 32, 512, 480, 245760
# Both functions remain byte-identical to the authenticated prior driver.
grade_samples, grade_run = previous.grade_samples, previous.grade_run
merge_authenticated = previous.merge_authenticated


def authenticate_one(directory):
    return previous.authenticate_one(directory)


def summarize_cell(prompts, level, domain, expected_prompts=PROMPTS_PER_CELL):
    """Same frozen estimator and RNG convention, resampling all cohort rows."""
    require(len(prompts) == expected_prompts, 'Expected every fixed problem in the cell')
    require(len({identity(p) for p in prompts}) == len(prompts), 'Duplicate problem in cell')
    require(all(identity(p)[:2] == (level, domain) for p in prompts), 'Mixed cell identities')
    require({len(p['keys']) for p in prompts} == {DRAWS}, 'Each problem requires exactly 512 draws')
    total = DRAWS
    grid = [2**i for i in range(total.bit_length())]
    seed = int(hashlib.sha256(f'{core.SEED}:{level}:{domain}'.encode()).hexdigest()[:8], 16)
    rng = core.np.random.default_rng(seed)
    indices = rng.integers(0, expected_prompts, size=(core.REPLICATES, expected_prompts))
    support_values = [p['support']['support_count'] for p in prompts]
    kinds = {p['support']['support_kind'] for p in prompts}
    points, vectors = [], {}
    counts = [list(Counter(key for key in p['keys'] if key is not None).values()) for p in prompts]
    for k in grid:
        results = [core.rarefaction(values, total, k) for values in counts]
        vectors[k] = {metric: [result[metric] for result in results] for metric in ('distinct', 'pass', 'breadth')}
        points.append({'k': k, **{metric: core.estimate(values, indices) for metric, values in vectors[k].items()}})
    tail = {'from_k': total // 2, 'to_k': total}
    for metric in ('distinct', 'pass', 'breadth'):
        tail[metric] = core.estimate(core.np.array(vectors[total][metric]) - core.np.array(vectors[total // 2][metric]), indices)
    observed = [{**{key: p[key] for key in ('level', 'domain', 'row_index', 'row_sha256', 'support')},
                 'responses': total, 'correct_draws': sum(key is not None for key in p['keys']),
                 'mode_counts': sorted(Counter(key for key in p['keys'] if key is not None).values(), reverse=True),
                 'prefix': {str(k): {'distinct': len({key for key in p['keys'][:k] if key is not None}),
                                    'pass': int(any(key is not None for key in p['keys'][:k]))} for k in grid}}
                for p in prompts]
    return {'domain': domain, 'level': level, 'n_prompts': len(prompts), 'max_draws': total,
            'support': {'mean': float(core.np.mean(support_values)), 'range': [min(support_values), max(support_values)],
                        'kind': 'exact' if kinds == {'exact'} else 'certified_lower_bound'},
            'points': points, 'tail': tail, 'prompts': observed,
            'bootstrap': {'seed': seed, 'replicates': core.REPLICATES, 'unit': 'whole prompt', 'pointwise': True}}


def load_support(support_path, prior):
    document = read_json(support_path)
    require(document['schema'] == 'gpt56-all-levels32-discovery-support-v1', 'Unexpected expanded support schema')
    for item in document['sources'].values():
        require(sha(ROOT / item['path']) == item['sha256'], 'Changed support-certificate source')
    certificate_path = Path(support_path).with_name('support_certificate_manifest.json')
    certificate = read_json(certificate_path)
    require(certificate['schema'] == 'gpt56-all-levels32-support-certificate-manifest-v1'
            and sha(support_path) == certificate['certificate']['sha256'], 'Changed support certificate output')
    require(certificate['sources'] == document['sources'], 'Support source bindings differ')
    for item in certificate['code_copies']:
        require(sha(ROOT / item['path']) == item['sha256'], 'Changed frozen support code copy')
    legacy_path = PRIOR / 'support_reference.json'
    require(sha(legacy_path) == SUPPORT_SHA256, 'Changed frozen prior support certificate')
    legacy = read_json(legacy_path)['references']
    require(all(document['references'].get(key) == ref for key, ref in legacy.items()),
            'Changed one of the 240 preserved support references')
    old = {identity(p): p['support'] for cell in prior['cells'] for p in cell['prompts']}
    references = {}
    for reference in document['references'].values():
        key = identity(reference)
        require(key not in references, 'Duplicate support reference')
        if key in old:
            reference = {**reference, 'row_sha256': reference.get('row_sha256', old[key]['row_sha256'])}
            require(reference == old[key], 'Changed prior analyzed support reference')
        require(bool(reference.get('row_sha256')), 'Support must bind its exact selected row')
        require(type(reference['support_count']) is int and reference['support_count'] > 0, 'Invalid known support count')
        require(reference['support_kind'] == ('exact' if key[1] == 'graph_coloring' else 'certified_lower_bound'),
                'Changed exact versus lower-bound support interpretation')
        references[key] = reference
    require(len(references) == PROMPTS and len(old) == 240 and set(old).issubset(references),
            'Expected 240 new and 240 preserved support references')
    # The certificate binds exact first-32 and next-16 source selection.
    selection_path = ROOT / document['sources']['all_rows']['path']
    selected = {identity(r): r for r in read_jsonl(selection_path)}
    require(set(selected) == set(references), 'Support differs from frozen selected row inventory')
    for key, row in selected.items():
        require(core.object_sha(row) == references[key]['row_sha256'], 'Support differs from frozen selected row bytes')
    return references, certificate_path


def validate_complete_pools(pools, support):
    require(len(pools) == len(support) == PROMPTS and set(pools) == set(support),
            'Expected the identical complete 480-problem sample/support inventory')
    counts = Counter()
    expected_cells = {(level, domain) for level in LEVELS for domain in DOMAINS}
    for key, pool in pools.items():
        require(key[:2] in expected_cells and identity(pool) == key, 'Unexpected or inconsistent problem identity')
        counts[key[:2]] += 1
        require(pool['row_sha256'] == support[key]['row_sha256'], 'Support binds a different problem revision')
        for grading in ('strict', 'normalization'):
            require(set(pool[grading]) == set(range(DRAWS)), 'Each problem requires all 512 gap-free draw slots')
    require(counts == Counter({cell: PROMPTS_PER_CELL for cell in expected_cells}),
            'Expected exactly 32 fixed problems in every one of the fifteen cells')


def build_report(run_dirs, support_path=BASE / 'support_reference.json', workers=4):
    require(sha(AMENDMENT) == AMENDMENT_SHA256, 'Changed prospective expansion amendment')
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
        require(old['directory'] in by_directory, 'Every prior response source must be retained')
        require(all(by_directory[old['directory']][k] == old[k]
                    for k in ('manifest', 'samples', 'grades', 'grading_audit', 'responses')),
                'Changed immutable prior response source')
    validate_complete_pools(pools, support)
    old_ids = {identity(p) for cell in prior['cells'] for p in cell['prompts']}
    analyses, cohorts = {}, {'retained_first16': {}, 'added_next16': {}}
    for grading in ('strict', 'normalization'):
        prompts = [{**pool, 'keys': [pool[grading][i] for i in range(DRAWS)], 'support': support[key]}
                   for key, pool in sorted(pools.items())]
        analyses[grading] = []
        for cohort in cohorts.values():
            cohort[grading] = []
        for level in LEVELS:
            for domain in DOMAINS:
                cell_prompts = [p for p in prompts if identity(p)[:2] == (level, domain)]
                analyses[grading].append(summarize_cell(cell_prompts, level, domain))
                for name, use_old in [('retained_first16', True), ('added_next16', False)]:
                    subset = [p for p in cell_prompts if (identity(p) in old_ids) == use_old]
                    cohorts[name][grading].append(summarize_cell(subset, level, domain, expected_prompts=16))
        frozen_cells = prior['cells' if grading == 'normalization' else 'strict_cells']
        require(cohorts['retained_first16'][grading] == frozen_cells,
                'Retained cohort no longer reproduces its complete frozen analysis')
    require(sum(s['responses'] for s in sources) == RESPONSES, 'Expected all 245,760 unique responses')
    return {'schema': 'gpt56-all-levels-discovery-v1', 'status': 'complete', 'model': 'gpt-5.6-sol',
            'served_model': 'gpt-5.6-sol', 'served_snapshot_header': 'gpt-5.6-sol-2026-07-09',
            'grading': 'normalized_secondary', 'wording': 'original', 'prompt_arm': 'original',
            'levels': list(LEVELS), 'cells': analyses['normalization'], 'strict_cells': analyses['strict'],
            'sources': sources, 'prior_analysis': binding(prior_path), 'new_support': binding(support_path),
            'prospective_expansion_amendment': binding(AMENDMENT),
            'support_certificate': binding(certificate_path), 'analyzer': binding(__file__),
            'shared_analyzer_driver': binding(DRIVER), 'shared_analyzer_core': binding(previous.CORE),
            'responses': RESPONSES, 'prompts': PROMPTS, 'retained_responses': 122880, 'added_responses': 122880,
            'cohort_sensitivity': {'interpretation': 'Descriptive first-16 versus next-16 comparison; different problems and collection times, not randomized collection-time groups or a causal time-effect estimate.',
                                   'cohorts': cohorts},
            'protocol': {'reasoning': 'medium', 'max_output_tokens': 8192, 'temperature_and_top_p': 'omitted',
                         'selection': 'First 32 problems per domain/level by outcome-independent SHA256 ranking with seed20260911; preserve all first16 and add ranks17 through32.',
                         'estimator': 'Exact rarefaction within each complete 512-response pool; average all32 problems equally, including failures.',
                         'bootstrap': {'replicates': core.REPLICATES, 'seed': core.SEED, 'unit': 'whole prompt',
                                       'strata': 'domain and level', 'pointwise': True},
                         'extension': 'User-requested exploratory fixed expansion from16 to32 problems per cell at512 responses, after observing the prior study; no outcome exclusions or early stopping within the expansion.'},
            'limitations': ['Finite-pool curves cannot establish asymptotic saturation or absence of unseen modes.',
                            'Graph support is exact; other available-mode references are certified lower bounds.',
                            'The extra16 problems per cell were collected later; deployment controls and provider-reported snapshot match but collection times differ.',
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
