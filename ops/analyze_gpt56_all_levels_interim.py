#!/usr/bin/env python3
"""Authenticate a frozen 128-draw interim view while the 512-draw study runs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import analyze_gpt56_all_levels_discovery as final

core = final.core
ROOT, BASE = final.ROOT, final.BASE
OUT = BASE / 'interim128'
DRAWS = 128


def snapshot_run(directory, target, selection_path):
    directory, target = Path(directory), Path(target)
    manifest = core.authenticate_manifest(directory)
    rows = {core.identity(r): r for r in core.read_jsonl(directory / 'rows.jsonl')}
    selection = {tuple(r['native_slot']): r['interim_index'] for r in core.read_json(selection_path)}
    samples = [r for r in core.read_jsonl(directory / 'samples.jsonl') if core.sample_identity(r) in selection]
    core.require({core.sample_identity(r) for r in samples} == set(selection), 'A prospectively selected interim slot has not completed')
    core.require(samples, 'Empty interim source')
    requests = {core.sample_identity(r): r for r in core.read_jsonl(directory / 'requests.jsonl')}
    groups = {r['group_id']: r for r in core.read_jsonl(directory / 'http_requests.jsonl')}
    sys.path[:0] = [str(directory / 'code/ops'), str(directory / 'code/src')]
    from frontier_modebench_contract import grade_response
    from frontier_modebench_normalization import normalize_and_grade
    grades, unique_inputs = final.grade_samples(rows, samples, grade_response, normalize_and_grade)
    native = core.load_native_auditor(directory)
    controls, providers, slots, pools = set(), set(), set(), {}
    for sample, grade in zip(samples, grades):
        slot = core.sample_identity(sample)
        core.require(slot not in slots, 'Duplicate interim slot')
        slots.add(slot)
        item, row = requests[slot], rows[core.identity(sample)]
        core.require(sample['request_sha256'] == item['request_sha256'] == core.object_sha(item['request']), 'Changed request')
        core.require(sample['row_sha256'] == item['row_sha256'] == core.object_sha(row), 'Changed row')
        request = item['request']
        control = {k: v for k, v in request.items() if k not in ('input', 'messages')}
        core.require(control.get('model') == 'gpt-5.6-sol' and control.get('reasoning') == {'effort': 'medium'}
                     and control.get('max_output_tokens') == 8192 and 'temperature' not in control and 'top_p' not in control,
                     'Changed generation controls')
        controls.add(core.object_sha(control))
        native.validate_completed(directory, item, groups[item['group_id']], sample, {})
        receipt = core.read_json(directory / sample['raw_receipt'])
        core.require(core.object_sha(receipt) == sample['raw_receipt_sha256'], 'Changed receipt')
        headers = {str(k).lower(): v for k, v in receipt['headers'].items()}
        core.require(headers.get('x-ms-served-model') == 'gpt-5.6-sol-2026-07-09', 'Changed snapshot')
        body = receipt.get('body', receipt.get('response', {}))
        provider = tuple(sample['provider_sample_identity'])
        core.require(body.get('model') == 'gpt-5.6-sol' and body.get('id') == provider[0]
                     and receipt['request_sha256'] == item['request_sha256'] and sample['sample_id'] in receipt['sample_ids'],
                     'Changed provider identity')
        core.require(provider not in providers, 'Duplicate provider response')
        providers.add(provider)
        key = core.identity(sample)
        pool = pools.setdefault(key, {'level': key[0], 'domain': key[1], 'row_index': key[2],
                                     'row_sha256': core.object_sha(row),
                                     'messages_sha256': core.object_sha(request.get('input', request.get('messages'))),
                                     'strict': {}, 'normalization': {}})
        core.require(pool['messages_sha256'] == core.object_sha(request.get('input', request.get('messages'))), 'Changed native prompt bytes')
        for grading in ('strict', 'normalization'):
            result = grade[grading]
            core.require(bool(result['verified']) == (result['canonical_key'] is not None), 'Malformed grade')
            pool[grading][selection[slot]] = result['canonical_key'] if result['verified'] else None
    core.require(len(controls) == 1, 'Mixed controls')
    target.mkdir(parents=True, exist_ok=False)
    for name, records in [('samples.jsonl', samples), ('grades.jsonl', grades),
                          ('requests.jsonl', [requests[core.sample_identity(s)] for s in samples])]:
        (target/name).write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in records))
    source = {'directory': str(directory.resolve()), 'manifest': core.binding(directory/'manifest.json'),
              'samples': core.binding(target/'samples.jsonl'), 'grades': core.binding(target/'grades.jsonl'),
              'requests': core.binding(target/'requests.jsonl'), 'responses': len(samples),
              'request_controls_sha256': controls.pop(), 'served_models': ['gpt-5.6-sol'],
              'served_snapshots': ['gpt-5.6-sol-2026-07-09'], 'native_receipts_authenticated': True,
              'view': 'First 128 planned HTTP-manifest requests per problem across collection stages; parent collection may continue.',
              'selection': core.binding(selection_path),
              'unique_verified_inputs': unique_inputs, 'analyzer': core.binding(__file__)}
    core.write_json(target/'authenticated.json', {'source': source, 'providers': sorted(providers), 'pools': list(pools.values())})
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-run', type=Path)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if args.snapshot_run:
        print(json.dumps(snapshot_run(args.snapshot_run, args.target, args.selection)))
        return
    prior = core.read_json(final.PRIOR/'analysis/analysis.json')
    support, certificate = final.load_support(BASE/'support_reference.json', prior)
    directories = [Path(s['directory']) for s in prior['sources']]
    directories += [Path(v['run_dir']) for v in core.read_json(BASE/'manifest.json')['collection_runs'].values()]
    jobs, assigned = [], {}
    selection_dir = OUT/'manifest_order_selection'
    selection_dir.mkdir(parents=True, exist_ok=False)
    for index, directory in enumerate(directories):
        requests = {r['sample_id']: r for r in core.read_jsonl(directory/'requests.jsonl')}
        selected = []
        for group in core.read_jsonl(directory/'http_requests.jsonl'):
            for sid in group['sample_ids']:
                request = requests[sid]
                key = core.identity(request)
                count = assigned.get(key, 0)
                if count < DRAWS:
                    selected.append({'native_slot': list(core.sample_identity(request)), 'interim_index': count})
                    assigned[key] = count + 1
        if selected:
            selection_path = selection_dir/f'{index:02d}_{directory.name}.json'
            core.write_json(selection_path, selected)
            jobs.append((directory, OUT/'manifest_order_sources'/f'{index:02d}_{directory.name}', selection_path))
    core.require(len(assigned) == 240 and set(assigned.values()) == {DRAWS}, 'Incomplete prospective interim selection')
    def run(job):
        directory, target, selection_path = job
        result = subprocess.run([sys.executable, __file__, '--snapshot-run', str(directory), '--target', str(target),
                                 '--selection', str(selection_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            raise RuntimeError(str(directory) + '\n' + result.stderr)
        return core.read_json(target/'authenticated.json')
    with ThreadPoolExecutor(max_workers=4) as executor:
        snapshots = list(executor.map(run, jobs))
    results = []
    for snap in snapshots:
        pools = {}
        for p in snap['pools']:
            for grading in ('strict', 'normalization'):
                p[grading] = {int(k): v for k, v in p[grading].items()}
            pools[core.identity(p)] = p
        results.append((snap['source'], pools, {tuple(p) for p in snap['providers']}))
    sources, pools = final.merge_authenticated(results)
    core.require(len(pools) == 240 and set(pools) == set(support), 'Interim prompt/support inventory mismatch')
    for key, pool in pools.items():
        core.require(pool['row_sha256'] == support[key]['row_sha256'], 'Support row mismatch')
        for grading in ('strict', 'normalization'):
            core.require(set(pool[grading]) == set(range(DRAWS)), 'Interim requires all 128 prospectively selected request slots')
    analyses = {}
    for grading in ('strict', 'normalization'):
        prompts = [{**p, 'keys': [p[grading][i] for i in range(DRAWS)], 'support': support[key]} for key,p in sorted(pools.items())]
        analyses[grading] = [core.summarize_cell([p for p in prompts if (p['level'], p['domain']) == (level, domain)], level, domain)
                             for level in final.LEVELS for domain in final.DOMAINS]
    report = {'schema': 'gpt56-all-levels-interim-v1', 'status': 'complete', 'model': 'gpt-5.6-sol',
              'served_model': 'gpt-5.6-sol', 'served_snapshot_header': 'gpt-5.6-sol-2026-07-09',
              'grading': 'normalized_secondary', 'wording': 'original', 'prompt_arm': 'original',
              'levels': [1,2,3], 'cells': analyses['normalization'], 'strict_cells': analyses['strict'],
              'prompts': 240, 'responses': 30720, 'sources': sources, 'analyzer': core.binding(__file__),
              'new_support': core.binding(BASE/'support_reference.json'), 'support_certificate': core.binding(certificate),
              'protocol': {'reasoning': 'medium', 'max_output_tokens': 8192, 'temperature_and_top_p': 'omitted',
                           'bootstrap': {'replicates': core.REPLICATES, 'seed': core.SEED, 'unit': 'whole prompt',
                                         'strata': 'domain and level', 'pointwise': True},
                           'selection': 'All sixteen fixed problems in each of fifteen cells; first 128 planned HTTP-manifest requests per problem across chronological collection stages, selected independently of outcomes and completion order. Original global slots and the within-interim ordering are retained in bound selection files.',
                           'extension': 'Explicit interim view of 128 prospectively ordered requests per problem while the fixed 512-draw follow-up continues.'}}
    core.require(sum(s['responses'] for s in sources) == 30720, 'Incomplete interim response total')
    core.write_json(OUT/'analysis.json', report)
    print(json.dumps({'status': 'complete', 'responses':30720, 'prompts':240, 'output':str(OUT/'analysis.json')}))

if __name__ == '__main__':
    main()
