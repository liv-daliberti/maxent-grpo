#!/usr/bin/env python3
"""Freeze a resumable additive Sol discovery stage without changing prior pools."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'ops/prepare_gpt56_five_domain_discovery.py'
ORIGINAL_FREEZE = ROOT / 'artifacts/modebench_discovery_five_domains_sol_20260912'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')


def load_helper():
    expected = json.loads((ORIGINAL_FREEZE / 'manifest.json').read_text())['artifact_sha256']['prepare_gpt56_five_domain_discovery.py']
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != expected:
        raise ValueError('Collection helper differs from original frozen source')
    spec = importlib.util.spec_from_file_location('_sol_discovery_stage_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(base, source, domains, start, stop, reason, workers=16):
    h = load_helper()
    base, source = Path(base).resolve(), Path(source).resolve()
    if (base / 'manifest.json').exists():
        prior = h.authenticate_base(base)
        protocol = json.loads((base / 'protocol.json').read_text())
        h.require(protocol['source_run'] == str(source) and protocol['domains'] == sorted(set(domains))
                  and protocol['draw_start_inclusive'] == start and protocol['draw_stop_exclusive'] == stop,
                  'Existing stage has different requested inputs')
        return prior
    h.require(not base.exists() or not any(base.iterdir()), 'Use a fresh output directory')
    h.require(start >= 64 and stop > start and (start & (start - 1)) == 0
              and (stop & (stop - 1)) == 0, 'Use increasing power-of-two global budgets from64')
    h.require(bool(reason.strip()) and 1 <= workers <= 64, 'Record the reason and valid planned concurrency')
    domains = sorted(set(domains))
    h.require(domains and all(d in DOMAINS for d in domains), 'Select named ModeBench domains')
    old = json.loads((source / 'manifest.json').read_text())
    h.authenticate(source, old)
    end_of_source = old.get('combined_draw_budget', old.get('sample_index_start', 0) + old['sample_count'])
    h.require(start == end_of_source, 'The additive interval must begin at the source combined draw budget')
    h.require(old['model'] == 'gpt-5.6-sol' and old['protocol'] == 'responses'
              and old['model_profile']['request_parameters'] == {'max_output_tokens': 8192, 'reasoning': {'effort': 'medium'}, 'store': False},
              'Preserve the original native Sol medium-reasoning controls')
    chosen = sorted([r for r in h.rows(source / 'rows.jsonl') if r['domain'] in domains], key=h.identity)
    cells = Counter((r['level'], r['domain']) for r in chosen)
    h.require(cells == Counter({(level, domain): 16 for level in (2, 3) for domain in domains}),
              'Keep all16 source prompts in every selected L2/L3 domain cell')
    templates = {}
    for item in h.rows(source / 'requests.jsonl'):
        key = h.identity(item)
        if item['domain'] in domains and (key not in templates or item['sample_index'] < templates[key]['sample_index']):
            templates[key] = item
    h.require(set(templates) == {h.identity(r) for r in chosen}, 'Source requests do not match selected prompts')
    base.mkdir(parents=True)
    h.write(base / 'selection.json', {'schema': 'sol-additive-discovery-selection-v1',
            'rule': 'All unchanged L2/L3 source prompts for the named domains; no within-domain outcome selection.',
            'source_rows': h.binding(source / 'rows.jsonl'), 'domains': domains, 'selected_count': len(chosen)})
    h.write(base / 'protocol.json', {'schema': 'sol-additive-discovery-stage-v1',
            'prepared_at_utc': h.now(), 'initial_state': 'prepared_not_launched',
            'source_run': str(source), 'source_manifest': h.binding(source / 'manifest.json'),
            'reason_for_stage': reason, 'domains': domains, 'levels': [2, 3], 'prompts_per_cell': 16,
            'wording': 'original', 'draw_start_inclusive': start, 'draw_stop_exclusive': stop,
            'new_draws_per_prompt': stop - start, 'combined_draw_budget': stop,
            'planned_new_requests': len(chosen) * (stop - start), 'planned_workers': workers,
            'activation': 'Preparation makes no model calls; collection is separately activated after assessment of the completed previous budget.',
            'stopping': 'Once activated, complete every frozen slot regardless of correctness or mode discovery; retain transport failures separately.',
            'analysis': 'Combine disjoint global sample indices in a source-bound analysis; preserve prior pools and stage timing.',
            'interpretation': 'Report observed late gains; no extrapolated plateau or claim that no rare unseen modes exist.'})
    out = base / 'hosted/gpt56sol/extension'
    out.mkdir(parents=True)
    shutil.copytree(source / 'code', out / 'code')
    for name in ('datasets.json', 'model_profile.json'):
        shutil.copyfile(source / name, out / name)
    h.write_rows(out / 'rows.jsonl', chosen)
    requests = []
    for row in chosen:
        reference = templates[h.identity(row)]
        h.require(reference['row_sha256'] == h.object_digest(row), 'Changed source row')
        for draw in range(start, stop):
            sid = f"L{row['level']}_{row['domain']}_{row['row_index']:03d}_{draw}"
            item = dict(reference)
            item.update(sample_index=draw, sample_id=sid, group_id=sid,
                        reference_sample_id=reference['sample_id'], experiment_condition='sol_additive_discovery_stage_v1')
            h.require(item['request_sha256'] == h.object_digest(item['request']), 'Changed payload hash')
            requests.append(item)
    requests.sort(key=lambda r: hashlib.sha256((f'sol-discovery-stage-{start}-{stop}:' + r['sample_id']).encode()).hexdigest())
    expected_count = len(chosen) * (stop - start)
    h.require(len(requests) == len({r['sample_id'] for r in requests}) == expected_count, 'Invalid fresh slot count')
    h.require(set(Counter(h.identity(r) for r in requests).values()) == {stop - start}, 'Unequal prompt budgets')
    groups = [{'group_id': r['group_id'], 'request': r['request'], 'request_sha256': r['request_sha256'],
               'sample_ids': [r['sample_id']], 'sample_count': 1, 'protocol': 'responses'} for r in requests]
    h.write_rows(out / 'requests.jsonl', requests)
    h.write_rows(out / 'http_requests.jsonl', groups)
    manifest = dict(old)
    manifest.update(prepared_at_utc=h.now(), experiment_condition='sol_additive_discovery_stage_v1',
                    cohort='extension', sample_count=stop - start, sample_index_start=start,
                    combined_draw_budget=stop, prompt_count=len(chosen), request_count=expected_count,
                    http_request_count=expected_count, reference_run=str(source),
                    reference_manifest_sha256=h.digest(source / 'manifest.json'),
                    source_selection=h.binding(base / 'selection.json'), protocol_binding=h.binding(base / 'protocol.json'),
                    notes=['Fresh additive global sample indices; exact native source requests and frozen grading preserved.',
                           'Preparation makes no calls; a full prior-budget assessment precedes activation.'])
    manifest['artifact_sha256'] = {n: h.digest(out / n) for n in ('datasets.json', 'model_profile.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl')}
    h.write(out / 'manifest.json', manifest)
    h.authenticate(out, manifest)
    shutil.copyfile(__file__, base / 'prepare_gpt56_discovery_extension.py')
    result = {'schema': 'sol-additive-discovery-stage-v1', 'frozen_at_utc': h.now(),
              'initial_state': 'prepared_not_launched', 'model_calls_at_preparation': 0,
              'collection_helper': h.binding(HELPER), 'source_run': str(source),
              'collection_runs': {'extension': {'run_dir': str(out), 'manifest': h.binding(out / 'manifest.json'),
                                               'new_samples': expected_count, 'sample_index_start': start}},
              'artifact_sha256': {n: h.digest(base / n) for n in ('selection.json', 'protocol.json', 'prepare_gpt56_discovery_extension.py')},
              'total_new_requests': expected_count, 'credentials_recorded': False}
    h.write(base / 'manifest.json', result)
    return h.authenticate_base(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'status', 'preflight', 'full'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--domain', action='append', choices=DOMAINS, dest='domains')
    parser.add_argument('--start', type=int)
    parser.add_argument('--stop', type=int, help='Exclusive global draw index; equal to resulting total budget.')
    parser.add_argument('--reason')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--credential-file', type=Path)
    args = parser.parse_args()
    h = load_helper()
    if args.command == 'prepare':
        if args.source_run is None or not args.domains or args.start is None or args.stop is None or not args.reason:
            parser.error('prepare requires --source-run, --domain, --start, --stop, and --reason')
        result = prepare(args.output, args.source_run, args.domains, args.start, args.stop, args.reason, args.workers)
        print(json.dumps({'prepared': result['collection_runs'], 'model_calls': 0}))
        return
    manifest = h.authenticate_base(args.output)
    out = Path(manifest['collection_runs']['extension']['run_dir'])
    if args.command == 'status':
        path = out / 'status.json'
        print(path.read_text() if path.exists() else json.dumps({'state': 'prepared_not_launched'}))
        return
    source = Path(manifest['source_run'])
    source_manifest = json.loads((source / 'manifest.json').read_text())
    state = json.loads((source / 'status.json').read_text())
    h.require(state.get('complete') is True and state.get('completed_samples') == source_manifest['request_count'],
              'Previous source pool must be complete before the next collection stage')
    h.require(h.digest(__file__) == manifest['artifact_sha256']['prepare_gpt56_discovery_extension.py'],
              'Run the exact frozen stage preparer/launcher source')
    h.collect(args.output, 'extension', args.command, args.credential_file, args.workers)


if __name__ == '__main__':
    main()
