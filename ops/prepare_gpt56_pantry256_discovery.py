#!/usr/bin/env python3
"""Prepare an additive, initially unlaunched Pantry 256-draw follow-up.

This stage adds draw indices 128-255 on the same 32 original-wording prompts.
Collection is a separate decision after assessing the complete 128-draw tail.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PRIOR_BASE = ROOT / 'artifacts/modebench_discovery_five_domains_sol_20260912'
BASE = PRIOR_BASE / 'pantry256_prospective'
HELPER_PATH = ROOT / 'ops/prepare_gpt56_five_domain_discovery.py'


def load_helper():
    import hashlib
    digest = hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()
    expected = json.loads((PRIOR_BASE / 'manifest.json').read_text())['artifact_sha256']['prepare_gpt56_five_domain_discovery.py']
    if digest != expected:
        raise ValueError('Original collection helper differs from its frozen source')
    spec = importlib.util.spec_from_file_location('_sol_five_domain_collection', HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def prepare(base=BASE):
    h = load_helper()
    base = Path(base).resolve()
    if (base / 'manifest.json').exists():
        return h.authenticate_base(base)
    h.require(not base.exists() or not any(base.iterdir()), 'Use a fresh prospective directory')
    source_base = h.authenticate_base(PRIOR_BASE)
    source = Path(source_base['collection_runs']['pantry_extension']['run_dir'])
    source_manifest = json.loads((source / 'manifest.json').read_text())
    chosen = h.rows(source / 'rows.jsonl')
    requests_by_prompt = {h.identity(r): r for r in h.rows(source / 'requests.jsonl') if r['sample_index'] == 64}
    h.require(len(chosen) == 32 and Counter((r['level'], r['domain']) for r in chosen)
              == Counter({(2, 'pantry_plan'): 16, (3, 'pantry_plan'): 16}), 'Unexpected Pantry cohort')
    h.require(source_manifest['sample_index_start'] == 64 and source_manifest['combined_draw_budget'] == 128,
              'Expected the existing 64-to-128 Pantry extension')
    base.mkdir(parents=True)
    h.write(base / 'selection.json', {'schema': 'sol-pantry256-fixed-selection-v1',
            'rule': 'All same 32 Pantry prompts from the original n64 cohort; no outcome-based prompt selection.',
            'source_rows': h.binding(source / 'rows.jsonl'),
            'original_rows': h.binding(h.PRIOR / 'rows.jsonl'), 'selected_count': 32})
    h.write(base / 'protocol.json', {'schema': 'sol-pantry256-prospective-v1',
            'prepared_at_utc': h.now(), 'initial_state': 'prepared_not_launched',
            'authorization': 'User permits larger GPT sampling budgets where needed; this preparation makes no model calls.',
            'activation': 'Assess the complete 128-draw Pantry tail before deciding whether to collect this stage.',
            'condition': 'Fixed 128 additional responses per unchanged original-wording Pantry prompt.',
            'levels': [2, 3], 'prompts_per_cell': 16, 'prompt_count': 32,
            'draw_indices': [128, 255], 'new_draws_per_prompt': 128, 'total_draw_budget': 256,
            'planned_new_requests': 4096, 'planned_workers': 16,
            'stopping': 'If activated, complete all frozen draws regardless of their outcomes; retain all failures and retries.',
            'analysis': 'Keep prior 0-63 and 64-127 pools intact; combine with new 128-255 only in a source-bound analysis.',
            'tail_reporting': 'Report actual 64-to-128 and 128-to-256 gains; do not extrapolate an asymptotic plateau.',
            'grading': 'Same frozen strict verifier and separately labeled frozen normalization sensitivity.',
            'timing': 'This is a successive deployment follow-up, not a simultaneously sampled pool.'})
    out = base / 'hosted/gpt56sol/pantry_extension256'
    out.mkdir(parents=True)
    shutil.copytree(source / 'code', out / 'code')
    for filename in ('datasets.json', 'model_profile.json', 'rows.jsonl'):
        shutil.copyfile(source / filename, out / filename)
    requests = []
    for row in chosen:
        reference = requests_by_prompt[h.identity(row)]
        for draw in range(128, 256):
            sid = f"L{row['level']}_{row['domain']}_{row['row_index']:03d}_{draw}"
            item = dict(reference)
            item.update(sample_index=draw, sample_id=sid, group_id=sid,
                        reference_sample_id=reference['sample_id'],
                        experiment_condition='sol_pantry256_followup_20260912_v1')
            requests.append(item)
    import hashlib
    requests.sort(key=lambda r: hashlib.sha256(('sol-pantry256-20260912:' + r['sample_id']).encode()).hexdigest())
    h.require(len(requests) == 4096 and len({r['sample_id'] for r in requests}) == 4096, 'Incorrect extension inventory')
    h.require(set(Counter(h.identity(r) for r in requests).values()) == {128}, 'Incomplete prompt budget')
    for item in requests:
        reference = requests_by_prompt[h.identity(item)]
        h.require(item['request'] == reference['request']
                  and item['request_sha256'] == h.object_digest(item['request']), 'Modified source payload')
    groups = [{'group_id': r['group_id'], 'request': r['request'], 'request_sha256': r['request_sha256'],
               'sample_ids': [r['sample_id']], 'sample_count': 1, 'protocol': 'responses'} for r in requests]
    h.write_rows(out / 'requests.jsonl', requests)
    h.write_rows(out / 'http_requests.jsonl', groups)
    manifest = dict(source_manifest)
    manifest.update(prepared_at_utc=h.now(), experiment_condition='sol_pantry256_followup_20260912_v1',
                    cohort='pantry_extension256', sample_count=128, sample_index_start=128,
                    combined_draw_budget=256, request_count=4096, http_request_count=4096,
                    reference_run=str(source), reference_manifest_sha256=h.digest(source / 'manifest.json'),
                    source_selection=h.binding(base / 'selection.json'), protocol_binding=h.binding(base / 'protocol.json'),
                    notes=['Prepared without model calls; activate only after complete n128 tail assessment.',
                           '128 fresh responses on each same Pantry prompt, draw indices128-255; no prior sample duplication.'])
    manifest['artifact_sha256'] = {n: h.digest(out / n) for n in ('datasets.json', 'model_profile.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl')}
    h.write(out / 'manifest.json', manifest)
    h.authenticate(out, manifest)
    shutil.copyfile(__file__, base / 'prepare_gpt56_pantry256_discovery.py')
    root_manifest = {'schema': 'sol-pantry256-prospective-v1', 'frozen_at_utc': h.now(),
                     'initial_state': 'prepared_not_launched', 'model_calls_at_preparation': 0,
                     'source_n128_extension': h.binding(source / 'manifest.json'),
                     'collection_helper': h.binding(HELPER_PATH),
                     'collection_runs': {'pantry_extension': {'run_dir': str(out), 'manifest': h.binding(out / 'manifest.json'),
                                                             'new_samples': 4096, 'sample_index_start': 128}},
                     'artifact_sha256': {n: h.digest(base / n) for n in ('selection.json', 'protocol.json', 'prepare_gpt56_pantry256_discovery.py')},
                     'total_new_requests': 4096, 'credentials_recorded': False}
    h.write(base / 'manifest.json', root_manifest)
    return h.authenticate_base(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'status', 'preflight', 'full'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    h = load_helper()
    h.require(1 <= args.workers <= 64, 'Workers must be 1-64')
    if args.command == 'prepare':
        result = prepare(args.base)
        print(json.dumps({'prepared': result['collection_runs'], 'model_calls': 0}))
    elif args.command == 'status':
        manifest = h.authenticate_base(args.base)
        out = Path(manifest['collection_runs']['pantry_extension']['run_dir'])
        path = out / 'status.json'
        print(path.read_text() if path.exists() else json.dumps({'state': 'prepared_not_launched', 'planned_new_requests': 4096}))
    else:
        source = PRIOR_BASE / 'hosted/gpt56sol/pantry_extension'
        state = json.loads((source / 'status.json').read_text())
        h.require(state.get('complete') is True and state.get('completed_samples') == 2048,
                  'Complete the previous Pantry extension before this separate collection stage')
        h.collect(args.base, 'pantry_extension', args.command, args.credential_file, args.workers)


if __name__ == '__main__':
    main()
