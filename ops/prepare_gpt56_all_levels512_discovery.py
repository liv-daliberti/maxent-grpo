#!/usr/bin/env python3
"""Freeze and collect the user's fixed all-level, all-domain Sol n512 design."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_all_levels_sol512_20260912'
PRIOR_BASE = ROOT / 'artifacts/modebench_discovery_five_domains_sol_20260912'
SOURCE = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
TEMPLATE = ROOT / 'artifacts/modebench_discovery_curves_20260911/hosted/gpt56sol/original'
HELPER = ROOT / 'ops/prepare_gpt56_five_domain_discovery.py'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LEVELS = (1, 2, 3)
OLD_BUDGETS = {'graph_coloring': 128, 'countdown': 128, 'python_factors': 64, 'mathir': 64, 'pantry_plan': 256}
PARAMETERS = {'max_output_tokens': 8192, 'reasoning': {'effort': 'medium'}, 'store': False}
SNAPSHOT = 'gpt-5.6-sol-2026-07-09'
CONDITION = 'sol_all_three_levels_five_domains_512_v1'


def load_helper():
    expected = json.loads((PRIOR_BASE / 'manifest.json').read_text())['artifact_sha256']['prepare_gpt56_five_domain_discovery.py']
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != expected:
        raise ValueError('Prior helper differs from its frozen source')
    spec = importlib.util.spec_from_file_location('_all_levels512_frozen_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(row):
    return row['level'], row['domain'], row['row_index']


def selection_hash(row):
    return hashlib.sha256(f"20260911\0{row['level']}\0{row['domain']}\0{row['row_index']}".encode()).hexdigest()


def selected_cell(source_rows, level, domain):
    candidates = [r for r in source_rows if (r['level'], r['domain']) == (level, domain)]
    if len(candidates) != 128 or {r['row_index'] for r in candidates} != set(range(128)):
        raise ValueError('Expected all128 distinct original prompt indices in each cell')
    ranked = sorted(candidates, key=lambda r: (selection_hash(r), identity(r)))
    return ranked[:16], ranked


def draw_start(level, domain):
    return 0 if level == 1 else OLD_BUDGETS[domain]


def expand(chosen, templates, start, stop=512):
    h = load_helper()
    requests = []
    for row in chosen:
        ref = templates[identity(row)]
        request = ref['request']
        h.require(ref['row_sha256'] == h.object_digest(row), 'Source row identity changed')
        h.require(ref['request_sha256'] == h.object_digest(request), 'Source request digest changed')
        h.require(request['model'] == 'gpt-5.6-sol'
                  and {k: v for k, v in request.items() if k not in ('input', 'model')} == PARAMETERS,
                  'Source generation controls differ from native Sol medium profile')
        h.require(len(request['input']) == 2 and [m['role'] for m in request['input']] == ['system', 'user'],
                  'Source must preserve the exact original system/user prompt')
        for draw in range(start, stop):
            sid = f"L{row['level']}_{row['domain']}_{row['row_index']:03d}_{draw}"
            requests.append({'level': row['level'], 'domain': row['domain'], 'row_index': row['row_index'],
                             'sample_index': draw, 'sample_id': sid, 'group_id': sid, 'choice_index': 0,
                             'row_sha256': ref['row_sha256'], 'request': request,
                             'request_sha256': ref['request_sha256'], 'reference_sample_id': ref['sample_id'],
                             'fresh_response_cohort': True, 'experiment_condition': CONDITION, 'prompt_arm': 'original'})
    requests.sort(key=lambda r: hashlib.sha256(('sol-all-levels512-order-v1:' + r['sample_id']).encode()).hexdigest())
    h.require(len(requests) == len(chosen) * (stop - start)
              and len({r['sample_id'] for r in requests}) == len(requests), 'Repeated or missing new slot')
    return requests


def cohort_order(work=None):
    names = [f'L{level}_{domain}' for level in LEVELS for domain in DOMAINS]
    work = work or {name: 16 * (512 - draw_start(int(name[1]), name[3:])) for name in names}
    ranked = sorted(names, key=lambda name: (-work[name], name))
    first = [next(name for name in ranked if name.startswith(f'L{level}_')) for level in LEVELS]
    first.sort(key=lambda name: (-work[name], name))
    return first + [name for name in ranked if name not in first]


def estimate_work(selected, source_samples):
    keys = {identity(row) for row in selected}
    observations = {}
    for sample in source_samples:
        if identity(sample) not in keys:
            continue
        name = f"L{sample['level']}_{sample['domain']}"
        item = observations.setdefault(name, {'latency': [], 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': []})
        item['latency'].append(sample['latency_seconds'])
        item['total_tokens'].append(sample['usage']['total_tokens'])
        for token in ('input_tokens', 'output_tokens'):
            item[token] += sample['usage'][token]
    result = {}
    for name, obs in observations.items():
        if len(obs['latency']) != 128:
            raise ValueError('Expected128 prior latency observations for every selected cell')
        level, domain = int(name[1]), name[3:]
        count = 16 * (512 - draw_start(level, domain))
        mean_latency = sum(obs['latency']) / len(obs['latency'])
        result[name] = {'new_requests': count, 'prior_latency_observations': 128,
                        'mean_prior_latency_seconds': mean_latency,
                        'estimated_serial_service_seconds': count * mean_latency,
                        'estimated_input_tokens': count * obs['input_tokens'] / 128,
                        'estimated_output_tokens': count * obs['output_tokens'] / 128,
                        'mean_prior_total_tokens': sum(obs['total_tokens']) / 128,
                        'prior_total_tokens_p95': sorted(obs['total_tokens'])[121],
                        'prior_total_tokens_max': max(obs['total_tokens'])}
    if len(result) != 15:
        raise ValueError('Missing execution latency cell')
    return result


def prepare(base=BASE):
    h = load_helper()
    base = Path(base).resolve()
    if (base / 'manifest.json').exists():
        return authenticate_base(base)
    h.require(not base.exists() or not any(base.iterdir()), 'Use a fresh n512 output directory')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    template_manifest = json.loads((TEMPLATE / 'manifest.json').read_text())
    h.authenticate(SOURCE, source_manifest)
    h.authenticate(TEMPLATE, template_manifest)
    profile = json.loads((TEMPLATE / 'model_profile.json').read_text())
    h.require(profile['model'] == 'gpt-5.6-sol' and profile['protocol'] == 'responses'
              and profile['request_parameters'] == PARAMETERS, 'Unexpected source model profile')
    all_rows = h.rows(SOURCE / 'rows.jsonl')
    templates = {identity(r): r for r in h.rows(SOURCE / 'requests.jsonl') if r['sample_index'] == 0}
    report_path = PRIOR_BASE / 'analysis/analysis.json'
    prior_report = json.loads(report_path.read_text())
    h.require(prior_report['status'] == 'complete' and prior_report['responses'] == 20480
              and prior_report['prompts'] == 160, 'Expected the complete existing five-domain report')
    old_slots, old_rows, baseline_runs = set(), {}, []
    for entry in prior_report['sources']:
        out = Path(entry['directory'])
        h.require(h.digest(out / 'manifest.json') == entry['manifest']['sha256']
                  and h.digest(out / 'samples.jsonl') == entry['samples']['sha256'], 'Changed existing completed pool')
        old_manifest = json.loads((out / 'manifest.json').read_text())
        h.authenticate(out, old_manifest)
        h.require(old_manifest['model_profile'] == profile, 'Existing pool has different native controls')
        for row in h.rows(out / 'rows.jsonl'):
            key = identity(row)
            h.require(key not in old_rows or old_rows[key] == row, 'Existing problem revisions differ')
            old_rows[key] = row
        for item in h.rows(out / 'requests.jsonl'):
            slot = (*identity(item), item['sample_index'])
            h.require(slot not in old_slots, 'Existing pools duplicate a response slot')
            old_slots.add(slot)
            h.require(item['request'] == templates[identity(item)]['request'], 'Existing wording differs from the held-out native source')
        baseline_runs.append({'directory': str(out), 'manifest': h.binding(out / 'manifest.json'),
                              'samples': h.binding(out / 'samples.jsonl'), 'responses': entry['responses']})
    h.require(len(old_slots) == 20480 and len(old_rows) == 160, 'Existing source inventory differs')
    chosen, cells, ledger = [], {}, []
    for level in LEVELS:
        for domain in DOMAINS:
            selected, ranked = selected_cell(all_rows, level, domain)
            cohort = f'L{level}_{domain}'
            cells[cohort] = sorted(selected, key=identity)
            chosen.extend(selected)
            for rank, row in enumerate(ranked, 1):
                ledger.append({'level': level, 'domain': domain, 'row_index': row['row_index'],
                               'selection_sha256': selection_hash(row), 'rank_in_cell': rank, 'selected': rank <= 16,
                               'row_sha256': h.object_digest(row)})
            if level > 1:
                for row in selected:
                    h.require(old_rows.get(identity(row)) == row, 'New selection would replace an existing L2/L3 problem')
                    existing = {slot[-1] for slot in old_slots if slot[:3] == identity(row)}
                    h.require(existing == set(range(OLD_BUDGETS[domain])), 'Existing response prefix has a gap')
    chosen.sort(key=identity)
    h.require(len(chosen) == 240, 'Expected all240 prompts')
    base.mkdir(parents=True)
    h.write_rows(base / 'rows.jsonl', chosen)
    h.write_rows(base / 'level1_rows.jsonl', [r for r in chosen if r['level'] == 1])
    h.write(base / 'selection.json', {'schema': 'sol-all-levels512-selection-v1',
            'source_rows': h.binding(SOURCE / 'rows.jsonl'), 'seed': 20260911,
            'rule': 'First16 per level/domain by SHA256(seed NUL level NUL domain NUL row_index), ties by identity.',
            'selection_uses_outcomes': False, 'candidates': ledger, 'selected_prompts': 240,
            'unchanged_existing_L2_L3_prompts': 160, 'new_L1_prompts': 80})
    estimates = estimate_work(chosen, h.rows(SOURCE / 'samples.jsonl'))
    order = cohort_order({name: row['estimated_serial_service_seconds'] for name, row in estimates.items()})
    h.write(base / 'execution_work_estimates.json', {'source_samples': h.binding(SOURCE / 'samples.jsonl'),
            'uses_answer_outcomes': False, 'purpose': 'Request latency and native usage metadata estimate execution work only.',
            'cells': estimates, 'cohort_order': order,
            'conservative_projected_TPM_at4000RPM': 4000 * max(row['mean_prior_total_tokens'] for row in estimates.values()),
            'soft_global_TPM_target': 4000000,
            'ideal_service_hours_at128_concurrency': sum(row['estimated_serial_service_seconds'] for row in estimates.values()) / 128 / 3600,
            'ideal_service_hours_at256_concurrency': sum(row['estimated_serial_service_seconds'] for row in estimates.values()) / 256 / 3600})
    protocol = {'schema': CONDITION, 'prepared_at_utc': h.now(),
                'authorization': 'User explicitly requests512 draws per problem for all three levels and all five domains.',
                'model': 'gpt-5.6-sol', 'served_snapshot_required': SNAPSHOT,
                'levels': list(LEVELS), 'domains': list(DOMAINS), 'prompts_per_cell': 16, 'prompts': 240,
                'final_draws_per_prompt': 512, 'final_total_responses': 122880,
                'retained_prior_responses': 20480, 'new_responses': 102400,
                'L1_draw_indices': [0, 511], 'L2_L3_prior_budgets': OLD_BUDGETS,
                'generation': profile, 'wording': 'exact original native system/user messages',
                'level1_interfaces': {d: source_manifest['profiles'][f'1/{d}'] for d in DOMAINS},
                'selection': 'Same outcome-independent SHA selection in all15 cells; every existing L2/L3 problem retained.',
                'reuse': 'Only the20,480 authenticated existing L2/L3 responses are reused. Original n8 responses, including L1, are excluded.',
                'analysis': 'Exact rarefaction at1,2,4,8,16,32,64,128,256,512 after every complete512-response prompt pool exists.',
                'stopping': 'Fixed512 budget for every problem, independent of correctness or observed mode discovery.',
                'grading': 'Same frozen strict verifier and separately labeled frozen normalization; L1 Pantry retains native six-bit support-mask decoding.',
                'uncertainty': '20000 whole-problem bootstrap replicates stratified by level/domain, seed20260911; pointwise intervals.',
                'execution': {'preflight': 'One authenticated sample per cohort, retained in production; sequential by default.',
                              'workers_per_cohort': 32, 'max_concurrent_cohorts': 4, 'max_request_concurrency': 128,
                              'cohort_order': order, 'order_basis': 'Longest estimated remaining request service first, with the longest cohort from each level among the first three; no outcome selection.',
                              'requests_per_minute_per_cohort': 500,
                              'ramp': {'enabled': True, 'initial_concurrent_cohorts': 4, 'maximum_concurrent_cohorts': 8,
                                       'maximum_request_concurrency': 256, 'minimum_completed_per_initial_cohort': 201,
                                       'maximum_transport_error_rate': 0.01, 'require_zero_HTTP429': True, 'require_zero_failed_groups': True,
                                       'maximum_projected_TPM': 4000000,
                                       'TPM_projection': '4000RPM times the largest prior cell or live initial-cohort mean native total tokens.',
                                       'amendment_before_extra_launches': True,
                                       'higher_cap': 'No cap above256 in this run; any future higher cap would require new evidence and a separate amendment.'}},
                'limitations': ['Successive collection stages differ in time; verify the same native snapshot on every accepted receipt.',
                                'Finite512-draw curves do not prove asymptotic saturation or absence of rare unseen modes.',
                                'No provider RNG-independence claim; incorrect/refused/truncated completions are retained.']}
    h.write(base / 'protocol.json', protocol)
    h.write(base / 'retained_prior_runs.json', {'analysis': h.binding(report_path),
            'collection_audit': h.binding(PRIOR_BASE / 'collection_completion_audit_graph128_pantry256.json'),
            'runs': baseline_runs, 'responses': 20480})
    registry, added = {}, set()
    for cohort in order:
        selected = cells[cohort]
        level, domain = selected[0]['level'], selected[0]['domain']
        start = draw_start(level, domain)
        requests = expand(selected, templates, start)
        out = base / 'hosted/gpt56sol' / cohort
        out.mkdir(parents=True)
        shutil.copytree(TEMPLATE / 'code', out / 'code')
        shutil.copyfile(TEMPLATE / 'model_profile.json', out / 'model_profile.json')
        shutil.copyfile(SOURCE / 'datasets.json', out / 'datasets.json')
        h.write_rows(out / 'rows.jsonl', selected)
        h.write_rows(out / 'requests.jsonl', requests)
        groups = [{'group_id': r['group_id'], 'request': r['request'], 'request_sha256': r['request_sha256'],
                   'sample_ids': [r['sample_id']], 'sample_count': 1, 'protocol': 'responses'} for r in requests]
        h.write_rows(out / 'http_requests.jsonl', groups)
        for item in requests:
            slot = (*identity(item), item['sample_index'])
            h.require(slot not in old_slots and slot not in added, 'New responses overlap an existing or planned slot')
            added.add(slot)
        manifest = {k: v for k, v in template_manifest.items() if k not in ('artifact_sha256', 'ablation_manifest_sha256',
                    'notes', 'parent_prompt_ablation_run', 'parent_prompt_ablation_manifest_sha256', 'preparer_sha256')}
        manifest.update(prepared_at_utc=h.now(), experiment_condition=CONDITION, cohort=cohort,
                        sample_count=512 - start, sample_index_start=start, combined_draw_budget=512,
                        prompt_count=16, request_count=len(requests), http_request_count=len(requests),
                        reference_run=str(SOURCE), reference_manifest_sha256=h.digest(SOURCE / 'manifest.json'),
                        source_selection=h.binding(base / 'selection.json'), protocol_binding=h.binding(base / 'protocol.json'),
                        retained_prior_runs=h.binding(base / 'retained_prior_runs.json'),
                        prompt_interface=source_manifest['profiles'][f'{level}/{domain}'],
                        notes=['Fixed additive global draw slots; all returned model outcomes count.',
                               'Original n8 data is not reused; prior L2/L3 pools are joined only in analysis.',
                               'Native prompt bytes, response surface, verifier, and model controls remain unchanged.'])
        manifest['artifact_sha256'] = {name: h.digest(out / name) for name in ('datasets.json', 'model_profile.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl')}
        h.write(out / 'manifest.json', manifest)
        h.authenticate(out, manifest)
        registry[cohort] = {'run_dir': str(out), 'manifest': h.binding(out / 'manifest.json'),
                            'level': level, 'domain': domain, 'new_samples': len(requests), 'sample_index_start': start}
    h.require(len(added) == 102400 and len(old_slots | added) == 122880, 'Incorrect final512 inventory')
    for row in chosen:
        key = identity(row)
        h.require({slot[-1] for slot in old_slots | added if slot[:3] == key} == set(range(512)), 'Final pool is not gap-free through511')
    shutil.copyfile(__file__, base / 'prepare_gpt56_all_levels512_discovery.py')
    test = ROOT / 'tests/test_gpt56_all_levels512_discovery.py'
    if test.exists():
        shutil.copyfile(test, base / test.name)
    artifact_names = ['rows.jsonl', 'level1_rows.jsonl', 'selection.json', 'protocol.json', 'retained_prior_runs.json', 'execution_work_estimates.json', 'prepare_gpt56_all_levels512_discovery.py']
    if test.exists():
        artifact_names.append(test.name)
    result = {'schema': CONDITION, 'frozen_at_utc': h.now(), 'status_at_preparation': 'prepared_not_launched',
              'collection_runs': registry, 'collection_helper': h.binding(HELPER),
              'artifact_sha256': {name: h.digest(base / name) for name in artifact_names},
              'new_requests': 102400, 'retained_responses': 20480, 'final_responses': 122880, 'prompts': 240,
              'model_calls_at_preparation': 0, 'credentials_recorded': False}
    h.write(base / 'manifest.json', result)
    return authenticate_base(base)


def authenticate_base(base, cohort=None):
    h = load_helper()
    base = Path(base).resolve()
    manifest = json.loads((base / 'manifest.json').read_text())
    h.authenticate(base, manifest)
    h.require(h.digest(__file__) == manifest['artifact_sha256']['prepare_gpt56_all_levels512_discovery.py'],
              'Run the unchanged frozen all-level512 collector source')
    entries = manifest['collection_runs']
    if cohort is not None:
        h.require(cohort in entries, 'Unknown n512 cohort')
        entries = {cohort: entries[cohort]}
    for entry in entries.values():
        out = Path(entry['run_dir'])
        h.require(h.digest(out / 'manifest.json') == entry['manifest']['sha256'], 'Changed cohort manifest')
        h.authenticate(out, json.loads((out / 'manifest.json').read_text()))
    return manifest


def audit_records(out):
    """Authenticate once per raw receipt, including the provider snapshot header."""
    h = load_helper()
    adapter = h.load_adapter(out)
    requests = {r['sample_id']: r for r in h.rows(out / 'requests.jsonl')}
    groups = {r['group_id']: r for r in h.rows(out / 'http_requests.jsonl')}
    expected = h.expected_returned_controls()
    records, providers = [], set()
    for path in sorted((out / 'sample_receipts').glob('*.json')):
        record = json.loads(path.read_text())
        sid = record['sample_id']
        h.require(sid in requests and path.name == sid + '.json', 'Unexpected saved response slot')
        item = requests[sid]
        cache = {}
        adapter.validate_completed(out, item, groups[item['group_id']], record, cache)
        receipt = cache[record['raw_receipt']]
        body = receipt['response']
        h.require({key: body.get(key) for key in expected} == expected, 'Served generation controls changed')
        headers = {key.lower(): value for key, value in receipt['headers'].items()}
        h.require(headers.get('x-ms-served-model') == SNAPSHOT, 'Served model snapshot changed')
        provider = tuple(record['provider_sample_identity'])
        h.require(provider not in providers, 'Duplicate native provider identity')
        providers.add(provider)
        records.append(record)
    return records


def collect_one(base, cohort, stage, credential_file=None, workers=32):
    h = load_helper()
    manifest = authenticate_base(base, cohort)
    entry = manifest['collection_runs'][cohort]
    out, expected_count = Path(entry['run_dir']), entry['new_samples']
    with (out / '.all_levels_orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = audit_records(out)
        if len(existing) == expected_count:
            return {'cohort': cohort, 'complete': True, 'authenticated_samples': expected_count, 'new_calls': 0}
        marker = out / 'preflight.json'
        if stage == 'preflight' and marker.exists():
            receipt = json.loads(marker.read_text())
            h.require(receipt['manifest_sha256'] == h.digest(out / 'manifest.json') and existing, 'Invalid existing preflight')
            for item in receipt['evidence']:
                h.require(h.digest(out / 'sample_receipts' / (item['sample_id'] + '.json')) == item['sample_receipt_sha256'], 'Changed preflight evidence')
            return {'cohort': cohort, 'preflight': 'authenticated_existing', 'authenticated_samples': len(existing), 'new_calls': 0}
        if stage == 'full':
            h.require(marker.exists() and existing, 'Complete and inspect the authenticated preflight before production')
            receipt = json.loads(marker.read_text())
            h.require(receipt['manifest_sha256'] == h.digest(out / 'manifest.json'), 'Preflight manifest changed')
            for item in receipt['evidence']:
                h.require(h.digest(out / 'sample_receipts' / (item['sample_id'] + '.json')) == item['sample_receipt_sha256'], 'Changed preflight evidence')
        env = os.environ.copy()
        env['AZURE_OPENAI_API_KEY'] = h.credential(credential_file)
        command = [sys.executable, str(out / 'code/ops/evaluate_native_prompt_ablation.py'), 'run',
                   '--model', 'gpt-5.6-sol', '--output', str(out), '--workers', str(workers),
                   '--request-timeout', '300', '--max-attempts', '8', '--rpm', '500']
        if stage == 'preflight':
            command += ['--max-new', '1']
        with (out / ('collection_' + stage + '.log')).open('a') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        env.pop('AZURE_OPENAI_API_KEY', None)
        h.require(process.returncode == 0, 'Native collection failed; retain evidence and inspect ' + cohort)
        records = audit_records(out)
        h.require(records and (stage != 'full' or len(records) == expected_count), 'Incomplete cohort: ' + cohort)
        if stage == 'preflight' and not marker.exists():
            h.write(marker, {'schema': 'sol-all-levels512-preflight-v1', 'checked_at_utc': h.now(),
                    'manifest_sha256': h.digest(out / 'manifest.json'), 'retained_in_production': True,
                    'served_snapshot': SNAPSHOT, 'served_controls': h.expected_returned_controls(),
                    'evidence': [{'sample_id': r['sample_id'], 'provider_sample_identity': r['provider_sample_identity'],
                                  'sample_receipt_sha256': h.digest(out / 'sample_receipts' / (r['sample_id'] + '.json'))} for r in records]})
        return {'cohort': cohort, 'stage': stage, 'authenticated_samples': len(records),
                'expected_samples': expected_count, 'served_snapshot': SNAPSHOT, 'complete': len(records) == expected_count}


def execution_limits(base, workers, concurrent, amendment=None):
    h = load_helper()
    h.require(1 <= workers <= 64 and 1 <= concurrent <= 15, 'Invalid concurrency')
    if workers == 32 and concurrent <= 4:
        return
    h.require(amendment is not None, 'Nondefault production workers require a prior execution amendment')
    document = json.loads(Path(amendment).read_text())
    h.require(document['manifest_sha256'] == h.digest(Path(base) / 'manifest.json')
              and document['workers_per_cohort'] == workers
              and document['max_concurrent_cohorts'] == concurrent
              and document['max_request_concurrency'] == workers * concurrent
              and document['before_changed_worker_launches'] is True, 'Execution amendment does not authorize this concurrency')
    h.require(workers * concurrent <= 256, 'Do not exceed the planned256-concurrency ramp ceiling')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'status', 'preflight', 'full'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--cohort', help='One named cohort; omit for the full fixed schedule.')
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--max-concurrent-cohorts', type=int, default=4)
    parser.add_argument('--execution-amendment', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.base)
        print(json.dumps({'prepared_cohorts': len(result['collection_runs']), 'new_requests': result['new_requests'],
                          'retained_responses': result['retained_responses'], 'final_responses': result['final_responses'], 'model_calls': 0}))
        return
    manifest = authenticate_base(args.base, args.cohort)
    protocol = json.loads((args.base / 'protocol.json').read_text())
    names = [args.cohort] if args.cohort else protocol['execution']['cohort_order']
    if args.command == 'status':
        for name in names:
            path = Path(manifest['collection_runs'][name]['run_dir']) / 'status.json'
            print(json.dumps({'cohort': name, 'status': json.loads(path.read_text()) if path.exists() else 'not_started'}))
        return
    if args.command == 'preflight':
        for name in names:
            print(json.dumps(collect_one(args.base, name, 'preflight', args.credential_file, 1)), flush=True)
        return
    execution_limits(args.base, args.workers, args.max_concurrent_cohorts, args.execution_amendment)
    # The executor has capacity for the authorized ramp but receives only four
    # initial tasks. Extra tasks are submitted after a written health amendment.
    with (args.base / '.production_scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        capacity = args.max_concurrent_cohorts
        ramp_allowed = not args.cohort and capacity == 4 and args.workers == 32
        maximum = 8 if ramp_allowed else capacity
        initial_names = names[:capacity]
        queued = list(names)
        errors = []
        with ThreadPoolExecutor(max_workers=maximum) as pool:
            pending = {}
            def submit_available():
                while queued and len(pending) < capacity and not errors:
                    name = queued.pop(0)
                    future = pool.submit(collect_one, args.base, name, 'full', args.credential_file, args.workers)
                    pending[future] = name
            submit_available()
            while pending:
                completed, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in completed:
                    name = pending.pop(future)
                    try:
                        print(json.dumps(future.result()), flush=True)
                    except Exception as error:
                        errors.append({'cohort': name, 'error': str(error)})
                        print(json.dumps(errors[-1]), flush=True)
                if ramp_allowed and capacity == 4 and not errors:
                    health = ramp_health(manifest, initial_names, args.base)
                    if health['eligible']:
                        amendment = args.base / 'execution_ramp_to256.json'
                        document = {'schema': 'sol-all-levels512-execution-ramp-v1', 'at_utc': load_helper().now(),
                                    'manifest_sha256': load_helper().digest(args.base / 'manifest.json'),
                                    'workers_per_cohort': 32, 'max_concurrent_cohorts': 8,
                                    'max_request_concurrency': 256, 'before_changed_worker_launches': True,
                                    'health_evidence': health, 'request_payloads_changed': False,
                                    'authorization': 'User fixed512 request and parent-authorized evidence-based128-to256 concurrency ramp.'}
                        if amendment.exists():
                            execution_limits(args.base, 32, 8, amendment)
                        else:
                            load_helper().write(amendment, document)
                        capacity = 8
                        print(json.dumps({'event': 'execution_ramp', 'concurrency': 256, 'amendment': str(amendment)}), flush=True)
                submit_available()
        if errors:
            raise RuntimeError(json.dumps({'errors': errors, 'unlaunched_cohorts': queued}))


def ramp_health(manifest, initial_names, base):
    observations = {}
    for name in initial_names:
        out = Path(manifest['collection_runs'][name]['run_dir'])
        path = out / 'status.json'
        if not path.exists():
            return {'eligible': False, 'reason': 'Initial cohort has no status'}
        status = json.loads(path.read_text())
        error_path = out / 'errors.jsonl'
        errors = [json.loads(line) for line in error_path.read_text().splitlines()] if error_path.exists() else []
        completed = status['completed_samples']
        observations[name] = {'completed_samples': completed, 'failed_groups': status['failed_groups_this_session'],
                              'HTTP429': sum(row.get('http_status') == 429 for row in errors),
                              'transport_error_attempts': len(errors), 'transport_error_rate': len(errors) / max(1, completed),
                              'mean_native_total_tokens': status.get('usage', {}).get('total_tokens', 0) / max(1, completed)}
    estimates = json.loads((Path(base) / 'execution_work_estimates.json').read_text())
    maximum_mean_tokens = max([row['mean_prior_total_tokens'] for row in estimates['cells'].values()]
                             + [row['mean_native_total_tokens'] for row in observations.values()])
    projected_TPM = 4000 * maximum_mean_tokens
    eligible = projected_TPM <= 4000000 and len(observations) == 4 and all(row['completed_samples'] >= 201 and row['failed_groups'] == 0
               and row['HTTP429'] == 0 and row['transport_error_rate'] <= .01 for row in observations.values())
    return {'eligible': eligible, 'initial_cohorts': observations,
            'conservative_projected_TPM_at4000RPM': projected_TPM, 'soft_global_TPM_target': 4000000}


if __name__ == '__main__':
    main()
