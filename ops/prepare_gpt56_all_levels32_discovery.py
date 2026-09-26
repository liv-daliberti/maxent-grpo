#!/usr/bin/env python3
"""Freeze and collect 32 problems per cell at 512 draws, retaining the first 16."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_all_levels_sol32x512_20260913'
PRIOR_BASE = ROOT / 'artifacts/modebench_discovery_all_levels_sol512_20260912'
SOURCE = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
TEMPLATE = ROOT / 'artifacts/modebench_discovery_curves_20260911/hosted/gpt56sol/original'
PRIOR_COLLECTOR = ROOT / 'ops/prepare_gpt56_all_levels512_discovery.py'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LEVELS = (1, 2, 3)
PARAMETERS = {'max_output_tokens': 8192, 'reasoning': {'effort': 'medium'}, 'store': False}
SNAPSHOT = 'gpt-5.6-sol-2026-07-09'
CONDITION = 'sol_all_three_levels_five_domains_32x512_v1'
SCRIPT = 'prepare_gpt56_all_levels32_discovery.py'


@lru_cache(maxsize=1)
def load_prior():
    manifest = json.loads((PRIOR_BASE / 'manifest.json').read_text())
    expected = manifest['artifact_sha256'][PRIOR_COLLECTOR.name]
    if hashlib.sha256(PRIOR_COLLECTOR.read_bytes()).hexdigest() != expected:
        raise ValueError('Prior collector differs from its frozen source')
    spec = importlib.util.spec_from_file_location('_all_levels32_prior_collector', PRIOR_COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_helper():
    return load_prior().load_helper()


def identity(row):
    return row['level'], row['domain'], row['row_index']


def selection_hash(row):
    return hashlib.sha256(f"20260911\0{row['level']}\0{row['domain']}\0{row['row_index']}".encode()).hexdigest()


def selected_cell(source_rows, level, domain):
    _, ranked = load_prior().selected_cell(source_rows, level, domain)
    return ranked[:16], ranked[16:32], ranked


def expand(chosen, templates):
    """Use all 512 fresh slots; original eight-draw samples are never reused."""
    requests = load_prior().expand(chosen, templates, 0, 512)
    for item in requests:
        item['experiment_condition'] = CONDITION
    requests.sort(key=lambda r: hashlib.sha256(('sol-all-levels32x512-order-v1:' + r['sample_id']).encode()).hexdigest())
    return requests


def check_binding(binding):
    h = load_helper()
    path = Path(binding['path'])
    if not path.is_absolute():
        path = ROOT / path
    h.require(h.digest(path) == binding['sha256'], 'Changed bound artifact: ' + str(path))
    return path


def retained_inventory(templates, profile):
    """Join only the completed n512 study, preserving its native receipt audit."""
    h = load_helper()
    load_prior().authenticate_base(PRIOR_BASE)
    audit_path = PRIOR_BASE / 'collection_completion_audit.json'
    report_path = PRIOR_BASE / 'analysis/analysis.json'
    audit = json.loads(audit_path.read_text())
    report = json.loads(report_path.read_text())
    check_binding(audit['collection_manifest'])
    h.require(audit['status'] == report['status'] == 'complete'
              and audit['total_authenticated_responses'] == report['responses'] == 122880
              and audit['unique_prompts'] == report['prompts'] == 240
              and audit['final_draws_per_prompt'] == 512
              and audit['global_provider_samples_unique'] and audit['global_sample_slots_gap_free'],
              'Expected the complete authenticated 240-problem, 512-draw study')
    audited = {entry['directory']: entry for entry in audit['retained_runs']}
    audited.update({entry['run_dir']: entry for entry in audit['runs'].values()})
    h.require(set(audited) == {entry['directory'] for entry in report['sources']}, 'Prior source inventory changed')
    slots, providers, old_rows, runs = set(), set(), {}, []
    for entry in report['sources']:
        out = Path(entry['directory'])
        evidence = audited[str(out)]
        for name in ('manifest', 'samples'):
            h.require(entry[name]['sha256'] == evidence[name]['sha256'], 'Prior analysis and receipt audit disagree')
            h.require(h.digest(out / (name + ('.json' if name == 'manifest' else '.jsonl'))) == entry[name]['sha256'],
                      'Changed completed prior ' + name)
        current = json.loads((out / 'manifest.json').read_text())
        h.authenticate(out, current)
        h.require(current['model_profile'] == profile, 'Retained native generation controls differ')
        for row in h.rows(out / 'rows.jsonl'):
            key = identity(row)
            h.require(key not in old_rows or old_rows[key] == row, 'Changed retained problem')
            h.require(templates[key]['row_sha256'] == h.object_digest(row), 'Retained row differs from source')
            old_rows[key] = row
        requests = {r['sample_id']: r for r in h.rows(out / 'requests.jsonl')}
        samples = h.rows(out / 'samples.jsonl')
        h.require(len(samples) == len(requests) == entry['responses'] == evidence['responses'], 'Incomplete retained run')
        local_ids = set()
        for item in samples:
            sid = item['sample_id']
            h.require(sid in requests and sid not in local_ids, 'Unknown or duplicate retained sample')
            local_ids.add(sid)
            request = requests[sid]
            key = identity(request)
            slot = (*key, request['sample_index'])
            h.require(slot not in slots, 'Retained runs repeat a sample slot')
            h.require(request['request'] == templates[key]['request'], 'Retained prompt wording changed')
            h.require(identity(item) == key and item['sample_index'] == request['sample_index'], 'Retained response slot changed')
            provider = tuple(item['provider_sample_identity'])
            h.require(provider not in providers, 'Duplicate retained provider identity')
            slots.add(slot)
            providers.add(provider)
        runs.append({'directory': str(out), 'manifest': h.binding(out / 'manifest.json'),
                     'samples': h.binding(out / 'samples.jsonl'), 'responses': len(samples)})
    by_problem = defaultdict(set)
    for level, domain, index, draw in slots:
        by_problem[(level, domain, index)].add(draw)
    h.require(len(slots) == 122880 and len(old_rows) == len(by_problem) == 240
              and all(draws == set(range(512)) for draws in by_problem.values()), 'Retained pools are not exactly 512 draws each')
    return old_rows, slots, {'analysis': h.binding(report_path), 'collection_audit': h.binding(audit_path),
                           'collection_manifest': h.binding(PRIOR_BASE / 'manifest.json'),
                           'runs': runs, 'responses': len(slots), 'prompts': len(old_rows)}


def estimate_work(selected, samples):
    keys = {identity(row) for row in selected}
    observations = defaultdict(list)
    for row in samples:
        if identity(row) in keys:
            observations[f"L{row['level']}_{row['domain']}"].append(row)
    result = {}
    for name, records in observations.items():
        load_helper().require(len(records) == 128, 'Expected 128 prior latency observations per new cell')
        latency = sum(row['latency_seconds'] for row in records) / 128
        result[name] = {'new_requests': 8192, 'prior_latency_observations': 128,
                        'mean_prior_latency_seconds': latency, 'estimated_serial_service_seconds': latency * 8192,
                        'mean_prior_total_tokens': sum(row['usage']['total_tokens'] for row in records) / 128,
                        'estimated_input_tokens': sum(row['usage']['input_tokens'] for row in records) * 64,
                        'estimated_output_tokens': sum(row['usage']['output_tokens'] for row in records) * 64}
    load_helper().require(len(result) == 15, 'Missing execution estimate cell')
    return result


def prepare(base=BASE):
    h = load_helper()
    base = Path(base).resolve()
    if (base / 'manifest.json').exists():
        return authenticate_base(base)
    h.require(not base.exists() or not any(base.iterdir()), 'Use a fresh 32x512 output directory')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    template_manifest = json.loads((TEMPLATE / 'manifest.json').read_text())
    h.authenticate(SOURCE, source_manifest)
    h.authenticate(TEMPLATE, template_manifest)
    profile = json.loads((TEMPLATE / 'model_profile.json').read_text())
    h.require(profile['model'] == 'gpt-5.6-sol' and profile['protocol'] == 'responses'
              and profile['request_parameters'] == PARAMETERS, 'Unexpected source generation controls')
    source_rows = h.rows(SOURCE / 'rows.jsonl')
    templates = {identity(r): r for r in h.rows(SOURCE / 'requests.jsonl') if r['sample_index'] == 0}
    old_rows, old_slots, retained = retained_inventory(templates, profile)
    selected, added_rows, cells, ledger = [], [], {}, []
    for level in LEVELS:
        for domain in DOMAINS:
            previous, added, ranked = selected_cell(source_rows, level, domain)
            h.require({identity(r) for r in previous} == {key for key in old_rows if key[:2] == (level, domain)},
                      'The first 16 selected problems must match the entire prior cell')
            h.require(all(old_rows[identity(row)] == row for row in previous), 'A retained problem changed')
            h.require(all(identity(row) not in old_rows for row in added), 'New problems overlap retained problems')
            cells[f'L{level}_{domain}'] = sorted(added, key=identity)
            selected.extend(previous + added)
            added_rows.extend(added)
            for rank, row in enumerate(ranked, 1):
                ledger.append({'level': level, 'domain': domain, 'row_index': row['row_index'],
                               'rank_in_cell': rank, 'selection_sha256': selection_hash(row),
                               'row_sha256': h.object_digest(row), 'selected': rank <= 32,
                               'retained': rank <= 16, 'new': 17 <= rank <= 32})
    selected.sort(key=identity)
    added_rows.sort(key=identity)
    h.require(len(selected) == 480 and len(added_rows) == 240, 'Incorrect 32x512 problem inventory')
    estimates = estimate_work(added_rows, h.rows(SOURCE / 'samples.jsonl'))
    order = load_prior().cohort_order({name: row['estimated_serial_service_seconds'] for name, row in estimates.items()})
    base.mkdir(parents=True)
    for name, rows in [('rows.jsonl', selected), ('new_rows.jsonl', added_rows),
                       ('retained_rows.jsonl', sorted(old_rows.values(), key=identity)),
                       ('level1_rows.jsonl', [row for row in selected if row['level'] == 1])]:
        h.write_rows(base / name, rows)
    h.write(base / 'selection.json', {'schema': 'sol-all-levels32x512-selection-v1', 'seed': 20260911,
            'source_rows': h.binding(SOURCE / 'rows.jsonl'), 'prior_selection': h.binding(PRIOR_BASE / 'selection.json'),
            'rule': 'First 32 per level/domain by SHA256(seed NUL level NUL domain NUL row_index), ties by identity; retain ranks 1-16 and add ranks 17-32.',
            'selection_uses_outcomes': False, 'candidates': ledger,
            'selected_prompts': 480, 'retained_prompts': 240, 'new_prompts': 240})
    h.write(base / 'retained_prior_runs.json', retained)
    h.write(base / 'execution_work_estimates.json', {'source_samples': h.binding(SOURCE / 'samples.jsonl'),
            'uses_answer_outcomes': False, 'purpose': 'Original eight-draw latency and usage metadata estimate execution work only; no responses are reused.',
            'cells': estimates, 'cohort_order': order, 'soft_global_TPM_target': 4000000,
            'conservative_projected_TPM_at4000RPM': 4000 * max(row['mean_prior_total_tokens'] for row in estimates.values()),
            'ideal_service_hours_at128_concurrency': sum(row['estimated_serial_service_seconds'] for row in estimates.values()) / 128 / 3600})
    execution = json.loads((PRIOR_BASE / 'protocol.json').read_text())['execution']
    execution['cohort_order'] = order
    protocol = {'schema': CONDITION, 'prepared_at_utc': h.now(),
                'authorization': 'User revised the extension to 32 problems per cell at 512 draws, prioritizing generalization across problems.',
                'model': 'gpt-5.6-sol', 'served_snapshot_required': SNAPSHOT,
                'levels': list(LEVELS), 'domains': list(DOMAINS), 'prompts_per_cell': 32, 'prompts': 480,
                'retained_prompts_per_cell': 16, 'new_prompts_per_cell': 16,
                'final_draws_per_prompt': 512, 'final_total_responses': 245760,
                'retained_prior_responses': 122880, 'new_responses': 122880, 'new_draw_indices': [0, 511],
                'generation': profile, 'wording': 'exact original native system/user messages',
                'selection': 'Outcome-independent seed20260911 SHA ranking; retain first16 and add ranks17-32 in each of15 cells.',
                'reuse': 'Retain all122880 authenticated responses from the completed240-problem study; collect512 fresh draws on each of240 new problems. Original eight-draw samples are excluded from every new pool.',
                'analysis': 'Exact rarefaction at1,2,4,8,16,32,64,128,256,512 after all480 complete512-response pools exist; average32 problems equally per cell.',
                'stopping': 'Fixed512 fresh draws for each new problem, independent of correctness or observed mode discovery; no replacement or extension to1024.',
                'grading': 'Unchanged frozen strict verifier and separately labeled frozen formatting normalization; retain native interfaces including L1 Pantry six-bit support-mask decoding.',
                'uncertainty': '20000 whole-problem bootstrap replicates stratified by level/domain, seed20260911; pointwise95% intervals.',
                'execution': execution,
                'limitations': ['The additional problems are collected later; every accepted receipt must report the same served snapshot.',
                                'Finite512-draw curves do not establish asymptotic saturation or absence of rare unseen modes.',
                                'No provider RNG-independence claim; incorrect, refused, and truncated completions remain in each pool.']}
    h.write(base / 'protocol.json', protocol)
    registry, new_slots = {}, set()
    for cohort in order:
        rows = cells[cohort]
        level, domain = rows[0]['level'], rows[0]['domain']
        requests = expand(rows, templates)
        out = base / 'hosted/gpt56sol' / cohort
        out.mkdir(parents=True)
        shutil.copytree(TEMPLATE / 'code', out / 'code')
        shutil.copyfile(TEMPLATE / 'model_profile.json', out / 'model_profile.json')
        shutil.copyfile(SOURCE / 'datasets.json', out / 'datasets.json')
        h.write_rows(out / 'rows.jsonl', rows)
        h.write_rows(out / 'requests.jsonl', requests)
        h.write_rows(out / 'http_requests.jsonl', [{'group_id': r['group_id'], 'request': r['request'],
                     'request_sha256': r['request_sha256'], 'sample_ids': [r['sample_id']],
                     'sample_count': 1, 'protocol': 'responses'} for r in requests])
        for item in requests:
            slot = (*identity(item), item['sample_index'])
            h.require(slot not in old_slots and slot not in new_slots, 'New sample slot overlaps retained or planned data')
            new_slots.add(slot)
        manifest = {k: v for k, v in template_manifest.items() if k not in ('artifact_sha256', 'ablation_manifest_sha256',
                    'notes', 'parent_prompt_ablation_run', 'parent_prompt_ablation_manifest_sha256', 'preparer_sha256')}
        manifest.update(prepared_at_utc=h.now(), experiment_condition=CONDITION, cohort=cohort,
                        sample_count=512, sample_index_start=0, combined_draw_budget=512,
                        prompt_count=16, combined_prompts_per_cell=32, request_count=8192, http_request_count=8192,
                        reference_run=str(SOURCE), reference_manifest_sha256=h.digest(SOURCE / 'manifest.json'),
                        source_selection=h.binding(base / 'selection.json'), protocol_binding=h.binding(base / 'protocol.json'),
                        retained_prior_runs=h.binding(base / 'retained_prior_runs.json'),
                        prompt_interface=source_manifest['profiles'][f'{level}/{domain}'],
                        notes=['Ranks17-32 receive512 fresh responses each; all returned outcomes count.',
                               'No original eight-draw outputs or retained first16-problem outputs enter these new pools.',
                               'Original native prompt bytes, response surface, verifier, and generation controls remain unchanged.'])
        manifest['artifact_sha256'] = {name: h.digest(out / name) for name in
                                      ('datasets.json', 'model_profile.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl')}
        h.write(out / 'manifest.json', manifest)
        h.authenticate(out, manifest)
        registry[cohort] = {'run_dir': str(out), 'manifest': h.binding(out / 'manifest.json'),
                            'level': level, 'domain': domain, 'new_samples': 8192, 'sample_index_start': 0}
    h.require(len(new_slots) == 122880 and len(old_slots | new_slots) == 245760, 'Incorrect final 32x512 inventory')
    by_problem = defaultdict(set)
    for level, domain, index, draw in old_slots | new_slots:
        by_problem[(level, domain, index)].add(draw)
    h.require(len(by_problem) == 480 and all(draws == set(range(512)) for draws in by_problem.values()), 'Final pools are not gap-free through511')
    amendment = ROOT / 'paper/preregistration/gpt56sol_generalization32x512_20260913.md'
    shutil.copyfile(amendment, base / amendment.name)
    test = ROOT / 'tests/test_gpt56_all_levels32_discovery.py'
    shutil.copyfile(__file__, base / SCRIPT)
    shutil.copyfile(test, base / test.name)
    names = ('rows.jsonl', 'new_rows.jsonl', 'retained_rows.jsonl', 'level1_rows.jsonl', 'selection.json',
             'protocol.json', 'retained_prior_runs.json', 'execution_work_estimates.json', SCRIPT, test.name, amendment.name)
    result = {'schema': CONDITION, 'frozen_at_utc': h.now(), 'status_at_preparation': 'prepared_not_launched',
              'collection_runs': registry, 'collection_helper': h.binding(load_prior().HELPER),
              'prior_collector': h.binding(PRIOR_COLLECTOR),
              'artifact_sha256': {name: h.digest(base / name) for name in names},
              'new_requests': 122880, 'retained_responses': 122880, 'final_responses': 245760,
              'prompts': 480, 'retained_prompts': 240, 'new_prompts': 240,
              'model_calls_at_preparation': 0, 'credentials_recorded': False}
    h.write(base / 'manifest.json', result)
    return authenticate_base(base)


def authenticate_base(base, cohort=None):
    h = load_helper()
    base = Path(base).resolve()
    manifest = json.loads((base / 'manifest.json').read_text())
    h.authenticate(base, manifest)
    h.require(h.digest(__file__) == manifest['artifact_sha256'][SCRIPT], 'Run the unchanged frozen 32x512 collector source')
    for name in ('collection_helper', 'prior_collector'):
        check_binding(manifest[name])
    retained = json.loads((base / 'retained_prior_runs.json').read_text())
    for name in ('analysis', 'collection_audit', 'collection_manifest'):
        check_binding(retained[name])
    for entry in retained['runs']:
        check_binding(entry['manifest'])
        check_binding(entry['samples'])
    entries = manifest['collection_runs']
    h.require(manifest['schema'] == CONDITION and len(entries) == 15 and manifest['new_requests'] == 122880
              and manifest['retained_responses'] == 122880 and manifest['final_responses'] == 245760
              and manifest['prompts'] == 480, 'Incorrect frozen 32x512 inventory')
    if cohort is not None:
        h.require(cohort in entries, 'Unknown 32x512 cohort')
        entries = {cohort: entries[cohort]}
    for name, entry in entries.items():
        out = Path(entry['run_dir'])
        h.require(out.resolve() == base / 'hosted/gpt56sol' / name, 'Unexpected cohort output directory')
        check_binding(entry['manifest'])
        h.authenticate(out, json.loads((out / 'manifest.json').read_text()))
    return manifest


def audit_records(out):
    return load_prior().audit_records(Path(out))


def validate_preflight(out, records):
    h = load_helper()
    receipt = json.loads((out / 'preflight.json').read_text())
    h.require(receipt['manifest_sha256'] == h.digest(out / 'manifest.json') and receipt['evidence'], 'Invalid preflight manifest or empty evidence')
    current = {row['sample_id']: row for row in records}
    for item in receipt['evidence']:
        sid = item['sample_id']
        h.require(sid in current and current[sid]['provider_sample_identity'] == item['provider_sample_identity']
                  and h.digest(out / 'sample_receipts' / (sid + '.json')) == item['sample_receipt_sha256'],
                  'Changed preflight evidence')


def write_preflight(out, records):
    h = load_helper()
    h.require(records, 'No authenticated preflight sample')
    h.write(out / 'preflight.json', {'schema': 'sol-all-levels32x512-preflight-v1', 'checked_at_utc': h.now(),
            'manifest_sha256': h.digest(out / 'manifest.json'), 'retained_in_production': True,
            'served_snapshot': SNAPSHOT, 'served_controls': h.expected_returned_controls(),
            'evidence': [{'sample_id': r['sample_id'], 'provider_sample_identity': r['provider_sample_identity'],
                          'sample_receipt_sha256': h.digest(out / 'sample_receipts' / (r['sample_id'] + '.json'))} for r in records]})


def collect_one(base, cohort, stage, credential_file=None, workers=32):
    h = load_helper()
    h.require(stage in ('preflight', 'full') and 1 <= workers <= 64, 'Invalid collection stage or workers')
    entry = authenticate_base(base, cohort)['collection_runs'][cohort]
    out, expected = Path(entry['run_dir']), entry['new_samples']
    with (out / '.all_levels_orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = audit_records(out)
        h.require(len(existing) <= expected, 'Too many authenticated samples')
        marker = out / 'preflight.json'
        if marker.exists():
            validate_preflight(out, existing)
        if stage == 'preflight' and existing:
            if not marker.exists():
                write_preflight(out, existing)
            return {'cohort': cohort, 'preflight': 'authenticated_existing', 'authenticated_samples': len(existing), 'new_calls': 0}
        if len(existing) == expected:
            return {'cohort': cohort, 'complete': True, 'authenticated_samples': expected, 'new_calls': 0}
        if stage == 'full':
            h.require(marker.exists() and existing, 'Complete and inspect the authenticated preflight before production')
        env = os.environ.copy()
        env['AZURE_OPENAI_API_KEY'] = h.credential(credential_file)
        command = [sys.executable, str(out / 'code/ops/evaluate_native_prompt_ablation.py'), 'run',
                   '--model', 'gpt-5.6-sol', '--output', str(out), '--workers', str(workers),
                   '--request-timeout', '300', '--max-attempts', '8', '--rpm', '500']
        if stage == 'preflight':
            command += ['--max-new', '1']
        try:
            with (out / ('collection_' + stage + '.log')).open('a') as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        finally:
            env.pop('AZURE_OPENAI_API_KEY', None)
        h.require(process.returncode == 0, 'Native collection failed; retain evidence and inspect ' + cohort)
        records = audit_records(out)
        h.require(records and len(records) <= expected and (stage != 'full' or len(records) == expected), 'Incomplete cohort: ' + cohort)
        if stage == 'preflight' and not marker.exists():
            write_preflight(out, records)
        return {'cohort': cohort, 'stage': stage, 'authenticated_samples': len(records),
                'expected_samples': expected, 'served_snapshot': SNAPSHOT, 'complete': len(records) == expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'status', 'preflight', 'full'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--cohort')
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--max-concurrent-cohorts', type=int, default=4)
    parser.add_argument('--execution-amendment', type=Path)
    args = parser.parse_args()
    args.base = args.base.resolve()
    if args.command == 'prepare':
        manifest = prepare(args.base)
        print(json.dumps({key: manifest[key] for key in ('new_requests', 'retained_responses', 'final_responses', 'prompts')} | {'model_calls': 0}))
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
    prior = load_prior()
    prior.execution_limits(args.base, args.workers, args.max_concurrent_cohorts, args.execution_amendment)
    with (args.base / '.production_scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        capacity = args.max_concurrent_cohorts
        ramp_allowed = not args.cohort and capacity == 4 and args.workers == 32
        maximum = 8 if ramp_allowed else capacity
        initial_names, queued, errors = names[:capacity], list(names), []
        with ThreadPoolExecutor(max_workers=maximum) as pool:
            pending = {}
            def submit_available():
                while queued and len(pending) < capacity and not errors:
                    name = queued.pop(0)
                    pending[pool.submit(collect_one, args.base, name, 'full', args.credential_file, args.workers)] = name
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
                    health = prior.ramp_health(manifest, initial_names, args.base)
                    if health['eligible']:
                        amendment = args.base / 'execution_ramp_to256.json'
                        if amendment.exists():
                            prior.execution_limits(args.base, 32, 8, amendment)
                        else:
                            load_helper().write(amendment, {'schema': 'sol-all-levels32x512-execution-ramp-v1',
                                'at_utc': load_helper().now(), 'manifest_sha256': load_helper().digest(args.base / 'manifest.json'),
                                'workers_per_cohort': 32, 'max_concurrent_cohorts': 8, 'max_request_concurrency': 256,
                                'before_changed_worker_launches': True, 'health_evidence': health, 'request_payloads_changed': False,
                                'authorization': 'Authorized32-problem expansion using the prior evidence-based128-to256 concurrency ramp.'})
                        capacity = 8
                        print(json.dumps({'event': 'execution_ramp', 'concurrency': 256, 'amendment': str(amendment)}), flush=True)
                submit_available()
        if errors:
            raise RuntimeError(json.dumps({'errors': errors, 'unlaunched_cohorts': queued}))


if __name__ == '__main__':
    main()
