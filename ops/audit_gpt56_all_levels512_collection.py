#!/usr/bin/env python3
"""Authenticate the completed fixed n512 collection without making API calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_all_levels_sol512_20260912'


def load_collector():
    path = ROOT / 'ops/prepare_gpt56_all_levels512_discovery.py'
    spec = importlib.util.spec_from_file_location('_n512_collection_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_binding(h, binding):
    h.require(h.digest(binding['path']) == binding['sha256'], 'Changed prior binding: ' + binding['path'])


def check_inventory(records, rows, expected_count=122880):
    """Reject reused provider outputs, duplicate slots, and incomplete prompt pools."""
    expected_rows = {(r['level'], r['domain'], r['row_index']) for r in rows}
    providers, slots, pools, cells = set(), set(), defaultdict(set), Counter()
    for record in records:
        key = record['level'], record['domain'], record['row_index']
        slot = (*key, record['sample_index'])
        provider = tuple(record['provider_sample_identity'])
        if key not in expected_rows or slot in slots or provider in providers:
            raise ValueError('Unexpected problem, duplicate global draw slot, or reused provider output')
        providers.add(provider)
        slots.add(slot)
        pools[key].add(record['sample_index'])
        cells[key[:2]] += 1
    if len(slots) != expected_count or set(pools) != expected_rows:
        raise ValueError('Incorrect response or prompt inventory')
    if any(indices != set(range(512)) for indices in pools.values()):
        raise ValueError('A prompt pool is not exactly the gap-free 0..511 interval')
    if len(expected_rows) != 240 or len(cells) != 15 or set(cells.values()) != {8192}:
        raise ValueError('Expected 16 prompts and 8192 responses per level/domain cell')
    return {'total_authenticated_responses': len(slots), 'unique_provider_samples': len(providers),
            'unique_prompts': len(pools), 'global_provider_samples_unique': True,
            'global_sample_slots_disjoint': True, 'global_sample_slots_gap_free': True,
            'final_draws_per_prompt': 512,
            'cells': {f'L{level}/{domain}': count for (level, domain), count in sorted(cells.items())}}


def audit_new_run(collector, name, entry):
    h = collector.load_helper()
    out = Path(entry['run_dir'])
    state = json.loads((out / 'status.json').read_text())
    expected = entry['new_samples']
    h.require(state['complete'] and state['completed_samples'] == expected
              and state['failed_groups_this_session'] == 0, 'Incomplete native run: ' + name)
    saved = h.rows(out / 'samples.jsonl')
    records = collector.audit_records(out)
    h.require(len(saved) == len(records) == expected, 'Incorrect native sample count: ' + name)
    h.require(len({r['sample_id'] for r in saved}) == expected, 'Repeated sample line: ' + name)
    h.require({r['sample_id']: r for r in saved} == {r['sample_id']: r for r in records},
              'Samples differ from authenticated native receipts: ' + name)
    errors = h.rows(out / 'errors.jsonl') if (out / 'errors.jsonl').exists() else []
    runner_errors = h.rows(out / 'runner_errors.jsonl') if (out / 'runner_errors.jsonl').exists() else []
    raw_count = sum(1 for _ in (out / 'raw_responses').glob('*.json'))
    h.require(not runner_errors, 'Native runner retained failed groups: ' + name)
    h.require(raw_count == expected + len(errors), 'Unaccounted raw attempt in ' + name)
    h.require(all(error.get('http_status') != 200 for error in errors), 'Successful response was classified as a transport error')
    result = {'run_dir': str(out), 'responses': expected,
              'manifest': h.binding(out / 'manifest.json'), 'samples': h.binding(out / 'samples.jsonl'),
              'status': h.binding(out / 'status.json'), 'preflight': h.binding(out / 'preflight.json'),
              'all_native_receipts_authenticated': True, 'samples_jsonl_equals_native_receipts': True,
              'served_snapshot': collector.SNAPSHOT, 'served_controls': h.expected_returned_controls(),
              'raw_attempt_count': raw_count, 'failed_groups': 0,
              'retained_transport_error_status_counts': dict(Counter(str(r.get('http_status')) for r in errors)),
              'successful_native_usage': state['usage']}
    if errors:
        result['transport_errors'] = h.binding(out / 'errors.jsonl')
    return name, result, saved


def audit(base=BASE, workers=4):
    collector = load_collector()
    h = collector.load_helper()
    base = Path(base).resolve()
    with ExitStack() as stack:
        # Refuse to attest completion while the scheduler or any collector owns
        # its normal execution lock. Locks are held until this audit is written.
        scheduler = stack.enter_context((base / '.production_scheduler.lock').open('r'))
        fcntl.flock(scheduler, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = collector.authenticate_base(base)
        for entry in manifest['collection_runs'].values():
            out = Path(entry['run_dir'])
            for name in ('.all_levels_orchestrator.lock', '.runner.lock'):
                handle = stack.enter_context((out / name).open('r'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = json.loads((base / 'retained_prior_runs.json').read_text())
        check_binding(h, prior['analysis'])
        check_binding(h, prior['collection_audit'])
        prior_audit = json.loads(Path(prior['collection_audit']['path']).read_text())
        h.require(prior_audit['total_authenticated_responses'] == 20480
                  and prior_audit['global_provider_samples_unique']
                  and prior_audit['global_sample_slots_disjoint']
                  and prior_audit['global_sample_slots_gap_free'], 'Prior native audit is incomplete')
        all_records, retained = [], []
        for entry in prior['runs']:
            check_binding(h, entry['manifest'])
            check_binding(h, entry['samples'])
            out = Path(entry['directory'])
            h.authenticate(out, json.loads((out / 'manifest.json').read_text()))
            records = h.rows(out / 'samples.jsonl')
            h.require(len(records) == entry['responses'], 'Retained native pool count changed')
            all_records.extend(records)
            retained.append(entry | {'authentication_basis': 'Unchanged manifest and samples bound to the completed prior native collection audit.'})
        h.require(len(all_records) == 20480, 'Incorrect retained response inventory')
        runs = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(audit_new_run, collector, name, entry)
                       for name, entry in manifest['collection_runs'].items()]
            for future in as_completed(futures):
                name, result, records = future.result()
                runs[name] = result
                all_records.extend(records)
                print(json.dumps({'authenticated_cohort': name, 'responses': len(records)}), flush=True)
        inventory = check_inventory(all_records, h.rows(base / 'rows.jsonl'))
        new_count = sum(r['responses'] for r in runs.values())
        h.require(new_count == 102400, 'Incorrect added response inventory')
        transport = Counter()
        usage = Counter()
        for run in runs.values():
            transport.update(run['retained_transport_error_status_counts'])
            usage.update({k: v for k, v in run['successful_native_usage'].items() if isinstance(v, int)})
        result = {'schema': 'sol-all-levels512-completion-v1', 'at_utc': h.now(), 'status': 'complete',
                  **inventory, 'new_authenticated_responses': new_count, 'retained_authenticated_responses': 20480,
                  'collection_manifest': h.binding(base / 'manifest.json'),
                  'prior_collection_audit': prior['collection_audit'], 'retained_runs': retained,
                  'runs': dict(sorted(runs.items())), 'new_raw_attempts': sum(r['raw_attempt_count'] for r in runs.values()),
                  'new_transport_error_status_counts': dict(transport), 'new_successful_native_usage': dict(usage),
                  'served_snapshot_required': collector.SNAPSHOT, 'served_controls': h.expected_returned_controls(),
                  'all_execution_locks_released_before_audit': True,
                  'audit_driver': h.binding(__file__), 'api_calls_by_audit': 0,
                  'semantic_regrading_and_normalization': 'Separate source-bound analysis; this audit authenticates collection identity and native controls.'}
        h.write(base / 'collection_completion_audit.json', result)
        print(json.dumps({'status': 'complete', **inventory, 'audit': str(base / 'collection_completion_audit.json')}), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--workers', type=int, default=4)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        parser.error('Audit workers must be between 1 and 8')
    audit(arguments.base, arguments.workers)
