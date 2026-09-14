#!/usr/bin/env python3
"""Authenticate the fixed 32-problem, 512-draw expansion without API calls.

Retained and newly collected native receipts are checked independently of the
collector's receipt loop before their disjoint problem inventories are joined.
"""
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
BASE = ROOT / 'artifacts/modebench_discovery_all_levels_sol32x512_20260913'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LEVELS = (1, 2, 3)
DRAWS = 512
RETAINED_RESPONSES = NEW_RESPONSES = 122880
FINAL_RESPONSES = 245760
SNAPSHOT = 'gpt-5.6-sol-2026-07-09'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_collector():
    path = ROOT / 'ops/prepare_gpt56_all_levels32_discovery.py'
    spec = importlib.util.spec_from_file_location('_all_levels32_collection_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(row):
    return row['level'], row['domain'], row['row_index']


def binding_path(item):
    path = Path(item['path'])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def same_binding(first, second):
    return first['sha256'] == second['sha256'] and binding_path(first) == binding_path(second)


def check_binding(h, item):
    require(h.digest(binding_path(item)) == item['sha256'], 'Changed retained binding: ' + item['path'])


def row_inventory(rows, prompts_per_cell):
    keys = [identity(row) for row in rows]
    cells = Counter(key[:2] for key in keys)
    require(len(keys) == len(set(keys)), 'Duplicate problem row in fixed selection')
    require(set(cells) == {(level, domain) for level in LEVELS for domain in DOMAINS}
            and set(cells.values()) == {prompts_per_cell},
            'Incorrect fixed domain, level, or problem count per cell')
    return set(keys)


def check_inventory(records, rows, prompts_per_cell=32):
    """Check every native provider identity and every fixed global draw slot."""
    expected_rows = row_inventory(rows, prompts_per_cell)
    providers, slots, pools, cells = set(), set(), defaultdict(set), Counter()
    for record in records:
        key = identity(record)
        draw = record['sample_index']
        require(type(draw) is int and 0 <= draw < DRAWS, 'Draw index outside the fixed 0..511 interval')
        provider = tuple(record['provider_sample_identity'])
        require(len(provider) == 2 and isinstance(provider[0], str) and bool(provider[0])
                and type(provider[1]) is int and provider[1] == 0, 'Malformed native provider identity')
        slot = (*key, draw)
        require(key in expected_rows and slot not in slots and provider not in providers,
                'Unexpected problem, duplicate global draw slot, or reused provider output')
        providers.add(provider)
        slots.add(slot)
        pools[key].add(draw)
        cells[key[:2]] += 1
    require(len(slots) == len(expected_rows) * DRAWS and set(pools) == expected_rows,
            'Incorrect response or problem inventory')
    require(all(indices == set(range(DRAWS)) for indices in pools.values()),
            'A problem pool is not exactly the gap-free 0..511 interval')
    require(set(cells.values()) == {prompts_per_cell * DRAWS}, 'Unequal per-cell response inventory')
    return {'total_authenticated_responses': len(slots), 'unique_provider_samples': len(providers),
            'unique_prompts': len(pools), 'prompts_per_cell': prompts_per_cell,
            'global_provider_samples_unique': True, 'global_sample_slots_disjoint': True,
            'global_sample_slots_gap_free': True, 'final_draws_per_prompt': DRAWS,
            'cells': {f'L{level}/{domain}': count for (level, domain), count in sorted(cells.items())}}


def check_expansion(retained, added, retained_rows, new_rows, all_rows):
    old_keys = row_inventory(retained_rows, 16)
    new_keys = row_inventory(new_rows, 16)
    all_keys = row_inventory(all_rows, 32)
    require(old_keys.isdisjoint(new_keys) and old_keys | new_keys == all_keys,
            'Added problems overlap retained problems or differ from the fixed union')
    expected = {identity(row): row for row in all_rows}
    require(all(expected[identity(row)] == row for row in [*retained_rows, *new_rows]),
            'A retained or new problem revision differs from the combined inventory')
    require(len(retained) == RETAINED_RESPONSES and len(added) == NEW_RESPONSES,
            'Expected exactly 122880 retained and 122880 new responses')
    require(all(identity(row) in old_keys for row in retained)
            and all(identity(row) in new_keys for row in added),
            'A response was reassigned across retained and newly selected problems')
    result = check_inventory([*retained, *added], all_rows)
    result['retained_and_new_problem_inventories_disjoint'] = True
    return result


def authenticate_receipts(h, out):
    """Use frozen native validation, then independently enforce Sol controls."""
    out = Path(out)
    adapter = h.load_adapter(out)
    request_rows = h.rows(out / 'requests.jsonl')
    group_rows = h.rows(out / 'http_requests.jsonl')
    requests = {row['sample_id']: row for row in request_rows}
    groups = {row['group_id']: row for row in group_rows}
    require(len(requests) == len(request_rows) and len(groups) == len(group_rows),
            'Duplicate request or HTTP group in a native manifest')
    expected_controls = h.expected_returned_controls()
    records, providers = [], set()
    for path in sorted((out / 'sample_receipts').glob('*.json')):
        record = json.loads(path.read_text())
        sid = record['sample_id']
        require(sid in requests and path.name == sid + '.json', 'Unexpected native receipt slot')
        item = requests[sid]
        cache = {}
        adapter.validate_completed(out, item, groups[item['group_id']], record, cache)
        receipt = cache[record['raw_receipt']]
        body = receipt['response']
        require({key: body.get(key) for key in expected_controls} == expected_controls,
                'Native served generation controls changed')
        headers = {key.lower(): value for key, value in receipt['headers'].items()}
        require(headers.get('x-ms-served-model') == SNAPSHOT, 'Native served model snapshot changed')
        provider = tuple(record['provider_sample_identity'])
        require(provider not in providers, 'Repeated native provider output within a run')
        providers.add(provider)
        records.append(record)
    return records


def authenticate_saved_run(h, out, expected):
    out = Path(out)
    h.authenticate(out, json.loads((out / 'manifest.json').read_text()))
    saved = h.rows(out / 'samples.jsonl')
    native = authenticate_receipts(h, out)
    require(len(saved) == len(native) == expected, 'Incorrect authenticated native sample count')
    require(len({row['sample_id'] for row in saved}) == expected,
            'Repeated sample line in samples.jsonl')
    require({row['sample_id']: row for row in saved} == {row['sample_id']: row for row in native},
            'Saved samples differ from authenticated native receipts')
    return saved


def audit_new_run(collector, name, entry):
    h = collector.load_helper()
    out = Path(entry['run_dir'])
    expected = entry['new_samples']
    require(expected == 8192, 'Every new cohort must contain 16 problems and 8192 responses')
    state = json.loads((out / 'status.json').read_text())
    require(state['complete'] is True and state['completed_samples'] == expected
            and state['failed_groups_this_session'] == 0, 'Incomplete native run: ' + name)
    saved = authenticate_saved_run(h, out, expected)
    errors = h.rows(out / 'errors.jsonl') if (out / 'errors.jsonl').exists() else []
    runner_errors = h.rows(out / 'runner_errors.jsonl') if (out / 'runner_errors.jsonl').exists() else []
    raw_count = sum(1 for _ in (out / 'raw_responses').glob('*.json'))
    require(not runner_errors, 'Native runner retained failed groups: ' + name)
    require(raw_count == expected + len(errors), 'Unaccounted raw attempt: ' + name)
    require(all(error.get('http_status') != 200 for error in errors),
            'A successful native response was classified as a transport error')
    usage = {key: sum(row['usage'][key] for row in saved)
             for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    require(state['usage'] == usage, 'Status usage differs from authenticated native usage')
    result = {'run_dir': str(out), 'responses': expected,
              'manifest': h.binding(out / 'manifest.json'), 'samples': h.binding(out / 'samples.jsonl'),
              'status': h.binding(out / 'status.json'), 'preflight': h.binding(out / 'preflight.json'),
              'all_native_receipts_authenticated': True, 'samples_jsonl_equals_native_receipts': True,
              'served_snapshot': SNAPSHOT, 'served_controls': h.expected_returned_controls(),
              'raw_attempt_count': raw_count, 'failed_groups': 0,
              'retained_transport_error_status_counts': dict(Counter(str(row.get('http_status')) for row in errors)),
              'successful_native_usage': usage}
    if errors:
        result['transport_errors'] = h.binding(out / 'errors.jsonl')
    return name, result, saved


def audit_retained_run(collector, entry):
    h = collector.load_helper()
    check_binding(h, entry['manifest'])
    check_binding(h, entry['samples'])
    records = authenticate_saved_run(h, Path(entry['directory']), entry['responses'])
    return entry | {'all_native_receipts_reauthenticated': True,
                    'authentication_basis': 'Unchanged prior bindings plus independent reauthentication of every frozen native receipt.'}, records


def lock_file(stack, path):
    handle = stack.enter_context(Path(path).open('r'))
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def audit(base=BASE, workers=4):
    require(type(workers) is int and 1 <= workers <= 8, 'Audit workers must be between 1 and 8')
    collector = load_collector()
    h = collector.load_helper()
    base = Path(base).resolve()
    with ExitStack() as stack:
        lock_file(stack, base / '.production_scheduler.lock')
        manifest = collector.authenticate_base(base)
        require(collector.SNAPSHOT == SNAPSHOT, 'Collector requested another native snapshot')
        expected_cells = {f'L{level}_{domain}' for level in LEVELS for domain in DOMAINS}
        require(set(manifest['collection_runs']) == expected_cells, 'Expected all fifteen new cohorts')
        for entry in manifest['collection_runs'].values():
            for name in ('.all_levels_orchestrator.lock', '.runner.lock'):
                lock_file(stack, Path(entry['run_dir']) / name)
        prior = json.loads((base / 'retained_prior_runs.json').read_text())
        for key in ('analysis', 'collection_audit', 'collection_manifest'):
            check_binding(h, prior[key])
        previous = json.loads(binding_path(prior['collection_audit']).read_text())
        require(previous['status'] == 'complete' and previous['total_authenticated_responses'] == RETAINED_RESPONSES
                and previous['unique_prompts'] == 240 and previous['final_draws_per_prompt'] == DRAWS
                and all(previous[key] is True for key in ('global_provider_samples_unique',
                    'global_sample_slots_disjoint', 'global_sample_slots_gap_free',
                    'all_execution_locks_released_before_audit')),
                'Prior native collection audit is incomplete')
        report = json.loads(binding_path(prior['analysis']).read_text())
        require(report['status'] == 'complete' and report['responses'] == RETAINED_RESPONSES
                and report['prompts'] == 240, 'Prior analysis is incomplete')
        expected_sources = {str(Path(row['directory']).resolve()): row for row in report['sources']}
        require(len(expected_sources) == len(prior['runs']) == 20
                and {str(Path(row['directory']).resolve()) for row in prior['runs']} == set(expected_sources),
                'Retained source runs differ from the complete prior analysis')
        prior_scheduler = binding_path(prior['collection_manifest']).parent / '.production_scheduler.lock'
        lock_file(stack, prior_scheduler)
        for entry in prior['runs']:
            out = Path(entry['directory'])
            reference = expected_sources[str(out.resolve())]
            require(entry['responses'] == reference['responses']
                    and all(same_binding(entry[key], reference[key]) for key in ('manifest', 'samples')),
                    'Retained source binding differs from the completed prior analysis')
            lock_file(stack, out / '.runner.lock')
            if (out / '.all_levels_orchestrator.lock').exists():
                lock_file(stack, out / '.all_levels_orchestrator.lock')
        retained_records, added_records, retained_runs, runs = [], [], [], {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(audit_retained_run, collector, entry): 'retained' for entry in prior['runs']}
            futures.update({pool.submit(audit_new_run, collector, name, entry): 'new'
                            for name, entry in manifest['collection_runs'].items()})
            for future in as_completed(futures):
                if futures[future] == 'retained':
                    entry, records = future.result()
                    retained_runs.append(entry)
                    retained_records.extend(records)
                    name = entry['directory']
                else:
                    name, result, records = future.result()
                    runs[name] = result
                    added_records.extend(records)
                print(json.dumps({'authenticated_cohort': name, 'stage': futures[future],
                                  'responses': len(records)}), flush=True)
        inventory = check_expansion(retained_records, added_records, h.rows(base / 'retained_rows.jsonl'),
                                    h.rows(base / 'new_rows.jsonl'), h.rows(base / 'rows.jsonl'))
        require(inventory['total_authenticated_responses'] == FINAL_RESPONSES, 'Incorrect completed expansion total')
        transport, usage = Counter(), Counter()
        for run in runs.values():
            transport.update(run['retained_transport_error_status_counts'])
            usage.update(run['successful_native_usage'])
        result = {'schema': 'sol-all-levels32x512-completion-v1', 'at_utc': h.now(), 'status': 'complete',
                  **inventory, 'new_authenticated_responses': len(added_records),
                  'retained_authenticated_responses': len(retained_records),
                  'collection_manifest': h.binding(base / 'manifest.json'),
                  'prior_collection_audit': prior['collection_audit'],
                  'retained_runs': sorted(retained_runs, key=lambda entry: entry['directory']),
                  'runs': dict(sorted(runs.items())),
                  'new_raw_attempts': sum(run['raw_attempt_count'] for run in runs.values()),
                  'new_transport_error_status_counts': dict(transport), 'new_successful_native_usage': dict(usage),
                  'served_snapshot_required': SNAPSHOT, 'served_controls': h.expected_returned_controls(),
                  'all_execution_locks_released_before_audit': True,
                  'all_retained_native_receipts_reauthenticated': True,
                  'audit_driver': h.binding(__file__), 'api_calls_by_audit': 0,
                  'semantic_regrading_and_normalization': 'Separate source-bound analysis; this audit authenticates collection identity and native controls.'}
        h.write(base / 'collection_completion_audit.json', result)
        print(json.dumps({'status': 'complete', **inventory, 'audit': str(base / 'collection_completion_audit.json')}), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    audit(args.base, args.workers)
