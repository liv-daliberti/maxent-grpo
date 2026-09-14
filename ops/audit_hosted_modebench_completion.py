#!/usr/bin/env python3
"""Authenticate completed hosted ModeBench evaluations across native providers.

Read-only for primary evidence; writes a separate audit and SHA-256 inventory.
Validates logical samples separately from physical HTTP groups (including n=8).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

RUNNERS = {
    'frontier-modebench-anthropic-messages-v1': 'ops/evaluate_claude_modebench.py',
    'frontier-modebench-native-chat-responses-v1': 'ops/evaluate_chat_frontier_modebench.py',
    'frontier-modebench-responses-v1': 'ops/evaluate_frontier_modebench.py',
}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_inventory(directory, expected_samples=15360):
    """Authenticate the frozen design and load its exact native receipt adapter."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, digest in manifest['artifact_sha256'].items():
        if file_sha(directory / name) != digest:
            raise ValueError('Frozen artifact changed: ' + name)
    for name, digest in manifest['code_sha256'].items():
        if file_sha(directory / 'code' / name) != digest:
            raise ValueError('Frozen code changed: ' + name)
    runner_name = RUNNERS.get(manifest['schema'])
    if runner_name is None or runner_name not in manifest['code_sha256']:
        raise ValueError('Unsupported or unfrozen native runner')
    name = '_hosted_receipt_adapter_' + sha(str(directory))[:16]
    spec = importlib.util.spec_from_file_location(name, directory / 'code' / runner_name)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    requests = read_jsonl(directory / 'requests.jsonl')
    expected = {r['sample_id']: r for r in requests}
    if len(requests) != expected_samples or len(expected) != expected_samples or manifest['request_count'] != expected_samples:
        raise ValueError('Unexpected or duplicate logical request inventory')
    rows = read_jsonl(directory / 'rows.jsonl')
    row_lookup = {(r['level'], r['domain'], r['row_index']): r for r in rows}
    if len(rows) != len(row_lookup):
        raise ValueError('Duplicate frozen prompt identity')
    for item in requests:
        identity = item['level'], item['domain'], item['row_index']
        if identity not in row_lookup or item['row_sha256'] != sha(row_lookup[identity]):
            raise ValueError('Logical request row identity mismatch')
        if item['request_sha256'] != sha(item['request']) or item['request']['model'] != manifest['model']:
            raise ValueError('Logical request payload identity mismatch')
    grouped = manifest['schema'] == 'frontier-modebench-native-chat-responses-v1'
    if grouped:
        group_list = read_jsonl(directory / 'http_requests.jsonl')
        groups = {g['group_id']: g for g in group_list}
        members = []
        if len(groups) != len(group_list) or len(groups) != manifest['http_request_count']:
            raise ValueError('Unexpected or duplicate HTTP group inventory')
        for group in groups.values():
            if group['request_sha256'] != sha(group['request']) or len(group['sample_ids']) != group['sample_count']:
                raise ValueError('Invalid HTTP group identity')
            choice_indices = set()
            for sample_id in group['sample_ids']:
                item = expected[sample_id]
                if item['group_id'] != group['group_id'] or item['request'] != group['request'] or item['request_sha256'] != group['request_sha256']:
                    raise ValueError('Logical sample/HTTP group mismatch')
                members.append(sample_id)
                choice_indices.add(item['choice_index'])
            if choice_indices != set(range(group['sample_count'])):
                raise ValueError('Invalid grouped choice inventory')
        if len(members) != len(expected) or set(members) != set(expected):
            raise ValueError('HTTP groups do not partition logical samples')
    else:
        groups = {sid: {'group_id': sid, 'request': item['request'],
                       'request_sha256': item['request_sha256'], 'sample_ids': [sid], 'sample_count': 1}
                  for sid, item in expected.items()}
    return {'directory': directory, 'manifest': manifest, 'runner': runner,
            'requests': requests, 'expected': expected, 'rows': row_lookup,
            'groups': groups, 'grouped': grouped}


def validate_sample(inventory, item, record, raw_cache):
    directory, runner = inventory['directory'], inventory['runner']
    if inventory['grouped']:
        group = inventory['groups'][item['group_id']]
        runner.validate_completed(directory, item, group, record, raw_cache)
    else:
        runner.validate_completed_record(directory, item, record)
    relative = record['raw_receipt']
    if relative not in raw_cache:
        raw_cache[relative] = json.loads((directory / relative).read_text())
    return raw_cache[relative]


def validate_native_records(inventory, records, io_workers=16):
    """Validate exact native receipts in parallel; never run scientific graders."""
    if not 1 <= io_workers <= 32:
        raise ValueError('I/O workers must be 1..32')
    raw_cache = {}
    def validate(record):
        item = inventory['expected'][record['sample_id']]
        validate_sample(inventory, item, record, raw_cache)
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        list(pool.map(validate, records))
    return raw_cache


def audit(directory, expected_samples=15360, io_workers=16):
    if not 1 <= io_workers <= 32:
        raise ValueError('I/O workers must be 1..32')
    inventory = load_inventory(directory, expected_samples)
    directory, manifest = inventory['directory'], inventory['manifest']
    expected, groups = inventory['expected'], inventory['groups']
    samples = read_jsonl(directory / 'samples.jsonl')
    actual = {sample['sample_id']: sample for sample in samples}
    if len(samples) != len(actual) or expected.keys() != actual.keys():
        raise ValueError('Evaluation is incomplete or contains duplicate/unexpected samples')
    atomic_paths = {p.stem: p for p in (directory / 'sample_receipts').glob('*.json')}
    if atomic_paths.keys() != expected.keys():
        raise ValueError('Atomic sample receipt inventory differs from frozen requests')
    raw_cache, response_groups = {}, {}
    provider_samples, raw_used = set(), set()
    snapshots, usage = Counter(), Counter()
    def validate_one(pair):
        sample_id, item = pair
        record = actual[sample_id]
        if json.loads(atomic_paths[sample_id].read_text()) != record:
            raise ValueError('JSONL and atomic sample receipt disagree: ' + sample_id)
        raw = validate_sample(inventory, item, record, raw_cache)
        return sample_id, item, record, raw

    # Only independent, read-only receipt validation runs in parallel. Logical
    # identity/collision decisions below remain ordered and deterministic.
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        authenticated = list(pool.map(validate_one, expected.items()))
    for sample_id, item, record, raw in authenticated:
        raw_used.add(record['raw_receipt'])
        response_id = raw['response'].get('id')
        if not isinstance(response_id, str) or not response_id:
            raise ValueError('Missing native response identifier')
        group_id = item.get('group_id', sample_id)
        if response_id in response_groups and response_groups[response_id] != group_id:
            raise ValueError('Native response ID reused across independent HTTP groups')
        response_groups[response_id] = group_id
        provider_id = response_id, item.get('choice_index', 0)
        if provider_id in provider_samples:
            raise ValueError('Duplicate native response/choice identity')
        provider_samples.add(provider_id)
        snapshots[raw.get('headers', {}).get('x-ms-served-model', 'not_returned')] += 1
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            usage[key] += (record.get('usage') or {}).get(key, 0) or 0
    events = read_jsonl(directory / 'events.jsonl')
    started = [((e.get('group_id') or e.get('sample_id')), e['attempt'])
               for e in events if e['event'] == 'request_started']
    if len(started) != len(set(started)):
        raise ValueError('Duplicate physical request-start identity')
    received, raw_status, files = set(), Counter(), {}
    inventory_paths = []
    for path in (directory / 'raw_responses').glob('*.json'):
        relative = str(path.relative_to(directory))
        raw = raw_cache.get(relative)
        if raw is None:
            raw = json.loads(path.read_text())
        group_id = raw.get('group_id') or raw.get('sample_id')
        if group_id not in groups or raw.get('relative_path') != relative:
            raise ValueError('Unexpected raw receipt group or path')
        group = groups[group_id]
        if inventory['grouped'] and raw.get('sample_ids') != group['sample_ids']:
            raise ValueError('Raw attempt sample inventory differs from frozen group')
        if raw.get('request_sha256') != group['request_sha256']:
            raise ValueError('Raw attempt request digest differs from frozen group')
        parsed_id, separator, attempt = path.stem.rpartition('__')
        if parsed_id != group_id or not separator or not attempt.isdigit() or int(attempt) != raw.get('attempt'):
            raise ValueError('Raw attempt filename differs from its metadata')
        attempt_id = group_id, raw['attempt']
        if attempt_id in received:
            raise ValueError('Duplicate physical attempt receipt')
        received.add(attempt_id)
        raw_status[str(raw.get('http_status') or raw.get('error_type'))] += 1
        inventory_paths.append(path)
    if set(started) != received:
        raise ValueError('Request starts and raw attempt receipts disagree')
    inventory_paths.extend(atomic_paths.values())
    for name in ('manifest.json', 'datasets.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl',
                 'model_profile.json', 'samples.jsonl', 'events.jsonl', 'status.json', 'errors.jsonl', 'runner_errors.jsonl'):
        if (directory / name).is_file():
            inventory_paths.append(directory / name)
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        digests = list(pool.map(file_sha, inventory_paths))
    files = {str(path.relative_to(directory)): digest for path, digest in zip(inventory_paths, digests)}
    result = {'schema': 'hosted-modebench-completion-audit-v1',
              'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'pass',
              'model': manifest['model'], 'native_schema': manifest['schema'],
              'expected_responses': len(expected), 'saved_samples': len(actual),
              'expected_http_groups': len(groups), 'unique_response_ids': len(response_groups),
              'unique_response_choice_ids': len(provider_samples), 'raw_success_receipts_used': len(raw_used),
              'raw_attempts': len(received), 'attempt_status_counts': dict(raw_status),
              'served_model_snapshots_by_logical_sample': dict(snapshots),
              'usage_counted_once_per_http_response': dict(usage),
              'evidence_file_count': len(files), 'evidence_inventory_sha256': sha(files),
              'checks': ['Frozen datasets, requests, grouped requests, and code hashes match.',
                         'Logical sample identities and HTTP groups partition the frozen test inventory.',
                         'Each sample has one atomic receipt and identical JSONL export.',
                         'Sample answer text, native metadata, usage, and response identity match its raw receipt.',
                         'Native response/choice identities are unique; shared response IDs occur only within one HTTP group.',
                         'Every logged physical request attempt has exactly one raw HTTP or transport-error receipt.'],
              'auditor_sha256': file_sha(Path(__file__)), 'independent_io_workers': io_workers}
    atomic(directory / 'evidence_file_sha256.json', files)
    atomic(directory / 'completion_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--io-workers', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, io_workers=args.io_workers), indent=2))
