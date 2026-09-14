#!/usr/bin/env python3
"""Read-only integrity audit and durable file inventory for a completed API run."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from evaluate_frontier_modebench import (
    atomic, file_sha, response_text, sha, validate_completed_record,
)


def audit(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, digest in manifest['artifact_sha256'].items():
        if file_sha(directory / name) != digest:
            raise ValueError('Frozen artifact changed: ' + name)
    for name, digest in manifest['code_sha256'].items():
        if file_sha(directory / 'code' / name) != digest:
            raise ValueError('Frozen code changed: ' + name)
    requests = [json.loads(line) for line in (directory / 'requests.jsonl').read_text().splitlines()]
    samples = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    expected = {r['sample_id']: r for r in requests}
    actual = {s['sample_id']: s for s in samples}
    if len(expected) != 15360 or len(samples) != len(actual) or expected.keys() != actual.keys():
        raise ValueError('Full evaluation must contain all 15,360 unique expected responses')
    response_ids = set()
    snapshots = Counter()
    raw_used = set()
    for sample_id, item in expected.items():
        record = actual[sample_id]
        saved = json.loads((directory / 'sample_receipts' / (sample_id + '.json')).read_text())
        if saved != record:
            raise ValueError('JSONL and atomic receipt disagree: ' + sample_id)
        validate_completed_record(directory, item, record)
        raw_used.add(record['raw_receipt'])
        raw = json.loads((directory / record['raw_receipt']).read_text())
        body = raw['response']
        if body['id'] in response_ids:
            raise ValueError('Repeated API response ID')
        response_ids.add(body['id'])
        snapshots[raw.get('headers', {}).get('x-ms-served-model', 'not_returned')] += 1
        if response_text(body) != record['text']:
            raise ValueError('Response extraction mismatch')
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    starts = {(e['sample_id'], e['attempt']) for e in events if e['event'] == 'request_started'}
    received = set()
    raw_status = Counter()
    files = {}
    for folder in ('raw_responses', 'sample_receipts'):
        for path in sorted((directory / folder).glob('*.json')):
            files[str(path.relative_to(directory))] = file_sha(path)
            if folder == 'raw_responses':
                raw = json.loads(path.read_text())
                received.add((raw['sample_id'], raw['attempt']))
                raw_status[str(raw.get('http_status') or raw.get('error_type'))] += 1
    if starts != received:
        raise ValueError('Request starts and raw attempt receipts disagree')
    for name in ('manifest.json', 'datasets.json', 'rows.jsonl', 'requests.jsonl',
                 'samples.jsonl', 'events.jsonl', 'status.json', 'errors.jsonl'):
        if (directory / name).exists():
            files[name] = file_sha(directory / name)
    atomic(directory / 'evidence_file_sha256.json', files)
    result = {
        'schema': 'frontier-modebench-completion-audit-v1',
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'pass', 'expected_responses': len(expected),
        'saved_samples': len(samples), 'unique_response_ids': len(response_ids),
        'raw_success_receipts_used': len(raw_used), 'raw_attempts': len(received),
        'attempt_status_counts': dict(raw_status), 'served_model_snapshots': dict(snapshots),
        'evidence_file_count': len(files), 'evidence_inventory_sha256': sha(files),
        'checks': ['Frozen request, dataset and code hashes match.',
                   'Every expected sample has one atomic receipt and identical JSONL export.',
                   'Saved response text, identity, provider metadata and usage match the exact raw response; verifier fields are internally consistent.',
                   'Every sample has a distinct API response ID.',
                   'Every logged started attempt has a raw response or transport-error receipt.'],
        'auditor_sha256': file_sha(Path(__file__)),
    }
    atomic(directory / 'completion_audit.json', result)
    return result


if __name__ == '__main__':
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else
                     'artifacts/frontier_modebench_gpt56sol_20260911')
    print(json.dumps(audit(directory), indent=2))
