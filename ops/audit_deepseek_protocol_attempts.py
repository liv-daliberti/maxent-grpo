#!/usr/bin/env python3
"""Classify authenticated DeepSeek physical attempts without changing evidence."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def audit(directory):
    directory = directory.resolve()
    completion = json.loads((directory / 'completion_audit.json').read_text())
    inventory = json.loads((directory / 'evidence_file_sha256.json').read_text())
    adapter = json.loads((directory / 'provider_protocol_adapter.json').read_text())
    if completion['status'] != 'pass' or completion['saved_samples'] != 15360:
        raise ValueError('Full terminal sample cohort has not passed completion audit')
    if compact_sha(inventory) != completion['evidence_inventory_sha256']:
        raise ValueError('Completion evidence inventory changed')
    source = directory / adapter['source_path']
    if digest(source) != adapter['source_sha256']:
        raise ValueError('Frozen runtime protocol predicate changed')
    spec = importlib.util.spec_from_file_location('_frozen_deepseek_protocol', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if digest(directory / 'samples.jsonl') != inventory['samples.jsonl']:
        raise ValueError('Terminal sample inventory changed')
    samples = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    used = {sample['raw_receipt'] for sample in samples}
    if len(samples) != len(used) or len(used) != 15360:
        raise ValueError('Expected 15,360 distinct terminal receipts')
    statuses = Counter()
    failures = []
    for relative, expected in inventory.items():
        if not relative.startswith('raw_responses/'):
            continue
        path = directory / relative
        if digest(path) != expected:
            raise ValueError('Raw attempt changed since completion audit')
        raw = json.loads(path.read_text())
        statuses[str(raw.get('http_status') or raw.get('error_type'))] += 1
        protocol_failure = module.provider_protocol_failure(raw)
        if relative in used and protocol_failure:
            raise ValueError('Malformed protocol attempt was included as a model draw')
        if raw.get('http_status') == 200 and relative not in used:
            if not protocol_failure:
                raise ValueError('Unclassified unused HTTP200 response')
            failures.append({'path': relative, 'sha256': expected, 'group_id': raw['group_id'],
                             'attempt': raw['attempt'], 'native_finish_reason': '',
                             'native_usage': None, 'outcome': 'nonterminal_provider_protocol_failure',
                             'possibly_billed': True, 'used_as_model_sample': False})
    for original in adapter['initial_receipts'].values():
        if inventory.get(original['path']) != original['sha256']:
            raise ValueError('Original motivating protocol failure changed')
    if dict(statuses) != completion['attempt_status_counts']:
        raise ValueError('Physical receipt counts differ from completion audit')
    result = {
        'schema': 'deepseek-final-protocol-attempt-audit-v1', 'status': 'pass',
        'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'api_calls': 0,
        'auditor_sha256': digest(Path(__file__)), 'completion_audit_sha256': digest(directory / 'completion_audit.json'),
        'evidence_inventory_sha256': completion['evidence_inventory_sha256'],
        'provider_protocol_adapter_sha256': digest(directory / 'provider_protocol_adapter.json'),
        'frozen_predicate_source_sha256': adapter['source_sha256'],
        'retained_terminal_samples': len(used), 'raw_attempt_receipts': sum(statuses.values()),
        'attempt_status_counts': dict(statuses), 'nonterminal_http200_protocol_failures': len(failures),
        'initial_protocol_failures': len(adapter['initial_receipts']),
        'additional_protocol_failures_after_recovery': len(failures) - len(adapter['initial_receipts']),
        'nonterminal_protocol_failures': sorted(failures, key=lambda item: item['path']),
        'interruption_accounting': completion['interruption_accounting'],
        'usage_note': 'Protocol usage:null and registered interrupted outcomes remain unknown and possibly billed; reported native token totals are subtotals.',
    }
    module.atomic(directory / 'provider_protocol_attempt_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    result = audit(parser.parse_args().directory)
    print(json.dumps({key: result[key] for key in ('status', 'retained_terminal_samples',
                     'raw_attempt_receipts', 'nonterminal_http200_protocol_failures',
                     'additional_protocol_failures_after_recovery', 'interruption_accounting')}, indent=2))
