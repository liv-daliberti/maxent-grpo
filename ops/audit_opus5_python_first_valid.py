#!/usr/bin/env python3
"""Independently authenticate the separate first-valid Python sampling condition."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from audit_hosted_modebench_completion import atomic, file_sha, load_inventory, read_jsonl, sha, validate_native_records


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same(left, right):
    return sha(left) == sha(right)


def audit(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    marker = json.loads((directory / 'analysis_complete.json').read_text())
    if (manifest['condition'] != 'opus5_python_plain_first_valid_v1' or marker['status'] != 'complete'
            or marker['summary_sha256'] != file_sha(directory / 'summary.json')
            or summary['manifest_sha256'] != file_sha(directory / 'manifest.json')):
        raise ValueError('Incomplete or changed first-valid condition')
    source = Path(manifest['source_run'])
    parent_marker = json.loads((source / 'analysis_complete.json').read_text())
    if (file_sha(source / 'analysis_complete.json') != manifest['source_analysis_marker_sha256']
            or file_sha(source / 'manifest.json') != manifest['source_manifest_sha256']
            or file_sha(source / 'normalized_samples.jsonl') != manifest['source_normalized_samples_sha256']):
        raise ValueError('Source condition changed since retry registration')
    for relative, expected in {**parent_marker['evidence_sha256'], **parent_marker['output_sha256']}.items():
        if file_sha(source / relative) != expected:
            raise ValueError('Original source evidence changed: ' + relative)
    for relative, expected in json.loads((source / 'postprocessing_source_manifest.json').read_text())['source_sha256'].items():
        if file_sha(source / relative) != expected:
            raise ValueError('Frozen source analysis code changed: ' + relative)
    source_inventory = load_inventory(source, expected_samples=3072)
    requests = source_inventory['expected']
    original = {record['sample_id']: record for record in read_jsonl(source / 'samples.jsonl')}
    parent_raw = validate_native_records(source_inventory, list(original.values()))
    audited = {record['sample_id']: record for record in read_jsonl(source / 'audited_primary_samples.jsonl')}
    normalized_cache = {record['strict_receipt_sha256']: record for record in read_jsonl(source / 'normalized_samples.jsonl')}
    normalizer_path = source / 'secondary_code/ops/frontier_modebench_normalization.py'
    normalizer_hash = file_sha(normalizer_path)
    if normalizer_hash != manifest['normalizer_sha256']:
        raise ValueError('Registered normalizer changed')
    initial = {}
    slots = []
    for sid, item in requests.items():
        strict = {**original[sid], **{name: audited[sid][name] for name in ('verified', 'canonical_key', 'graded_text')}}
        cached = normalized_cache[sha(strict)]
        if cached['normalization_source_sha256'] != normalizer_hash:
            raise ValueError('Source normalization belongs to a different rule set')
        norm = cached['normalization']
        initial[sid] = strict, norm
        slots.append({'sample_id': sid, 'request_sha256': item['request_sha256'], 'row_sha256': item['row_sha256'],
                      'initial_raw_receipt': original[sid]['raw_receipt'],
                      'initial_raw_file_sha256': file_sha(source / original[sid]['raw_receipt']),
                      'initial_record': strict, 'initial_normalization': norm, 'initial_accepted': norm['verified']})
    if (len(slots) != 3072 or sha(slots) != manifest['source_slots_sha256']
            or not same(slots, read_jsonl(directory / 'slot_inventory.jsonl'))
            or file_sha(directory / 'rows.jsonl') != file_sha(source / 'rows.jsonl')):
        raise ValueError('First-valid slot registration differs from source reconstruction')
    failed = {sid for sid, (_, norm) in initial.items() if not norm['verified']}
    if len(failed) != 3 or any(initial[sid][0]['stop_reason'] != 'refusal' for sid in failed):
        raise ValueError('Retry selection differs from the three original failed slots')
    runner = directory / 'code/retry_opus5_python_until_valid.py'
    if file_sha(runner) != manifest['retry_runner_sha256']:
        raise ValueError('Frozen retry runner changed')
    rows = source_inventory['rows']
    sys.path.insert(0, str(source / 'code/src'))
    sys.path.insert(0, str(source / 'code/ops'))
    from frontier_modebench_contract import grade_response
    native = source_inventory['runner']
    native.warm_python_worker()
    normalizer = load_module('_first_valid_audit_normalizer', normalizer_path)
    statistics = load_module('_first_valid_audit_statistics', source / 'secondary_code/ops/summarize_frontier_modebench.py')
    starts = {}
    for event in read_jsonl(directory / 'events.jsonl'):
        if event.get('event') != 'request_started':
            continue
        identity = event['sample_id'], event['attempt']
        if identity in starts or identity[0] not in failed or not 2 <= identity[1] <= 33:
            raise ValueError('Duplicate or unregistered retry request start')
        if event['request_sha256'] != requests[identity[0]]['request_sha256']:
            raise ValueError('Retry request-start payload binding differs')
        starts[identity] = event
    raw_attempts = {}
    grades = {}
    response_ids = {record['response_id'] for record in original.values()}
    statuses = Counter()
    for path in sorted((directory / 'raw_responses').glob('*.json')):
        raw = json.loads(path.read_text())
        identity = raw['sample_id'], raw['attempt']
        sid, attempt = identity
        relative = str(path.relative_to(directory))
        if (identity in raw_attempts or identity not in starts or sid not in failed
                or relative != f'raw_responses/{sid}__{attempt:02d}.json'
                or raw['relative_path'] != relative or raw['request'] != requests[sid]['request']
                or raw['request_sha256'] != requests[sid]['request_sha256']
                or raw['started_at_utc'] != starts[identity]['at_utc']
                or datetime.fromisoformat(raw['received_at_utc']) < datetime.fromisoformat(raw['started_at_utc'])):
            raise ValueError('Retry raw receipt does not bind to its registered request')
        if file_sha(path) != summary['raw_evidence_sha256'].get(relative):
            raise ValueError('Retry raw receipt changed since summary')
        raw_attempts[identity] = raw
        statuses[str(raw.get('http_status') or raw.get('error_type'))] += 1
        if raw.get('http_status') != 200:
            continue
        native.validate_raw_receipt(requests[sid], raw, relative_path=relative)
        response_id = raw['response']['id']
        if response_id in response_ids:
            raise ValueError('Response identity reused across original/retry attempts')
        response_ids.add(response_id)
        grade_path = directory / 'attempt_grades' / f'{sid}__{attempt:02d}.json'
        saved = json.loads(grade_path.read_text())
        if (saved['sample_id'] != sid or saved['attempt'] != attempt or saved['raw_file_sha256'] != file_sha(path)
                or saved['normalizer_sha256'] != normalizer_hash):
            raise ValueError('Retry grade source/raw binding differs')
        native.validate_completed_record(directory, requests[sid], saved['record'])
        item = requests[sid]
        row = rows[item['level'], item['domain'], item['row_index']]
        regraded = native.grade_receipt(item, raw, row, grade_response)
        fields = ('verified', 'canonical_key', 'graded_text')
        if not same({name: saved['record'][name] for name in fields}, {name: regraded[name] for name in fields}):
            raise ValueError('Independent strict retry grade differs')
        norm = normalizer.normalize_and_grade(row, regraded['text'], strict_grade=regraded, grader=grade_response)
        if not same(norm, saved['normalization']):
            raise ValueError('Independent normalized retry grade differs')
        confirmation = grade_response(row['level'], row['domain'], row, norm['graded_text'])
        if not same({name: confirmation[name] for name in fields},
                    {name: saved['independent_repeat_grade'][name] for name in fields}):
            raise ValueError('Independent executable-verifier confirmation differs')
        grades[identity] = saved
    if set(starts) != set(raw_attempts):
        raise ValueError('Request start lacks a retained raw outcome')
    expected_grade_files = {f'{sid}__{attempt:02d}.json' for sid, attempt in grades}
    if {path.name for path in (directory / 'attempt_grades').glob('*.json')} != expected_grade_files:
        raise ValueError('Unexpected or missing retry grade receipt')
    selections = read_jsonl(directory / 'selected_samples.jsonl')
    if (len(selections) != 3072 or {item['sample_id'] for item in selections} != set(requests)
            or file_sha(directory / 'selected_samples.jsonl') != summary['selected_samples_sha256']):
        raise ValueError('Selected sample inventory differs')
    first_valid = {}
    for sid in failed:
        attempts = sorted(attempt for sample_id, attempt in raw_attempts if sample_id == sid)
        if not attempts or attempts != list(range(2, max(attempts) + 1)):
            raise ValueError('Missing physical attempt in registered retry order')
        valid = [attempt for attempt in attempts if (sid, attempt) in grades and grades[sid, attempt]['normalization']['verified']]
        if len(valid) != 1 or valid[0] != max(attempts):
            raise ValueError('Collection did not stop at the first observed valid retry')
        first_valid[sid] = valid[0]
    grouped = defaultdict(list)
    for selected in selections:
        sid = selected['sample_id']
        initial_record, initial_norm = initial[sid]
        if sid in failed:
            grade = grades[sid, first_valid[sid]]
            expected = {'sample_id': sid, 'source': 'retry', 'attempt': first_valid[sid], 'record': grade['record'],
                        'normalization': grade['normalization'], 'evidence_root': str(directory)}
        else:
            expected = {'sample_id': sid, 'source': 'initial', 'attempt': 1, 'record': initial_record,
                        'normalization': initial_norm, 'evidence_root': str(source)}
        if not same(selected, expected):
            raise ValueError('Selected record differs from first-valid reconstruction')
        record = {**selected['record'], **{name: selected['normalization'][name] for name in ('verified', 'canonical_key', 'graded_text')}}
        grouped[record['level'], record['domain'], record['row_index']].append(record)
    if len(grouped) != 384 or any(len(group) != 8 or {r['sample_index'] for r in group} != set(range(8)) for group in grouped.values()):
        raise ValueError('Selected condition lacks complete eight-draw prompts')
    rng = np.random.default_rng(20260911)
    for level in (1, 2, 3):
        stats = [statistics.prompt_statistics(row, grouped[identity]) for identity, row in rows.items() if identity[0] == level]
        cell, _ = statistics.bootstrap_cell(stats, 2000, rng)
        if not same(cell, summary['levels'][str(level)]):
            raise ValueError('Independent first-valid occupancy statistics differ')
    if (summary['valid_selected_draws'] != 3072 or summary['initial_valid_draws'] != 3069
            or summary['new_physical_attempts'] != len(raw_attempts) or summary['new_terminal_responses'] != len(grades)
            or summary['accepted_new_responses'] != 3
            or summary['new_failed_terminal_responses'] != sum(not grade['normalization']['verified'] for grade in grades.values())):
        raise ValueError('Reported attempt or selection counts differ from reconstruction')
    evidence = {relative: file_sha(directory / relative) for relative in
                ['manifest.json', 'slot_inventory.jsonl', 'rows.jsonl', 'events.jsonl', 'selected_samples.jsonl', 'summary.json', 'analysis_complete.json']}
    evidence.update({str(path.relative_to(directory)): file_sha(path) for folder in ('raw_responses', 'attempt_grades', 'code')
                     for path in (directory / folder).glob('*.json' if folder != 'code' else '*.py')})
    result = {'schema': 'opus5-first-valid-independent-audit-v1', 'status': 'pass',
              'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'api_calls': 0,
              'auditor_sha256': file_sha(Path(__file__)), 'original_source_slots_authenticated': 3072,
              'valid_selected_slots': 3072, 'complete_prompts': 384, 'initial_valid_slots_reused': 3069,
              'retry_slots_selected': 3, 'new_physical_attempts': len(raw_attempts), 'new_terminal_responses': len(grades),
              'attempt_status_counts': dict(statuses), 'first_valid_retry_attempt_by_slot': first_valid,
              'all_original_evidence_unchanged': True, 'all_retained_retries_independently_regraded': True,
              'all_level_statistics_and_bootstraps_reconstructed': True, 'evidence_sha256': evidence,
              'interpretation': 'Eight selected valid outputs per prompt, with variable request effort. This is validity-conditioned breadth, not unconditional accuracy or fixed-budget sampling.'}
    atomic(directory / 'independent_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    result = audit(parser.parse_args().directory)
    print(json.dumps({key: result[key] for key in ('status', 'valid_selected_slots', 'new_physical_attempts',
                     'new_terminal_responses', 'first_valid_retry_attempt_by_slot')}, indent=2))
