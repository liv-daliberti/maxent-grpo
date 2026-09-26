#!/usr/bin/env python3
"""Separate first-valid sampling condition for the revised Opus 5 Python run.

Retain the initial 3,072 draws as immutable evidence. Reuse initial valid draws,
then obtain the first frozen-normalizer/verifier-valid answer for each of the
three remaining slots. Every failed attempt remains saved. This condition
measures validity-conditioned breadth, not unconditional model accuracy.
"""
from __future__ import annotations
from collections import Counter
import fcntl
import getpass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/frontier_modebench_claude_opus5_python_plain_20260911'
OUTPUT = ROOT / 'artifacts/frontier_modebench_opus5_python_first_valid_20260911'
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_chat_frontier_modebench import atomic, append, file_sha, now, sha


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def key(item):
    return item['level'], item['domain'], item['row_index'], item['sample_index']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def source_inventory():
    marker = json.loads((SOURCE / 'analysis_complete.json').read_text())
    if marker['status'] != 'complete' or marker['received_responses'] != 3072:
        raise ValueError('The complete revised prompt condition must be audited first')
    for name, digest in {**marker['evidence_sha256'], **marker['output_sha256']}.items():
        if file_sha(SOURCE / name) != digest:
            raise ValueError('Source evidence or analysis changed: ' + name)
    analysis_sources = json.loads((SOURCE / 'postprocessing_source_manifest.json').read_text())
    for name, digest in analysis_sources['source_sha256'].items():
        if file_sha(SOURCE / name) != digest:
            raise ValueError('Frozen postprocessing source changed: ' + name)
    normalizer_sha256 = file_sha(SOURCE / 'secondary_code/ops/frontier_modebench_normalization.py')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for name, digest in manifest['code_sha256'].items():
        if file_sha(SOURCE / 'code' / name) != digest:
            raise ValueError('Frozen source changed')
    requests = {r['sample_id']: r for r in read_jsonl(SOURCE / 'requests.jsonl')}
    raw = {r['sample_id']: r for r in read_jsonl(SOURCE / 'samples.jsonl')}
    audited = {r['sample_id']: r for r in read_jsonl(SOURCE / 'audited_primary_samples.jsonl')}
    normalized = {key(r): r for r in read_jsonl(SOURCE / 'normalized_samples.jsonl')}
    rows = {(r['level'], r['domain'], r['row_index']): r for r in read_jsonl(SOURCE / 'rows.jsonl')}
    if not (len(requests) == len(raw) == len(audited) == len(normalized) == 3072 and len(rows) == 384):
        raise ValueError('Incorrect parent condition size')
    evidence_hashes = json.loads((SOURCE / 'evidence_file_sha256.json').read_text())
    slots = []
    for sid, request in requests.items():
        if file_sha(SOURCE / raw[sid]['raw_receipt']) != evidence_hashes[raw[sid]['raw_receipt']]:
            raise ValueError('Original raw response changed')
        strict = {**raw[sid], **{name: audited[sid][name] for name in ['verified', 'canonical_key', 'graded_text']}}
        norm = normalized[key(strict)]
        if (norm['strict_receipt_sha256'] != sha(strict)
                or norm['normalization_source_sha256'] != normalizer_sha256
                or request['request_sha256'] != sha(request['request'])):
            raise ValueError('Parent normalization or request binding differs')
        slots.append({'sample_id': sid, 'request_sha256': request['request_sha256'],
                      'row_sha256': request['row_sha256'], 'initial_raw_receipt': raw[sid]['raw_receipt'],
                      'initial_raw_file_sha256': file_sha(SOURCE / raw[sid]['raw_receipt']),
                      'initial_record': strict, 'initial_normalization': norm['normalization'],
                      'initial_accepted': norm['normalization']['verified']})
    failed = [slot for slot in slots if not slot['initial_accepted']]
    if len(failed) != 3 or any(slot['initial_record']['stop_reason'] != 'refusal' for slot in failed):
        raise ValueError('This registered condition is restricted to the three remaining revised-Python refusals')
    return manifest, requests, rows, slots


def validate_attempt_receipt(item, receipt, relative):
    if (receipt.get('sample_id') != item['sample_id'] or receipt.get('relative_path') != relative
            or receipt.get('request_sha256') != item['request_sha256']
            or receipt.get('request') != item['request']):
        raise ValueError('Retry attempt request identity/payload changed')
    stem = Path(relative).stem
    sid, separator, attempt = stem.rpartition('__')
    if (Path(relative).parent != Path('raw_responses') or not separator or sid != item['sample_id']
            or not attempt.isdigit() or receipt.get('attempt') != int(attempt) or not 2 <= int(attempt) <= 33):
        raise ValueError('Retry physical attempt identity changed')


def validate_existing_attempts(requests, slots):
    failed = {slot['sample_id'] for slot in slots if not slot['initial_accepted']}
    starts = {}
    for event in read_jsonl(OUTPUT / 'events.jsonl'):
        if event.get('event') != 'request_started':
            continue
        identity = event['sample_id'], event['attempt']
        if identity in starts or identity[0] not in failed or not 2 <= identity[1] <= 33:
            raise ValueError('Duplicate or unregistered retry request start')
        if event.get('request_sha256') != requests[identity[0]]['request_sha256']:
            raise ValueError('Retry start request binding changed')
        starts[identity] = event
    received = set()
    for path in (OUTPUT / 'raw_responses').glob('*.json'):
        receipt = json.loads(path.read_text())
        sid, attempt = receipt.get('sample_id'), receipt.get('attempt')
        if sid not in failed or (sid, attempt) in received:
            raise ValueError('Unexpected or duplicated saved retry receipt')
        validate_attempt_receipt(requests[sid], receipt, str(path.relative_to(OUTPUT)))
        event = starts.get((sid, attempt))
        if event is None or receipt.get('started_at_utc') != event['at_utc']:
            raise ValueError('Saved retry receipt lacks its timestamp-bound request start')
        received.add((sid, attempt))
    if set(starts) != received:
        raise ValueError('Interrupted retry request has no saved outcome; preserve the unknown attempt and review it before resuming')
    return set(starts)


def main():
    import httpx
    import numpy as np
    source_manifest, requests, rows, slots = source_inventory()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.runner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = OUTPUT / 'code' / Path(__file__).name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists() and file_sha(frozen) != file_sha(__file__):
            raise ValueError('Retry-condition source changed')
        if not frozen.exists():
            shutil.copyfile(__file__, frozen)
        normalizer_path = SOURCE / 'secondary_code/ops/frontier_modebench_normalization.py'
        manifest = {'schema': 'frontier-first-valid-condition-v1', 'condition': 'opus5_python_plain_first_valid_v1',
                    'model': 'claude-opus-5', 'source_run': str(SOURCE),
                    'source_manifest_sha256': file_sha(SOURCE / 'manifest.json'),
                    'source_analysis_marker_sha256': file_sha(SOURCE / 'analysis_complete.json'),
                    'source_normalized_samples_sha256': file_sha(SOURCE / 'normalized_samples.jsonl'),
                    'normalizer_sha256': file_sha(normalizer_path), 'source_code_sha256': source_manifest['code_sha256'],
                    'retry_runner_sha256': file_sha(frozen), 'source_slots_sha256': sha(slots),
                    'initial_draws': 3072, 'initial_valid_draws': 3069, 'initial_failed_slots': 3,
                    'target_valid_draws_per_prompt': 8, 'prompt_count': 384,
                    'success_rule': 'Previously frozen formatting normalization and unchanged executable verifier',
                    'selection_rule': 'Retain each initial valid draw; otherwise accept the first subsequent valid draw for that exact slot.',
                    'prompt_feedback': False, 'request_payload_changes': [], 'training': False,
                    'max_new_physical_attempts_per_failed_slot': 32,
                    'interpretation': 'Validity-conditioned sampling with variable request effort; accepted-set correctness is fixed by design and is not an unconditional accuracy estimate.'}
        manifest_path = OUTPUT / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Registered first-valid condition changed')
        if not manifest_path.exists():
            atomic(manifest_path, manifest)
            with (OUTPUT / 'slot_inventory.jsonl').open('x') as handle:
                for slot in slots:
                    handle.write(json.dumps(slot, sort_keys=True) + '\n')
            shutil.copyfile(SOURCE / 'rows.jsonl', OUTPUT / 'rows.jsonl')
        if read_jsonl(OUTPUT / 'slot_inventory.jsonl') != slots or file_sha(OUTPUT / 'rows.jsonl') != file_sha(SOURCE / 'rows.jsonl'):
            raise ValueError('Saved first-valid slot inventory or rows changed')
        started_attempts = validate_existing_attempts(requests, slots)
        if (OUTPUT / 'analysis_complete.json').exists():
            print((OUTPUT / 'analysis_complete.json').read_text())
            return
        sys.path.insert(0, str(SOURCE / 'code/src'))
        sys.path.insert(0, str(SOURCE / 'code/ops'))
        from frontier_modebench_contract import grade_response
        native = module('_retry_native_claude', SOURCE / 'code/ops/evaluate_claude_modebench.py')
        normalizer = module('_retry_frozen_normalizer', normalizer_path)
        statistics = module('_retry_frozen_statistics', SOURCE / 'secondary_code/ops/summarize_frontier_modebench.py')
        append(OUTPUT / 'events.jsonl', native.warm_python_worker())
        api_key = os.environ.get('AZURE_OPENAI_API_KEY') or getpass.getpass('Azure API key (hidden): ')
        if not api_key:
            raise ValueError('Missing credential')
        (OUTPUT / 'raw_responses').mkdir(exist_ok=True)
        (OUTPUT / 'attempt_grades').mkdir(exist_ok=True)
        selections = {}
        all_ids = {slot['initial_record']['response_id'] for slot in slots}
        with httpx.Client(headers={'x-api-key': api_key, 'anthropic-version': '2023-06-01'},
                          timeout=httpx.Timeout(300, connect=30), follow_redirects=False) as client:
            for slot in slots:
                sid = slot['sample_id']
                if slot['initial_accepted']:
                    selections[sid] = {'source': 'initial', 'attempt': 1, 'record': slot['initial_record'],
                                       'normalization': slot['initial_normalization'], 'evidence_root': str(SOURCE)}
                    continue
                item = requests[sid]
                row = rows[item['level'], item['domain'], item['row_index']]
                for attempt in range(2, 34):
                    relative = f'raw_responses/{sid}__{attempt:02d}.json'
                    path = OUTPUT / relative
                    if path.exists():
                        receipt = json.loads(path.read_text())
                    else:
                        receipt = {'sample_id': sid, 'attempt': attempt, 'request': item['request'],
                                   'request_sha256': item['request_sha256'], 'relative_path': relative,
                                   'started_at_utc': now()}
                        if (sid, attempt) in started_attempts:
                            raise ValueError('Attempt already started without a reusable receipt; do not repeat an unknown call')
                        started_attempts.add((sid, attempt))
                        append(OUTPUT / 'events.jsonl', {'event': 'request_started', 'sample_id': sid,
                               'attempt': attempt, 'at_utc': receipt['started_at_utc'], 'request_sha256': item['request_sha256']})
                        start = time.monotonic()
                        try:
                            response = client.post(source_manifest['endpoint'], json=item['request'])
                            try:
                                body = response.json()
                            except ValueError:
                                body = {'non_json_response': response.text}
                            receipt.update(http_status=response.status_code, response=body, http_body_text=response.text,
                                           headers={k: '[REDACTED]' if k.lower() in {'authorization', 'api-key', 'x-api-key', 'set-cookie'} else v
                                                    for k, v in response.headers.items()})
                        except (httpx.TimeoutException, httpx.TransportError) as error:
                            receipt.update(http_status=None, error_type=type(error).__name__, error=str(error), possibly_billed=True)
                        receipt.update(latency_seconds=time.monotonic() - start, received_at_utc=now())
                        receipt = json.loads(json.dumps(receipt).replace(api_key, '[REDACTED]'))
                        atomic(path, receipt)
                    validate_attempt_receipt(item, receipt, relative)
                    if receipt.get('http_status') != 200:
                        if receipt.get('http_status') not in (None, 408, 409, 425, 429, 500, 502, 503, 504):
                            raise RuntimeError('Nonretryable API failure; see ' + str(path))
                        time.sleep(min(attempt, 10))
                        continue
                    native.validate_raw_receipt(item, receipt, relative_path=relative)
                    record = native.grade_receipt(item, receipt, row, grade_response)
                    normalized = normalizer.normalize_and_grade(row, record['text'], strict_grade=record, grader=grade_response)
                    confirmation = grade_response(row['level'], row['domain'], row, normalized['graded_text'])
                    if (confirmation['verified'], confirmation['canonical_key']) != (normalized['verified'], normalized['canonical_key']):
                        raise ValueError('Frozen verifier was unstable; preserve attempt for independent review')
                    grade = {'sample_id': sid, 'attempt': attempt, 'record': record,
                             'normalization': normalized, 'independent_repeat_grade': confirmation,
                             'raw_file_sha256': file_sha(path), 'normalizer_sha256': file_sha(normalizer_path)}
                    grade_path = OUTPUT / 'attempt_grades' / (sid + f'__{attempt:02d}.json')
                    if grade_path.exists():
                        saved = json.loads(grade_path.read_text())
                        if any(saved.get(name) != grade[name] for name in ['sample_id', 'attempt', 'raw_file_sha256', 'normalizer_sha256']):
                            raise ValueError('Saved retry grade source/receipt binding changed')
                        native.validate_completed_record(OUTPUT, item, saved['record'])
                        if (any(saved['record'].get(name) != record.get(name) for name in ['verified', 'canonical_key', 'graded_text'])
                                or saved['normalization'] != normalized
                                or any(saved['independent_repeat_grade'].get(name) != confirmation.get(name)
                                       for name in ['verified', 'canonical_key', 'graded_text'])):
                            raise ValueError('Saved retry grade differs from independent regrade; retain it for review')
                        record, normalized = saved['record'], saved['normalization']
                    else:
                        atomic(grade_path, grade)
                    if record['response_id'] in all_ids:
                        raise ValueError('Provider reused a response identity')
                    all_ids.add(record['response_id'])
                    if normalized['verified']:
                        selections[sid] = {'source': 'retry', 'attempt': attempt, 'record': record,
                                           'normalization': normalized, 'evidence_root': str(OUTPUT)}
                        break
                if sid not in selections:
                    raise RuntimeError('Registered retry limit reached without a valid answer for ' + sid)
                atomic(OUTPUT / 'progress.json', {'updated_at_utc': now(), 'valid_slots_processed': len(selections), 'expected_slots': 3072})
        if len(selections) != 3072:
            raise ValueError('The validity-conditioned inventory is incomplete')
        with (OUTPUT / 'selected_samples.jsonl').open('w') as handle:
            for sid in requests:
                handle.write(json.dumps({'sample_id': sid, **selections[sid]}, sort_keys=True) + '\n')
        grouped = {}
        for selected in selections.values():
            record = {**selected['record'], **{name: selected['normalization'][name] for name in ['verified', 'canonical_key', 'graded_text']}}
            grouped.setdefault((record['level'], record['domain'], record['row_index']), []).append(record)
        rng = np.random.default_rng(20260911)
        levels = {}
        for level in [1, 2, 3]:
            stats = [statistics.prompt_statistics(row, grouped[identity]) for identity, row in rows.items() if identity[0] == level]
            cell, _ = statistics.bootstrap_cell(stats, 2000, rng)
            levels[str(level)] = cell
        raw_paths = sorted((OUTPUT / 'raw_responses').glob('*.json'))
        retry_grades = [json.loads(path.read_text()) for path in sorted((OUTPUT / 'attempt_grades').glob('*.json'))]
        source_inventory()  # Recheck that every original source and audit remains unchanged.
        report = {'status': 'complete', 'condition': manifest['condition'], 'completed_at_utc': now(),
                  'initial_draws': 3072, 'initial_valid_draws': 3069, 'valid_selected_draws': 3072,
                  'new_physical_attempts': len(raw_paths), 'new_terminal_responses': len(retry_grades),
                  'new_failed_terminal_responses': sum(not r['normalization']['verified'] for r in retry_grades),
                  'accepted_new_responses': sum(s['source'] == 'retry' for s in selections.values()),
                  'accepted_set_accuracy_note': '100% after frozen normalization by selection rule; not unconditional model accuracy.',
                  'levels': levels, 'all_original_evidence_unchanged': True,
                  'manifest_sha256': file_sha(manifest_path), 'selected_samples_sha256': file_sha(OUTPUT / 'selected_samples.jsonl'),
                  'raw_evidence_sha256': {str(path.relative_to(OUTPUT)): file_sha(path) for path in raw_paths}}
        atomic(OUTPUT / 'summary.json', report)
        atomic(OUTPUT / 'analysis_complete.json', {'status': 'complete', 'completed_at_utc': now(),
                'valid_selected_draws': 3072, 'new_physical_attempts': len(raw_paths), 'summary_sha256': file_sha(OUTPUT / 'summary.json'),
                'independent_audit': 'pending separate verification of source selection and retry attempts'})
        print(json.dumps({k: report[k] for k in ['status', 'valid_selected_draws', 'new_physical_attempts', 'new_failed_terminal_responses', 'accepted_new_responses']}, indent=2))


if __name__ == '__main__':
    main()
