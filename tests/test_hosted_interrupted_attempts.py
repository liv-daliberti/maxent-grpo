"""Integrity guards for explicitly registered interrupted evaluation requests."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
from audit_hosted_interrupted_attempts import sha, validate_attempt_accounting


def fixture():
    old = {'event': 'request_started', 'group_id': 'a', 'sample_ids': ['a'], 'attempt': 1,
           'request_sha256': 'payload', 'at_utc': '2026-09-11T16:20:00+00:00'}
    resume = {'event': 'session_started', 'at_utc': '2026-09-11T16:37:00+00:00'}
    new = {**old, 'at_utc': '2026-09-11T16:37:01+00:00'}
    events = [old, resume, new]
    raw = {('a', 1): {'group_id': 'a', 'sample_ids': ['a'], 'attempt': 1, 'request_sha256': 'payload',
                     'started_at_utc': '2026-09-11T16:37:01.010000+00:00', 'received_at_utc': '2026-09-11T16:38:00+00:00'}}
    registration = {'schema': 'hosted-interrupted-attempt-registration-v1', 'resume_event_index': 1,
                    'resume_event': resume, 'pre_resume_events_sha256': sha(events[:1]), 'orphan_attempt_count': 1,
                    'orphan_starts': [{'event_index': 0, 'event': copy.deepcopy(old), 'event_sha256': sha(old),
                                       'outcome': 'unknown', 'usage': 'unknown', 'possibly_billed': True}]}
    return events, raw, registration


def test_registered_orphan_and_timestamp_bound_replacement_pass():
    result = validate_attempt_accounting(*fixture())
    assert result['logged_physical_request_starts'] == result['accounted_physical_attempts'] == 2
    assert result['registered_interrupted_attempts_with_unknown_outcome'] == 1
    assert result['possibly_billed_unknown_attempts'] == 1


@pytest.mark.parametrize('change', ['prefix', 'orphan', 'resume', 'unknown_usage', 'duplicate_registration',
                                  'extra_start', 'extra_duplicate', 'old_raw', 'missing_raw', 'payload',
                                  'inventory', 'receipt_time', 'missing_replacement'])
def test_unaccounted_or_ambiguous_evidence_is_rejected(change):
    events, raw, registration = fixture()
    if change == 'prefix': events[0]['at_utc'] = '2026-09-11T16:19:00+00:00'
    elif change == 'orphan': registration['orphan_starts'][0]['event_sha256'] = 'changed'
    elif change == 'resume': registration['resume_event'] = {**registration['resume_event'], 'pid': 123}
    elif change == 'unknown_usage': registration['orphan_starts'][0]['usage'] = 0
    elif change == 'duplicate_registration': registration['orphan_starts'] *= 2
    elif change == 'extra_start': events.append({**events[-1], 'group_id': 'b'})
    elif change == 'extra_duplicate': events.append(dict(events[-1]))
    elif change == 'old_raw': raw[('a', 1)]['started_at_utc'] = '2026-09-11T16:20:01+00:00'
    elif change == 'missing_raw': raw.clear()
    elif change == 'payload': raw[('a', 1)]['request_sha256'] = 'different'
    elif change == 'inventory': raw[('a', 1)]['sample_ids'] = ['b']
    elif change == 'receipt_time': raw[('a', 1)]['received_at_utc'] = '2026-09-11T16:20:00+00:00'
    elif change == 'missing_replacement': events.pop()
    with pytest.raises(ValueError): validate_attempt_accounting(events, raw, registration)


@pytest.mark.parametrize('tamper', [None, 'registration', 'helper_source', 'unregistered_start'])
def test_frozen_completion_adapter_preserves_receipt_checks(tmp_path, tamper):
    import hashlib
    import importlib.util
    import json
    import shutil
    import test_hosted_modebench_completion as original

    root = Path(__file__).resolve().parents[1]
    source = root / 'artifacts/frontier_modebench_deepseek_v4_pro_20260911/interruption_analysis_code/ops'
    original.fixture(tmp_path, protocol='responses', count=2)
    code = tmp_path / 'interruption_analysis_code/ops'
    code.mkdir(parents=True)
    for name in ('audit_hosted_interrupted_attempts.py', 'audit_hosted_modebench_completion.py'):
        shutil.copy2(source / name, code / name)
    events = original.auditor.read_jsonl(tmp_path / 'events.jsonl')
    for event in events:
        raw_path = tmp_path / 'raw_responses' / f"{event['group_id']}__01.json"
        raw = json.loads(raw_path.read_text())
        event.update(request_sha256=raw['request_sha256'], sample_ids=raw['sample_ids'],
                     at_utc='2026-09-11T16:37:01+00:00')
        raw.update(started_at_utc='2026-09-11T16:37:01.010000+00:00',
                   received_at_utc='2026-09-11T16:38:00+00:00')
        raw_path.write_text(json.dumps(raw))
    samples = original.auditor.read_jsonl(tmp_path / 'samples.jsonl')
    for sample in samples:
        raw = json.loads((tmp_path / sample['raw_receipt']).read_text())
        sample['raw_receipt_sha256'] = sha(raw)
        (tmp_path / 'sample_receipts' / (sample['sample_id'] + '.json')).write_text(json.dumps(sample))
    original.write_jsonl(tmp_path / 'samples.jsonl', samples)
    old = {**events[0], 'at_utc': '2026-09-11T16:20:00+00:00'}
    resume = {'event': 'session_started', 'at_utc': '2026-09-11T16:37:00+00:00'}
    events = [old, resume, *events]
    registration = {'schema': 'hosted-interrupted-attempt-registration-v1', 'resume_event_index': 1,
                    'resume_event': resume, 'pre_resume_events_sha256': sha(events[:1]), 'orphan_attempt_count': 1,
                    'orphan_starts': [{'event_index': 0, 'event': old, 'event_sha256': sha(old),
                                       'outcome': 'unknown', 'usage': 'unknown', 'possibly_billed': True}]}
    registration_path = tmp_path / 'interrupted_attempt_registration.json'
    registration_path.write_text(json.dumps(registration))
    file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = {'schema': 'hosted-completion-audit-adapter-v1',
               'registration': registration_path.name, 'registration_sha256': file_sha(registration_path),
               'source_sha256': {str(path.relative_to(tmp_path)): file_sha(path) for path in code.glob('*.py')}}
    (tmp_path / 'completion_audit_adapter.json').write_text(json.dumps(adapter))
    if tamper == 'registration': registration_path.write_text(json.dumps({**registration, 'changed': True}))
    elif tamper == 'helper_source':
        with (code / 'audit_hosted_interrupted_attempts.py').open('a') as handle: handle.write('\n# changed\n')
    elif tamper == 'unregistered_start': events.append(dict(events[-1]))
    original.write_jsonl(tmp_path / 'events.jsonl', events)
    spec = importlib.util.spec_from_file_location('interrupted_completion_test', code / 'audit_hosted_modebench_completion.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tamper:
        with pytest.raises(ValueError): module.audit(tmp_path, expected_samples=2)
    else:
        result = module.audit(tmp_path, expected_samples=2)
        assert result['status'] == 'pass'
        assert result['saved_samples'] == result['unique_response_choice_ids'] == 2
        assert result['interruption_accounting']['logged_physical_request_starts'] == 3
        assert result['interruption_accounting']['registered_interrupted_attempts_with_unknown_outcome'] == 1
        import summarize_frontier_comparison as comparison
        inventory = json.loads((tmp_path / 'evidence_file_sha256.json').read_text())
        comparison.validate_interruption_provenance(tmp_path, result, inventory)
        registration_path.write_text(json.dumps({**registration, 'altered_after_audit': True}))
        with pytest.raises(ValueError, match='registration changed since audit'):
            comparison.validate_interruption_provenance(tmp_path, result, inventory)
