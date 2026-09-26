"""Independent offline integrity audit fixtures for native and grouped receipts."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import audit_hosted_modebench_completion as auditor
import evaluate_claude_modebench as claude
import evaluate_chat_frontier_modebench as chat


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))


def fixture(tmp_path, protocol='anthropic', count=2, domain='countdown'):
    native_chat = protocol != 'anthropic'
    runner = chat if native_chat else claude
    model = 'FW-Kimi-K3' if protocol == 'grouped' else 'gpt-5.4' if protocol == 'responses' else 'claude-opus-5'
    row = {'level': 1, 'domain': domain, 'row_index': 0, 'problem': 'Puzzle', 'answer': 'Private', 'metadata': {}}
    if domain == 'python_factors':
        row['answer'] = {'verifier': 'python_factor_function', 'python_version': 'factor-v1', 'cases': [6, 8]}
    messages = [{'role': 'system', 'content': 'Frozen'}, {'role': 'user', 'content': 'Puzzle'}]
    if native_chat:
        profile = chat.default_profile(model)
        if protocol == 'grouped':
            profile['request_parameters']['n'] = count
        # Historical grouped receipts remain auditable even when new grouped
        # generation is disabled by the live runner's profile validator.
        payload = ({'model': model, 'messages': messages, **profile['request_parameters']}
                   if protocol == 'grouped' else chat.native_request(messages, profile))
    else:
        payload = claude.native_request(messages, 8192, model)
    items = []
    for i in range(count):
        item = {'sample_id': f'L1_{domain}_000_{i}', 'level': 1, 'domain': domain,
                'row_index': 0, 'sample_index': i, 'row_sha256': auditor.sha(row),
                'request': payload, 'request_sha256': auditor.sha(payload)}
        if native_chat:
            item.update(group_id='group_n8' if protocol == 'grouped' else item['sample_id'],
                        choice_index=i if protocol == 'grouped' else 0)
        items.append(item)
    groups = {}
    for item in items:
        gid = item.get('group_id', item['sample_id'])
        groups.setdefault(gid, {'group_id': gid, 'request': payload, 'request_sha256': auditor.sha(payload),
                               'sample_ids': [], 'sample_count': 0,
                               'protocol': 'responses' if protocol == 'responses' else 'chat_completions'})
        groups[gid]['sample_ids'].append(item['sample_id'])
        groups[gid]['sample_count'] += 1
    (tmp_path / 'raw_responses').mkdir()
    (tmp_path / 'sample_receipts').mkdir()
    def grade(*args):
        if domain == 'python_factors':
            return {'verified': False, 'canonical_key': None, 'graded_text': args[-1]}
        return {'verified': True, 'canonical_key': 'mode1', 'graded_text': args[-1]}
    samples, events = [], []
    lookup = {r['sample_id']: r for r in items}
    for gid, group in groups.items():
        response = {'id': 'response_' + gid, 'model': model}
        if protocol == 'anthropic':
            response.update(type='message', role='assistant', stop_reason='end_turn',
                            content=[{'type': 'thinking', 'thinking': 'hidden'}, {'type': 'text', 'text': r'\boxed{lambda n: 2}' if domain == 'python_factors' else 'correct'}],
                            usage={'input_tokens': 10, 'output_tokens': 20})
        elif protocol == 'responses':
            response.update(status='completed', output=[{'type': 'message', 'content': [{'type': 'output_text', 'text': 'correct'}]}],
                            usage={'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30})
        else:
            response.update(choices=[{'index': i, 'message': {'role': 'assistant', 'content': 'correct'},
                                      'finish_reason': 'stop'} for i in range(group['sample_count'])],
                            usage={'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 99})
        relative = f'raw_responses/{gid}__01.json'
        raw = {'relative_path': relative, 'attempt': 1, 'request_sha256': auditor.sha(payload),
               'http_status': 200, 'response': response, 'latency_seconds': .1}
        key = 'group_id' if native_chat else 'sample_id'
        raw[key] = gid
        if native_chat:
            raw['sample_ids'] = group['sample_ids']
        auditor.atomic(tmp_path / relative, raw)
        events.append({'event': 'request_started', key: gid, 'attempt': 1})
        for sid in group['sample_ids']:
            item = lookup[sid]
            record = (chat.grade_receipt(item, group, raw, row, grade) if native_chat else
                      claude.grade_receipt(item, raw, row, grade))
            auditor.atomic(tmp_path / 'sample_receipts' / (sid + '.json'), record)
            samples.append(record)
    write_jsonl(tmp_path / 'rows.jsonl', [row])
    write_jsonl(tmp_path / 'requests.jsonl', items)
    write_jsonl(tmp_path / 'samples.jsonl', samples)
    write_jsonl(tmp_path / 'events.jsonl', events)
    names = ['rows.jsonl', 'requests.jsonl']
    if native_chat:
        write_jsonl(tmp_path / 'http_requests.jsonl', list(groups.values()))
        names.append('http_requests.jsonl')
    runner_name = 'ops/evaluate_chat_frontier_modebench.py' if native_chat else 'ops/evaluate_claude_modebench.py'
    runner_copy = tmp_path / 'code' / runner_name
    runner_copy.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / runner_name, runner_copy)
    manifest = {'schema': runner.SCHEMA, 'model': model, 'request_count': count,
                'http_request_count': len(groups),
                'artifact_sha256': {name: auditor.file_sha(tmp_path / name) for name in names},
                'code_sha256': {runner_name: auditor.file_sha(runner_copy)}}
    auditor.atomic(tmp_path / 'manifest.json', manifest)
    return items, samples, groups


@pytest.mark.parametrize('protocol,count', [('anthropic', 2), ('responses', 2), ('grouped', 8)])
def test_complete_native_run_and_grouped_choices_pass(tmp_path, protocol, count):
    fixture(tmp_path, protocol, count)
    result = auditor.audit(tmp_path, count)
    assert result['status'] == 'pass'
    assert result['saved_samples'] == result['unique_response_choice_ids'] == count
    assert result['unique_response_ids'] == (1 if protocol == 'grouped' else count)
    assert result['usage_counted_once_per_http_response']['total_tokens'] == (99 if protocol == 'grouped' else 60)
    assert (tmp_path / 'evidence_file_sha256.json').is_file()


def test_incomplete_export_never_gets_pass(tmp_path):
    _, samples, _ = fixture(tmp_path)
    write_jsonl(tmp_path / 'samples.jsonl', samples[:1])
    with pytest.raises(ValueError, match='incomplete'):
        auditor.audit(tmp_path, 2)
    assert not (tmp_path / 'completion_audit.json').exists()


def test_extra_atomic_receipt_is_detected(tmp_path):
    _, samples, _ = fixture(tmp_path)
    auditor.atomic(tmp_path / 'sample_receipts' / 'extra.json', samples[0])
    with pytest.raises(ValueError, match='Atomic sample receipt inventory'):
        auditor.audit(tmp_path, 2)


def test_unreceipted_request_start_is_detected(tmp_path):
    fixture(tmp_path)
    with (tmp_path / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps({'event': 'request_started', 'sample_id': 'L1_countdown_000_0', 'attempt': 2}) + '\n')
    with pytest.raises(ValueError, match='starts and raw'):
        auditor.audit(tmp_path, 2)


def test_duplicate_response_id_across_independent_calls_is_detected(tmp_path):
    _, samples, _ = fixture(tmp_path)
    second = samples[1]
    raw_path = tmp_path / second['raw_receipt']
    raw = json.loads(raw_path.read_text())
    raw['response']['id'] = samples[0]['response_id']
    auditor.atomic(raw_path, raw)
    second['response_id'] = raw['response']['id']
    second['raw_receipt_sha256'] = auditor.sha(raw)
    auditor.atomic(tmp_path / 'sample_receipts' / (second['sample_id'] + '.json'), second)
    write_jsonl(tmp_path / 'samples.jsonl', samples)
    with pytest.raises(ValueError, match='reused across independent'):
        auditor.audit(tmp_path, 2)


def test_frozen_request_drift_fails_before_receipt_validation(tmp_path):
    fixture(tmp_path)
    with (tmp_path / 'requests.jsonl').open('a') as handle:
        handle.write('{}\n')
    with pytest.raises(ValueError, match='Frozen artifact changed'):
        auditor.audit(tmp_path, 2)


def test_parallel_io_preserves_serial_evidence_inventory(tmp_path):
    fixture(tmp_path, protocol='grouped', count=8)
    serial = auditor.audit(tmp_path, 8, io_workers=1)
    parallel = auditor.audit(tmp_path, 8, io_workers=16)
    assert serial['evidence_inventory_sha256'] == parallel['evidence_inventory_sha256']
    for key in ('saved_samples', 'unique_response_ids', 'unique_response_choice_ids',
                'raw_attempts', 'usage_counted_once_per_http_response'):
        assert serial[key] == parallel[key]
