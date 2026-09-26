"""Offline native transport, immutable-profile, and paid-response recovery checks."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
import evaluate_chat_frontier_modebench as runner


def body(model='grok-4.3', n=1, text='correct'):
    return {'id': 'native_response_1', 'model': model, 'object': 'chat.completion',
            'choices': [{'index': i, 'message': {'role': 'assistant', 'content': text,
                        'reasoning_content': 'private reasoning is retained, never graded'},
                        'finish_reason': 'stop'} for i in range(n)],
            'usage': {'prompt_tokens': 8, 'completion_tokens': 1, 'total_tokens': 240,
                      'completion_tokens_details': {'reasoning_tokens': 231}}}


def fixture(tmp_path, monkeypatch, model='grok-4.3', n=1):
    profile = runner.default_profile(model)
    if n != 1:
        profile['request_parameters']['n'] = n
    messages = [{'role': 'system', 'content': 'Frozen system'}, {'role': 'user', 'content': 'Frozen puzzle'}]
    payload = runner.native_request(messages, profile)
    row = {'level': 1, 'domain': 'countdown', 'row_index': 0,
           'problem': 'Frozen puzzle', 'answer': 'Not sent to model', 'metadata': {}}
    gid = 'L1_countdown_000_0' if n == 1 else 'L1_countdown_000_n8'
    items = [{'level': 1, 'domain': 'countdown', 'row_index': 0, 'sample_index': i,
              'sample_id': f'L1_countdown_000_{i}', 'row_sha256': runner.sha(row),
              'request': payload, 'request_sha256': runner.sha(payload), 'group_id': gid,
              'choice_index': i if n > 1 else 0} for i in range(n)]
    group = {'group_id': gid, 'request': payload, 'request_sha256': runner.sha(payload),
             'sample_ids': [i['sample_id'] for i in items], 'sample_count': n, 'protocol': profile['protocol']}
    tmp_path.mkdir(exist_ok=True)
    for name, records in [('rows.jsonl', [row]), ('requests.jsonl', items), ('http_requests.jsonl', [group])]:
        runner.write_jsonl(tmp_path / name, records)
    runner.atomic(tmp_path / 'model_profile.json', profile)
    manifest = {'model': model, 'model_profile': profile, 'model_profile_sha256': runner.sha(profile),
                'samples_per_http_request': n, 'endpoint': profile['endpoint'],
                'reference_run': str(runner.REFERENCE.resolve()), 'code_sha256': {},
                'artifact_sha256': {name: runner.file_sha(tmp_path / name) for name in
                    ('rows.jsonl', 'requests.jsonl', 'http_requests.jsonl', 'model_profile.json')}}
    runner.atomic(tmp_path / 'manifest.json', manifest)
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'OFFLINE_TEST_CREDENTIAL')
    graded = []
    def grade(level, domain, supplied_row, text):
        graded.append(text)
        return {'verified': text == 'correct', 'canonical_key': 'mode1' if text == 'correct' else None, 'graded_text': text}
    monkeypatch.setattr(runner, 'load_frozen_grader', lambda output: grade)
    return profile, row, items, group, grade, graded


def raw(group, response=None):
    return {'group_id': group['group_id'], 'sample_ids': group['sample_ids'], 'attempt': 1,
            'request_sha256': group['request_sha256'], 'relative_path': f"raw_responses/{group['group_id']}__01.json",
            'http_status': 200, 'response': response or body(group['request']['model'], group['sample_count']),
            'latency_seconds': .25}


def install_client(monkeypatch, outcomes):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['follow_redirects'] is False
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            return None
        async def post(self, url, *, json):
            calls.append((url, json))
            assert outcomes, 'Unexpected HTTP call while recovering saved paid response'
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    return calls


def execute(path, model='grok-4.3', **kwargs):
    return asyncio.run(runner.run(path, model, workers=1, request_timeout=1, max_attempts=kwargs.pop('max_attempts', 1), rpm=0, **kwargs))


def test_content_only_excludes_reasoning_and_refusal_blocks():
    b = body()
    assert runner.response_text(b, 'chat_completions') == 'correct'
    b['choices'][0]['message']['content'] = [{'type': 'text', 'text': 'answer'}, {'type': 'reasoning', 'text': 'secret'}]
    assert runner.response_text(b, 'chat_completions') == 'answer'
    r = {'output': [{'type': 'reasoning', 'content': [{'type': 'output_text', 'text': 'secret'}]},
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'answer'}, {'type': 'refusal', 'refusal': 'x'}]}]}
    assert runner.response_text(r, 'responses') == 'answer'


def test_usage_preserves_grok_nonadditive_accounting(tmp_path, monkeypatch):
    _, row, items, group, grade, _ = fixture(tmp_path, monkeypatch)
    receipt = raw(group)
    record = runner.grade_receipt(items[0], group, receipt, row, grade)
    assert record['usage'] == {'input_tokens': 8, 'output_tokens': 1, 'total_tokens': 240}
    assert record['native_usage']['completion_tokens_details']['reasoning_tokens'] == 231
    assert record['temperature'] is None
    assert record['reasoning'] is None
    assert record['text'] == 'correct'
    assert 'messages' not in record['requested_settings']
    assert receipt['response']['choices'][0]['message']['reasoning_content']


@pytest.mark.parametrize('changes', [
    {'n': 2}, {'n': 8}, {'messages': []}, {'tools': []}, {'max_tokens': 0},
    {'max_completion_tokens': 8192}, {'stream': True},
])
def test_profile_rejects_unsupported_or_overriding_settings(changes):
    profile = runner.default_profile('grok-4.3')
    profile['request_parameters'].update(changes)
    with pytest.raises(ValueError):
        runner.validate_profile(profile)


def test_resume_profile_and_model_changes_fail_before_http(tmp_path, monkeypatch):
    profile, *_ = fixture(tmp_path, monkeypatch)
    calls = install_client(monkeypatch, [])
    changed = copy.deepcopy(profile)
    changed['request_parameters']['reasoning_effort'] = 'high'
    with pytest.raises(ValueError, match='profile'):
        execute(tmp_path, profile=changed)
    with pytest.raises(ValueError, match='deployment'):
        execute(tmp_path, model='DeepSeek-V4-Pro')
    with pytest.raises(ValueError, match='different model'):
        runner.prepare(tmp_path, changed)
    assert not calls


def test_saved_paid_response_recovers_without_http(tmp_path, monkeypatch):
    _, _, _, group, _, graded = fixture(tmp_path, monkeypatch)
    receipt = raw(group)
    runner.atomic(tmp_path / receipt['relative_path'], receipt)
    calls = install_client(monkeypatch, [])
    assert execute(tmp_path) == 0
    assert graded == ['correct'] and calls == []
    records = runner.read_jsonl(tmp_path / 'samples.jsonl')
    assert len(records) == 1 and records[0]['verified']
    assert execute(tmp_path) == 0  # Reuses authenticated atomic grade.
    assert graded == ['correct'] and calls == []


def test_native_choice_mismatch_is_retained_and_fails_closed(tmp_path, monkeypatch):
    _, _, _, group, _, graded = fixture(tmp_path, monkeypatch, 'FW-Kimi-K3', 1)
    calls = install_client(monkeypatch, [httpx.Response(200, json=body('FW-Kimi-K3', 8))])
    assert execute(tmp_path, 'FW-Kimi-K3') == 2
    assert len(calls) == 1 and not graded
    assert len(list((tmp_path / 'raw_responses').glob('*.json'))) == 1
    calls = install_client(monkeypatch, [])
    assert execute(tmp_path, 'FW-Kimi-K3') == 2
    assert not calls


def test_native_length_finish_is_received_incomplete_not_retried(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    b = body(text='unfinished')
    b['choices'][0]['finish_reason'] = 'length'
    calls = install_client(monkeypatch, [httpx.Response(200, json=b)])
    assert execute(tmp_path) == 0
    record = runner.read_jsonl(tmp_path / 'samples.jsonl')[0]
    assert record['response_status'] == 'incomplete' and not record['verified']
    assert len(calls) == 1


def test_http_retry_receipts_and_native_bodies_are_durable(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'retry_delay', lambda *args: 0)
    b = body()
    calls = install_client(monkeypatch, [httpx.Response(429, json={'error': 'rate limited'}),
        httpx.Response(200, json=b, headers={'x-provider': 'native', 'set-cookie': 'sensitive'})])
    assert execute(tmp_path, max_attempts=2) == 0
    assert len(calls) == 2
    errors = runner.read_jsonl(tmp_path / 'errors.jsonl')
    assert errors[0]['http_status'] == 429
    receipts = [json.loads(p.read_text()) for p in sorted((tmp_path / 'raw_responses').glob('*.json'))]
    assert json.loads(receipts[1]['http_body_text']) == b
    assert receipts[1]['headers']['x-provider'] == 'native'
    assert receipts[1]['headers']['set-cookie'] == '[REDACTED]'


def test_nonretryable_model_mismatch_preserves_response_and_stops(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    calls = install_client(monkeypatch, [httpx.Response(200, json=body('wrong-deployment'))])
    assert execute(tmp_path, max_attempts=3) == 2
    assert len(calls) == 1
    assert list((tmp_path / 'raw_responses').glob('*.json'))


def test_corrupted_grade_fails_without_api_replacement(tmp_path, monkeypatch):
    _, row, items, group, grade, _ = fixture(tmp_path, monkeypatch)
    receipt = raw(group)
    runner.atomic(tmp_path / receipt['relative_path'], receipt)
    record = runner.grade_receipt(items[0], group, receipt, row, grade)
    record['text'] = 'tampered'
    runner.atomic(tmp_path / 'sample_receipts' / (items[0]['sample_id'] + '.json'), record)
    calls = install_client(monkeypatch, [])
    with pytest.raises(ValueError, match='native response'):
        execute(tmp_path)
    assert not calls


def test_responses_protocol_keeps_native_reasoning_usage(tmp_path, monkeypatch):
    _, _, _, _, _, graded = fixture(tmp_path, monkeypatch, 'gpt-5.4')
    response = {'id': 'response_gpt54', 'model': 'gpt-5.4', 'status': 'completed',
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'correct'}]}],
                'reasoning': {'effort': 'medium'}, 'temperature': 1.0, 'top_p': .98,
                'usage': {'input_tokens': 12, 'output_tokens': 9, 'total_tokens': 21,
                          'output_tokens_details': {'reasoning_tokens': 4}}}
    calls = install_client(monkeypatch, [httpx.Response(200, json=response)])
    assert execute(tmp_path, 'gpt-5.4') == 0
    record = runner.read_jsonl(tmp_path / 'samples.jsonl')[0]
    assert record['usage'] == response['usage']
    assert record['reasoning'] == {'effort': 'medium'} and graded == ['correct']
    assert calls[0][0].endswith('/responses')
    assert calls[0][1]['store'] is False
