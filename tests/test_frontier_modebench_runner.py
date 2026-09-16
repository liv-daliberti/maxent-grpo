"""Offline persistence and API-recovery checks; never issue external requests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import evaluate_frontier_modebench as runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def run_fixture(tmp_path, monkeypatch):
    row = {'cell_id': 'level1/countdown', 'level': 1, 'domain': 'countdown',
           'row_index': 0, 'problem': 'Public puzzle', 'answer': 'Private answer',
           'metadata': {'answer_mode_count': 3}}
    payload = {'model': 'gpt-5.6-sol', 'input': [{'role': 'user', 'content': 'Public puzzle'}]}
    item = {'level': 1, 'domain': 'countdown', 'row_index': 0, 'sample_index': 0,
            'sample_id': 'L1_countdown_000_0', 'row_sha256': runner.sha(row),
            'request_sha256': runner.sha(payload), 'request': payload}
    for name, value in [('rows.jsonl', row), ('requests.jsonl', item), ('datasets.json', {})]:
        write_json(tmp_path / name, value)
    manifest = {'schema': runner.SCHEMA, 'max_output_tokens': 8192,
                'artifact_sha256': {name: runner.file_sha(tmp_path / name)
                                    for name in ['rows.jsonl', 'requests.jsonl', 'datasets.json']},
                'code_sha256': {}}
    write_json(tmp_path / 'manifest.json', manifest)
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'OFFLINE_TEST_CREDENTIAL')
    graded = []
    def grade(level, domain, supplied_row, text):
        graded.append((level, domain, supplied_row, text))
        return {'verified': text == 'correct', 'canonical_key': 'mode1' if text == 'correct' else None,
                'graded_text': text}
    monkeypatch.setitem(sys.modules, 'frontier_modebench_contract', SimpleNamespace(grade_response=grade))
    return row, item, graded


def body(model='gpt-5.6-sol', text='correct'):
    return {'id': 'resp_offline', 'status': 'completed', 'model': model,
            'output': [{'type': 'reasoning', 'summary': []},
                       {'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30}}


def raw_receipt(item, response=None):
    return {'sample_id': item['sample_id'], 'attempt': 1,
            'request_sha256': item['request_sha256'],
            'relative_path': f"raw_responses/{item['sample_id']}__01.json",
            'http_status': 200, 'response': response or body(), 'latency_seconds': 0.1}


def install_client(monkeypatch, outcomes):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            self.options = kwargs
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            return None
        async def post(self, url, *, json):
            calls.append((url, json))
            assert outcomes, 'Unexpected paid-call attempt in an offline recovery test'
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    return calls


def run_once(path, *, attempts=2):
    return asyncio.run(runner.run(path, workers=1, request_timeout=1, max_attempts=attempts, max_new=0))


def test_response_text_uses_only_output_text():
    response = {'output': [{'type': 'reasoning', 'content': [{'type': 'output_text', 'text': 'hidden'}]},
                           {'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'ignored'},
                                                           {'type': 'output_text', 'text': 'answer'}]}]}
    assert runner.response_text(response) == 'answer'
    assert runner.response_text({'output': []}) == ''


def test_prepare_rejects_frozen_dataset_drift(tmp_path):
    write_json(tmp_path / 'rows.jsonl', {'problem': 'Original'})
    write_json(tmp_path / 'manifest.json', {'max_output_tokens': 8192,
               'artifact_sha256': {'rows.jsonl': runner.file_sha(tmp_path / 'rows.jsonl')}})
    write_json(tmp_path / 'rows.jsonl', {'problem': 'Changed'})
    with pytest.raises(ValueError, match='Frozen artifact changed: rows.jsonl'):
        runner.prepare(tmp_path, 8192)


def test_prepare_rejects_changed_output_budget(tmp_path):
    write_json(tmp_path / 'manifest.json', {'max_output_tokens': 8192, 'artifact_sha256': {}})
    with pytest.raises(ValueError, match='different output budget'):
        runner.prepare(tmp_path, 4096)


def test_grade_retains_exact_raw_receipt_reference_and_request_identity(tmp_path, monkeypatch):
    row, item, graded = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item)
    record = runner.grade_receipt(item, raw, row, sys.modules['frontier_modebench_contract'].grade_response)
    assert record['raw_receipt'] == raw['relative_path']
    assert record['request_sha256'] == item['request_sha256']
    assert record['row_sha256'] == item['row_sha256']
    assert record['response_id'] == raw['response']['id']
    assert record['usage'] == raw['response']['usage']
    assert record['verified'] is True
    assert 'request' not in record
    assert graded == [(1, 'countdown', row, 'correct')]


def test_recovery_grades_saved_http_response_without_new_request(tmp_path, monkeypatch):
    row, item, graded = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item)
    write_json(tmp_path / raw['relative_path'], raw)
    calls = install_client(monkeypatch, [])
    assert run_once(tmp_path) == 0
    assert calls == []
    assert graded == [(1, 'countdown', row, 'correct')]
    record = json.loads((tmp_path / 'sample_receipts' / f"{item['sample_id']}.json").read_text())
    assert record['raw_receipt'] == raw['relative_path']
    assert json.loads((tmp_path / 'status.json').read_text())['complete'] is True
    assert len((tmp_path / 'samples.jsonl').read_text().splitlines()) == 1


def test_recovery_rebuilds_interrupted_append_without_duplicate_or_paid_call(tmp_path, monkeypatch):
    _, item, _ = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item)
    write_json(tmp_path / raw['relative_path'], raw)
    calls = install_client(monkeypatch, [])
    assert run_once(tmp_path) == 0
    # Atomic per-sample receipts are durable even if the convenience JSONL append was interrupted.
    (tmp_path / 'samples.jsonl').write_text('{"truncated":')
    assert run_once(tmp_path) == 0
    assert calls == []
    records = [json.loads(line) for line in (tmp_path / 'samples.jsonl').read_text().splitlines()]
    assert [r['sample_id'] for r in records] == [item['sample_id']]


def test_http_429_retries_and_retains_both_receipts(tmp_path, monkeypatch):
    _, item, _ = run_fixture(tmp_path, monkeypatch)
    calls = install_client(monkeypatch, [httpx.Response(429, json={'error': 'rate limit'},
                                                      headers={'retry-after': '3'}),
                                        httpx.Response(200, json=body())])
    waits = []
    async def no_wait(seconds):
        waits.append(seconds)
    monkeypatch.setattr(runner.asyncio, 'sleep', no_wait)
    assert run_once(tmp_path) == 0
    assert len(calls) == 2 and waits == [3]
    first = json.loads((tmp_path / 'raw_responses' / f"{item['sample_id']}__01.json").read_text())
    second = json.loads((tmp_path / 'raw_responses' / f"{item['sample_id']}__02.json").read_text())
    assert first['http_status'] == 429 and second['http_status'] == 200
    record = json.loads((tmp_path / 'sample_receipts' / f"{item['sample_id']}.json").read_text())
    assert record['raw_receipt'] == second['relative_path']


def test_http_timeout_retains_possibly_billed_evidence_before_recovery(tmp_path, monkeypatch):
    _, item, _ = run_fixture(tmp_path, monkeypatch)
    calls = install_client(monkeypatch, [httpx.ReadTimeout('offline timeout'), httpx.Response(200, json=body())])
    async def no_wait(_):
        return None
    monkeypatch.setattr(runner.asyncio, 'sleep', no_wait)
    assert run_once(tmp_path) == 0
    assert len(calls) == 2
    first = json.loads((tmp_path / 'raw_responses' / f"{item['sample_id']}__01.json").read_text())
    assert first['http_status'] is None and first['possibly_billed'] is True
    assert first['error_type'] == 'ReadTimeout'


def test_nonretryable_error_stops_without_grading_or_leaking_key(tmp_path, monkeypatch):
    _, item, graded = run_fixture(tmp_path, monkeypatch)
    calls = install_client(monkeypatch, [httpx.Response(401, json={'error': 'OFFLINE_TEST_CREDENTIAL'})])
    assert run_once(tmp_path) == 2
    assert len(calls) == 1 and graded == []
    assert not list((tmp_path / 'sample_receipts').glob('*.json'))
    assert 'OFFLINE_TEST_CREDENTIAL' not in (tmp_path / 'errors.jsonl').read_text()
    assert '[REDACTED]' in (tmp_path / 'errors.jsonl').read_text()


def test_recovery_rejects_raw_request_identity_mismatch(tmp_path, monkeypatch):
    _, item, graded = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item)
    raw['request_sha256'] = 'different_request'
    write_json(tmp_path / raw['relative_path'], raw)
    calls = install_client(monkeypatch, [])
    assert run_once(tmp_path) == 2
    assert calls == [] and graded == []
    assert not list((tmp_path / 'sample_receipts').glob('*.json'))


def test_recovery_rejects_unexpected_saved_model(tmp_path, monkeypatch):
    _, item, graded = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item, body(model='unexpected-deployment'))
    write_json(tmp_path / raw['relative_path'], raw)
    calls = install_client(monkeypatch, [])
    assert run_once(tmp_path) == 2
    assert calls == [] and graded == []
    assert not list((tmp_path / 'sample_receipts').glob('*.json'))


@pytest.mark.parametrize('field,changed', [
    ('text', 'tampered answer'),
    ('response_id', 'resp_wrong'),
    ('usage', {'total_tokens': 999}),
    ('request_sha256', 'wrong_request'),
    ('row_sha256', 'wrong_row'),
    ('raw_receipt_sha256', 'wrong_receipt'),
    ('raw_receipt', '../outside.json'),
])
def test_completed_receipt_tampering_fails_before_any_network(tmp_path, monkeypatch, field, changed):
    _, item, _ = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item)
    write_json(tmp_path / raw['relative_path'], raw)
    calls = install_client(monkeypatch, [])
    assert run_once(tmp_path) == 0
    saved_path = tmp_path / 'sample_receipts' / f"{item['sample_id']}.json"
    saved = json.loads(saved_path.read_text())
    saved[field] = changed
    write_json(saved_path, saved)
    with pytest.raises(ValueError):
        run_once(tmp_path)
    assert calls == []


def test_completed_receipt_requires_its_raw_http_evidence(tmp_path, monkeypatch):
    _, item, _ = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item)
    write_json(tmp_path / raw['relative_path'], raw)
    calls = install_client(monkeypatch, [])
    assert run_once(tmp_path) == 0
    (tmp_path / raw['relative_path']).unlink()
    with pytest.raises(FileNotFoundError):
        run_once(tmp_path)
    assert calls == []


def test_old_receipt_without_new_digest_still_checks_every_response_field(tmp_path, monkeypatch):
    _, item, _ = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item)
    write_json(tmp_path / raw['relative_path'], raw)
    calls = install_client(monkeypatch, [])
    assert run_once(tmp_path) == 0
    saved_path = tmp_path / 'sample_receipts' / f"{item['sample_id']}.json"
    saved = json.loads(saved_path.read_text())
    saved.pop('raw_receipt_sha256')
    write_json(saved_path, saved)
    assert run_once(tmp_path) == 0
    raw['response']['model'] = 'unexpected-deployment'
    write_json(tmp_path / raw['relative_path'], raw)
    with pytest.raises(ValueError, match='Unexpected returned model'):
        run_once(tmp_path)
    assert calls == []


def test_fresh_unexpected_model_stops_and_preserves_raw_receipt(tmp_path, monkeypatch):
    _, item, graded = run_fixture(tmp_path, monkeypatch)
    calls = install_client(monkeypatch, [httpx.Response(200, json=body(model='unexpected-deployment'))])
    assert run_once(tmp_path) == 2
    assert len(calls) == 1 and graded == []
    raw = json.loads((tmp_path / 'raw_responses' / f"{item['sample_id']}__01.json").read_text())
    assert raw['response']['model'] == 'unexpected-deployment'
    assert not list((tmp_path / 'sample_receipts').glob('*.json'))


def test_recovered_sample_id_mismatch_never_uses_wrong_answer(tmp_path, monkeypatch):
    _, item, graded = run_fixture(tmp_path, monkeypatch)
    raw = raw_receipt(item)
    raw['sample_id'] = 'other_sample'
    write_json(tmp_path / raw['relative_path'], raw)
    calls = install_client(monkeypatch, [])
    assert run_once(tmp_path) == 2
    assert calls == [] and graded == []
