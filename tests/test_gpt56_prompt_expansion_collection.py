"""Offline checks for paid collection identity, native controls and recovery."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

OPS = Path(__file__).resolve().parents[1] / 'ops'
sys.path.insert(0, str(OPS))
import prepare_gpt56_prompt_expansion as prepare
import run_gpt56_prompt_expansion as run


def request():
    return {'sample_id': 'T0P0__L1_countdown_000_0', 'original_sample_id': 'L1_countdown_000_0',
            'request_sha256': 'payload-hash', 'temperature_condition': 0.0,
            'request': {'model': 'gpt-5.6-sol'}}


def receipt(item, **changes):
    raw = {'sample_id': item['sample_id'], 'attempt': 1, 'request_sha256': item['request_sha256'],
           'relative_path': f"raw_responses/{item['sample_id']}__01.json", 'http_status': 200,
           'response': {'status': 'completed', 'model': 'gpt-5.6-sol', 'temperature': 0.0,
                        'top_p': .98, 'reasoning': {'effort': 'none'}},
           'headers': {'x-ms-served-model': run.SERVED_MODEL}}
    raw.update(changes)
    return raw


@pytest.mark.parametrize('field,value', [('temperature', .5), ('top_p', 1), ('reasoning', {'effort': 'medium'}), ('model', 'other')])
def test_returned_control_drift_is_rejected(field, value):
    item = request()
    raw = receipt(item)
    raw['response'][field] = value
    with pytest.raises(ValueError):
        run.validate_raw_receipt(item, raw)


def test_served_snapshot_drift_is_rejected():
    item = request()
    raw = receipt(item, headers={'x-ms-served-model': 'new-snapshot'})
    with pytest.raises(ValueError, match='snapshot'):
        run.validate_raw_receipt(item, raw)


@pytest.mark.parametrize('status', ['completed', 'incomplete'])
def test_valid_native_outcomes_include_truncation_and_refusal(status):
    item = request()
    raw = receipt(item)
    raw['response']['status'] = status
    raw['response']['output'] = [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'Declined'}]}]
    run.validate_raw_receipt(item, raw)


def dispatch(tmp_path, item, attempt=1):
    run.append(tmp_path / 'events.jsonl', {'event': 'request_started', 'sample_id': item['sample_id'],
               'attempt': attempt, 'request_sha256': item['request_sha256']})


def test_interrupted_unknown_dispatch_blocks_automatic_retry(tmp_path):
    item = request()
    dispatch(tmp_path, item)
    with pytest.raises(ValueError, match='Ambiguous interrupted dispatch'):
        run.audit_attempts(tmp_path, [item])


def test_durable_native_answer_can_recover_without_paid_repeat(tmp_path):
    item = request()
    raw = receipt(item)
    dispatch(tmp_path, item)
    run.atomic(tmp_path / raw['relative_path'], raw)
    history = run.audit_attempts(tmp_path, [item])
    assert history[item['sample_id']] == [raw]


def test_duplicate_received_answers_are_rejected(tmp_path):
    item = request()
    for attempt in (1, 2):
        dispatch(tmp_path, item, attempt)
        raw = receipt(item, attempt=attempt, relative_path=f"raw_responses/{item['sample_id']}__{attempt:02d}.json")
        run.atomic(tmp_path / raw['relative_path'], raw)
    with pytest.raises(ValueError, match='Duplicate received'):
        run.audit_attempts(tmp_path, [item])


def test_transport_failure_is_retained_as_attempt_not_model_outcome(tmp_path):
    item = request()
    dispatch(tmp_path, item)
    raw = receipt(item, http_status=None, response=None, possibly_billed=True)
    run.atomic(tmp_path / raw['relative_path'], raw)
    assert run.audit_attempts(tmp_path, [item])[item['sample_id']][0]['possibly_billed']


def test_full_collection_cannot_bypass_missing_preflight(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.gate(tmp_path, {}, [request()] * 5)


def test_approved_plan_digest_is_immutable(tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    with pytest.raises(ValueError, match='Approved plan digest'):
        prepare.derive_requests(tmp_path)


def test_real_plan_reconstructs_all_exact_interleaved_payload_hashes():
    plan, reference, original, rows, requests = prepare.derive_requests(prepare.PLAN_DIR)
    assert len(rows) == 360
    assert len(requests) == 14400
    assert len({item['sample_id'] for item in requests}) == 14400
    assert all(tuple(x['temperature_condition'] for x in requests[start:start + 5]) == prepare.TEMPERATURES
               for start in range(0, len(requests), 5))
    assert len({x['original_sample_id'] for x in requests}) == 2880
    assert all(x['request']['max_output_tokens'] == 8192 and x['request']['store'] is False for x in requests)


@pytest.fixture
def fake_collection(tmp_path, monkeypatch):
    """Exercise real dispatch/recovery with ten slots and a local fake client."""
    import types
    source = prepare.DEFAULT_OUTPUT
    manifest = json.loads((source / 'manifest.json').read_text())
    requests = prepare.read_lines(source / 'requests.jsonl')[:10]
    all_rows = {prepare.identity(row): row for row in prepare.read_lines(source / 'rows.jsonl')}
    rows = {prepare.identity(item): all_rows[prepare.identity(item)] for item in requests}
    run.atomic(tmp_path / 'manifest.json', manifest)
    monkeypatch.setattr(run, 'validate_inventory', lambda output: (manifest, rows, requests))
    monkeypatch.setitem(sys.modules, 'frontier_modebench_contract', types.SimpleNamespace(
        grade_response=lambda level, domain, row, text: {'verified': False, 'canonical_key': None}))

    class FakeClient:
        calls = []
        response_override = None

        def __init__(self, **kwargs):
            assert kwargs['headers'] == {'api-key': 'fake-credential'}
            assert kwargs['follow_redirects'] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, endpoint, *, json):
            assert endpoint == run.native.ENDPOINT
            self.calls.append(json)
            body = {'id': 'fake-response-' + str(len(self.calls)), 'status': 'completed', 'model': 'gpt-5.6-sol',
                    'temperature': json['temperature'], 'top_p': .98, 'reasoning': {'effort': 'none'},
                    'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'invalid answer'}]}]}
            if self.response_override:
                body.update(self.response_override)
            return types.SimpleNamespace(status_code=200, headers={'x-ms-served-model': run.SERVED_MODEL}, json=lambda: body)

    return tmp_path, manifest, requests, FakeClient


def test_retained_preflight_gate_then_full_dispatch_and_resume(fake_collection):
    import asyncio
    output, manifest, requests, client = fake_collection
    result = asyncio.run(run.collect(output, 'preflight', 1, 600, 'fake-credential', client_factory=client))
    assert result == 0 and len(client.calls) == 5
    assert [item['temperature'] for item in client.calls] == list(run.TEMPERATURES)
    assert (output / 'gate.json').is_file()
    assert len(list((output / 'sample_receipts').glob('*.json'))) == 5
    assert all(not record['verified'] for record in run.load_completed(output, requests).values())
    result = asyncio.run(run.collect(output, 'full', 1, 600, 'fake-credential', client_factory=client))
    assert result == 0 and len(client.calls) == 10
    result = asyncio.run(run.collect(output, 'full', 1, 600, 'fake-credential', client_factory=client))
    assert result == 0 and len(client.calls) == 10
    assert len(run.audit_attempts(output, requests)) == 10
    assert len(prepare.read_lines(output / 'samples.jsonl')) == 10


def test_recover_received_preflight_without_repeating_api_call(fake_collection):
    import asyncio
    output, manifest, requests, client = fake_collection
    item = requests[0]
    dispatch(output, item)
    raw = receipt(item, latency_seconds=.1)
    raw['response']['temperature'] = item['temperature_condition']
    run.atomic(output / raw['relative_path'], raw)
    result = asyncio.run(run.collect(output, 'preflight', 1, 600, 'fake-credential', client_factory=client))
    assert result == 0 and len(client.calls) == 4
    assert run.load_completed(output, requests)[item['sample_id']]['raw_receipt'] == raw['relative_path']


def test_fresh_control_drift_stops_further_dispatch_and_is_never_recollected(fake_collection):
    import asyncio
    output, manifest, requests, client = fake_collection
    client.response_override = {'top_p': 1.0}
    result = asyncio.run(run.collect(output, 'preflight', 1, 600, 'fake-credential', client_factory=client))
    assert result == 2 and len(client.calls) == 1
    assert len(list((output / 'raw_responses').glob('*.json'))) == 1
    assert not list((output / 'sample_receipts').glob('*.json'))
    assert not (output / 'gate.json').exists()
    with pytest.raises(ValueError, match='top_p'):
        asyncio.run(run.collect(output, 'preflight', 1, 600, 'fake-credential', client_factory=client))
    assert len(client.calls) == 1


def test_transport_views_retain_error_attempts_and_all_dispatch_bindings(tmp_path, monkeypatch):
    import complete_gpt56_prompt_expansion_views as views
    requests = []
    for temperature in run.TEMPERATURES:
        item = request()
        item['sample_id'] = prepare.slug(temperature).upper() + '__L1_countdown_000_0'
        item['temperature_condition'] = temperature
        requests.append(item)
    run.atomic(tmp_path / 'manifest.json', {'test': True})
    monkeypatch.setattr(views, 'verify_collector_bindings', lambda output: {'fake-test': 'fixture'})
    monkeypatch.setattr(views, 'validate_inventory', lambda output: ({}, {}, requests))
    monkeypatch.setattr(views, 'load_completed', lambda output, reqs: dict.fromkeys(range(14400)))
    for index, item in enumerate(requests):
        view = tmp_path / 'arms' / prepare.slug(item['temperature_condition'])
        view.mkdir(parents=True)
        prepare.write_lines(view / 'requests.jsonl', [item])
        run.atomic(view / 'manifest.json', {'expansion_view_schema': 'gpt56-temperature-expansion-arm-view-v1',
                   'temperature': item['temperature_condition'], 'request_count': 2880,
                   'artifact_sha256': {'requests.jsonl': run.file_sha(view / 'requests.jsonl')}})
        for attempt in range(1, 3 if index == 0 else 2):
            dispatch(tmp_path, item, attempt)
            raw = receipt(item, attempt=attempt, relative_path=f"raw_responses/{item['sample_id']}__{attempt:02d}.json")
            raw['response']['temperature'] = item['temperature_condition']
            if index == 0 and attempt == 1:
                raw.update(http_status=503, response={'error': 'service unavailable'})
                run.append(tmp_path / 'errors.jsonl', raw)
            run.atomic(tmp_path / raw['relative_path'], raw)
    results = views.complete_views(tmp_path)
    assert [entry['raw_attempts'] for entry in results] == [2, 1, 1, 1, 1]
    assert views.complete_views(tmp_path) == results
    first = tmp_path / 'arms/t0p0'
    assert len(prepare.read_lines(first / 'events.jsonl')) == 2
    assert len(prepare.read_lines(first / 'errors.jsonl')) == 1
    assert len(list((first / 'raw_responses').glob('*.json'))) == 2
    changed = first / 'events.jsonl'
    changed.write_text(changed.read_text() + '{}\n')
    with pytest.raises(ValueError, match='view log differs'):
        views.complete_views(tmp_path)


def test_transport_export_uses_exact_sealed_collection_dependencies():
    import complete_gpt56_prompt_expansion_views as views
    bindings = views.verify_collector_bindings(prepare.DEFAULT_OUTPUT)
    assert set(bindings) == {'ops/run_gpt56_prompt_expansion.py', 'ops/prepare_gpt56_prompt_expansion.py', 'ops/evaluate_frontier_modebench.py'}


@pytest.mark.parametrize('module_name', ['collection_runner', 'collection_preparer', 'native'])
def test_transport_export_rejects_changed_live_dependency(tmp_path, monkeypatch, module_name):
    import complete_gpt56_prompt_expansion_views as views
    module = views.collection_runner.native if module_name == 'native' else getattr(views, module_name)
    changed = tmp_path / 'changed.py'
    changed.write_text('# Different source bytes must never interpret sealed receipts.\n')
    monkeypatch.setattr(module, '__file__', str(changed))
    with pytest.raises(ValueError, match='Loaded collection dependency differs from sealed bytes'):
        views.verify_collector_bindings(prepare.DEFAULT_OUTPUT)
