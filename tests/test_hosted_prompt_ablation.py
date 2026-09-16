import copy
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from prepare_hosted_prompt_ablation import adapt_items, sha, source_profile
from run_hosted_prompt_ablation import credential


def fixture(model='gpt-5.6-sol'):
    request = {'model': model, 'input': [{'role': 'system', 'content': 'Format. Hint.'},
               {'role': 'user', 'content': 'Immutable task.'}], 'reasoning': {'effort': 'medium'},
               'max_output_tokens': 8192, 'store': False}
    item = {'sample_id': 'L2_mathir_000_0', 'sample_index': 0, 'level': 2, 'domain': 'mathir',
            'row_index': 0, 'row_sha256': 'same-row', 'request': request, 'request_sha256': sha(request)}
    manifest = {'model': model, 'endpoint': 'https://liv.services.ai.azure.com/openai/v1/responses'}
    return item, source_profile(manifest, request)


def test_only_system_changes_and_source_is_preserved():
    original, profile = fixture()
    before = copy.deepcopy(original)
    template = copy.deepcopy(original)
    template['request']['input'][0]['content'] = 'Format.'
    template['request_sha256'] = sha(template['request'])
    result = adapt_items([original], [template], profile, 'neutral')[0]
    assert original == before
    assert result['request']['input'][1] == before['request']['input'][1]
    assert result['request']['reasoning'] == {'effort': 'medium'}
    assert result['request']['max_output_tokens'] == 8192
    assert result['request']['input'][0]['content'] == 'Format.'
    assert result['reference_request_sha256'] == original['request_sha256']


@pytest.mark.parametrize('mutation', ['task', 'row', 'original_hint', 'sampling'])
def test_rejects_confounding_mutations(mutation):
    original, profile = fixture()
    template = copy.deepcopy(original)
    if mutation == 'task':
        template['request']['input'][1]['content'] = 'Different problem.'
    elif mutation == 'row':
        template['row_sha256'] = 'different'
    elif mutation == 'original_hint':
        template['request']['input'][0]['content'] = 'Other hint.'
    elif mutation == 'sampling':
        profile['request_parameters']['reasoning'] = {'effort': 'none'}
    template['request_sha256'] = sha(template['request'])
    with pytest.raises(ValueError):
        adapt_items([original], [template], profile, 'original')


def test_adapter_only_admits_gpt56_without_changing_transport():
    old = (ROOT / 'ops/evaluate_chat_frontier_modebench.py').read_text()
    new = (ROOT / 'ops/evaluate_native_prompt_ablation.py').read_text()
    old_body = old[old.index('from __future__'):]
    expected = old_body.replace("MODELS = ('FW-Kimi-K3', 'grok-4.3', 'DeepSeek-V4-Pro', 'gpt-5.4')",
                                "MODELS = ('FW-Kimi-K3', 'grok-4.3', 'DeepSeek-V4-Pro', 'gpt-5.4', 'gpt-5.6-sol')")
    expected = expected.replace("protocol = 'responses' if model == 'gpt-5.4' else 'chat_completions'",
                                "protocol = 'responses' if model in ('gpt-5.4', 'gpt-5.6-sol') else 'chat_completions'")
    assert new[new.index('from __future__'):] == expected


def test_credential_requires_explicit_private_source(tmp_path, monkeypatch):
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError):
        credential()
    p = tmp_path / 'credential'
    p.write_text('synthetic-test-value')
    p.chmod(0o644)
    with pytest.raises(ValueError):
        credential(p)
    p.chmod(0o600)
    assert credential(p) == 'synthetic-test-value'


def recovery_fixture(tmp_path, *, completed=1, total=3):
    import json
    import shutil
    import evaluate_native_prompt_ablation as native
    output = tmp_path / 'run'
    output.mkdir()
    original, profile = fixture()
    row = {'level': 2, 'domain': 'mathir', 'row_index': 0,
           'problem': 'Immutable task.', 'answer': 'Private verifier spec', 'metadata': {}}
    items, groups = [], []
    for draw in range(total):
        item = copy.deepcopy(original)
        item.update(sample_id=f'L2_mathir_000_{draw}', sample_index=draw,
                    group_id=f'L2_mathir_000_{draw}', choice_index=0, row_sha256=sha(row),
                    prompt_arm='original', experiment_condition='prompt_hint_ablation_v1')
        group = {'group_id': item['group_id'], 'request': item['request'],
                 'request_sha256': item['request_sha256'], 'sample_ids': [item['sample_id']],
                 'sample_count': 1, 'protocol': profile['protocol']}
        items.append(item)
        groups.append(group)
        if draw < completed:
            raw = {'group_id': group['group_id'], 'sample_ids': group['sample_ids'], 'attempt': 1,
                   'request_sha256': group['request_sha256'],
                   'relative_path': f"raw_responses/{group['group_id']}__01.json",
                   'http_status': 200, 'latency_seconds': .25,
                   'response': {'id': f'offline_response_{draw}', 'model': 'gpt-5.6-sol',
                                'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
                                    'content': [{'type': 'output_text', 'text': 'A;C'}]}],
                                'usage': {'input_tokens': 8, 'output_tokens': 3, 'total_tokens': 11}}}
            grade = lambda level, domain, supplied_row, text: {
                'verified': True, 'canonical_key': 'test-mode', 'graded_text': text}
            record = native.grade_receipt(item, group, raw, row, grade)
            native.atomic(output / raw['relative_path'], raw)
            native.atomic(output / 'sample_receipts' / (item['sample_id'] + '.json'), record)
    for name, records in [('rows.jsonl', [row]), ('requests.jsonl', items), ('http_requests.jsonl', groups)]:
        native.write_jsonl(output / name, records)
    adapter = output / 'code/ops/evaluate_native_prompt_ablation.py'
    adapter.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / 'ops/evaluate_native_prompt_ablation.py', adapter)
    native.atomic(output / 'manifest.json', {
        'model': profile['model'], 'request_count': total,
        'artifact_sha256': {name: native.file_sha(output / name)
                            for name in ('rows.jsonl', 'requests.jsonl', 'http_requests.jsonl')},
        'code_sha256': {'ops/evaluate_native_prompt_ablation.py': native.file_sha(adapter)}})
    # Deliberately stale status is not authoritative completion evidence.
    native.atomic(output / 'status.json', {'completed_samples': 0})
    return {'model_id': 'gpt56sol', 'model': 'gpt-5.6-sol', 'arm': 'original', 'run_dir': str(output)}


def forbid_generation(monkeypatch):
    import run_hosted_prompt_ablation as orchestration
    def fail(*args, **kwargs):
        raise AssertionError('Recovery must not request a credential or launch any generator')
    monkeypatch.setattr(orchestration, 'credential', fail)
    monkeypatch.setattr(orchestration.subprocess, 'Popen', fail)


def test_missing_preflight_marker_recovers_atomic_sample_without_new_generation(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import run_hosted_prompt_ablation as orchestration
    entry = recovery_fixture(tmp_path)
    monkeypatch.setattr(orchestration, 'validate_inventory', lambda base: {'runs': [entry]})
    forbid_generation(monkeypatch)
    args = SimpleNamespace(base=tmp_path, stage='preflight', credential_file=None, workers=1)
    assert orchestration.run(args) == 0
    marker_path = Path(entry['run_dir']) / 'preflight_result.json'
    marker = json.loads(marker_path.read_text())
    assert marker['terminal_samples'] == 1 and marker['recovered_without_generation'] is True
    assert len(marker['authenticated_samples']) == 1
    assert marker['authenticated_samples'][0]['provider_sample_identity'] == ['offline_response_0', 0]
    # A second recovery is idempotent and also cannot launch a paid request.
    unchanged = marker_path.read_bytes()
    assert orchestration.run(args) == 0
    assert marker_path.read_bytes() == unchanged


@pytest.mark.parametrize('mutation', ['saved_text', 'row_hash', 'raw_model', 'arm', 'unknown_sample'])
def test_preflight_recovery_rejects_corrupt_or_misattributed_receipts(tmp_path, monkeypatch, mutation):
    import json
    from types import SimpleNamespace
    import run_hosted_prompt_ablation as orchestration
    entry = recovery_fixture(tmp_path)
    output = Path(entry['run_dir'])
    sample = output / 'sample_receipts/L2_mathir_000_0.json'
    value = json.loads(sample.read_text())
    if mutation == 'saved_text':
        value['text'] = 'Changed answer'
    elif mutation == 'row_hash':
        value['row_sha256'] = 'changed'
    elif mutation == 'raw_model':
        raw_path = output / value['raw_receipt']
        raw = json.loads(raw_path.read_text())
        raw['response']['model'] = 'different-provider-model'
        raw_path.write_text(json.dumps(raw))
    elif mutation == 'arm':
        value['prompt_arm'] = 'neutral'
    else:
        value['sample_id'] = 'L2_mathir_001_0'
    sample.write_text(json.dumps(value))
    monkeypatch.setattr(orchestration, 'validate_inventory', lambda base: {'runs': [entry]})
    forbid_generation(monkeypatch)
    with pytest.raises(ValueError):
        orchestration.run(SimpleNamespace(base=tmp_path, stage='preflight', credential_file=None, workers=1))
    assert not (output / 'preflight_result.json').exists()


def test_preflight_recovery_uses_frozen_adapter_and_rejects_adapter_drift(tmp_path, monkeypatch):
    import evaluate_native_prompt_ablation as native
    import run_hosted_prompt_ablation as orchestration
    entry = recovery_fixture(tmp_path)
    def unfrozen(*args, **kwargs):
        raise AssertionError('Must authenticate with the frozen adapter')
    monkeypatch.setattr(native, 'validate_completed', unfrozen)
    assert orchestration.recover_preflight(entry)['terminal_samples'] == 1
    adapter = Path(entry['run_dir']) / 'code/ops/evaluate_native_prompt_ablation.py'
    adapter.write_text(adapter.read_text() + '\n# drift\n')
    with pytest.raises(ValueError, match='Frozen code changed'):
        orchestration.recover_preflight(entry)


def test_full_stage_accepts_two_authenticated_samples_after_interrupted_preflight(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import run_hosted_prompt_ablation as orchestration
    entry = recovery_fixture(tmp_path, completed=2)
    monkeypatch.setattr(orchestration, 'validate_inventory', lambda base: {'runs': [entry]})
    def ready_for_remaining_collection(*args, **kwargs):
        raise RuntimeError('Reached credential step for remaining authorized collection')
    monkeypatch.setattr(orchestration, 'credential', ready_for_remaining_collection)
    with pytest.raises(RuntimeError, match='Reached credential step'):
        orchestration.run(SimpleNamespace(base=tmp_path, stage='full', credential_file=None, workers=1))
    assert orchestration.recover_preflight(entry)['terminal_samples'] == 2


def test_full_stage_cannot_trust_preflight_marker_without_atomic_samples(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import evaluate_native_prompt_ablation as native
    import run_hosted_prompt_ablation as orchestration
    entry = recovery_fixture(tmp_path, completed=0)
    native.atomic(Path(entry['run_dir']) / 'preflight_result.json', {'exit_code': 0, 'terminal_samples': 1})
    monkeypatch.setattr(orchestration, 'validate_inventory', lambda base: {'runs': [entry]})
    forbid_generation(monkeypatch)
    with pytest.raises(ValueError, match='at least one authenticated'):
        orchestration.run(SimpleNamespace(base=tmp_path, stage='full', credential_file=None, workers=1))
