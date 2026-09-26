"""Execution proof guards use CPU fixtures and never call a scheduler or model."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('recovery_execution_audit_test',
    Path(__file__).resolve().parents[1] / 'ops/audit_modebench_level3_recovery_execution.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    Path(path).write_text(json.dumps(value))


def account(job_id='42', state='COMPLETED', exit_code='0:0'):
    return {'JobIDRaw': job_id, 'State': state, 'ExitCode': exit_code, 'Partition': 'all',
            'AllocCPUS': '6', 'ReqMem': '48G', 'Timelimit': '01:00:00', 'NodeList': 'node007',
            'Start': '2026-09-08T20:00:00', 'End': '2026-09-08T20:30:00',
            'ReqTRES': 'cpu=6,gres/gpu:rtx_6000=1,mem=48G',
            'AllocTRES': 'cpu=6,gres/gpu:rtx_6000=1,mem=48G'}


@pytest.mark.parametrize('state,exit_code,expected', [
    ('PENDING', '0:0', 'pending'), ('RUNNING', '0:0', 'pending'),
    ('COMPLETED', '0:0', 'complete'), ('COMPLETED', '1:0', 'failed'),
    ('TIMEOUT', '0:0', 'failed'), ('FAILED', '1:0', 'failed'), ('CANCELLED', '0:0', 'failed')])
def test_failed_or_timed_out_jobs_do_not_wait_forever(state, exit_code, expected):
    assert audit.completion_status({'42': account(state=state, exit_code=exit_code)})[0] == expected


def test_same_id_accounting_history_preserves_previous_epoch(monkeypatch):
    rows = [account(state='PREEMPTED'), account()]
    output = '\n'.join('|'.join(row[field] for field in audit.ACCOUNT_FIELDS) for row in rows)
    monkeypatch.setattr(audit.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=output))
    actual = audit.scheduler_records(['42'])['42']
    assert actual['State'] == 'COMPLETED'
    assert actual['accounting_history'] == rows


@pytest.mark.parametrize('kind', ['missing', 'unexpected', 'active_history'])
def test_ambiguous_accounting_cannot_qualify(monkeypatch, kind):
    rows = [] if kind == 'missing' else [account('99')] if kind == 'unexpected' else [account(state='RUNNING'), account()]
    output = '\n'.join('|'.join(row[field] for field in audit.ACCOUNT_FIELDS) for row in rows)
    monkeypatch.setattr(audit.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=output))
    with pytest.raises(ValueError):
        audit.scheduler_records(['42'])


def runtime_fixture():
    runtime = {'gpu_names': ['Quadro RTX 6000'], 'attention_backend': 'XFORMERS', 'engine': 'V0',
        'partition_preempt_mode': 'OFF', 'scheduler': {'JobId': '42', 'JobState': 'RUNNING',
        'Partition': 'all', 'NumCPUs': '6', 'MinMemoryNode': '48G', 'TimeLimit': '01:00:00',
        'NodeList': 'node007', 'TresPerNode': 'gpu:rtx_6000:1', 'Restarts': '0'}}
    job = {'old_job_id': '1', 'name': '3b_mathir_d2', '_model_path': '/fixed/model'}
    event = {'event': 'recovery_worker_authenticated', 'old_job_id': '1', 'new_job_id': '42',
             'cell': job['name'], 'recovery_seal_sha256': audit.SEAL_SHA256,
             'scientific_seal_sha256': audit.SCIENTIFIC_SHA256, 'runtime': deepcopy(runtime),
             'resumed_same_new_id': False}
    engine = "Initializing a V0 LLM engine (v0.8.4) model='/fixed/model', dtype=torch.float16, tensor_parallel_size=1, quantization=None\nUsing XFormers backend."
    return job, runtime, event, engine


def test_first_and_resumed_runtime_epochs_remain_honest():
    job, runtime, event, engine = runtime_fixture()
    resumed = deepcopy(event)
    resumed['resumed_same_new_id'] = True
    resumed['runtime']['scheduler']['Restarts'] = '1'
    for events in ([event], [event, resumed], [resumed]):
        audit.verify_runtime_evidence(job, {'new_job_id': '42'}, {'runtime': runtime}, account(),
                                      '\n'.join(map(json.dumps, events)) + '\n' + engine)


@pytest.mark.parametrize('change', ['partition', 'gpu', 'id', 'engine', 'model', 'dtype', 'allocation', 'exit'])
def test_false_actual_runtime_cannot_qualify(change):
    job, runtime, event, engine = runtime_fixture()
    acct = account()
    if change == 'partition': runtime['scheduler']['Partition'] = 'lowprio'
    elif change == 'gpu': runtime['gpu_names'] = ['A6000']
    elif change == 'id': event['new_job_id'] = '1'
    elif change == 'engine': engine = engine.replace('V0', 'V1')
    elif change == 'model': engine = engine.replace('/fixed/model', '/another/model')
    elif change == 'dtype': engine = engine.replace('float16', 'bfloat16')
    elif change == 'allocation': acct['NodeList'] = 'node008'
    elif change == 'exit': acct['ExitCode'] = '1:0'
    with pytest.raises(ValueError):
        audit.verify_runtime_evidence(job, {'new_job_id': '42'}, {'runtime': runtime}, acct,
                                      json.dumps(event) + '\n' + engine)


@pytest.fixture(params=[64, 128])
def complete_samples(tmp_path, request):
    from test_modebench_level3_independent_evaluator import evaluator, make_task, LLM, Tokenizer
    from oat_drgrpo.math_grader import validated_modebench_outcome_key
    task = make_task(tmp_path, count=request.param)
    task.update(seeds=[6328000, 6328001, 6328002, 6328003], batch_size=8)
    receipt = evaluator.evaluate_task(LLM(), Tokenizer(), task,
        model={'label': '3b', 'vllm_version': '0.8.4'}, code=evaluator.code_identity(),
        grader=validated_modebench_outcome_key,
        params_factory=lambda args, *_: SimpleNamespace(seed=args.seed, n=8))
    taskpath = tmp_path / 'tasks.json'
    write(taskpath, [task])
    job = {'tasks': str(taskpath), 'output': task['output'], 'domain': task['domain'], 'model_label': '3b'}
    helper = SimpleNamespace(validate_snapshot=lambda *a, **kw: {
        'batch_count': request.param // 8 * 4, 'expected_batches': request.param // 8 * 4,
        'existing_receipt_sha256': audit.digest(task['output'])})
    return helper, job, receipt, request.param


def test_full_64_and128_rows_regrade_and_match_every_committed_draw(complete_samples):
    helper, job, _, count = complete_samples
    _, attempts = audit.verify_completed_samples(helper, job, {})
    assert attempts == count * 32


@pytest.mark.parametrize('change', ['receipt_text', 'coherent_false_grade', 'missing_draw', 'row_metadata', 'summary'])
def test_saved_receipt_or_coherent_batch_tampering_is_rejected(complete_samples, change):
    helper, job, receipt, _ = complete_samples
    batchpath = sorted(Path(job['output'] + '.batches').glob('seed-*.json'))[0]
    batch = audit.read(batchpath)
    if change == 'receipt_text':
        receipt['prompt_results'][0]['draws'][0]['attempts'][0]['text'] = 'different'
    elif change == 'coherent_false_grade':
        attempt = batch['draws'][0]['attempts'][0]
        attempt.update(canonical_key='fabricated', verified=True)
        receipt['prompt_results'][0]['draws'][0] = deepcopy(batch['draws'][0])
        batch['draws_sha256'] = audit.sha(batch['draws'])
        write(batchpath, batch)
    elif change == 'missing_draw':
        batchpath.unlink()
    elif change == 'row_metadata':
        receipt['prompt_results'][0]['row_metadata']['candidate_id'] = 'changed'
    else:
        receipt['metrics']['distinct8'] = 123
    write(job['output'], receipt)
    with pytest.raises(ValueError):
        audit.verify_completed_samples(helper, job, {})


@pytest.mark.parametrize('state', ['PENDING', 'TIMEOUT'])
def test_incomplete_campaign_never_publishes(tmp_path, monkeypatch, state):
    monkeypatch.setattr(audit, 'ATTESTATION', tmp_path / 'attestation.json')
    monkeypatch.setattr(audit, 'recovery_helper', lambda: None)
    monkeypatch.setattr(audit, 'authenticate_submission_chain', lambda _: (None, {}, {}, {'jobs': [{'new_job_id': '42'}]}, {}))
    monkeypatch.setattr(audit, 'scheduler_records', lambda _: {'42': account(state=state)})
    result = audit.audit_recovery_execution(publish=True)
    assert result['status'] == ('pending' if state == 'PENDING' else 'failed')
    assert not audit.ATTESTATION.exists()


def test_changed_chain_and_coherent_false_attestation_fail(tmp_path, monkeypatch):
    proof = tmp_path / 'proof.json'
    source = tmp_path / 'source'
    source.write_text('original')
    saved = {'schema': audit.SCHEMA, 'status': 'complete', 'files_sha256': {str(source): audit.digest(source)},
             'jobs': [{'attempts_regraded': count * 32, 'final_snapshot': {'stable_identity': {'rows': count}}} for count in [64] + [128] * 10],
             'all_attempts_regraded_with_original_grader': True, 'attempts': 43008, 'created_at': 'fixed'}
    write(proof, saved)
    monkeypatch.setattr(audit, 'ATTESTATION', proof)
    current = deepcopy({key: value for key, value in saved.items() if key != 'created_at'})
    current['all_attempts_regraded_with_original_grader'] = False
    current['jobs'] = [{**record, 'attempts_regraded': 0} for record in current['jobs']]
    monkeypatch.setattr(audit, 'audit_recovery_execution', lambda **kw: current)
    assert audit.validate_completion_attestation(proof)['attempts'] == 43008
    source.write_text('changed')
    with pytest.raises(ValueError, match='execution chain changed'):
        audit.validate_completion_attestation(proof)
    saved['files_sha256'][str(source)] = audit.digest(source)
    saved['attempts'] = 40960
    write(proof, saved)
    with pytest.raises(ValueError, match='reauthenticated current execution'):
        audit.validate_completion_attestation(proof)


def test_changed_canonical_ledger_fails_before_any_scheduler_call(tmp_path, monkeypatch):
    path = tmp_path / 'ledger.json'
    path.write_text('{}')
    monkeypatch.setattr(audit, 'LEDGER', path)
    with pytest.raises(ValueError, match='submission ledger changed'):
        audit.authenticate_submission_chain(None)


def test_publication_cannot_skip_original_grader_replay():
    with pytest.raises(ValueError, match='publishing requires full original-grader replay'):
        audit.audit_recovery_execution(publish=True, regrade=False)


def test_unchanged_completion_reauthentication_skips_only_grader_calls(complete_samples, monkeypatch):
    helper, job, _, count = complete_samples
    monkeypatch.setattr(audit, 'grade_completed_receipt', lambda *_: (_ for _ in ()).throw(AssertionError('grader replay unnecessary')))
    assert audit.verify_completed_samples(helper, job, {}, regrade=False)[1] == count * 32


def test_original_grader_rejects_a_coherent_claimed_success(complete_samples):
    _, job, receipt, _ = complete_samples
    rows, _ = audit.load_rows(audit.read(job['tasks'])[0])
    attempt = receipt['prompt_results'][0]['draws'][0]['attempts'][0]
    attempt.update(canonical_key='fabricated', verified=True)
    with pytest.raises(ValueError, match='original grader disagrees'):
        audit.grade_completed_receipt(receipt, rows)


def test_preserved_samples_require_original_runtime_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'CAMPAIGN', tmp_path)
    folder = tmp_path / 'source_evidence/1'
    folder.mkdir(parents=True)
    outpath, errpath = folder / '1.out', folder / '1.err'
    _, _, _, engine = runtime_fixture()
    event = {'event': 'worker_authenticated', 'job_id': '1', 'cell': 'original',
             'seal_sha256': 'a' * 64, 'gpu_names': ['Quadro RTX 6000']}
    outpath.write_text(json.dumps(event) + '\n' + engine)
    errpath.write_text('')
    job = {'old_job_id': '1', 'name': 'original', 'old_seal_sha256': 'a' * 64,
           'source_evidence': {str(path): audit.digest(path) for path in (outpath, errpath)}}
    audit.verify_preserved_runtime(job, {'batch_count': 1}, '/fixed/model')
    event['gpu_names'] = ['A6000']
    outpath.write_text(json.dumps(event) + '\n' + engine)
    job['source_evidence'][str(outpath)] = audit.digest(outpath)
    with pytest.raises(ValueError, match='original worker GPU'):
        audit.verify_preserved_runtime(job, {'batch_count': 1}, '/fixed/model')
