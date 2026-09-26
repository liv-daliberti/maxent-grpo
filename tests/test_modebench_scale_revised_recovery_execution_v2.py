"""Synthetic terminal-policy tests; no audit, grading or scheduler execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path('/n/fs/similarity/maxent-grpo')
SOURCE = ROOT/'artifacts/verify_modebench_scale_revised_recovery_execution_v2_20260912.py'


@pytest.fixture
def u():
    spec = importlib.util.spec_from_file_location('_test_recovery_v2', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(u, tmp_path, monkeypatch):
    observation = u.read(u.OBSERVATION)
    python_observation = u.read(u.PYTHON_OBSERVATION)
    monkeypatch.setattr(u, 'observed_python', lambda: deepcopy(python_observation))
    monkeypatch.setattr(u, 'observed_pantry', lambda: deepcopy(observation))
    root = u.base.RECOVERY
    plan = {'prepared_at_utc':'2026-09-12T17:01:00+00:00'}
    submitted = {'array_job_id':31259131,'at_utc':'2026-09-12T17:02:32.393064+00:00'}
    runtimes = [{'environment':{'SLURM_JOB_ID':raw},'hostname':node+'.ionic.cs.princeton.edu','at_utc':at}
        for raw,node,at in [('31259132','node202','2026-09-12T17:05:52+00:00'),
                            ('31259131','node203','2026-09-12T17:40:53+00:00')]]
    rows = deepcopy(observation['cell_terminal_rows'])
    rows.extend(deepcopy(python_observation['cell_terminal_rows']))
    logs = {str(root/'logs'/f'31259131_{i}{suffix}'):'a'*64 for i in (0,1) for suffix in ('.out','.err')}
    logs.update({name:digest for name,digest in observation['files_sha256'].items() if '/logs/' in name})
    record = {'schema':u.base.TERMINAL_SCHEMA,'array_job_id':31259131,'command':u.base.terminal_command(31259131),
        'returncode':0,'stderr':'','environment':{'TZ':'UTC'},'observer_host':'spin.cs.princeton.edu',
        'captured_at_utc':'2026-09-12T20:02:50+00:00','logs_sha256':logs}
    checked = []
    monkeypatch.setattr(u.base, 'pins_check', lambda pins: checked.append(dict(pins)))
    monkeypatch.setattr(u, 'read', lambda path: {'at_utc':'2026-09-12T17:02:32.100000+00:00'})
    value = SimpleNamespace(u=u,root=root,plan=plan,submitted=submitted,runtimes=runtimes,
                            rows=rows,record=record,checked=checked,observation=observation)
    return value


def terminal(case):
    case.record['stdout'] = '\n'.join('|'.join(row) for row in case.rows)+'\n'
    return case.u.recovery_terminal(case.root,None,case.plan,case.submitted,case.runtimes,case.record)


def test_literal_observed_failure_preserved_without_rewriting_input(case):
    original = deepcopy(case.rows)
    result = terminal(case)
    assert case.rows == original
    assert result[0]['state'] == 'FAILED'
    assert result[0]['exit_code'] == '1:0'
    assert result[0]['exit_cause'] == 'unknown'
    assert result[0]['scheduler_success'] is False
    assert result[1]['state'] == 'FAILED'
    assert result[1]['exit_code'] == '143:0'
    assert result[1]['exit_cause'] == 'unknown'
    assert result[1]['scheduler_success'] is False
    assert len(case.checked) == 1


@pytest.mark.parametrize('index,state,code',[(3,'RUNNING','0:0'),(3,'FAILED','1:0'),
    (3,'COMPLETED','0:0'),(3,'NODE_FAIL','1:0'),(3,'COMPLETED','1:0'),
    (4,'FAILED','1:0'),(5,'FAILED','1:0'),(0,'COMPLETED','0:0'),(1,'COMPLETED','0:0'),(2,'FAILED','1:0')])
def test_no_future_or_normalized_failure_policy(case,index,state,code):
    case.rows[index][2:4] = [state,code]
    with pytest.raises(ValueError): terminal(case)


@pytest.mark.parametrize('row,field,value',[(0,0,'other'),(0,5,'2026-09-12T17:37:53'),
    (1,10,'2026-09-12T17:02:32'),(3,0,'other'),(3,6,'node202'),(3,7,'7'),
    (3,8,'64G'),(3,9,'cpu=6,cpu=6'),(3,11,'mltheory'),(3,12,'all'),
    (4,4,'2026-09-12T17:00:00'),(5,5,'2026-09-12T19:00:01'),
    (3,10,'2026-09-12T17:01:00')])
def test_exact_observed_identity_resources_and_chronology(case,row,field,value):
    case.rows[row][field] = value
    with pytest.raises(ValueError): terminal(case)


@pytest.mark.parametrize('field,value',[('array_job_id',31258973),('observer_host','wash.cs.princeton.edu'),
    ('returncode',1),('stderr','scheduler unavailable'),('environment',{'TZ':'EST'}),
    ('captured_at_utc','2026-09-12T18:00:00+00:00')])
def test_exact_successful_final_observation_required(case,field,value):
    case.record[field] = value
    with pytest.raises(ValueError): terminal(case)


@pytest.mark.parametrize('change',['missing_step','duplicate_step','wrong_array','missing_runtime',
    'wrong_runtime_raw','wrong_runtime_node','missing_log','changed_pantry_log','overlap'])
def test_no_incomplete_or_different_execution(case,change):
    if change == 'missing_step': case.rows.pop()
    elif change == 'duplicate_step': case.rows[-1] = deepcopy(case.rows[-2])
    elif change == 'wrong_array': case.submitted['array_job_id'] = 31258973
    elif change == 'missing_runtime': case.runtimes.pop()
    elif change == 'wrong_runtime_raw': case.runtimes[1]['environment']['SLURM_JOB_ID'] = '999'
    elif change == 'wrong_runtime_node': case.runtimes[1]['hostname'] = 'node202'
    elif change == 'missing_log': case.record['logs_sha256'].pop(next(iter(case.record['logs_sha256'])))
    elif change == 'changed_pantry_log': case.record['logs_sha256'][str(case.root/'logs/31259131_0.out')] = 'b'*64
    else:
        for row in case.rows[3:]: row[4] = '2026-09-12T17:37:53'
    with pytest.raises(ValueError): terminal(case)


def test_certificate_keeps_full_scientific_coverage_but_records_recovery_failure(case,monkeypatch):
    original = {'schema':'native','scheduler_success':False,'recovery_scheduler_success':True,
        'recovery_executions':terminal(case),'completed_receipts':8,'completed_batches':864,
        'attempts_validated':54400,'new_grader_invocations':54400,'scientific_outputs_complete':True}
    monkeypatch.setattr(case.u, '_original_build_certificate', lambda root: deepcopy(original))
    result = case.u.build_certificate(case.root)
    assert result['recovery_scheduler_success'] is False
    assert result['observed_failed_recovery_cells'] == [0,1]
    assert result['observed_python_failure_sha256'] == case.u.PYTHON_OBSERVATION_SHA
    assert result['observed_pantry_failure_sha256'] == case.u.OBSERVATION_SHA
    assert all(result[key] == original[key] for key in original if key != 'recovery_scheduler_success')


def test_inspection_adds_base_and_observation_closure_without_scientific_replay(case,monkeypatch):
    expected = SimpleNamespace(pins={'native':'c'*64})
    calls = []
    monkeypatch.setattr(case.u, '_original_inspect', lambda root,digest: (calls.append((root,digest)),expected)[1])
    result = case.u.inspect(case.root,'d'*64)
    assert calls == [(case.root,'d'*64)]
    assert result.pins[str(case.u.BASE)] == case.u.BASE_SHA
    assert result.pins[str(case.u.OBSERVATION)] == case.u.OBSERVATION_SHA
    assert all(result.pins[name] == digest for name,digest in case.observation['files_sha256'].items())


def test_original_audit_and_readonly_verification_entrypoints_remain_inherited(u):
    assert u.audit is u.base.audit
    assert u.register is u.base.register
    assert u.verify_existing is u.base.verify_existing
    assert u.base.inspect is u.inspect
    assert u.base.build_certificate is u.build_certificate


@pytest.mark.parametrize('field,value',[('full_array_terminal',True),('scientific_audit_performed',True),
    ('exit_cause','proven child shutdown'),('array_index',1),('completed_receipts',3)])
def test_observation_cannot_be_relabelled_as_audit_or_other_event(u,monkeypatch,field,value):
    observation = u.read(u.OBSERVATION)
    observation[field] = value
    monkeypatch.setattr(u, 'read', lambda path: observation)
    with pytest.raises(ValueError): u.observed_pantry()


def test_literal_cli_dispatch_uses_adapter_and_does_not_audit_on_verify(u,monkeypatch,capsys):
    value = {'schema':'test','status':'verified','array_job_id':31259131,'files_sha256':{'x':'y'}}
    calls = []
    monkeypatch.setattr(u.sys, 'argv', [str(u.SOURCE),'verify'])
    monkeypatch.setattr(u, 'verify_existing', lambda root: (calls.append(root),value)[1])
    monkeypatch.setattr(u, 'audit', lambda *args: pytest.fail('must not audit during verify'))
    u.main()
    assert calls == [u.base.RECOVERY]
    assert 'verified' in capsys.readouterr().out


def test_literal_cli_refuses_audit_without_final_terminal_sha(u,monkeypatch):
    monkeypatch.setattr(u.sys, 'argv', [str(u.SOURCE),'audit'])
    monkeypatch.setattr(u, 'audit', lambda *args: pytest.fail('must not claim without final capture'))
    with pytest.raises(ValueError,match='explicit final terminal SHA'): u.main()


@pytest.fixture
def mixed_workflow(u,tmp_path,monkeypatch):
    path = ROOT/'tests/test_modebench_scale_revised_recovery_execution.py'
    spec = importlib.util.spec_from_file_location('_recovery_v2_inherited_scratch_fixtures',path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    monkeypatch.setattr(fixtures,'load',lambda:u.base)
    t = fixtures.t.__wrapped__(tmp_path,monkeypatch)
    # Retain the actual adapter source identity; every mutable workflow record
    # and every stub grader call belongs only to the scratch recovery root.
    monkeypatch.setattr(u.base,'SOURCE',u.SOURCE)
    w = fixtures.workflow.__wrapped__(t,monkeypatch)
    scientific_stub = u.base.inspect
    monkeypatch.setattr(u,'_original_inspect',scientific_stub)
    monkeypatch.setattr(u.base,'inspect',u.inspect)
    w.context.recovery_executions = [
        {'state':'FAILED','exit_code':'1:0','scheduler_success':False,'exit_cause':'unknown'},
        {'state':'FAILED','exit_code':'143:0','scheduler_success':False,'exit_cause':'unknown'}]
    w.adapter = u
    return w


def test_inherited_eight_task_audit_and_adapter_certificate_compose_once(mixed_workflow):
    w = mixed_workflow
    u = w.adapter
    result = u.audit(w.t.root,w.digest)
    registration = u.base.read(w.t.root/'execution_audit/registration.json')
    assert registration['script'] == str(u.SOURCE)
    assert registration['script_sha256'] == u.base.file_sha(u.SOURCE)
    assert registration['terminal_policy'] == u.POLICY
    assert registration['files_sha256'][str(u.BASE)] == u.BASE_SHA
    assert registration['files_sha256'][str(u.OBSERVATION)] == u.OBSERVATION_SHA
    assert w.graders == [(i,tier) for i in (0,1) for tier in range(4)]
    assert result['new_grader_invocations'] == 54400
    assert result['completed_receipts'] == 8 and result['completed_batches'] == 864
    assert result['recovery_scheduler_success'] is False
    assert result['recovery_executions'][0]['exit_cause'] == 'unknown'
    assert u.verify_existing(w.t.root) == result
    assert len(w.graders) == 8
    with pytest.raises(ValueError,match='already claimed'): u.audit(w.t.root,w.digest)
    assert len(w.graders) == 8


def test_adapter_composition_refuses_missing_final_receipt_before_grader_claim(mixed_workflow):
    w = mixed_workflow
    w.state.fail_inspect = True
    with pytest.raises(ValueError,match='incomplete'): w.adapter.audit(w.t.root,w.digest)
    assert not w.graders
    assert not (w.t.root/'execution_audit/claim.json').exists()


def test_adapter_composition_keeps_partial_grader_failure_and_never_replays(mixed_workflow):
    w = mixed_workflow
    w.state.fail_grade = (1,1)
    with pytest.raises(ValueError,match='grader disagrees'): w.adapter.audit(w.t.root,w.digest)
    assert len(w.graders) == 6
    assert (w.t.root/'execution_audit/failure.json').is_file()
    with pytest.raises(ValueError,match='already claimed'): w.adapter.audit(w.t.root,w.digest)
    assert len(w.graders) == 6
    assert not (w.t.root/'execution_reconciliation.json').exists()


@pytest.mark.parametrize('field,value',[('full_array_terminal',False),('scientific_audit_performed',True),
    ('exit_cause','proven child shutdown'),('array_index',0),('completed_receipts',3),('source_terminal_sha256','0'*64)])
def test_python_observation_cannot_be_relabelled_or_detached_from_actual_terminal(u,monkeypatch,field,value):
    observation=u.read(u.PYTHON_OBSERVATION);observation[field]=value
    monkeypatch.setattr(u,'read',lambda path:observation)
    with pytest.raises(ValueError):u.observed_python()


def test_both_exact_failed_cell_observations_are_in_audit_closure(mixed_workflow):
    w=mixed_workflow;u=w.adapter;result=u.audit(w.t.root,w.digest)
    assert result['observed_failed_recovery_cells']==[0,1]
    assert result['observed_python_failure_sha256']==u.PYTHON_OBSERVATION_SHA
    assert result['files_sha256'][str(u.PYTHON_OBSERVATION)]==u.PYTHON_OBSERVATION_SHA
    assert result['files_sha256'][str(u.FINAL_TERMINAL)]==u.FINAL_TERMINAL_SHA
    assert [r['exit_code'] for r in result['recovery_executions']]==['1:0','143:0']
