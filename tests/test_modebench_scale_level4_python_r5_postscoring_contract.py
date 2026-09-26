"""Scratch-only prospective guards: no scheduler, generator, grader or fitter."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r4_postscoring_contract', ROOT/'artifacts/modebench_scale_level4_python_r5_postscoring_contract_20260913.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@pytest.fixture
def terminal(tmp_path):
    root = tmp_path/'development'; (root/'logs').mkdir(parents=True)
    (root/'submission_intent.json').write_text(json.dumps({'at_utc':'2026-09-13T10:00:01+00:00'}))
    logs={}
    for suffix in ('.out','.err'):
        path=root/'logs'/('70000001_0'+suffix);path.write_text('scratch-only synthetic evidence\n');logs[str(path)]=m.sha(path)
    plan={'created_at':'2026-09-13T10:00:00+00:00'}
    submitted={'array_job_id':70000001,'at_utc':'2026-09-13T10:00:03+00:00'}
    runtime={'array_job_id':70000001,'array_index':0,'at_utc':'2026-09-13T10:00:05+00:00',
        'environment':{'SLURM_JOB_ID':'70000001'},'hostname':'node202.cs.princeton.edu',
        'portable_evidence_staging':{'at_utc':'2026-09-13T10:00:04+00:00'},
        'gpu_metadata_probe':{'started_at_utc':'2026-09-13T10:00:04.500000+00:00'}}
    outcome={'array_job_id':70000001,'array_index':0,'returncode':0,
             'started_at_utc':'2026-09-13T10:00:06+00:00','finished_at_utc':'2026-09-13T10:09:59+00:00'}
    lines=[]
    for suffix in ('','.batch','.extern'):
        lines.append('|'.join(['70000001'+suffix,'70000001_0'+suffix,'COMPLETED','0:0',
            '2026-09-13T10:00:04','2026-09-13T10:10:00','node202','6','60G',
            'cpu=6,node=1,mem=60G,gres/gpu=2,gres/gpu:a5000=2','2026-09-13T10:00:02','allcs','cs']))
    record={'schema':m.TERMINAL_SCHEMA,'array_job_id':70000001,'command':m.terminal_command(70000001),
        'environment':{'TZ':'UTC'},'returncode':0,'stderr':'','stdout':'\n'.join(lines),
        'observer_host':m.CAPTURE_HOST,'observer_uid':m.UID,'observer_pid':999,
        'captured_at_utc':'2026-09-13T10:11:00+00:00','logs_sha256':logs}
    return root,plan,submitted,runtime,outcome,record

def test_successful_actual_shape_retains_resources_and_truthful_identity(terminal):
    value=m.terminal_execution(*terminal)
    assert value['scheduler_success'] is True
    assert value['array_job_id']==70000001 and value['state']=='COMPLETED' and value['exit_code']=='0:0'
    assert value['account']=='allcs' and value['partition']=='cs' and value['node']=='node202'

@pytest.mark.parametrize('state,code',[('FAILED','143:0'),('FAILED','1:0'),('RUNNING','0:0'),('COMPLETED','143:0'),('CANCELLED','0:15')])
@pytest.mark.parametrize('line',[0,1,2])
def test_no_failed_r3_or_other_terminal_exception_carries_forward(terminal,state,code,line):
    record=terminal[-1];rows=[r.split('|') for r in record['stdout'].splitlines()]
    rows[line][2:4]=[state,code];record['stdout']='\n'.join('|'.join(r) for r in rows)
    with pytest.raises(ValueError,match='complete0'):m.terminal_execution(*terminal)

@pytest.mark.parametrize('field,value',[
    ('observer_host','spin.cs.princeton.edu'),('observer_uid',363431),('observer_uid',True),
    ('observer_pid',False),('returncode',False),('returncode',1),('environment',{}),
    ('schema','modebench_scale_level4_python_r3_terminal_accounting_v1')])
def test_truthful_current_observer_and_exact_capture_contract_required(terminal,field,value):
    terminal[-1][field]=value
    with pytest.raises(ValueError,match='capture'):m.terminal_execution(*terminal)

@pytest.mark.parametrize('value',[1,143,False,None])
def test_evaluator_must_return_literal_zero(terminal,value):
    terminal[-2]['returncode']=value
    with pytest.raises(ValueError,match='evaluator exit0'):m.terminal_execution(*terminal)

@pytest.mark.parametrize('job',[31260226,True,0,-1,'70000001'])
def test_old_or_ambiguous_job_is_rejected(job):
    with pytest.raises(ValueError):m.terminal_command(job)

@pytest.mark.parametrize('needle,replacement',[
    ('mem=60G','mem=48G'),('gres/gpu=2','gres/gpu=1'),('|6|60G|','|4|60G|'),
    ('|allcs|cs','|other|cs'),('|allcs|cs','|allcs|other')])
def test_original_resource_contract_is_retained(terminal,needle,replacement):
    terminal[-1]['stdout']=terminal[-1]['stdout'].replace(needle,replacement)
    with pytest.raises(ValueError,match='resources'):m.terminal_execution(*terminal)

def test_changed_log_bytes_reject_even_with_successful_accounting(terminal):
    Path(next(iter(terminal[-1]['logs_sha256']))).write_text('changed')
    with pytest.raises(ValueError,match='input changed'):m.terminal_execution(*terminal)

def test_terminal_must_follow_evaluator(terminal):
    terminal[-2]['finished_at_utc']='2026-09-13T10:12:00+00:00'
    with pytest.raises(ValueError,match='chronology'):m.terminal_execution(*terminal)

@pytest.fixture
def audit():
    return {'schema':m.AUDIT_SCHEMA,'status':m.AUDIT_STATUS,'terminal_policy':m.POLICY,'array_job_id':70000001,
        'terminal_sha256':'a'*64,'scheduler_success':True,'evaluator_returncode':0,
        'scientific_outputs_complete':True,'completed_receipts':4,'completed_batches':400,
        'attempts_validated':24704,'new_grader_invocations':24704,'summaries':[{}]*4,'grader_audits':[{}]*4,
        'fit_performed':False,'model_sampling_performed':False,'source_selection_performed':False}

def test_complete_audit_shape_is_readonly_prerequisite_only(audit):
    assert m.require_complete_audit(audit,70000001,'a'*64) is audit

@pytest.mark.parametrize('field,value',[
    ('scheduler_success',False),('status','verified_scientific_outputs_with_failed_execution'),
    ('terminal_policy','exact_observed_31260226_failed_143_evaluator_0_only'),
    ('completed_receipts',3),('completed_batches',399),('attempts_validated',24703),
    ('new_grader_invocations',0),('fit_performed',True),('source_selection_performed',True),
    ('model_sampling_performed',True),('evaluator_returncode',False),('terminal_sha256','b'*64)])
def test_partial_or_wrong_actual_audit_cannot_qualify_for_fit(audit,field,value):
    audit[field]=value
    with pytest.raises(ValueError,match='complete actual'):m.require_complete_audit(audit,70000001,'a'*64)

@pytest.mark.parametrize('prefix',[
    ['var/seed_paper_eval/paper310/bin/python','-B','/absolute/source.py','fit'],
    [str(m.PYTHON),'-B','artifacts/source.py','fit'],
    [str(m.PYTHON),'-B','/absolute/source.py','audit'],
    [str(m.PYTHON),'/absolute/source.py','fit'],
])
def test_relative_or_wrong_outer_prefix_rejected_before_any_state(monkeypatch,prefix):
    monkeypatch.setattr(m.socket,'gethostname',lambda:m.HOST)
    monkeypatch.setattr(m.os,'uname',lambda:SimpleNamespace(nodename=m.HOST))
    monkeypatch.setattr(m.os,'getuid',lambda:m.UID);monkeypatch.setattr(m.os,'geteuid',lambda:m.UID)
    monkeypatch.setattr(m,'sha',lambda p:m.NEUTRAL_SHA)
    monkeypatch.setattr(Path,'read_bytes',lambda p:('\0'.join(prefix)+'\0').encode())
    with pytest.raises(ValueError,match='full absolute'):m.outer_command_guard('/absolute/source.py','fit')


def test_native_pinned_venv_style_symlink_is_authenticated_without_path_rewrite(tmp_path):
    actual=tmp_path/'python3.10';actual.write_bytes(b'scratch interpreter bytes')
    literal=tmp_path/'python';literal.symlink_to(actual)
    m.check_pins({str(literal):m.sha(literal)},guest=False)
    actual.write_bytes(b'changed interpreter bytes')
    with pytest.raises(ValueError,match='input changed'):
        m.check_pins({str(literal):'0'*64},guest=False)


def bind_authorized_terminal(module, monkeypatch, root, plan, submitted, runtime, outcome, record):
    """Synthetic authorization/decision archive, never production evidence."""
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return path
    job=submitted['array_job_id'];monkeypatch.setattr(module,'OBSERVED_JOB',job)
    monkeypatch.setattr(module,'DEVELOPMENT',root)
    rows=[line.split('|') for line in record['stdout'].splitlines()]
    for row in rows:
        row[2:4]=['COMPLETED','0:0'] if row[1].endswith('.extern') else ['FAILED','143:0']
    record['stdout']='\n'.join('|'.join(row) for row in rows)+'\n'
    consumed={'plan.json':plan,'submission_result.json':submitted,'runtime/0.json':runtime,
              'runtime/0.evaluator_exit.json':outcome,'terminal_accounting.json':record}
    for raw,value in consumed.items():put(root/raw,value)
    # Revision 5 links decision -> authorization, because the authorization was
    # recorded before the decision existed. Revision 4 pinned the other direction.
    authorization=put(root.parent/'authorization.json',{
        'schema':'modebench_scale_level4_python_r5_user_authorization_v1','authorized_job_id':job,
        'files_sha256':{}})
    decision=put(root.parent/'decision.json',{
        'schema':'modebench_scale_level4_python_r5_observed_terminal_decision_v1',
        'status':'observed_failed_scheduler_evaluator_exit0_pending_scientific_audit',
        'actual_job_id':job,'scheduler_success':False,'exit_cause':'post_completion_teardown',
        'evaluator_returncode':0,'user_authorization_sha256':module.sha(authorization),
        'fit_performed':False,'model_sampling_performed':False,'native_audit_performed':False,
        'literal_terminal_stdout':record['stdout'],
        'files_sha256':{**record['logs_sha256'],**{str(root/raw):module.sha(root/raw) for raw in consumed}}})
    for name,path in [('OBSERVED_DECISION',decision),('AUTHORIZATION',authorization)]:
        monkeypatch.setattr(module,name,path);monkeypatch.setattr(module,name+'_SHA',module.sha(path))
    return SimpleNamespace(decision=decision,authorization=authorization,put=put,consumed=consumed)

@pytest.fixture
def authorized(terminal,monkeypatch):
    return terminal,bind_authorized_terminal(m,monkeypatch,*terminal)

def test_exact_authorized_failed143_retains_literal_failure_and_user_evidence(authorized):
    terminal,evidence=authorized;value=m.terminal_execution(*terminal)
    assert value['scheduler_success'] is False and value['state']=='FAILED' and value['exit_code']=='143:0'
    assert value['exit_cause']=='post_completion_teardown' and value['evaluator_returncode']==0
    assert value['user_authorization_sha256']==m.sha(evidence.authorization)
    assert value['observed_terminal_decision_sha256']==m.sha(evidence.decision)

@pytest.mark.parametrize('kind',['authorization','decision','terminal','runtime','sidecar','plan','submission','log'])
def test_authorized_exception_rejects_any_changed_actual_or_permission_bytes(authorized,kind):
    terminal,e=authorized;root=terminal[0]
    paths={'authorization':e.authorization,'decision':e.decision,
        'terminal':root/'terminal_accounting.json','runtime':root/'runtime/0.json',
        'sidecar':root/'runtime/0.evaluator_exit.json','plan':root/'plan.json',
        'submission':root/'submission_result.json','log':Path(next(iter(terminal[-1]['logs_sha256'])))}
    path=paths[kind];path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='input changed'):m.terminal_execution(*terminal)

@pytest.mark.parametrize('index',[1,2,3,4,5])
def test_exact_authorized_pins_do_not_allow_different_in_memory_records(authorized,index):
    terminal,_=authorized;value=copy.deepcopy(terminal[index]);value['unreviewed_field']=True
    changed=list(terminal);changed[index]=value
    with pytest.raises(ValueError,match='only exact observed'):m.terminal_execution(*changed)

@pytest.mark.parametrize('field,value',[('scheduler_success',True),('exit_cause','unknown'),
    ('evaluator_returncode',False),('fit_performed',True),('actual_job_id',123)])
def test_even_rehashed_decision_cannot_normalize_or_expand_failed_outcome(authorized,monkeypatch,field,value):
    terminal,e=authorized;decision=m.read(e.decision);decision[field]=value;e.put(e.decision,decision)
    monkeypatch.setattr(m,'OBSERVED_DECISION_SHA',m.sha(e.decision))
    # r5 links decision -> authorization; rewriting the authorization here would break
    # the decision's user_authorization_sha256 pin, so it is deliberately left alone.
    with pytest.raises(ValueError,match='literal observed failed'):m.terminal_execution(*terminal)

@pytest.mark.parametrize('suffix',['','.batch','.extern'])
def test_even_rehashed_authorized_rows_must_keep_exact_failed143_and_extern0(authorized,monkeypatch,suffix):
    terminal,e=authorized;record=terminal[-1];rows=[r.split('|') for r in record['stdout'].splitlines()]
    row=next(row for row in rows if row[1]==str(terminal[2]['array_job_id'])+'_0'+suffix)
    row[2:4]=['FAILED','143:0'] if suffix=='.extern' else ['COMPLETED','0:0']
    record['stdout']='\n'.join('|'.join(r) for r in rows)+'\n';e.put(terminal[0]/'terminal_accounting.json',record)
    d=m.read(e.decision);d['literal_terminal_stdout']=record['stdout'];d['files_sha256'][str(terminal[0]/'terminal_accounting.json')]=m.sha(terminal[0]/'terminal_accounting.json');e.put(e.decision,d)
    monkeypatch.setattr(m,'OBSERVED_DECISION_SHA',m.sha(e.decision))
    # r5 links decision -> authorization; rewriting the authorization here would break
    # the decision's user_authorization_sha256 pin, so it is deliberately left alone.
    with pytest.raises(ValueError,match='exact authorized failed143'):m.terminal_execution(*terminal)

def test_authorization_for_other_job_is_rejected_even_with_rehashed_file(authorized,monkeypatch):
    terminal,e=authorized;v=m.read(e.authorization);v['authorized_job_id']+=1;e.put(e.authorization,v)
    monkeypatch.setattr(m,'AUTHORIZATION_SHA',m.sha(e.authorization))
    with pytest.raises(ValueError,match='exact-job user authorization'):m.terminal_execution(*terminal)

def test_complete_authorized_audit_still_requires_all_native_checks(authorized,audit):
    terminal,_=authorized;audit.update(m.execution_classification(terminal[2]['array_job_id']))
    audit['terminal_sha256']=m.sha(terminal[0]/'terminal_accounting.json')
    assert m.require_complete_audit(audit,audit['array_job_id'],audit['terminal_sha256']) is audit
    audit['new_grader_invocations']=0
    with pytest.raises(ValueError,match='complete actual'):m.require_complete_audit(audit,audit['array_job_id'],audit['terminal_sha256'])


# --- Real-record binding ------------------------------------------------------
# Every other test here builds a synthetic record and stamps it FROM the constant
# under test, so the fixture moves with the constant and can never confront it with
# what is actually on disk. That blind spot hid an r4 exit_cause literal in the
# auditor's gate and then an r4 TERMINAL_SCHEMA here. These read the sealed records.

REAL_TERMINAL = ROOT/'var/artifacts/modebench_scale_level4_python_r5_20260913/development/terminal_accounting.json'


def _real(path):
    import json
    return json.loads(Path(path).read_text())


def test_terminal_schema_matches_the_sealed_capture_on_disk():
    assert m.TERMINAL_SCHEMA == _real(REAL_TERMINAL)['schema']


def test_capture_host_matches_the_sealed_capture_on_disk():
    assert m.CAPTURE_HOST == _real(REAL_TERMINAL)['observer_host']


def test_observed_job_matches_the_sealed_capture_and_submission_on_disk():
    root=ROOT/'var/artifacts/modebench_scale_level4_python_r5_20260913/development'
    assert m.OBSERVED_JOB == _real(REAL_TERMINAL)['array_job_id']
    assert m.OBSERVED_JOB == _real(root/'submission_result.json')['array_job_id']


def test_live_host_constant_matches_the_machine_this_audit_will_run_on():
    import socket
    assert m.HOST == socket.gethostname()


def test_capture_host_and_live_host_are_not_conflated():
    """They were one constant until the audit moved machines. If a future edit
    collapses them, the contract would assert the capture happened where the audit
    ran, which is falsification that passes silently because both sides agree."""
    assert m.HOST != m.CAPTURE_HOST
    assert _real(REAL_TERMINAL)['observer_host'] != m.HOST
