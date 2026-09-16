"""Offline fixtures for exact-array classification and additive storage gates."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
import e122_storage_with_modebench_array_20260911 as a


def num(value=None):
    return {'set':value is not None,'infinite':False,'number':value or 0}


@pytest.fixture
def fixture(monkeypatch):
    qualified={'plan_path':str(ROOT/'var/artifacts/test_array/plan.json'),'plan_sha256':'a'*64,
        'cells':[{'array_task_id':i,'recommended_future_reserve_bytes':a.PER_TASK_BYTES,
            'model_label':'7b' if i<5 else '14b','model_source':'cached/model',
            'cell_id':'cell'+str(i),'outputs':[{'output':str(ROOT/'var/data/inference'/str(i)/'receipt.json')}]}
            for i in range(10)]}
    plan={'submit_command':['sbatch','--array=0-9%1','worker.slurm']}
    active={'job_id':31245163,'array_job_id':num(a.ARRAY_ID),'array_task_id':num(4),
        'array_task_string':'','name':'modebench-scale-v3-dev','job_state':['RUNNING'],
        'state_reason':'None','tres_per_node':'gres/gpu:a5000:2',
        'command':str(ROOT/'var/artifacts/test_array/worker.slurm'),
        'submit_line':'sbatch --array=0-9%1 worker.slurm'}
    pending={**active,'job_id':a.ARRAY_ID,'array_task_id':num(),'array_task_string':'5-9%1',
        'job_state':['PENDING'],'state_reason':'JobArrayTaskLimit'}
    ordinary={'job_id':123,'array_job_id':num(0),'array_task_id':num(),
        'array_task_string':'','name':'training','job_state':['RUNNING'],
        'state_reason':'None','tres_per_node':'gres/gpu:a100:1'}
    snap={'jobs':[active,pending,ordinary],'errors':[]}
    monkeypatch.setattr(a.original.base,'canonical_registry',lambda:{'123':{'run_dir':'training'}})
    monkeypatch.setattr(a,'systems_certificate',lambda:None)
    return SimpleNamespace(qualified=qualified,plan=plan,snapshot=snap,active=active,pending=pending,ordinary=ordinary)


def test_keeps_all_six_true_array_identities_and_scalar_writer(fixture):
    training, inference=a.classify_snapshot(fixture.snapshot,fixture.qualified,fixture.plan)
    assert [r['job_id'] for r in training['jobs']]==['123']
    assert [r['job_id'] for r in inference]==[f'{a.ARRAY_ID}_{i}' for i in range(4,10)]
    assert inference[0]['numeric_job_id']=='31245163'
    assert all(r['numeric_job_id'] is None for r in inference[1:])
    assert [r['model_label'] for r in inference]==['7b']+['14b']*5
    assert sum(r['future_reserve_bytes'] for r in inference)==192*1024**3
    assert all(r['checkpoint_credit_bytes']==0 for r in inference)


@pytest.mark.parametrize('text', ['0-99','5-4','5-9%2','5,5','5-7,7-9','*','', '5-9:2'])
def test_unknown_or_duplicate_array_range_is_rejected(text):
    with pytest.raises(ValueError):a.array_indices(text)


@pytest.mark.parametrize('field,value', [('command','other/worker'),('submit_line','sbatch changed'),
    ('tres_per_node','gres/gpu:a100:1'),('name','unregistered-inference')])
def test_exact_array_command_and_resources_are_required(fixture,field,value):
    fixture.active[field]=value
    with pytest.raises(ValueError):a.classify_snapshot(fixture.snapshot,fixture.qualified,fixture.plan)


def test_unknown_array_is_never_dropped(fixture):
    fixture.active['array_job_id']=num(999)
    with pytest.raises(ValueError,match='unmapped'):a.classify_snapshot(fixture.snapshot,fixture.qualified,fixture.plan)


def test_pending_range_overlapping_running_task_is_rejected(fixture):
    fixture.pending['array_task_string']='4-9%1'
    with pytest.raises(ValueError,match='duplicate'):a.classify_snapshot(fixture.snapshot,fixture.qualified,fixture.plan)


def test_running_aggregate_cannot_hide_multiple_writers(fixture):
    fixture.pending['job_state']=['RUNNING']
    with pytest.raises(ValueError,match='unresolved'):a.classify_snapshot(fixture.snapshot,fixture.qualified,fixture.plan)


@pytest.fixture
def report_fixture(fixture,monkeypatch):
    monkeypatch.setattr(a,'certificate',lambda:(fixture.qualified,fixture.plan))
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=json.dumps(fixture.snapshot)))
    original={'schema':'e122_shared_storage_admission_v1','status':'approved','allowed':True,
        'errors':[],'unknown_writers':[],'required_bytes':500*1024**3,'free_bytes':800*1024**3,
        'external_peak_reserve_bytes':215*1024**3,'free_inodes':20000,'reservations':[],
        'writer_profiles':[{'job_id':'123','model_choice':'3b'}],
        'writer_job_ids':['123'],'e122_terminal_reserve_bytes':125*1024**3,
        'e122_peak_reserve_bytes':96*1024**3,'shared_headroom_bytes':64*1024**3,'policy':'original fixed policy'}
    calls=[]
    def storage(**kwargs):calls.append(kwargs);return copy.deepcopy(original)
    monkeypatch.setattr(a.original,'storage_report',storage)
    return original,calls


def test_storage_adds_full_inference_reserve_and_preserves_all_original_gates(report_fixture):
    previous,calls=report_fixture
    result=a.storage_report(include_held_job_ids=['31158682'])
    assert result['allowed'] and result['required_bytes']==692*1024**3
    assert result['external_peak_reserve_bytes']==407*1024**3
    assert len(result['writer_job_ids'])==7 and '123' in result['writer_job_ids']
    assert calls[0]['include_held_job_ids']==['31158682']
    assert len(calls[0]['external_snapshot']['jobs'])==1
    assert result['e122_terminal_reserve_bytes']==125*1024**3
    assert result['e122_peak_reserve_bytes']==96*1024**3
    assert result['shared_headroom_bytes']==64*1024**3


def test_additional_inference_can_block_otherwise_passing_budget(report_fixture):
    previous,_=report_fixture;previous['free_bytes']=691*1024**3
    result=a.storage_report()
    assert not result['allowed'] and result['status']=='rejected' and result['blocked_reason']=='waiting_disk'


def test_unknown_training_writer_failure_is_never_excused(report_fixture):
    previous,_=report_fixture
    previous.update(status='rejected',allowed=False,errors=['unmapped GPU writer'],unknown_writers=['777'],required_bytes=None)
    result=a.storage_report()
    assert not result['allowed'] and result['errors']==['unmapped GPU writer'] and result['unknown_writers']==['777']


def test_certificate_change_fails_closed_before_scheduler(monkeypatch):
    monkeypatch.setattr(a,'certificate',lambda:(_ for _ in ()).throw(ValueError('source pin changed')))
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kwargs:pytest.fail('changed source reached scheduler'))
    result=a.storage_report()
    assert not result['allowed'] and result['blocked_reason']=='unresolved_storage_safety'


def test_no_inference_means_no_additional_charge(fixture,report_fixture):
    fixture.snapshot['jobs']=[fixture.ordinary]
    result=a.storage_report()
    assert result['allowed'] and result['evaluation_future_reserve_bytes']==0 and result['required_bytes']==500*1024**3


def test_bounded_zip_is_scoped_to_budget_call(report_fixture,monkeypatch):
    before=a.original.base.zipfile.ZipFile
    original_storage=a.original.storage_report
    def scoped(**kwargs):
        assert a.original.base.zipfile.ZipFile is a.bounded_zip.BoundedZipFile
        return original_storage(**kwargs)
    monkeypatch.setattr(a.original,'storage_report',scoped)
    assert a.storage_report()['allowed']
    assert a.original.base.zipfile.ZipFile is before


def test_bounded_zip_is_restored_after_budget_exception(report_fixture,monkeypatch):
    before=a.original.base.zipfile.ZipFile
    def broken(**kwargs):
        assert a.original.base.zipfile.ZipFile is a.bounded_zip.BoundedZipFile
        raise ValueError('fixture budget failed')
    monkeypatch.setattr(a.original,'storage_report',broken)
    assert not a.storage_report()['allowed']
    assert a.original.base.zipfile.ZipFile is before


@pytest.fixture
def systems(monkeypatch,fixture):
    value={'job_id':a.SYSTEMS_ID,'command':'/exact/systems.slurm','name':'e124-7b-systems',
        'gres':'gres/gpu:a6000:1','comment':'exact-comment','submit_command':['sbatch','/exact/systems.slurm'],
        'run_dir':str(ROOT/'var/artifacts/e124_qwen7b_three_level/systems'),
        'plan_path':str(ROOT/'var/artifacts/e124_qwen7b_three_level/systems/plan.json'),'plan_sha256':'b'*64}
    raw={**fixture.ordinary,'job_id':a.SYSTEMS_ID,'name':value['name'],'command':value['command'],
        'tres_per_node':value['gres'],'submit_line':'sbatch /exact/systems.slurm',
        'comment':value['comment'],'account':'mltheory','partition':'lowprio',
        'job_state':['PENDING'],'state_reason':'Resources'}
    fixture.snapshot['jobs'].append(raw)
    monkeypatch.setattr(a,'systems_certificate',lambda:value)
    return value,raw


def test_exact_pending_systems_gets_full_reserve_and_keeps_identity(fixture,systems,report_fixture):
    result=a.storage_report()
    assert result['benchmark_future_reserve_bytes']==220*1024**3
    assert result['required_bytes']==912*1024**3 and not result['allowed']
    row=result['benchmark_writers'][0]
    assert row['job_id']==row['numeric_job_id']==str(a.SYSTEMS_ID)
    assert row['model_choice']=='7b' and row['checkpoint_credit_bytes']==0
    assert len(result['writer_job_ids'])==8


@pytest.mark.parametrize('field,value',[('command','changed'),('comment','changed'),
    ('submit_line','sbatch changed'),('account','other'),('partition','other'),
    ('tres_per_node','gres/gpu:a100:1')])
def test_changed_systems_writer_fails_closed(fixture,systems,report_fixture,field,value):
    systems[1][field]=value
    result=a.storage_report()
    assert not result['allowed'] and result['blocked_reason']=='unresolved_storage_safety'


def test_systems_certificate_pin_change_fails_closed(monkeypatch):
    monkeypatch.setattr(a.original,'digest',lambda p:'changed')
    with pytest.raises(ValueError,match='systems certificate changed'):a.systems_certificate()
