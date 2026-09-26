"""Budget and failure-gate regressions for the additive E124 JSON adapter."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import e124_storage_with_modebench_array_20260911 as a


@pytest.fixture
def context(tmp_path, monkeypatch):
    (tmp_path / 'var/data').mkdir(parents=True)
    (tmp_path / 'var/artifacts/e124_qwen7b_three_level/systems').mkdir(parents=True)
    monkeypatch.setattr(a.base, 'ROOT', tmp_path)
    monkeypatch.setattr(a.base.os, 'statvfs', lambda _: SimpleNamespace(
        f_bavail=2000 * a.base.GIB, f_frsize=1, f_favail=100000))
    monkeypatch.setattr(a, 'pins', lambda: ({}, {}, {}))
    monkeypatch.setattr(a.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='{}'))
    training = {'jobs': [], 'registry': {}, 'benchmark_writers': []}
    inference = [dict(job_id='31243495_4', future_reserve_bytes=32*a.base.GIB)]
    monkeypatch.setattr(a.arrays, 'classify_snapshot', lambda *args: (copy.deepcopy(training), copy.deepcopy(inference)))
    plan = {'root': str(tmp_path), 'cells': [dict(cell_id='science', model_size='7b',
            run_dir=str(tmp_path / 'var/data/e124/test'))], 'benchmark_cells': [
            dict(cell_id='systems', model_size='7b', run_dir=str(tmp_path / 'var/artifacts/e124_qwen7b_three_level/systems'))]}
    return SimpleNamespace(root=tmp_path, plan=plan, training=training, inference=inference)


def test_original_gate_plus_complete_shared_reserves(context, monkeypatch):
    required = (220+64+125+96+32)*a.base.GIB
    monkeypatch.setattr(a.base.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=required, f_frsize=1, f_favail=100000))
    report=a.storage_report(context.plan)
    assert report['allowed'] and report['required_bytes']==required
    assert report['original_required_bytes']==(220+64)*a.base.GIB
    monkeypatch.setattr(a.base.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=required-1, f_frsize=1, f_favail=100000))
    report=a.storage_report(context.plan)
    assert not report['allowed'] and report['blocked_reason']=='waiting_disk'


def test_released_systems_consumes_own_slot_and_full_peak(context):
    row=context.plan['benchmark_cells'][0]
    context.training['benchmark_writers']=[dict(job_id='31161634', run_dir=row['run_dir'],
        state='PENDING', gres='gres/gpu:a6000:1', reason='Resources')]
    tx={'rows': {'systems': {'status':'released','job_id':31161634}}}
    report=a.controller_storage_report(context.plan,tx)
    assert not report['allowed'] and report['blocked_reason']=='own_concurrency_cap'
    assert report['own_live_count']==1 and report['own_live_reserve_bytes']==220*a.base.GIB


def test_pending_training_writer_is_still_charged(context):
    context.training['jobs']=[dict(job_id='123', state='PENDING', gres='gres/gpu:1',
        reason='Priority', run_dir=str(context.root/'var/data/external'), model_choice='3b')]
    report=a.storage_report(context.plan)
    assert report['external_reserve_bytes']==82*a.base.GIB
    assert report['required_bytes']==(220+64+82+125+96+32)*a.base.GIB


def test_unknown_training_writer_stays_blocked(context):
    context.training['jobs']=[dict(job_id='123', state='RUNNING', gres='gres/gpu:1', reason='None')]
    report=a.storage_report(context.plan)
    assert not report['allowed'] and 'unmapped external GPU writer' in report['errors'][0]


def test_changed_array_identity_stays_blocked(context, monkeypatch):
    def fail(*args): raise ValueError('unmapped or changed array GPU writer')
    monkeypatch.setattr(a.arrays, 'classify_snapshot', fail)
    report=a.storage_report(context.plan)
    assert not report['allowed'] and report['blocked_reason']=='unresolved_storage_safety'


def test_source_pin_failure_precedes_scheduler(context, monkeypatch):
    def fail(): raise ValueError('source pin changed')
    monkeypatch.setattr(a, 'pins', fail)
    monkeypatch.setattr(a.subprocess, 'run', lambda *args,**kwargs: pytest.fail('queried after failed pin'))
    assert not a.storage_report(context.plan)['allowed']


def test_low_inode_gate_is_preserved(context, monkeypatch):
    monkeypatch.setattr(a.base.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=2000*a.base.GIB, f_frsize=1, f_favail=9999))
    report=a.storage_report(context.plan)
    assert not report['allowed'] and report['blocked_reason']=='insufficient_inodes'


def test_original_zip_restored_on_budget_exception(context, monkeypatch):
    previous=a.base.zipfile.ZipFile
    def fail(*args,**kwargs):
        assert a.base.zipfile.ZipFile is a.arrays.bounded_zip.BoundedZipFile
        raise ValueError('budget failed')
    monkeypatch.setattr(a.base, 'storage_report', fail)
    assert not a.storage_report(context.plan)['allowed']
    assert a.base.zipfile.ZipFile is previous


@pytest.mark.parametrize('state',['released','release_intent','requeue_intent','held_retry'])
def test_ambiguous_or_retry_owned_intents_keep_slot(context,state):
    tx={'rows':{'science':{'status':state,'job_id':111}}}
    report=a.controller_storage_report(context.plan,tx)
    assert report['own_live_count']==1 and not report['allowed']
    report=a.controller_storage_report(context.plan,tx,releasing='science')
    assert report['own_live_count']==0 and report['allowed']
