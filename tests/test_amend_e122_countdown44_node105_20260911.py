from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops/exp_scaling'))
import amend_e122_countdown44_node105_20260911 as m


def test_pool_is_exact_additive_node105_only():
    assert m.NEW_NODES-m.OLD_NODES=={'node105'}
    assert set(m.JOBS)=={'31158682','31158683','31158684'}


def test_prior_hold_or_started_allocation_is_preserved():
    for text in ('Priority=0 Reason=JobHeldUser RunTime=00:00:00 Restarts=0',
                 'Priority=1 Reason=Priority RunTime=00:00:01 Restarts=0',
                 'Priority=1 Reason=Priority RunTime=00:00:00 Restarts=1'):
        with pytest.raises(RuntimeError):m.require_released(text)
    m.require_released('Priority=1 Reason=Priority RunTime=00:00:00 Restarts=0')


def test_changed_resource_fails_before_scheduler_update(monkeypatch):
    original={k:'same' for k in m.prior.PRESERVE}
    original.update(JobId='31158682',MinMemoryNode='128G',NumCPUs='8',TresPerNode='gres/gpu:1',
                    TimeLimit='1-12:00:00',NumNodes='1-1')
    monkeypatch.setattr(m.c,'field',lambda record,key:record[key])
    monkeypatch.setattr(m.prior.base,'submit_tokens',lambda record:['fixed'])
    monkeypatch.setattr(m.prior.base,'exports',lambda tokens:{'SEED':'44'})
    cell={'environment':{'SEED':'44'}}
    m.audit(original,original,'31158682',cell)
    for key,value in [('MinMemoryNode','40G'),('NumCPUs','4'),('Account','mltheory'),('TimeLimit','1:00:00')]:
        with pytest.raises(RuntimeError):m.audit({**original,key:value},original,'31158682',cell)


def test_uncertain_prior_apply_is_never_repeated(monkeypatch,tmp_path):
    from contextlib import nullcontext
    monkeypatch.setattr(m,'ART',tmp_path)
    monkeypatch.setattr(m,'verify_plan',lambda pin:{})
    monkeypatch.setattr(m.recovery,'admission_locks',nullcontext)
    monkeypatch.setattr(m.c,'locked',lambda path:nullcontext())
    (tmp_path/'apply.intent.json').write_text('{}')
    monkeypatch.setattr(m.c,'command',lambda *args:pytest.fail('duplicate mutation'))
    with pytest.raises(RuntimeError,match='never repeat'):m.apply('pin')


def test_only_oldest_job_gets_owner_route_and_peers_retain_e118_pool():
    assert m.ROUTES['31158682']=={'nodes':['node105'],'Account':'mltheory','Partition':'mltheory','QOS':'medium'}
    for jid in ('31158683','31158684'):
        assert m.OLD_NODES <= set(m.ROUTES[jid]['nodes'])
        assert m.ROUTES[jid]['Account']=='allcs' and m.ROUTES[jid]['Partition']=='lowprio'


def test_owner_partition_can_only_change_derived_billing(monkeypatch):
    original={k:'same' for k in m.prior.PRESERVE}
    original.update(JobId='31158682',MinMemoryNode='128G',NumCPUs='8',TresPerNode='gres/gpu:1',
                    TimeLimit='1-12:00:00',NumNodes='1-1',Account='allcs',Partition='lowprio',QOS='medium',
                    ReqTRES='cpu=8,mem=128G,node=1,billing=3,gres/gpu=1')
    monkeypatch.setattr(m.c,'field',lambda record,key:record[key])
    monkeypatch.setattr(m.prior.base,'submit_tokens',lambda record:['fixed'])
    monkeypatch.setattr(m.prior.base,'exports',lambda tokens:{'SEED':'44'})
    after={**original,'Account':'mltheory','Partition':'mltheory','ReqTRES':original['ReqTRES'].replace('billing=3','billing=25')}
    m.audit(after,original,'31158682',{'environment':{'SEED':'44'}},after=True)
    after['ReqTRES']=after['ReqTRES'].replace('mem=128G','mem=40G')
    with pytest.raises(RuntimeError):m.audit(after,original,'31158682',{'environment':{'SEED':'44'}},after=True)


def test_unknown_or_duplicate_run_directory_writer_blocks(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(m.recovery,'verify_plan',lambda:{'storage_module':'fixture'})
    report={'allowed':True,'errors':[],'writer_profiles':[{'job_id':'31158682','run_dir':'exact'}],
            'observed_at':'now','writer_identity_sha256':'pin','required_bytes':1,'margin_bytes':2}
    monkeypatch.setattr(m.recovery.importlib,'import_module',lambda name:SimpleNamespace(storage_report=lambda:report))
    assert m.sole_writer('31158682',{'run_dir':'exact'})['writer']['job_id']=='31158682'
    report['writer_profiles'].append({'job_id':'other','run_dir':'exact'})
    with pytest.raises(RuntimeError):m.sole_writer('31158682',{'run_dir':'exact'})
    report.update(allowed=False,errors=['unknown writer'])
    with pytest.raises(RuntimeError):m.sole_writer('31158682',{'run_dir':'exact'})
