"""Exercise the finite burst's scheduler side effects and fail-closed behavior."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
spec=importlib.util.spec_from_file_location('e122_burst_test',ROOT/'ops/exp_scaling/burst_e122_graph43_20260910.py')
burst=importlib.util.module_from_spec(spec);spec.loader.exec_module(burst)


def fixture(monkeypatch,tmp_path,*,fail_job=None,storage=True,issues=None):
    art=tmp_path/'burst';art.mkdir();planpath=art/'plan.json';planpath.write_text('{}')
    journal=tmp_path/'journal';journal.mkdir()
    selected=[{'job_id':str(j),'cell':{'identity':a}} for j,a in burst.JOBS.items()]
    binding={'pinned':'campaign'}
    plan={'binding':binding,'jobs':[j['job_id'] for j in selected]}
    calls=[];released=[]
    monkeypatch.setattr(burst,'ART',art);monkeypatch.setattr(burst,'PLAN',planpath);monkeypatch.setattr(burst,'JOURNAL',journal)
    monkeypatch.setattr(burst,'verify_plan',lambda:plan)
    monkeypatch.setattr(burst,'authenticate',lambda:({'binding':binding},selected))
    monkeypatch.setattr(burst.launcher,'audit_held',lambda j,c:'held')
    monkeypatch.setattr(burst.launcher,'audit_held_record',lambda raw,j,c:raw)
    monkeypatch.setattr(burst,'latest_heartbeat',lambda:({'observed_at':'2026-09-10T22:00:00+00:00','reserved_unfinished_slots':4},0))
    monkeypatch.setattr(burst,'route_released',lambda j:{'job_id':j['job_id'],'status':'route_expanded'})
    def status(campaign,root):
        return {'issues':issues or [],'unknown':0,'needs_operator_review_job_ids':[],
            'reserved_unfinished_slots':4+len(released),'storage':{'can_release':storage},
            'staged_held_job_ids':[j['job_id'] for j in selected if j['job_id'] not in released]}
    monkeypatch.setattr(burst.controller,'status',status)
    def run(argv,**kwargs):
        calls.append(argv)
        assert kwargs['timeout']<=3
        if argv[:2]==['scontrol','release']:
            assert burst.controller.MAX_ACTIVE==8
            if argv[-1]==fail_job:return subprocess.CompletedProcess(argv,1,'','rejected')
            released.append(argv[-1]);return subprocess.CompletedProcess(argv,0,'','')
        assert argv[:4]==['scontrol','show','job','-dd']
        return subprocess.CompletedProcess(argv,0,'held','')
    monkeypatch.setattr(burst.subprocess,'run',run)
    return art,journal,calls,released


def test_exact_balanced_four_released_once_with_original_format_journals(monkeypatch,tmp_path):
    art,journal,calls,released=fixture(monkeypatch,tmp_path)
    burst.apply()
    assert released==[str(j) for j in burst.JOBS]
    assert burst.controller.MAX_ACTIVE==4
    for jid in released:
        intent=json.loads((journal/'jobs'/f'{jid}.intent.json').read_text())
        result=json.loads((journal/'jobs'/f'{jid}.result.json').read_text())
        assert intent['schema']=='e122_level3_release_intent_v1'
        assert result['intent_sha256']==burst.controller.digest(journal/'jobs'/f'{jid}.intent.json')
        assert result['returncode']==0 and result['error'] is None
    before=len(calls)
    with pytest.raises(RuntimeError,match='already claimed'):burst.apply()
    assert len(calls)==before


def test_failed_release_consumes_intent_and_stops_remaining_jobs(monkeypatch,tmp_path):
    art,journal,calls,released=fixture(monkeypatch,tmp_path,fail_job='31158699')
    with pytest.raises(RuntimeError,match='Uncertain release'):burst.apply()
    assert released==['31158698'] and burst.controller.MAX_ACTIVE==4
    assert json.loads((journal/'jobs/31158699.result.json').read_text())['returncode']==1
    assert not (journal/'jobs/31158700.intent.json').exists()


@pytest.mark.parametrize('kwargs',[{'storage':False},{'issues':['ambiguous_release:old']}])
def test_fresh_storage_or_controller_issue_prevents_any_release(monkeypatch,tmp_path,kwargs):
    art,journal,calls,released=fixture(monkeypatch,tmp_path,**kwargs)
    with pytest.raises(RuntimeError):burst.apply()
    assert not calls and not released and burst.controller.MAX_ACTIVE==4


def test_recorded_aggregate_storage_block_prevents_scheduler_calls(monkeypatch,tmp_path):
    art,journal,calls,released=fixture(monkeypatch,tmp_path)
    (art/'storage_blocked.json').write_text('{"status":"blocked"}')
    with pytest.raises(RuntimeError,match='storage-blocked'):burst.apply()
    assert not calls and not released and not (art/'burst.intent.json').exists()
