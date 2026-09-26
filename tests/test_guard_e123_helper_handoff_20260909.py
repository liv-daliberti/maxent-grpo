"""E123-only guard handoffs preserve recovery state and reject logic drift."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
spec=importlib.util.spec_from_file_location('handoff',ROOT/'ops/exp_scaling/handoff_campaign_guard_e123_helpers_20260909.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_e123_extension_preserves_all_unrelated_logic():
    before=m.BEFORE.read_text();after=m.CAMPAIGN.read_text()
    proof=m.presentation_audit(before,after)
    assert proof['outside_e123_status_extension_ast_equal']
    assert proof['guarded_e118_e119_e120_functions_constants_and_resolvers_unchanged']
    for changed in [after.replace('def e120_continuation_jobs(', 'def changed_e120_continuation_jobs('),
                    after.replace('E120_CONTINUATIONS =', 'CHANGED_E120_CONTINUATIONS ='),
                    after.replace('    release_snapshot = level3_release_status(ledger, payload)',
                                  '    release_snapshot = erase_science(ledger, payload)')]:
        with pytest.raises(AssertionError,match='outside the reviewed E123'):
            m.presentation_audit(before,changed)


def test_handoff_changes_only_hashes_and_retains_retry_state(tmp_path,monkeypatch):
    art=tmp_path/'handoff';art.mkdir();p=tmp_path/'plan.json';t=tmp_path/'transaction.json'
    original_plan={'helper_sha256':{str(m.CAMPAIGN):'old','other.py':'untouched'},'rows':[{'initial_checkpoint_step':768,'max_requeues':24}]}
    p.write_text(json.dumps(original_plan))
    jobs={'101':{'status':'monitoring','attempts':[{'released':True,'checkpoint_before_release':{'step':864}}],'last_resume_step':864},'102':{'status':'manual_stop','attempts':[]}}
    original_tx={'plan_sha256':m.sha(p),'deadline_utc':'2026-09-10T12:00:00+00:00','cleanup_deadline_utc':'2026-09-10T13:05:00+00:00','jobs':jobs,'events':[{'at':'old'}]}
    t.write_text(json.dumps(original_tx));monkeypatch.setattr(m,'ART',art);monkeypatch.setattr(m,'save',lambda *_:None)
    item={'old_job_id':1,'guard_plan':str(p),'guard_tx':str(t),'deadline_utc':original_tx['deadline_utc'],'cleanup_deadline_utc':original_tx['cleanup_deadline_utc']}
    m.stage_images({'after_campaign_sha256':'new'},item)
    after_plan=json.loads(Path(item['images']['plan_after']).read_text());after_tx=json.loads(Path(item['images']['tx_after']).read_text())
    expected={**original_plan,'helper_sha256':{**original_plan['helper_sha256'],str(m.CAMPAIGN):'new'}}
    assert after_plan==expected
    assert after_tx=={**original_tx,'plan_sha256':item['images']['plan_after_sha256']}
    assert json.loads(p.read_text())==original_plan and json.loads(t.read_text())==original_tx


def test_handoff_refuses_inflight_science_transition(tmp_path,monkeypatch):
    p=tmp_path/'plan.json';t=tmp_path/'transaction.json';p.write_text('{}');t.write_text(json.dumps({'plan_sha256':m.sha(p),'jobs':{'101':{'attempts':[{'released':False}]}}}))
    monkeypatch.setattr(m,'ART',tmp_path)
    with pytest.raises(AssertionError,match='unfinished science mutation'):
        m.stage_images({}, {'old_job_id':1,'guard_plan':str(p),'guard_tx':str(t)})


@pytest.mark.parametrize('state', [{'status':'fallback_in_progress'}, {'retirement_in_progress':True}])
def test_handoff_refuses_unfinished_fallback_or_retirement(tmp_path,monkeypatch,state):
    p=tmp_path/'plan.json';t=tmp_path/'transaction.json';p.write_text('{}')
    t.write_text(json.dumps({'plan_sha256':m.sha(p),'jobs':{'101':dict(state,attempts=[])}}))
    monkeypatch.setattr(m,'ART',tmp_path)
    with pytest.raises(AssertionError,match='transition must finish'):
        m.stage_images({}, {'old_job_id':1,'guard_plan':str(p),'guard_tx':str(t)})
