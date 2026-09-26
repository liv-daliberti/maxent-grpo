"""Offline supervisor policy/receipt gates; no Hub or scheduler mutations."""
import copy,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'var/artifacts/hf_archive_review_20260911/run_paced_archive715.py'
spec=importlib.util.spec_from_file_location('supervisor_under_test',PATH)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def state():return {'pending_attempt':None,'blocked_reason':None,'cooldown_until':4605}
def failure(kind='RepositoryCommitRateLimited',model='experiments/E114/test'):
    return {'model':model,'phase':'model' if model else 'catalog','error_type':kind,'http_status':None}
def rejected(summary='Archive E114/test'):
    return {'status':'definitively_rejected_granular_limit','summary':summary,'cooldown_until':4605}
def refused(summary='Archive E114/other'):
    return {'status':'deferred_before_http','summary':summary,'cooldown_until':4605}


def test_latest_diagnostic_uses_last_rejected_commit_and_granular_hour():
    rows=[json.loads(line) for line in s.DIAGNOSTIC.read_text().splitlines()]
    assert s.utc(s.latest_cooldown(rows))=='2026-09-11T18:03:28.382026+00:00'
    assert all('"error"' in row['body'] for row in rows) # Stored sanitization deliberately need not be valid JSON.

@pytest.mark.parametrize('field,value',[('status',500),('method','GET'),('host','evil.example'),('path','/commit/branch'),('body','429 general api limit')])
def test_unclassified_initial_diagnostic_rejected(field,value):
    row=json.loads(s.DIAGNOSTIC.read_text().splitlines()[0]);row[field]=value
    with pytest.raises(s.pacing.CommitPacingUnsafe):s.latest_cooldown([row])


def test_exact_fresh_rejection_and_nosend_match():
    s.validate_cycle_failures([failure(),failure('RepositoryCommitCooldown','experiments/E114/other')],
        [rejected()],[refused()],state())


def test_catalog_context_match():
    s.validate_cycle_failures([failure(model=None)],[rejected('Index 421 verified experimental models')],[],state())

@pytest.mark.parametrize('change',['unknown','wrong_model','missing_receipt','old_cooldown','pending','blocked','duplicate','nosend_only'])
def test_unexplained_or_uncertain_cycle_is_never_retried(change):
    failures=[failure()];results=[rejected()];refusals=[];current=state()
    if change=='unknown':failures[0]['error_type']='HfHubHTTPError'
    elif change=='wrong_model':results[0]['summary']='Archive different'
    elif change=='missing_receipt':results=[]
    elif change=='old_cooldown':results[0]['cooldown_until']=1
    elif change=='pending':current['pending_attempt']=0
    elif change=='blocked':current['blocked_reason']='uncertain_request'
    elif change=='duplicate':failures.append(failure())
    elif change=='nosend_only':failures=[failure('RepositoryCommitCooldown')];results=[];refusals=[refused('Archive E114/test')]
    with pytest.raises(s.pacing.CommitPacingUnsafe):s.validate_cycle_failures(failures,results,refusals,current)

@pytest.mark.parametrize('mode',['missing','different','matching'])
def test_successful_commit_requires_matching_engine_receipt(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(s,'ALL',tmp_path)
    record={'repo_prefix':'experiments/E114/test','archive_id':'abc'}
    result={'summary':'Archive E114/test','status':'committed','commit_sha':'a'*40}
    if mode!='missing':
        path=tmp_path/'models/abc/upload.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'commit_sha':('a' if mode=='matching' else 'b')*40}))
    if mode=='matching':s.check_success_receipts([result],[record])
    else:
        with pytest.raises(s.pacing.CommitPacingUnsafe):s.check_success_receipts([result],[record])


def test_fixed_safety_and_plan_pins_still_exact():
    for path,expected in s.FIXED.items():assert s.sha(path)==expected


def test_transfer_runs707_then715_only_after_each_complete(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'STATE',tmp_path/'state');monkeypatch.setattr(s,'validate_pins',lambda plan:None)
    monkeypatch.setattr(s,'progress',lambda *args,**kwargs:None)
    plans={n:tmp_path/f'plan{n}.json' for n in [707,715]}
    for p in plans.values():p.write_text('{"models":[]}')
    monkeypatch.setattr(s,'PLANS',plans)
    events=[]
    runner=SimpleNamespace(run=lambda path,*args,**kwargs:events.append(('run',Path(path).name,kwargs)))
    monkeypatch.setattr(s,'load',lambda *args:runner)
    monkeypatch.setattr(s,'complete_target',lambda target:events.append(('complete',target)) or {'status':'batch_finished'})
    pacer=SimpleNamespace(directory=tmp_path/'pacer',snapshot=lambda:{'schema':s.pacing.SCHEMA,'repo_id':s.pacing.REPO,
        'minimum_spacing_seconds':20,'granular_cooldown_seconds':3605,'last_attempt_at':0,'cooldown_until':0,
        'next_sequence':0,'pending_attempt':None,'blocked_reason':None})
    s.transfer({'targets':[707,715],'lifetime_seconds':60,'max_cycles_per_target':2},pacer)
    assert events==[('run','plan707.json',{'workers':6}),('complete',707),('run','plan715.json',{'workers':6}),('complete',715)]


def test_incomplete707_never_starts715(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'STATE',tmp_path/'state');monkeypatch.setattr(s,'validate_pins',lambda plan:None)
    monkeypatch.setattr(s,'progress',lambda *args,**kwargs:None)
    plans={n:tmp_path/f'plan{n}.json' for n in [707,715]}
    for p in plans.values():p.write_text('{"models":[]}')
    monkeypatch.setattr(s,'PLANS',plans)
    calls=[]
    monkeypatch.setattr(s,'load',lambda *args:SimpleNamespace(run=lambda path,*args,**kwargs:calls.append(path)))
    def incomplete(target):raise s.pacing.CommitPacingUnsafe('target_batch_not_fully_complete')
    monkeypatch.setattr(s,'complete_target',incomplete)
    pacer=SimpleNamespace(directory=tmp_path/'pacer',snapshot=lambda:{'schema':s.pacing.SCHEMA,'repo_id':s.pacing.REPO,
        'minimum_spacing_seconds':20,'granular_cooldown_seconds':3605,'last_attempt_at':0,'cooldown_until':0,
        'next_sequence':0,'pending_attempt':None,'blocked_reason':None})
    with pytest.raises(s.pacing.CommitPacingUnsafe):s.transfer({'targets':[707,715],'lifetime_seconds':60,'max_cycles_per_target':2},pacer)
    assert calls==[str(plans[707])]


def test_failed_offline_closing_cannot_build_or_publish(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'STATE',tmp_path);monkeypatch.setattr(s,'complete_target',lambda target:None)
    monkeypatch.setattr(s,'validate_pins',lambda plan:None);monkeypatch.setattr(s,'progress',lambda *args,**kwargs:None)
    run=Mock(return_value=SimpleNamespace(returncode=1));monkeypatch.setattr(s.subprocess,'run',run)
    publisher=Mock();monkeypatch.setattr(s,'load',publisher)
    with pytest.raises(s.pacing.CommitPacingUnsafe):s.closing({},None,None)
    assert run.call_count==1 and not publisher.called
    assert '--require-complete' in run.call_args[0][0]


def test_interrupted_service_cannot_blindly_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'STATE',tmp_path);monkeypatch.setattr(s,'reviewed_plan',lambda:{})
    (tmp_path/'run_started.json').write_text('{}')
    install=Mock();monkeypatch.setattr(s.pacing,'install_pacing',install)
    with pytest.raises(s.pacing.CommitPacingUnsafe,match='interrupted_supervisor'):s.run()
    assert not install.called

@pytest.mark.parametrize('mode',['passed','failed_links','failed_offline','failed_remote','pending','changed_token','failed_public_bytes'])
def test_credential_cleanup_only_after_all_verified_under_lock(tmp_path,monkeypatch,mode):
    all_dir=tmp_path/'all';all_dir.mkdir();docs=tmp_path/'docs';docs.mkdir();art=tmp_path/'art';art.mkdir()
    service=tmp_path/'state';service.mkdir();token=tmp_path/'private_token';token.write_text('fake private test secret')
    unrelated=tmp_path/'other_token';unrelated.write_text('retain this')
    for name,value in [('ALL',all_dir),('DOCS',docs),('ART',art),('STATE',service),('TOKEN',token)]:monkeypatch.setattr(s,name,value)
    monkeypatch.setattr(s,'validate_pins',lambda plan:None);monkeypatch.setattr(s,'complete_target',lambda target:None)
    identity={'inode':1};monkeypatch.setattr(s,'private_token_identity',lambda:{'inode':2} if mode=='changed_token' else identity)
    links={'status':'failed' if mode=='failed_links' else 'verified','commit_sha':'a'*40,'checks':[{'status':'verified'}],'unique_urls':1}
    link_path=art/'links.json';link_path.write_text(json.dumps(links))
    final={'status':'verified','models':715,'public_download_sha256_checked':mode!='failed_public_bytes','public_files':[{}]*38,
        'commit_sha':'a'*40,'public_links_report':str(link_path),'unique_public_links_verified':1}
    receipt=docs/'final715_publication/publication_verified.json';receipt.parent.mkdir();receipt.write_text(json.dumps(final))
    (art/'final_offline_retirement_audit.json').write_text(json.dumps({'status':'failed' if mode=='failed_offline' else 'complete','all_715_retired_verified':True,'errors':[]}))
    (art/'remote_head_final.json').write_text(json.dumps({'status':'failed' if mode=='failed_remote' else 'verified','audited_model_count':715,'errors':[]}))
    pacer=SimpleNamespace(snapshot=lambda:{'pending_attempt':1 if mode=='pending' else None,'blocked_reason':None})
    if mode=='passed':
        s.cleanup_private_token({'delete_private_token_after_verified_publication':True},identity,pacer)
        assert not token.exists() and (service/'private_token_cleanup.complete.json').exists()
    else:
        with pytest.raises(s.pacing.CommitPacingUnsafe):s.cleanup_private_token({'delete_private_token_after_verified_publication':True},identity,pacer)
        assert token.exists() and not (service/'private_token_cleanup.complete.json').exists()
    assert unrelated.read_text()=='retain this'
