"""Synthetic final admissions: original read-only native/legacy gates, no science."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_execution_provenance import release as legacy_release, write

ROOT=Path(__file__).resolve().parents[1]


def change(path,fn):
    path=Path(path);value=json.loads(path.read_text());fn(value);path.chmod(0o644);write(path,value)


@pytest.fixture
def world(legacy_release,tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('_scratch_combined_publisher',ROOT/'artifacts/publish_modebench_scale_combined_admission_draft_20260912.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    original_load=m.load
    core=original_load(m.CORE,m.CORE_SHA,'_scratch_combined_native_core')
    legacy=original_load(m.LEGACY,m.LEGACY_SHA,'_scratch_combined_legacy')
    release=legacy_release.root;source=legacy_release.source;state=tmp_path/'action'
    legacy_release.campaign.unlink()
    canonical=tmp_path/'templates.py';canonical.write_text('old')
    view=write(tmp_path/'view.json',{'synthetic_frozen_mapping':True})
    lock=tmp_path/'controller.lock';lock.write_text('synthetic_lock')
    for name,value in {'STATE':state,'RELEASE':release,'PARENT':source,'HISTORICAL_RELEASE':tmp_path/'historical_release',
            'VIEW':view,'REVIEW':tmp_path/'final_review.json','DESTINATION':tmp_path/'destination.json',
            'DESTINATION_HELPER':tmp_path/'destination_provider.py'}.items():monkeypatch.setattr(m,name,value)
    m.DESTINATION_HELPER.write_text('# synthetic destination provider\n')
    monkeypatch.setattr(m,'DESTINATION_HELPER_SHA',m.digest(m.DESTINATION_HELPER))
    monkeypatch.setattr(m.first,'CANONICAL',canonical);monkeypatch.setattr(m.first,'OLD_SHA',m.digest(canonical))
    monkeypatch.setattr(m.first,'LOCK',lock);monkeypatch.setattr(m.first,'LOCK_INODE',lock.stat().st_ino)
    current=[];blocked=set();forbidden=[];providers={};records={}
    def bomb(*args,**kwargs):forbidden.append('science');raise AssertionError('forbidden science/controller call')
    for name in ('_sweep','audit_source'):monkeypatch.setattr(core,name,bomb)
    for obj,name in ((core.revision,'confirm_domain'),(core.revision,'fit_domain'),(core.original,'confirm_domain')):
        if hasattr(obj,name):monkeypatch.setattr(obj,name,bomb)
    monkeypatch.setattr(core,'frozen_identity',lambda root,level,domain,kind:m.read(core.domain_paths(root,level,domain)['dataset']/'identity.json'))
    def disjoint(root,level,domain):
        current.append((level,domain))
        if (level,domain) in blocked:raise ValueError('current source overlap')
        return {'files_sha256':{}}
    monkeypatch.setattr(core,'verify_dataset',disjoint)
    for level in m.LEVELS:
        admission=release/level/'admission.json';saved=m.read(admission)
        manifest=release/level/'source_manifest.json'
        change(manifest,lambda v:v.update(schema=core.MANIFEST_SCHEMA,targets={'preserved':'synthetic'},files_sha256={}))
        write(manifest.with_name('source_manifest.sha256.json'),{'sha256':m.digest(manifest)})
        for domain in m.DOMAINS:
            paths=core.domain_paths(source,level,domain)
            change(paths['receipt'],lambda v:v.update(status='complete',level=level,domain=domain,split='eval'))
            change(paths['audit'],lambda v:v.update(receipt_sha256=m.digest(paths['receipt'])))
        saved['targets']={'preserved':'synthetic'}
        saved['files_sha256']={p:m.digest(p) for p in saved['files_sha256']}
        write(admission,saved);(release/level/'README.md').write_text('synthetic admitted level links\n')
        completion=tmp_path/(level+'_completion');provider=tmp_path/(level+'_provider.py');provider.write_text('# synthetic reviewed completion implementation\n')
        provider_review=write(tmp_path/(level+'_review.json'),{'schema':'synthetic_completion_review_v1','status':'reviewed',
            'files_sha256':{str(provider):m.digest(provider)}})
        result=write(completion/'admit_action/result.json',{'status':level+'_admitted','level':level,
            'admission_path':str(admission),'admission_sha256':m.digest(admission),'new_grader_calls':0,
            'combined_release_published':False,'files_sha256':{str(admission):m.digest(admission),**saved['files_sha256']}})
        certificate=write(completion/'registration.json',{'schema':'synthetic_execution_v1','status':'actual_complete_synthetic',
            'array_job_id':600+list(m.LEVELS).index(level),'files_sha256':{str(result):m.digest(result)}})
        providers[str(provider)]=SimpleNamespace(SOURCE=provider,REVIEW=provider_review,STATE=completion,RELEASE=release,
            verify_existing=lambda root,p=result: m.read(p))
        records[level]={'completion_provider':str(provider),'completion_provider_sha256':m.digest(provider),
            'completion_review':str(provider_review),'completion_review_sha256':m.digest(provider_review),
            'completion_root':str(completion),'admission_result_path':str(result),'admission_result_sha256':m.digest(result),
            'execution_certificate':str(certificate),'execution_certificate_sha256':m.digest(certificate),
            'execution_schema':'synthetic_execution_v1','execution_status':'actual_complete_synthetic',
            'array_job_id':600+list(m.LEVELS).index(level),'admission_sha256':m.digest(admission)}
    destination={'effective_required_final_execution_provenance':{'path':str(release/'execution_provenance.json')},
        'historical_validation':{'release_argument':str(m.HISTORICAL_RELEASE)},'scientific_validation':{'release_argument':str(release)},
        'required_status':'verified','files_sha256':{str(m.DESTINATION_HELPER):m.digest(m.DESTINATION_HELPER)}}
    write(m.DESTINATION,destination);monkeypatch.setattr(m,'DESTINATION_SHA',m.digest(m.DESTINATION))
    monkeypatch.setattr(m,'SEALED',{m.FIRST:m.FIRST_SHA,m.CORE:m.CORE_SHA,m.LEGACY:m.LEGACY_SHA,
        m.DESTINATION_HELPER:m.DESTINATION_HELPER_SHA,m.DESTINATION:m.DESTINATION_SHA})
    def loader(path,sha,name):
        path=Path(path)
        assert m.digest(path)==sha
        if path==m.CORE:return core
        if path==m.LEGACY:return legacy
        if path==m.DESTINATION_HELPER:return SimpleNamespace(verify=lambda *,guest:deepcopy(destination))
        if str(path) in providers:return providers[str(path)]
        return original_load(path,sha,name)
    monkeypatch.setattr(m,'load',loader)
    def refresh():
        pins={str(p):m.digest(p) for p in (m.SOURCE,m.TESTS,m.VIEW,*m.SEALED)}
        for level,r in records.items():
            for field,h in (('completion_provider','completion_provider_sha256'),('completion_review','completion_review_sha256'),
                    ('admission_result_path','admission_result_sha256'),('execution_certificate','execution_certificate_sha256')):
                r[h]=m.digest(r[field]);pins[r[field]]=r[h]
            r['admission_sha256']=m.digest(release/level/'admission.json')
            for field in ('completion_review','admission_result_path','execution_certificate'):
                pins.update(m.read(r[field])['files_sha256'])
        write(m.REVIEW,{'schema':'modebench_scale_combined_admission_final_review_v1','status':'reviewed',
            'scope':'actual_both_level_admissions_and_execution_certificates','levels':deepcopy(records),
            'release_root':str(release),'destination_registration_sha256':m.DESTINATION_SHA,'files_sha256':pins})
        return m.digest(m.REVIEW)
    refresh()
    owner={'pid':41000,'uid':os.getuid(),'start_ticks':'100','state':'S','command':[]}
    held=[];active=[];host=[];fence={'lost':False};dispatch={'code':0,'error':None}
    @contextmanager
    def lifetime():
        assert held==[];held.append(8)
        try:yield 8
        finally:held.clear()
    def assert_fence(fd):
        if fd!=8 or held!=[8] or fence['lost']:raise ValueError('actual inherited fence lost')
    def host_guard(*,guest):
        host.append(guest)
        if canonical.read_text()!=('old' if guest else 'neutral'):raise ValueError('wrong actual host source view')
    def identity(pid):
        if active and pid!=owner['pid']:
            command=active[-1][active[-1].index('--')+1:];command.insert(1,'-B')
            return {**deepcopy(owner),'pid':41001,'start_ticks':'101','command':command}
        return deepcopy(owner)
    monkeypatch.setattr(m.first,'lifetime_fence',lifetime);monkeypatch.setattr(m.first,'assert_fence',assert_fence)
    monkeypatch.setattr(m.first,'host_guard',host_guard);monkeypatch.setattr(m.first,'process_identity',identity)
    def authenticate(path):
        assert canonical.read_text()=='neutral';host.append('authenticate_outside')
    monkeypatch.setattr(m.first,'load_module',lambda path,name:SimpleNamespace(authenticate=authenticate))
    def subprocess_run(command,**kwargs):
        assert kwargs['pass_fds']==(8,) and held==[8];active.append(command);canonical.write_text('old')
        try:
            if dispatch['error']:raise dispatch['error']
            m.guest(state,8,owner['pid'],owner['start_ticks'],m.digest(m.REVIEW))
        except (ValueError,KeyError,FileNotFoundError) as error:
            dispatch['guest_error']=error
            return SimpleNamespace(returncode=1)
        finally:active.pop();canonical.write_text('neutral')
        return SimpleNamespace(returncode=dispatch['code'])
    monkeypatch.setattr(m,'subprocess',SimpleNamespace(run=subprocess_run))
    def run():
        canonical.write_text('neutral');owner['command']=m.outer_command(state,m.digest(m.REVIEW));return m.run(state,m.digest(m.REVIEW))
    def verify():canonical.write_text('old');return m.verify_existing(state)
    return SimpleNamespace(m=m,core=core,legacy=legacy,state=state,release=release,source=source,records=records,
        providers=providers,current=current,blocked=blocked,forbidden=forbidden,refresh=refresh,owner=owner,
        held=held,active=active,host=host,fence=fence,dispatch=dispatch,run=run,verify=verify,canonical=canonical)


def files_under(path):return {str(p):p.read_bytes() for p in path.rglob('*') if p.is_file() and not p.is_symlink()}


def test_real_native_preflight_reads_ten_and_writes_nothing(world):
    w=world;before=files_under(w.release.parent)
    result=w.m.preflight(w.m.digest(w.m.REVIEW))
    assert len(result['audits'])==10 and len(w.current)==10 and w.forbidden==[]
    assert files_under(w.release.parent)==before and not w.state.exists()
    assert set(result['campaign']['levels'])=={'level4','level5'}
    assert set(result['campaign']['files_sha256'])=={str(w.release/l/'admission.json') for l in w.m.LEVELS}


@pytest.mark.parametrize('defect',['missing_l5','failed_l5','incomplete_receipt','missing_audit','partial_audit',
    'wrong_audit_route','audit_not_pinned','admission_object','bad_manifest_checksum','missing_link','overlap','data_changed'])
def test_any_incomplete_or_changed_level_refuses_before_state_or_campaign(world,defect):
    w=world;m=w.m;admission=w.release/'level5/admission.json';paths=w.core.domain_paths(w.source,'level5','pantry')
    if defect=='missing_l5':admission.unlink()
    elif defect=='failed_l5':change(admission,lambda v:v.update(difficulty_matched=False))
    elif defect=='incomplete_receipt':change(paths['receipt'],lambda v:v.update(status='partial'))
    elif defect=='missing_audit':paths['audit'].unlink()
    elif defect=='partial_audit':change(paths['audit'],lambda v:v.update(original_grader_replayed_attempts=4095))
    elif defect=='wrong_audit_route':change(paths['audit'],lambda v:v.update(level='level4'))
    elif defect=='audit_not_pinned':change(admission,lambda v:v['files_sha256'].pop(str(paths['audit'])))
    elif defect=='admission_object':change(admission,lambda v:v.update(targets={'invented':'target'}))
    elif defect=='bad_manifest_checksum':write(w.release/'level5/source_manifest.sha256.json',{'sha256':'0'*64})
    elif defect=='missing_link':(w.release/'level5/dataset/pantry').unlink()
    elif defect=='overlap':w.blocked.add(('level5','pantry'))
    elif defect=='data_changed':(paths['dataset']/'eval.jsonl').write_text('{}\n')
    # Re-pin deliberately where possible so semantic boundaries, not only hashes, reject.
    if defect not in ('missing_l5','missing_audit','data_changed'):
        saved=m.read(admission)
        if defect in ('incomplete_receipt','partial_audit','wrong_audit_route'):
            for p in (paths['receipt'],paths['audit']):saved['files_sha256'][str(p)]=m.digest(p)
        write(admission,saved)
        result=Path(w.records['level5']['admission_result_path'])
        change(result,lambda v:v.update(admission_sha256=m.digest(admission),files_sha256={str(admission):m.digest(admission),**saved['files_sha256']}))
        cert=Path(w.records['level5']['execution_certificate']);change(cert,lambda v:v['files_sha256'].update({str(result):m.digest(result)}))
        w.refresh()
    w.canonical.write_text('neutral');before=files_under(w.release.parent)
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):w.run()
    assert not w.state.exists() and not (w.release/'admission.json').exists()
    assert files_under(w.release.parent)==before and w.forbidden==[]


@pytest.mark.parametrize('defect',['component_only','one_level','missing_dependency','missing_actual_cert','failed_completion',
    'unreviewed_completion','invented_job','wrong_destination','circular_future_pin','wrong_provider_route'])
def test_final_actual_review_and_completed_action_contract_required(world,defect):
    w=world;m=w.m;r=w.records['level5']
    if defect=='component_only':change(m.REVIEW,lambda v:v.update(scope='component_tests_only'))
    elif defect=='one_level':change(m.REVIEW,lambda v:v['levels'].pop('level5'))
    elif defect=='missing_dependency':change(m.REVIEW,lambda v:v['files_sha256'].pop(str(m.CORE)))
    elif defect=='missing_actual_cert':change(m.REVIEW,lambda v:v['files_sha256'].pop(r['execution_certificate']))
    elif defect=='failed_completion':
        change(r['admission_result_path'],lambda v:v.update(status='needs_new_development_revision'));w.refresh()
    elif defect=='unreviewed_completion':
        change(r['completion_review'],lambda v:v.update(status='draft'));w.refresh()
    elif defect=='invented_job':
        change(r['execution_certificate'],lambda v:v.update(array_job_id=None));w.refresh()
    elif defect=='wrong_destination':change(m.REVIEW,lambda v:v.update(release_root=str(m.HISTORICAL_RELEASE)))
    elif defect=='circular_future_pin':change(m.REVIEW,lambda v:v['files_sha256'].update({str(w.state/'result.json'):'0'*64}))
    elif defect=='wrong_provider_route':w.providers[r['completion_provider']].RELEASE=m.HISTORICAL_RELEASE
    with pytest.raises((ValueError,KeyError)):m.preflight(m.digest(m.REVIEW))
    assert not w.state.exists() and not (w.release/'admission.json').exists() and w.forbidden==[]


def test_publish_native_campaign_once_then_verify_readonly_without_any_grader(world):
    w=world;result=w.run();assert result['goal_completion_claimed'] is False
    assert result['execution_provenance_published'] is False and not (w.release/'execution_provenance.json').exists()
    assert len(w.current)==10 and w.forbidden==[] and 'authenticate_outside' in w.host
    w.canonical.write_text('old');before=files_under(w.release.parent);assert w.verify()==result
    assert files_under(w.release.parent)==before and len(w.current)==20
    with pytest.raises(ValueError,match='already attempted'):w.run()
    assert w.m.read(w.state/'exit.json')['returncode']==0


@pytest.mark.parametrize('destination',['existing_campaign','broken_campaign_link','existing_provenance','broken_provenance_link'])
def test_all_nonfresh_destinations_refuse_without_action_state(world,destination):
    w=world;path=w.release/('execution_provenance.json' if 'provenance' in destination else 'admission.json')
    if 'broken' in destination:path.symlink_to(w.release/'missing')
    else:write(path,{'already':'exists'})
    with pytest.raises(ValueError):w.run()
    assert 'fresh' in str(w.dispatch['guest_error'])
    assert not w.state.exists() and w.forbidden==[]


def test_postwrite_legacy_failure_preserves_admission_claim_and_actual_exit(world,monkeypatch):
    w=world
    def reject(root):raise ValueError('native postwrite validation failure')
    monkeypatch.setattr(w.legacy,'validate_release',reject)
    with pytest.raises(ValueError,match='did not complete'):w.run()
    assert (w.release/'admission.json').is_file() and (w.state/'claim.json').is_file()
    failure=w.m.read(w.state/'failure.json');assert failure['published_admission_preserved'] is True
    assert failure['detail']=='native postwrite validation failure' and w.m.read(w.state/'exit.json')['returncode']==1
    assert (w.state/'outer_failure.json').exists() and not (w.state/'result.json').exists()
    with pytest.raises(ValueError,match='already attempted'):w.run()


def test_nonzero_wrapper_retains_completed_result_and_blocks_acceptance_and_retry(world):
    w=world;w.dispatch['code']=143
    with pytest.raises(ValueError,match='did not complete'):w.run()
    assert (w.state/'result.json').is_file() and (w.release/'admission.json').is_file()
    exited=w.m.read(w.state/'exit.json');assert exited['returncode']==143 and exited['result_sha256']==w.m.digest(w.state/'result.json')
    with pytest.raises(ValueError,match='explicit reconciliation'):w.verify()
    with pytest.raises(ValueError,match='already attempted'):w.run()


@pytest.mark.parametrize('defect',['fence','owner','host'])
def test_live_preclaim_fence_owner_and_host_fail_closed(world,defect,monkeypatch):
    w=world
    if defect=='fence':w.fence['lost']=True
    elif defect=='owner':monkeypatch.setattr(w.m.first,'process_identity',lambda pid:{**w.owner,'command':['wrong']})
    else:monkeypatch.setattr(w.m.first,'host_guard',lambda **kw:(_ for _ in ()).throw(ValueError('wrong actual host')))
    with pytest.raises(ValueError):w.run()
    assert not w.state.exists() and not (w.release/'admission.json').exists()


def test_readonly_history_does_not_compare_worker_local_device_or_require_live_owner(world,monkeypatch):
    w=world;result=w.run()
    class OtherHostLock:
        def stat(self):return SimpleNamespace(st_dev=99999)
    monkeypatch.setattr(w.m.first,'LOCK',OtherHostLock())
    monkeypatch.setattr(w.m.first,'process_identity',lambda pid:(_ for _ in ()).throw(AssertionError('historical PID replay')))
    monkeypatch.setattr(w.m.first,'assert_fence',lambda fd:(_ for _ in ()).throw(AssertionError('historical fence replay')))
    assert w.verify()==result


@pytest.mark.parametrize('field,value',[('lock_inode',0),('view_manifest_sha256','0'*64),('host','wash.cs.princeton.edu'),
    ('command',['fake']),('pid',41001),('state','Z')])
def test_relinked_runtime_tampering_cannot_pass_static_predicates(world,field,value):
    w=world;w.run();m=w.m
    change(w.state/'runtime.json',lambda v:v.update({field:value}))
    for path,key in ((w.state/'intent.json','runtime_sha256'),(w.state/'guest_runtime.json','outer_runtime_sha256')):
        change(path,lambda v:v.update({key:m.digest(w.state/'runtime.json')}))
    with pytest.raises((ValueError,KeyError)):w.verify()


def test_saved_result_must_pin_original_preflight_and_claim(world):
    w=world;w.run();m=w.m
    change(w.state/'result.json',lambda v:v['files_sha256'].pop(str(w.state/'claim.json')))
    change(w.state/'exit.json',lambda v:v.update(result_sha256=m.digest(w.state/'result.json')))
    with pytest.raises(ValueError,match='result changed'):w.verify()


def test_fence_loss_after_child_result_keeps_raw_zero_exit_and_refuses_success(world,monkeypatch):
    w=world;original=w.m.subprocess.run
    def lose_after_result(*args,**kwargs):
        result=original(*args,**kwargs);w.fence['lost']=True;return result
    monkeypatch.setattr(w.m.subprocess,'run',lose_after_result)
    with pytest.raises(ValueError,match='fence lost'):w.run()
    assert w.m.read(w.state/'exit.json')['returncode']==0 and (w.state/'result.json').exists()
    assert w.m.read(w.state/'outer_failure.json')['published_admission_preserved'] is True
    with pytest.raises(ValueError,match='explicit reconciliation'):w.verify()


def test_current_overlap_after_admission_refuses_readonly_without_rewriting(world):
    w=world;w.run();w.blocked.add(('level4','python_factors'));w.canonical.write_text('old')
    before=files_under(w.release.parent)
    with pytest.raises(ValueError,match='current source overlap'):w.verify()
    assert files_under(w.release.parent)==before and w.forbidden==[]


@pytest.mark.parametrize('change_count',[-1,1])
def test_only_exact_frozen_runner_extra_B_is_accepted(world,change_count):
    w=world;w.run()
    def wrong(v):
        if change_count<0:v['command'].pop(1)
        else:v['command'].insert(1,'-B')
    change(w.state/'guest_runtime.json',wrong)
    with pytest.raises(ValueError,match='ledger changed'):w.verify()


def test_partial_action_state_blocks_rerun_without_recreating_any_file(world):
    w=world;write(w.state/'intent.json',{'partial_attempt':True});w.canonical.write_text('neutral')
    before=files_under(w.release.parent)
    with pytest.raises(ValueError,match='already attempted'):w.run()
    assert files_under(w.release.parent)==before and not (w.release/'admission.json').exists()


def test_execution_provenance_published_later_does_not_rewrite_historical_action(world):
    w=world;result=w.run();write(w.release/'execution_provenance.json',{'synthetic_separate_final_verifier':'verified'})
    w.canonical.write_text('old');before=files_under(w.release.parent)
    assert w.verify()==result and files_under(w.release.parent)==before
    assert result['execution_provenance_published'] is False
