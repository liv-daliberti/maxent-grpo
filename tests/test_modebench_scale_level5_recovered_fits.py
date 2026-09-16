"""Synthetic audited recovery and two native-fit stubs; no real actions."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_ready_fits import fixture as prior_fixture,write,change

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(prior_fixture,monkeypatch):
    f=prior_fixture;m=f.a;tmp=f.state.parent
    spec=importlib.util.spec_from_file_location('scratch_recovered_fits',ROOT/'artifacts/continue_modebench_scale_level5_recovered_fits_20260912.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    monkeypatch.setattr(a,'engine',m)
    prior_review=m.REVIEW
    for name in ('SOURCE','TESTS','SCOPE','SCHEMA','RESULT_SCHEMA'):
        monkeypatch.setattr(m,name,getattr(a,name))
    for name,value in {'REVIEW':tmp/'recovered_review.json','PRIOR_REVIEW':prior_review,'RECOVERY':tmp/'recovery',
        'AUDITOR':tmp/'auditor.py','AUDITOR_TESTS':tmp/'auditor_tests.py','AUDITOR_REVIEW':tmp/'auditor_review.json',
        'SOURCE_ROOTS':{d:tmp/d for _,d,_ in m.SCOPE}}.items():monkeypatch.setattr(m,name,value,raising=False)
    m.AUDITOR.write_text('scratch readonly audit verifier');m.AUDITOR_TESTS.write_text('scratch audit tests')
    plan=m.read(m.PLAN);plan['cells'] += [None,None];manifest=m.read(m.MANIFEST);receipts=[]
    for index,domain,raw in m.SCOPE:
        root=m.SOURCE_ROOTS[domain];tasks=[];path=tmp/f'tasks{index}.json'
        for tier in range(4):
            output=root/'level5/results/development'/domain/f'difficulty_{tier}.json'
            write(output,{'status':'complete','level':'level5','domain':domain,'split':'dev'});receipts.append(output)
            tasks.append({'output':str(output)})
        write(path,tasks);plan['cells'][index]={'level':'level5','domain':domain,'source_root':str(root),'tasks':str(path)}
        manifest['sources'][domain]={'source_kind':'domain_revision_v1','source_root':str(root)}
    write(m.PLAN,plan);write(m.MANIFEST,manifest)
    monkeypatch.setattr(m.BASE,'REVISED_PLAN_SHA',m.sha(m.PLAN));monkeypatch.setattr(m.BASE,'SOURCE_PINS',{m.MANIFEST:m.sha(m.MANIFEST)})
    monkeypatch.setattr(m,'SEALED',{p:m.sha(p) for p in m.SEALED})
    # Re-pin the scratch predecessor review after the synthetic original plan update.
    old=m.read(prior_review);old['files_sha256']={p:m.sha(p) for p in old['files_sha256']};write(prior_review,old)
    terminal=m.RECOVERY/'terminal_accounting.json';write(terminal,{'actual_states':['FAILED','FAILED'],'scratch_only':True})
    reports=[];pins={str(p):m.sha(p) for p in [f.source,m.AUDITOR,terminal,*receipts]}
    for index in (0,1):
        for tier in range(4):
            p=m.RECOVERY/'execution_audit/tiers'/f'{index}_{tier}.json';write(p,{'scratch_complete_audit':True})
            reports.append({'path':str(p),'sha256':m.sha(p)});pins[str(p)]=m.sha(p)
    executions=[];cells=[]
    for recovery_index,(index,domain,raw) in enumerate(m.SCOPE):
        executions.append({'recovery_index':recovery_index,'original_index':index,'array_job_id':31259131,
            'job_id':'31259131_'+str(recovery_index),'job_id_raw':raw,'state':'FAILED',
            'exit_code':'1:0' if recovery_index==0 else '143:0','scheduler_success':False,'exit_cause':'unknown'})
        cells.append({'recovery_index':recovery_index,'original_index':index,'domain':domain,
            'source_root':str(m.SOURCE_ROOTS[domain]),'completed_receipts':4})
    certificate=m.RECOVERY/'execution_reconciliation.json'
    proof={'schema':'modebench_scale_revised_recovery_execution_reconciliation_v1',
        'status':'verified_scientific_outputs_with_execution_recovery','array_job_id':31259131,
        'scheduler_success':False,'recovery_scheduler_success':False,'scientific_outputs_complete':True,
        'observed_failed_recovery_cells':[0,1],
        'completed_receipts':8,'completed_batches':864,'attempts_validated':54400,'terminal_sha256':m.sha(terminal),
        'fit_performed':False,'selection_performed':False,'recovery_executions':executions,'cells':cells,'grader_audits':reports,'files_sha256':pins}
    write(certificate,proof)
    write(m.AUDITOR_REVIEW,{'schema':'modebench_scale_revised_recovery_execution_v2_independent_review_v1','status':'reviewed',
        'files_sha256':{str(p):m.sha(p) for p in (m.AUDITOR,m.AUDITOR_TESTS,f.source)}})
    reviewpins={**old['files_sha256'],**m.read(m.AUDITOR_REVIEW)['files_sha256']}
    reviewpins.update({str(p):m.sha(p) for p in (m.SOURCE,m.TESTS,*m.SEALED,m.AUDITOR,m.AUDITOR_TESTS,m.AUDITOR_REVIEW,
        certificate,terminal,prior_review)})
    write(m.REVIEW,{'schema':'modebench_scale_level5_recovered_fits_independent_review_v1','status':'reviewed',
        'files_sha256':reviewpins,'recovery_auditor_sha256':m.sha(m.AUDITOR),
        'recovery_certificate_sha256':m.sha(certificate),'recovery_terminal_sha256':m.sha(terminal)})
    for name in ('reviewed','entries_and_proofs','guest','verify_existing'):monkeypatch.setattr(m,name,getattr(a,name))
    f.owner['command']=[str(m.PYTHON),'-B',str(m.SOURCE),'fit']
    oldload=m.load_module;verification=[]
    def verify(root):
        assert root==m.RECOVERY;verification.append(root);return m.read(certificate)
    def exclusions(root,level,domain,carried):
        assert domain in ('pantry','python_factors') and level=='level5';f.exclusions.append(domain)
        return {str(f.source):m.sha(f.source)}
    def load(path,name):
        if path==m.AUDITOR:return SimpleNamespace(verify_existing=verify)
        if path==m.CORE:
            value=oldload(path,name);value.validate_revision_pool_exclusions=exclusions;return value
        return oldload(path,name)
    monkeypatch.setattr(m,'load_module',load)
    return SimpleNamespace(a=a,m=m,f=f,certificate=certificate,terminal=terminal,receipts=receipts,
        verification=verification,run=lambda:a.run(f.state,m.sha(m.REVIEW)))


def test_only_recovered_two_domains_fit_once_after_full_audit_and_preserve_failed_scheduler(fixture):
    x=fixture;result=x.run()
    assert x.f.calls==['pantry','python_factors'] and x.verification==[x.m.RECOVERY]
    assert x.f.exclusions==x.f.calls and result['failed_domains']==[]
    registration=x.m.read(x.f.state/'registration.json')
    assert registration['revisions'][0]['recovery_execution']['state']=='FAILED'
    assert [(v['recovery_execution']['state'],v['recovery_execution']['exit_code'],
             v['recovery_execution']['scheduler_success'],v['recovery_execution']['exit_cause'])
            for v in registration['revisions']]==[('FAILED','1:0',False,'unknown'),('FAILED','143:0',False,'unknown')]
    assert registration['files_sha256'][str(x.f.state/'action/runtime.json')]==x.m.sha(x.f.state/'action/runtime.json')
    assert x.a.verify_existing(x.f.state)==result and x.verification==[x.m.RECOVERY]
    assert x.m.read(x.f.state/'registration.json')['revisions']==registration['revisions']
    assert x.m.read(x.certificate)['observed_failed_recovery_cells']==[0,1]
    assert x.f.calls==['pantry','python_factors']


@pytest.mark.parametrize('failed',[['pantry'],['python_factors'],['pantry','python_factors']])
def test_completed_negative_fits_preserved_and_second_independent_fit_still_runs(fixture,failed):
    x=fixture;x.f.failures.update(failed);v=x.run()
    assert v['failed_domains']==failed and v['status']=='needs_new_development_revision'
    assert x.f.calls==['pantry','python_factors'] and x.a.verify_existing(x.f.state)==v


@pytest.mark.parametrize('domain',['pantry','python_factors'])
def test_native_exception_stops_preserves_claim_and_forbids_retry(fixture,domain):
    x=fixture;x.f.exceptions.add(domain)
    with pytest.raises(ValueError,match='scratch fitter exception'):x.run()
    assert (x.f.state/'action/failure.json').exists()
    assert (x.f.state/'fits'/domain/'claim.json').exists()
    before=list(x.f.calls)
    with pytest.raises(ValueError,match='already attempted'):x.run()
    assert x.f.calls==before
    if domain=='python_factors':assert (x.f.state/'fits/pantry/result.json').exists()


@pytest.mark.parametrize('code',[1,143,-15])
def test_nonzero_wrapper_preserves_both_written_decisions_without_success_waiver(fixture,code):
    x=fixture;x.f.dispatch['code']=code
    with pytest.raises(ValueError,match='explicit reconciliation'):x.run()
    assert x.f.calls==['pantry','python_factors']
    assert x.m.read(x.f.state/'action/exit.json')['returncode']==code
    with pytest.raises(ValueError,match='reconciliation'):x.a.verify_existing(x.f.state)


@pytest.mark.parametrize('kind',['certificate_absent','certificate_changed','auditor_changed','terminal_changed',
    'auditor_review_draft','auditor_review_schema','review_draft','missing_certificate_pin','missing_dependency_pin','missing_final_digest'])
def test_unfinalized_or_changed_recovery_review_refuses_before_action_claim(fixture,kind):
    x=fixture;m=x.m
    if kind=='certificate_absent':x.certificate.unlink()
    elif kind=='certificate_changed':change(x.certificate,status='pending')
    elif kind=='auditor_changed':m.AUDITOR.write_text('changed')
    elif kind=='terminal_changed':change(x.terminal,actual_states=['RUNNING','RUNNING'])
    elif kind.startswith('auditor_review'):change(m.AUDITOR_REVIEW,**({'status':'draft'} if kind.endswith('draft') else {'schema':'wrong'}))
    else:
        v=m.read(m.REVIEW)
        if kind=='review_draft':v['status']='draft'
        elif kind=='missing_certificate_pin':v['files_sha256'].pop(str(x.certificate))
        elif kind=='missing_dependency_pin':v['files_sha256'].pop(str(x.f.source))
        else:v.pop('recovery_terminal_sha256')
        write(m.REVIEW,v)
    with pytest.raises((ValueError,FileNotFoundError)):x.run()
    assert not x.f.calls and not x.f.state.exists()


@pytest.mark.parametrize('kind',['scope','complete','receipts','batches','attempts','report','receipt_pin','execution_index','cell_root'])
def test_complete_original_grader_scope_is_required_even_with_explicit_rehashed_certificate(fixture,kind):
    x=fixture;m=x.m;v=m.read(x.certificate)
    if kind=='scope':v['array_job_id']=31259795
    elif kind=='complete':v['scientific_outputs_complete']=False
    elif kind=='receipts':v['completed_receipts']=7
    elif kind=='batches':v['completed_batches']=863
    elif kind=='attempts':v['attempts_validated']=1
    elif kind=='report':v['grader_audits'].pop()
    elif kind=='receipt_pin':v['files_sha256'].pop(str(x.receipts[0]))
    elif kind=='execution_index':v['recovery_executions'][0]['original_index']=3
    else:v['cells'][0]['source_root']='wrong'
    write(x.certificate,v);review=m.read(m.REVIEW);review['files_sha256'][str(x.certificate)]=m.sha(x.certificate)
    review['recovery_certificate_sha256']=m.sha(x.certificate);write(m.REVIEW,review)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls


@pytest.mark.parametrize('domain',['pantry','python_factors'])
def test_existing_recipe_never_gets_refitted(fixture,domain):
    x=fixture;root=x.m.SOURCE_ROOTS[domain];write(root/'level5/recipes'/(domain+'.json'),{'prior':True})
    with pytest.raises(ValueError,match='unfitted'):x.run()
    assert not x.f.calls


def test_static_verification_uses_pinned_recorded_runtime_not_current_nfs_device(fixture,monkeypatch):
    x=fixture;result=x.run()
    class ForeignLock:
        def stat(self):raise AssertionError('static proof must not inspect current-host device')
    monkeypatch.setattr(x.m.first,'LOCK',ForeignLock())
    assert x.a.verify_existing(x.f.state)==result and x.f.calls==['pantry','python_factors']


@pytest.mark.parametrize('kind',['runtime','exit','guest_argv','result','recipe','claim','missing_pin'])
def test_static_fit_proof_rejects_changed_or_partial_record_without_any_refit(fixture,kind):
    x=fixture;x.run();m=x.m;r=x.f.state
    if kind=='runtime':change(r/'action/runtime.json',lock_device=0)
    elif kind=='exit':change(r/'action/exit.json',returncode=1)
    elif kind=='guest_argv':
        v=m.read(r/'action/guest_runtime.json');v['command'].remove('-B');write(r/'action/guest_runtime.json',v)
    elif kind=='result':change(r/'action/result.json',status='invented')
    elif kind=='recipe':change(m.SOURCE_ROOTS['pantry']/'level5/recipes/pantry.json',development_fit_pass=False)
    elif kind=='claim':change(r/'fits/pantry/claim.json',source_root='wrong')
    else:
        v=m.read(r/'registration.json');v['files_sha256'].pop(str(r/'action/runtime.json'));write(r/'registration.json',v)
        write(r/'registration.sha256.json',{'sha256':m.sha(r/'registration.json')})
    with pytest.raises(ValueError):x.a.verify_existing(r)
    assert x.f.calls==['pantry','python_factors']


def test_live_fence_and_owner_guards_remain_original_sealed_functions(fixture,monkeypatch):
    x=fixture
    assert Path(x.m.guest_guard.__code__.co_filename)==x.a.PRIOR
    assert Path(x.m.fit_registered.__code__.co_filename)==x.a.PRIOR
    assert Path(x.m.run.__code__.co_filename)==x.a.PRIOR
    def lost(fd):raise ValueError('lost inherited fence')
    monkeypatch.setattr(x.m,'assert_fence',lost)
    with pytest.raises(ValueError,match='lost inherited fence'):x.run()
    assert not x.f.calls
