"""Synthetic complete r3 proof and one native-fit stub; no actual science."""
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
    spec=importlib.util.spec_from_file_location('scratch_python_r3_fit',ROOT/'artifacts/continue_modebench_scale_level4_python_r3_fit_20260912.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    monkeypatch.setattr(a,'engine',m)
    prior_review=m.REVIEW
    for name in ('SOURCE','TESTS','SCOPE','SCHEMA','RESULT_SCHEMA'):
        monkeypatch.setattr(m,name,getattr(a,name))
    for name,value in {'REVIEW':tmp/'python_fit_review.json','PRIOR_REVIEW':prior_review,
        'PREPARATION':tmp/'preparation','RECOVERY':tmp/'preparation/development',
        'AUDITOR':tmp/'auditor.py','AUDITOR_TESTS':tmp/'auditor_tests.py','AUDITOR_REVIEW':tmp/'auditor_review.json',
        'REVISION_ROOT':tmp/'python_factors'}.items():monkeypatch.setattr(m,name,value,raising=False)
    m.AUDITOR.write_text('scratch readonly audit verifier');m.AUDITOR_TESTS.write_text('scratch audit tests')
    root=m.REVISION_ROOT;write(root/'protocol.json',{'unchanged_r3_targets':True})
    terminal=m.RECOVERY/'terminal_accounting.json';write(terminal,{'actual_state':'FAILED','exit_code':'143:0','scratch_only':True})
    runtime=m.RECOVERY/'runtime/0.json';sidecar=m.RECOVERY/'runtime/0.evaluator_exit.json'
    write(runtime,{'scratch_only':True,'array_job_id':31260226})
    write(sidecar,{'scratch_only':True,'returncode':0})
    monkeypatch.setattr(a,'OBSERVED_SHA',{'execution_terminal_sha256':m.sha(terminal),
        'execution_runtime_sha256':m.sha(runtime),'execution_evaluator_exit_sha256':m.sha(sidecar)})
    tasks=[];summaries=[];reports=[];receipts=[];sources=[]
    pins={str(p):m.sha(p) for p in [f.source,m.AUDITOR,terminal,runtime,sidecar,root/'protocol.json']}
    for tier in range(4):
        output=root/'level4/results/development/python_factors'/f'difficulty_{tier}.json'
        source=root/'level4/pools/python_factors'/f'difficulty_{tier}.jsonl';write(source,{'scratch_rows':193})
        write(output,{'status':'complete','level':'level4','domain':'python_factors','split':'dev'})
        receipts.append(output);sources.append(source)
        tasks.append({'output':str(output),'rows_jsonl':str(source),'level':'level4','domain':'python_factors',
            'split':'dev','batch_size':8,'row_offset':0,'row_limit':0,
            'interface':'modebench_qwen_scale_independent_v1','seeds':[7203000,7203001,7203002,7203003]})
        summaries.append({'tier':tier,'receipt':str(output),'source':str(source),'receipt_sha256':m.sha(output),
            'source_sha256':m.sha(source),'rows':193,'batches':100,'attempts':6176})
        p=m.RECOVERY/'execution_audit/tiers'/f'{tier}.json';write(p,{'scratch_complete_audit':True})
        reports.append({'path':str(p),'sha256':m.sha(p),'tier':tier,'new_grader_invocations':6176})
        pins.update({str(q):m.sha(q) for q in (output,source,p)})
    cell_path=m.PREPARATION/'development_cell.json';tasks_path=m.PREPARATION/'development_tasks.json'
    cell={'id':'level4_python_factors_r3_dev','level':'level4','domain':'python_factors','source_root':str(root),
        'source_kind':'domain_revision_v1','phase':'dev','model_label':'7b','tasks':tasks}
    write(cell_path,cell);write(tasks_path,tasks)
    prep=m.PREPARATION/'certificate.json'
    write(prep,{'schema':'modebench_scale_level4_python_r3_registration_v1','status':'ready_for_fresh_python_r3_development',
        'revision_root':str(root),'level':'level4','domains':['python_factors'],'development_cell':str(cell_path),
        'development_tasks':str(tasks_path),'source_binding_performed':False})
    pins.update({str(p):m.sha(p) for p in (prep,cell_path,tasks_path)})
    execution={'array_job_id':31260226,'array_index':0,'job_id_raw':'31260226','job_id':'31260226_0',
        'state':'FAILED','exit_code':'143:0','scheduler_success':False,'evaluator_returncode':0,'exit_cause':'unknown'}
    certificate=m.RECOVERY/'execution_reconciliation.json'
    write(certificate,{'schema':'modebench_scale_level4_python_r3_execution_reconciliation_v1',
        'status':'verified_scientific_outputs_with_failed_execution','array_job_id':31260226,
        'terminal_policy':a.POLICY,'scheduler_success':False,'scientific_outputs_complete':True,
        'exit_cause':'unknown','evaluator_returncode':0,'completed_receipts':4,'completed_batches':400,
        'attempts_validated':24704,'new_grader_invocations':24704,'terminal_sha256':m.sha(terminal),
        'fit_performed':False,'source_selection_performed':False,'model_sampling_performed':False,
        'execution':execution,'summaries':summaries,'grader_audits':reports,'files_sha256':pins})
    write(m.AUDITOR_REVIEW,{'schema':'modebench_scale_level4_python_r3_execution_independent_review_v1','status':'reviewed',
        'files_sha256':{str(p):m.sha(p) for p in (m.AUDITOR,m.AUDITOR_TESTS,f.source)}})
    monkeypatch.setattr(m,'PROOF_FIELDS',{
        'execution_auditor_sha256':m.AUDITOR,'execution_auditor_review_sha256':m.AUDITOR_REVIEW,
        'execution_certificate_sha256':certificate,'execution_terminal_sha256':terminal,
        'execution_runtime_sha256':runtime,'execution_evaluator_exit_sha256':sidecar,
        'preparation_certificate_sha256':prep},raising=False)
    old=m.read(prior_review)
    reviewpins={**old['files_sha256'],**m.read(m.AUDITOR_REVIEW)['files_sha256']}
    reviewpins.update({str(p):m.sha(p) for p in (m.SOURCE,m.TESTS,*m.SEALED,m.AUDITOR,m.AUDITOR_TESTS,m.AUDITOR_REVIEW,*m.PROOF_FIELDS.values(),prior_review)})
    write(m.REVIEW,{'schema':'modebench_scale_level4_python_r3_fit_independent_review_v1','status':'reviewed','files_sha256':reviewpins,'array_job_id':31260226,'job_id_raw':'31260226','terminal_policy':a.POLICY,
        **{field:m.sha(path) for field,path in m.PROOF_FIELDS.items()}})
    for name in ('reviewed','entries_and_proofs','collect_inputs','fit_registered','guest','verify_existing'):
        monkeypatch.setattr(m,name,getattr(a,name))
    f.owner['command']=[str(m.PYTHON),'-B',str(m.SOURCE),'fit']
    oldload=m.load_module;verification=[]
    def verify(root):
        assert root==m.RECOVERY;verification.append(root);return m.read(certificate)
    def paths(root):
        return {'recipe':root/'level4/recipes/python_factors.json','dataset':root/'level4/dataset',
            'confirmation':root/'level4/results/confirmation.json'}
    def fit(root):
        assert root==m.REVISION_ROOT and f.held==[8]
        assert (f.state/'fits/python_factors/claim.json').is_file()
        f.calls.append('python_factors')
        if 'python_factors' in f.exceptions:raise ValueError('scratch fitter exception')
        value={'level':'level4','domain':'python_factors','development_fit_pass':'python_factors' not in f.failures,
            'input_sha256':{str(f.source):m.sha(f.source)}}
        write(paths(root)['recipe'],value);return value
    def exclusions(root,level,domain,carried):
        assert root==m.REVISION_ROOT and level=='level4' and domain=='python_factors'
        f.exclusions.append(domain);return {str(f.source):m.sha(f.source)}
    def load(path,name):
        if path==m.AUDITOR:return SimpleNamespace(verify_existing=verify)
        if path==m.REVISION:return SimpleNamespace(fit_domain=fit)
        if path==m.CORE:return SimpleNamespace(revision=SimpleNamespace(paths=paths),validate_revision_pool_exclusions=exclusions)
        return oldload(path,name)
    monkeypatch.setattr(m,'load_module',load)
    return SimpleNamespace(a=a,m=m,f=f,certificate=certificate,terminal=terminal,receipts=receipts,sources=sources,
        prep=prep,cell=cell_path,tasks=tasks_path,verification=verification,run=lambda:a.run(f.state,m.sha(m.REVIEW)))


def test_only_python_r3_fits_once_after_full_audit_preserving_actual_scheduler(fixture):
    x=fixture;result=x.run()
    assert x.f.calls==['python_factors'] and x.verification==[x.m.RECOVERY] and x.f.exclusions==x.f.calls
    assert result['status']=='level4_python_r3_development_gates_passed' and result['failed_domains']==[]
    registration=x.m.read(x.f.state/'registration.json');e=registration['revisions'][0]
    assert registration['completed_receipts']==4 and len(registration['revisions'])==1
    assert e['execution']['state']=='FAILED' and e['execution']['exit_code']=='143:0'
    assert e['execution']['scheduler_success'] is False and e['execution']['evaluator_returncode']==0
    assert e['execution']['exit_cause']=='unknown'
    assert registration['files_sha256'][str(x.f.state/'action/runtime.json')]==x.m.sha(x.f.state/'action/runtime.json')
    assert x.a.verify_existing(x.f.state)==result and x.verification==[x.m.RECOVERY]
    assert x.f.calls==['python_factors'] and not result['confirmation_performed'] and not result['publication_performed']
    with pytest.raises(ValueError,match='already attempted'):x.run()
    assert x.f.calls==['python_factors']


def test_completed_negative_python_fit_saved_without_retry_or_other_fit(fixture):
    x=fixture;x.f.failures.add('python_factors');v=x.run()
    assert v['failed_domains']==['python_factors'] and v['status']=='needs_new_development_revision'
    assert x.f.calls==['python_factors'] and x.a.verify_existing(x.f.state)==v


def test_native_exception_preserves_claim_and_forbids_retry(fixture):
    x=fixture;x.f.exceptions.add('python_factors')
    with pytest.raises(ValueError,match='scratch fitter exception'):x.run()
    assert (x.f.state/'action/failure.json').exists() and (x.f.state/'fits/python_factors/claim.json').exists()
    with pytest.raises(ValueError,match='already attempted'):x.run()
    assert x.f.calls==['python_factors']


@pytest.mark.parametrize('code',[1,143,-15])
def test_nonzero_wrapper_preserves_saved_recipe_and_actual_code(fixture,code):
    x=fixture;x.f.dispatch['code']=code
    with pytest.raises(ValueError,match='explicit reconciliation'):x.run()
    assert x.f.calls==['python_factors'] and x.m.read(x.f.state/'action/exit.json')['returncode']==code
    with pytest.raises(ValueError,match='reconciliation'):x.a.verify_existing(x.f.state)


@pytest.mark.parametrize('kind',['certificate_absent','certificate_changed','auditor_changed','terminal_changed',
    'auditor_review_draft','auditor_review_schema','review_draft','missing_certificate_pin','missing_dependency_pin','review_cycle'])
def test_changed_or_incomplete_actual_review_refuses_before_action_claim(fixture,kind):
    x=fixture;m=x.m
    if kind=='certificate_absent':x.certificate.unlink()
    elif kind=='certificate_changed':change(x.certificate,status='pending')
    elif kind=='auditor_changed':m.AUDITOR.write_text('changed')
    elif kind=='terminal_changed':change(x.terminal,actual_state='COMPLETED')
    elif kind.startswith('auditor_review'):change(m.AUDITOR_REVIEW,**({'status':'draft'} if kind.endswith('draft') else {'schema':'wrong'}))
    else:
        v=m.read(m.REVIEW)
        if kind=='review_draft':v['status']='draft'
        elif kind=='missing_certificate_pin':v['files_sha256'].pop(str(x.certificate))
        elif kind=='missing_dependency_pin':v['files_sha256'].pop(str(x.f.source))
        else:v['files_sha256'][str(x.f.state/'future.json')]='a'*64
        write(m.REVIEW,v)
    with pytest.raises((ValueError,FileNotFoundError)):x.run()
    assert not x.f.calls and not x.f.state.exists()


def repin(x):
    """Scratch-only rehash exposes semantic predicates, never actual records."""
    m=x.m;proof=m.read(x.certificate)
    proof['files_sha256']={p:m.sha(p) for p in proof['files_sha256']};write(x.certificate,proof)
    v=m.read(m.REVIEW);v['files_sha256']={p:m.sha(p) for p in v['files_sha256']}
    v.update({field:m.sha(path) for field,path in m.PROOF_FIELDS.items()});write(m.REVIEW,v)


@pytest.mark.parametrize('kind',['scope','complete','receipts','batches','attempts','report','receipt_pin',
    'execution_id','false_success','evaluator_code','summary_source','summary_rows','prep_root','cell_domain',
    'task_seed','task_limit','receipt_status'])
def test_exact_native_python_scope_required_even_when_scratch_certificate_rehashed(fixture,kind):
    x=fixture;m=x.m;v=m.read(x.certificate)
    if kind=='scope':v['array_job_id']=31260227
    elif kind=='complete':v['scientific_outputs_complete']=False
    elif kind=='receipts':v['completed_receipts']=3
    elif kind=='batches':v['completed_batches']=303
    elif kind=='attempts':v['attempts_validated']=1
    elif kind=='report':v['grader_audits'].pop()
    elif kind=='receipt_pin':v['files_sha256'].pop(str(x.receipts[0]))
    elif kind=='execution_id':v['execution']['job_id']='31260227_0'
    elif kind=='false_success':v['execution']['scheduler_success']=True
    elif kind=='evaluator_code':v['execution']['evaluator_returncode']=1
    elif kind=='summary_source':v['summaries'][0]['source']='wrong'
    elif kind=='summary_rows':v['summaries'][0]['rows']=128
    elif kind=='prep_root':change(x.prep,revision_root='old_r2')
    elif kind=='cell_domain':change(x.cell,domain='mathir')
    elif kind.startswith('task_'):
        tasks=m.read(x.tasks);tasks[0]['seeds' if kind=='task_seed' else 'row_limit']=[0] if kind=='task_seed' else 128
        write(x.tasks,tasks);change(x.cell,tasks=tasks)
    else:change(x.receipts[0],status='partial')
    write(x.certificate,v);repin(x)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls


def test_existing_recipe_never_gets_refitted(fixture):
    x=fixture;write(x.m.REVISION_ROOT/'level4/recipes/python_factors.json',{'prior':True})
    with pytest.raises(ValueError,match='unfitted'):x.run()
    assert not x.f.calls


def test_static_verification_does_not_use_current_host_nfs_device(fixture,monkeypatch):
    x=fixture;result=x.run()
    class ForeignLock:
        def stat(self):raise AssertionError('static proof must not inspect current-host device')
    monkeypatch.setattr(x.m.first,'LOCK',ForeignLock())
    assert x.a.verify_existing(x.f.state)==result and x.f.calls==['python_factors']


@pytest.mark.parametrize('kind',['runtime','exit','guest_argv','result','recipe','claim','missing_runtime_pin','missing_result'])
def test_partial_or_tampered_saved_action_rejected_without_refit(fixture,kind):
    x=fixture;x.run();m=x.m;r=x.f.state
    if kind=='runtime':change(r/'action/runtime.json',lock_device=0)
    elif kind=='exit':change(r/'action/exit.json',returncode=1)
    elif kind=='guest_argv':
        v=m.read(r/'action/guest_runtime.json');v['command'].remove('-B');write(r/'action/guest_runtime.json',v)
    elif kind=='result':change(r/'action/result.json',status='invented')
    elif kind=='recipe':change(m.REVISION_ROOT/'level4/recipes/python_factors.json',development_fit_pass=False)
    elif kind=='claim':change(r/'fits/python_factors/claim.json',source_root='wrong')
    elif kind=='missing_result':(r/'action/result.json').unlink()
    else:
        v=m.read(r/'registration.json');v['files_sha256'].pop(str(r/'action/runtime.json'));write(r/'registration.json',v)
        write(r/'registration.sha256.json',{'sha256':m.sha(r/'registration.json')})
    with pytest.raises((ValueError,FileNotFoundError)):x.a.verify_existing(r)
    assert x.f.calls==['python_factors']


def test_live_fence_and_owner_reuse_sealed_functions(fixture,monkeypatch):
    x=fixture
    assert Path(x.m.guest_guard.__code__.co_filename)==x.a.PRIOR
    assert Path(x.m.run.__code__.co_filename)==x.a.PRIOR
    def lost(fd):raise ValueError('lost inherited fence')
    monkeypatch.setattr(x.m,'assert_fence',lost)
    with pytest.raises(ValueError,match='lost inherited fence'):x.run()
    assert not x.f.calls


@pytest.mark.parametrize('drift',[{'lock_device':0},{'start_ticks':'different'},{'command':['wrong']},{'host':'wash.cs.princeton.edu'}])
def test_live_guard_rejects_owner_or_fence_drift_before_fit(fixture,drift):
    x=fixture;x.f.dispatch['drift']=drift
    with pytest.raises(ValueError,match='owner/view/fence'):x.run()
    assert not x.f.calls


@pytest.mark.parametrize('field',['execution_auditor_sha256','execution_auditor_review_sha256','execution_certificate_sha256',
    'execution_terminal_sha256','execution_runtime_sha256','execution_evaluator_exit_sha256',
    'preparation_certificate_sha256','array_job_id','job_id_raw','terminal_policy'])
def test_no_unobserved_future_proof_can_authorize_fit(fixture,field):
    x=fixture;v=x.m.read(x.m.REVIEW);v.pop(field);write(x.m.REVIEW,v)
    with pytest.raises(ValueError):x.run()
    assert not x.f.state.exists() and not x.f.calls


@pytest.mark.parametrize('kind',['completed','failed1','unknown_cause_changed','top_success','top_cause',
    'policy','evaluator_signal','wrong_array_review','wrong_raw_review','missing_runtime_pin','missing_sidecar_pin'])
def test_only_observed_143_evaluator_zero_policy_even_with_scratch_reviewed_hashes(fixture,kind):
    x=fixture;v=x.m.read(x.certificate)
    if kind=='completed':
        v['status']='verified_scientific_outputs_with_successful_execution';v['scheduler_success']=True
        v['execution'].update(state='COMPLETED',exit_code='0:0',scheduler_success=True)
    elif kind=='failed1':v['execution']['exit_code']='1:0'
    elif kind=='unknown_cause_changed':v['execution']['exit_cause']='cleanup'
    elif kind=='top_success':v['scheduler_success']=True
    elif kind=='top_cause':v['exit_cause']='cleanup'
    elif kind=='policy':v['terminal_policy']='any_failed_execution'
    elif kind=='evaluator_signal':v['execution']['evaluator_returncode']=-15
    elif kind=='wrong_array_review':
        v['array_job_id']=31260227;review=x.m.read(x.m.REVIEW);review['array_job_id']=31260227;write(x.m.REVIEW,review)
    elif kind=='wrong_raw_review':
        v['execution']['job_id_raw']='31260228';review=x.m.read(x.m.REVIEW);review['job_id_raw']='31260228';write(x.m.REVIEW,review)
    else:
        field='execution_runtime_sha256' if kind=='missing_runtime_pin' else 'execution_evaluator_exit_sha256'
        v['files_sha256'].pop(str(x.m.PROOF_FIELDS[field]))
    write(x.certificate,v);repin(x)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls


@pytest.mark.parametrize('field',['execution_terminal_sha256','execution_runtime_sha256','execution_evaluator_exit_sha256'])
def test_known_observed_bytes_cannot_be_rebound_by_a_new_review(fixture,field):
    x=fixture;path=x.m.PROOF_FIELDS[field];change(path,altered=True);repin(x)
    with pytest.raises(ValueError,match='exact observed'):x.run()
    assert not x.f.calls
