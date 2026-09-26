"""Scratch-only joins for existing actions; never execute real collector proofs."""
from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path('/n/fs/similarity/maxent-grpo')
SOURCE = ROOT/'artifacts/verify_modebench_scale_recorded_actions_20260912.py'
spec = importlib.util.spec_from_file_location('_test_recorded_actions', SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))
    return path


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(n): return datetime(2026,9,12,12,0,n,tzinfo=timezone.utc).isoformat()


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'SOURCE',tmp_path/'collector.py');m.SOURCE.write_text('scratch source')
    monkeypatch.setattr(m,'TESTS',tmp_path/'test.py');m.TESTS.write_text('scratch tests')
    monkeypatch.setattr(m,'CANONICAL',tmp_path/'frozen.py');m.CANONICAL.write_text('scratch frozen')
    monkeypatch.setattr(m,'FROZEN_SHA',digest(m.CANONICAL))
    stable=tmp_path/'input.txt';stable.write_text('scientific input')
    basic={str(stable):digest(stable)}
    values={}; paths=set()
    for key,(root,relative) in m.RESULTS.items():
        p=m.state(root)/relative
        if key in m.FIT_SCOPES:
            owner,level,decisions=m.FIT_SCOPES[key];records=[]
            for domain,passed in decisions.items():
                recipe=m.state(owner)/'recipes'/f'{domain}.json'
                write(recipe,{'input_sha256':basic,'domain':domain,'development_fit_pass':passed})
                records.append({'domain':domain,'development_fit_pass':passed,'recipe_path':str(recipe),'recipe_sha256':digest(recipe)})
            value={'level':level,'status':'needs_new_development_revision','failed_domains':[d for d,v in decisions.items() if not v], 'fits':records}
            reg=m.state(owner)/'registration.json';write(reg,{'files_sha256':basic});paths.add(reg)
        else:
            domains={'l4_freeze':['graph_coloring','pantry'],'mathir_freeze':['mathir'],'python_freeze':['python_factors'],
                     'graph_prep':['graph_coloring'],'python_prep':['python_factors']}[key]
            value={'status':key+'_saved','domains':domains,'files_sha256':basic}
        write(p,value);paths.add(p);values[key]=value
    for key,(root,relative,owner) in m.ACTIONS.items():
        directory=m.state(root)/relative
        runtime={'at_utc':stamp(2),'host':'spin.cs.princeton.edu','uid':5,'fence_fd':7,
                 'source_sha256':m.COMPONENTS[owner]['source_sha256']}
        rp=write(directory/'runtime.json',runtime)
        command=['python','runner','--','python','-B','script','guest']
        ip=write(directory/'intent.json',{'at_utc':stamp(3),'command':command,'runtime_sha256':digest(rp)})
        guest=command[3:];guest.insert(1,'-B')
        gp=write(directory/'guest_runtime.json',{'at_utc':stamp(4),'host':runtime['host'],'uid':5,'fence_fd':7,
                  'outer_runtime_sha256':digest(rp),'command':guest})
        ep=write(directory/'exit.json',{'at_utc':stamp(5),'returncode':0})
        result=values.get(key,{'status':'registered'})
        op=write(directory/'result.json',result);paths.update((rp,ip,gp,ep,op))
    math=m.state('mathir_check');extra={str(stable):digest(stable)}
    mp=write(math/'stdout.json',{'original_certificate':values['mathir_freeze'],'additive_evidence_pins':extra})
    err=math/'stderr.txt';err.write_text('')
    mi=write(math/'intent.json',{'at_utc':stamp(6),'source_sha256':m.COMPONENTS['mathir_static']['source_sha256'],
          'review_sha256':m.COMPONENTS['mathir_static']['review_sha256'],'observer_host':'spin.cs.princeton.edu',
          'new_grader_invocations':0,'new_model_calls':0})
    me=write(math/'exit.json',{'actual_returncode':0,'at_utc':stamp(7),'source_sha256':m.COMPONENTS['mathir_static']['source_sha256'],
          'stdout_sha256':digest(mp),'stderr_sha256':digest(err)})
    paths.update((mp,err,mi,me))
    py=m.state('python_prep'); directory=py/'development'
    plan={'provider_sha256':m.COMPONENTS['python_prep']['source_sha256'],'dependency_ids':[31254520],
          'phase':'dev','level':'level4','inputs_sha256':basic,'scientific_inputs_sha256':basic}
    pp=write(directory/'plan.json',plan)
    submitted={'array_job_id':31260226,'at_utc':stamp(13),'returncode':0}
    sp=write(directory/'submission_result.json',submitted)
    contract={'schema':'portable','shared_source':str(stable),'historical_destination':'/tmp/exact_history.json','sha256':digest(stable)}
    staging={**contract,'at_utc':stamp(14),'observer_host':'node202.ionic.cs.princeton.edu','observer_uid':363432,
        'created_destination':True,'status':'created_exact_diagnostic_copy','overwrite_permitted':False,'science_or_live_fence_changed':False}
    observed={'status':'actual_portable_development_inputs_verified','source_sha256':m.COMPONENTS['python_transport']['source_sha256'],
        'model_calls':0,'tasks':4,'cells':['level4_python_factors_r3_dev'],'readiness_sha256':digest(py/'certificate.json'),
        'host':staging['observer_host'],'staging':deepcopy(staging)}
    op=write(py/'node202_static_readiness_preflight.json',{'returncode':1,'model_calls':0,'completed_at_utc':stamp(10)})
    np=write(py/'node202_portable_readiness_preflight.json',{'returncode':0,'model_calls':0,'new_dataset_or_grader_actions':False,
        'source_sha256':m.COMPONENTS['python_transport']['source_sha256'],'started_at_utc':stamp(11),'completed_at_utc':stamp(12),'stdout':json.dumps(observed)})
    rp=write(directory/'runtime/0.json',{'array_job_id':31260226,'array_index':0,'plan_sha256':digest(pp),
        'submission_result_sha256':digest(sp),'readiness_sha256':digest(py/'certificate.json'),'portable_evidence_staging':staging,
        'hostname':staging['observer_host'],'status':'validated_before_unchanged_evaluator_subprocess','at_utc':stamp(15)})
    paths.update((pp,sp,op,np,rp))
    called=[]
    def verified(root,*,fresh):
        assert root==py and fresh is False;called.append(('verified',fresh));return plan,None,None,None,None
    def identity(root): assert root==py;called.append(('submission_identity',));return submitted
    modules={'mathir_static':SimpleNamespace(evidence_pins=lambda:extra),
        'python_transport':SimpleNamespace(verified=verified,submission_identity=identity),
        'portable':SimpleNamespace(contract=lambda:contract)}
    monkeypatch.setattr(m,'ACTUAL',{str(p.relative_to(tmp_path)):digest(p) for p in paths})
    monkeypatch.setattr(m,'reviewed_closure',lambda:dict(basic))
    monkeypatch.setattr(m,'modules',lambda:modules)
    monkeypatch.setattr(m,'readonly_proofs',lambda _:deepcopy(values))
    return SimpleNamespace(root=tmp_path,values=values,modules=modules,paths=paths,called=called,plan=plan,submission=submitted)


def test_full_synthetic_collection_keeps_failures_and_post_exit_records(scratch):
    value=m.verify()
    assert value['status']=='verified_recorded_actions_only'
    assert len(value['actions'])==10
    assert value['new_grader_invocations']==value['new_model_calls']==value['new_fit_publications']==0
    assert value['final_execution_provenance'] is value['admission_performed'] is False
    assert value['saved_decisions']['l4_fits']['decisions']['python_factors'] is False
    assert value['saved_decisions']['recovered_fits']['decisions']=={'pantry':False,'python_factors':True}
    assert value['saved_decisions']['graph_fit']['decisions']=={'graph_coloring':False}
    assert value['python_development_submission']['terminal_state_claimed'] is False
    for root,relative,_ in m.ACTIONS.values():
        for n in ('exit.json','result.json','runtime.json','intent.json','guest_runtime.json'):
            assert str(m.state(root)/relative/n) in value['files_sha256']
    assert scratch.called==[('verified',False),('submission_identity',)]


@pytest.mark.parametrize('kind',['bad_hash','missing_exit','wrapper_failure','guest_argv','guest_link','source','host','chronology','failure_record'])
def test_damaged_or_failed_old_action_stops_before_any_publication(scratch,kind):
    directory=m.state('graph_fit')/'action'
    path=directory/'exit.json'; value=m.read(path)
    if kind=='bad_hash':path.write_text('{}')
    elif kind=='missing_exit':path.unlink()
    elif kind=='failure_record':write(directory/'failure.json',{'error':'unknown'})
    else:
        if kind=='wrapper_failure':value['returncode']=143
        else:
            path=directory/'guest_runtime.json';value=m.read(path)
            if kind=='guest_argv':value['command'].append('changed')
            elif kind=='guest_link':value['outer_runtime_sha256']='0'*64
            elif kind=='host':value['host']='wash.cs.princeton.edu'
            elif kind=='chronology':value['at_utc']=stamp(1)
            elif kind=='source':path=directory/'runtime.json';value=m.read(path);value['source_sha256']='0'*64
        write(path,value)
        # Even when a synthetic root trusted pin is rebound, semantic gates hold.
        m.ACTUAL[str(path.relative_to(scratch.root))]=digest(path)
    with pytest.raises((ValueError,FileNotFoundError)):m.verify()


@pytest.mark.parametrize('kind',['promote_failed','omit_domain','duplicate_domain','wrong_level','wrong_failed_list','recipe_drift','different_provider_result'])
def test_saved_fit_cannot_be_reselected_or_relabelled(scratch,kind):
    values=deepcopy(scratch.values);v=values['l4_fits']
    if kind=='promote_failed':v['fits'][2]['development_fit_pass']=True
    elif kind=='omit_domain':v['fits'].pop()
    elif kind=='duplicate_domain':v['fits'][2]=deepcopy(v['fits'][0])
    elif kind=='wrong_level':v['level']='level5'
    elif kind=='wrong_failed_list':v['failed_domains']=[]
    elif kind=='recipe_drift':Path(v['fits'][0]['recipe_path']).write_text('{}')
    else:v['status']='a_new_result'
    if kind!='different_provider_result':write(m.state('l4_first')/'actions/fit/result.json',v)
    with pytest.raises(ValueError):m.saved_proofs(values,{})


@pytest.mark.parametrize('key,value',[('actual_returncode',143),('source_sha256','0'*64),('stdout_sha256','0'*64)])
def test_mathir_readonly_observation_preserves_its_actual_zero(scratch,key,value):
    p=m.state('mathir_check')/'exit.json';v=m.read(p);v[key]=value;write(p,v)
    with pytest.raises(ValueError):m.mathir_observation(scratch.values['mathir_freeze'],scratch.modules['mathir_static'],{})


@pytest.mark.parametrize('kind',['wrong_job','wrong_dependency','wrong_plan','wrong_staging_hash','overwrite','invented_completion','preflight_failure','preflight_scope'])
def test_python_submission_and_staging_never_imply_current_scientific_completion(scratch,kind):
    py=m.state('python_prep');p=py/'development/runtime/0.json';v=m.read(p)
    if kind=='wrong_job':scratch.submission['array_job_id']=99
    elif kind=='wrong_dependency':scratch.plan['dependency_ids']=[1]
    elif kind=='preflight_failure':p=py/'node202_portable_readiness_preflight.json';v=m.read(p);v['returncode']=1
    elif kind=='preflight_scope':p=py/'node202_portable_readiness_preflight.json';v=m.read(p);obs=json.loads(v['stdout']);obs['tasks']=1;v['stdout']=json.dumps(obs)
    elif kind=='wrong_plan':v['plan_sha256']='0'*64
    elif kind=='wrong_staging_hash':v['portable_evidence_staging']['sha256']='0'*64
    elif kind=='overwrite':v['portable_evidence_staging']['overwrite_permitted']=True
    else:v['status']='scientific_outputs_complete'
    write(p,v)
    with pytest.raises(ValueError):m.python_submission(scratch.modules,{})


def test_future_receipts_batches_and_terminal_are_outside_stable_scope(scratch):
    before=m.verify();directory=m.state('python_prep')/'development'
    write(directory/'terminal_accounting.json',{'future':'untrusted'})
    write(directory/'runtime/0.evaluator_exit.json',{'returncode':143})
    write(directory/'results/difficulty_0.json',{'future':'untrusted'})
    write(directory/'results/difficulty_0.json.batches/new.json',{'future':'untrusted'})
    assert m.verify()==before


def test_frozen_view_is_required_before_static_dispatch(scratch,monkeypatch):
    m.CANONICAL.write_text('neutral host source')
    monkeypatch.setattr(m,'modules',lambda:pytest.fail('must fail before proof dispatch'))
    with pytest.raises(ValueError,match='frozen source'):m.verify()


def test_static_dispatch_uses_exact_adapters_and_only_readonly_apis(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    calls=[]
    def verify(name):
        def called(root,*,guest):
            assert guest is True;calls.append((name,root));return {'proof':name}
        return called
    frozen=SimpleNamespace(successful_action=lambda _:pytest.fail('old NFS predicate replayed'))
    prep=SimpleNamespace(verify=verify('python_prep'))
    def historical(module,root):
        assert module is frozen;calls.append(('l4_historical',root));return {'old':'freeze'}
    prep.frozen_successful_action=historical
    def frozen_verify(root,*,guest):
        assert guest is True;assert frozen.successful_action(root)=={'old':'freeze'}
        calls.append(('l4_freeze',root));return {'proof':'l4_freeze'}
    frozen.verify=frozen_verify
    graph=SimpleNamespace(verify=verify('graph_prep'),_first=SimpleNamespace(verify_existing=lambda *a,**k:pytest.fail('raw L5 device predicate')))
    def install(provider):
        assert provider is graph;calls.append(('graph_adapter',None));provider._first.verify_existing=verify('l5_fits');return provider
    mods={'first':SimpleNamespace(verify_fits=verify('l4_fits')),'python_prep':prep,'l4_freeze':frozen,
          'graph_prep':graph,'graph_static':SimpleNamespace(install_readonly_verifiers=install),
          'mathir_static':SimpleNamespace(verify=verify('mathir_freeze')),
          'recovered_fits':SimpleNamespace(verify_existing=verify('recovered_fits')),
          'python_freeze':SimpleNamespace(verify=verify('python_freeze')),
          'graph_fit':SimpleNamespace(verify_existing=verify('graph_fit'))}
    result=m.readonly_proofs(mods)
    assert set(result)==set(m.RESULTS)
    assert [name for name,_ in calls]==['l4_fits','l4_historical','l4_freeze','graph_adapter','l5_fits','graph_prep',
        'mathir_freeze','recovered_fits','python_freeze','graph_fit','python_prep']
    assert dict(calls)['l5_fits']==dict(calls)['mathir_freeze']==m.state('l5_ready')


@pytest.fixture
def tiny_reviews(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'SOURCE',tmp_path/'collector.py');monkeypatch.setattr(m,'TESTS',tmp_path/'collector_test.py')
    src=tmp_path/'source.py';src.write_text('source')
    test=tmp_path/'test.py';test.write_text('test')
    childsrc=tmp_path/'old.py';childsrc.write_text('old')
    child=write(tmp_path/'old_review.json',{'schema':'prior_review_v1','status':'pending_historical_only',
        'files_sha256':{str(childsrc):digest(childsrc)}})
    review=write(tmp_path/'review.json',{'schema':'component_review_v1','status':'reviewed',
        'files_sha256':{str(src):digest(src),str(test):digest(test),str(child):digest(child)}})
    info={'source':'source.py','tests':'test.py','review':'review.json','source_sha256':digest(src),
          'tests_sha256':digest(test),'review_sha256':digest(review)}
    monkeypatch.setattr(m,'COMPONENTS',{'one':info})
    return SimpleNamespace(info=info,src=src,test=test,review=review,child=child,childsrc=childsrc)


def test_direct_source_tests_review_and_pending_ancestor_are_all_pinned(tiny_reviews):
    p=tiny_reviews;pins=m.reviewed_closure()
    assert set(pins)=={str(x) for x in (p.src,p.test,p.review,p.child,p.childsrc)}
    assert m.read(p.child)['status']=='pending_historical_only'


@pytest.mark.parametrize('kind',['missing_test','wrong_test_pin','wrong_source','pending_leaf','nested_drift','self_pin','future_collector_pin'])
def test_review_closure_fails_closed_without_promoting_old_or_draft_reviews(tiny_reviews,kind):
    p=tiny_reviews;v=m.read(p.review)
    if kind=='missing_test':del v['files_sha256'][str(p.test)]
    elif kind=='wrong_test_pin':v['files_sha256'][str(p.test)]='0'*64
    elif kind=='wrong_source':p.src.write_text('changed')
    elif kind=='pending_leaf':v['status']='pending'
    elif kind=='nested_drift':p.childsrc.write_text('changed')
    elif kind=='self_pin':v['files_sha256'][str(p.review)]='0'*64
    else:m.SOURCE.write_text('draft');v['files_sha256'][str(m.SOURCE)]=digest(m.SOURCE)
    write(p.review,v);p.info['review_sha256']=digest(p.review)
    with pytest.raises(ValueError):m.reviewed_closure()


def test_duplicate_hash_conflicts_are_rejected(tmp_path):
    p=tmp_path/'proof';p.write_text('new')
    with pytest.raises(ValueError,match='conflicting'):m.merge({str(p):'0'*64},{str(p):digest(p)})


def test_scope_contains_no_future_fit_receipt_terminal_or_publication_api():
    assert set(m.RESULTS)=={'l4_fits','l4_freeze','l5_fits','mathir_freeze','recovered_fits','python_freeze','graph_prep','graph_fit','python_prep'}
    assert not any('terminal' in name or '.batches/' in name or '/results/development/' in name or 'evaluator_exit' in name for name in m.ACTUAL)
    tree=ast.parse(SOURCE.read_text())
    forbidden={'audit_source','confirm_domain','receipt_scores','fit_domain','freeze_dataset','freeze_passed',
               'prepare','submit','worker','stage','run','Popen','system','publish_level','_sweep','atomic_new',
               'write_text','write_bytes','unlink','mkdir','assert_fence','host_guard'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            assert getattr(node.func,'attr',getattr(node.func,'id',None)) not in forbidden
