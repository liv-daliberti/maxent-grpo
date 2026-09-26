"""Scratch future-PASS release composition; no actual freeze, bind or heldout."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level4_python_r3_fit import fixture as python_fixture, prior_fixture, write, repin
from test_modebench_scale_level4_python_r4_fit_observed_cpu import fixture as fit_fixture, r3_fixture, repin as repin_r4, base_fixture, observed_base_authorized

ROOT=Path(__file__).resolve().parents[1]

def change(path,fn):
    v=json.loads(Path(path).read_text());fn(v);write(path,v)

@pytest.fixture
def x(fit_fixture,monkeypatch):
    migration=fit_fixture
    f=SimpleNamespace(**vars(migration.x));f.a=migration.a;f.run=migration.run;tmp=f.f.state.parent
    spec=importlib.util.spec_from_file_location('_scratch_l4_release',ROOT/'artifacts/continue_modebench_scale_level4_first_release_r4_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    state=tmp/'release_action';parent=tmp/'campaign';release=tmp/'release_v2';python=f.m.REVISION_ROOT
    def paths(root,level,domain):
        root=Path(root);return {'protocol':root/'protocol.json','recipe':root/level/'recipes'/(domain+'.json'),
            'dataset':root/level/'dataset'/domain,'receipt':root/level/'results/confirmation'/(domain+'.json'),
            'audit':root/level/'confirmation'/(domain+'.json')}
    sources={};retained={};counts={'train':384,'dev':128,'eval':128}
    def dataset(root,domain):
        q=paths(root,'level4',domain);identity={'splits':{k:{'rows':n,'rows_sha256':k} for k,n in counts.items()}}
        write(q['dataset']/'identity.json',identity)
        for split in counts:write(q['dataset']/(split+'.jsonl'),{'scratch_split':split})
        return identity
    write(parent/'protocol.json',{'models':{'7b':{'path':'synthetic7b'}}})
    for domain in a.RETAINED:
        kind='campaign_v1' if domain in ('countdown','mathir') else 'domain_revision_v1'
        r=parent if kind=='campaign_v1' else tmp/domain
        if kind!='campaign_v1':write(r/'protocol.json',{'draw_labels':{'eval':[900,901,902,903]}})
        write(paths(r,'level4',domain)['recipe'],{'development_fit_pass':True})
        identity=dataset(r,domain);source={'source_kind':kind,'source_root':str(r),'protocol_sha256':a.sha(r/'protocol.json'),'parent_recipe_sha256':'synthetic'}
        sources[domain]=source;retained[domain]={'source':source,'dataset':str(paths(r,'level4',domain)['dataset']),
            'identity_sha256':a.sha(paths(r,'level4',domain)['dataset']/'identity.json'),'splits':identity['splits']}
    change(python/'protocol.json',lambda v:v.update(draw_labels={'eval':[7204500,7204501,7204502,7204503]}))
    change(f.prep,lambda v:v.update(preserved_level4_sources=retained,preserved_python_failure_sha256='old-python-failed'))
    repin(f)
    for proof_root in (migration.a.AUDIT_ROOT,migration.a.READONLY_ROOT):
        change(proof_root/'exit.json',lambda v:v.update(certificate_sha256=f.m.sha(f.certificate)))
    repin_r4(migration)
    f.run()
    for key,value in {'STATE':state,'PARENT':parent,'RELEASE':release,'SOURCE_MANIFEST':release/'level4/source_manifest.json',
        'PYTHON_ROOT':python,'PYTHON_RECIPE':python/'level4/recipes/python_factors.json','FIT_STATE':f.f.state,
        'FIT_RESULT':f.f.state/'action/result.json','PREPARATION_STATE':f.m.PREPARATION,'PREPARATION_SHA':a.sha(f.prep),
        'FIT_REVIEW':f.m.REVIEW,'REVIEW':tmp/'release_review.json','MANIFEST':f.m.VIEW,'LOCK':f.m.first.LOCK,
        'LOCK_INODE':f.m.first.LOCK_INODE,'assert_fence':f.m.assert_fence,'lifetime_fence':f.m.lifetime_fence}.items():monkeypatch.setattr(a,key,value)
    prep=SimpleNamespace(FAILED_RECIPE_SHA='old-python-failed',development_inputs=lambda *args,**kwargs:
        {'files_sha256':{str(f.prep):a.sha(f.prep)},'cell':f.m.read(f.cell)})
    reviewed_calls=[]
    def reviewed(expected,fitsha,*,guest):
        assert expected==a.sha(a.REVIEW) and fitsha==a.sha(a.FIT_PROVIDER)
        reviewed_calls.append(guest)
        return prep,f.a,{str(f.f.source):a.sha(f.f.source),str(a.REVIEW):a.sha(a.REVIEW)}
    write(a.REVIEW,{'scratch_final_actual_pass_contract':True})
    original_reviewed=a.reviewed
    monkeypatch.setattr(a,'reviewed',reviewed)
    calls=[];fail=set();overlap=set();new_sources={};auth=[]
    def frozen(root,level,domain,kind):
        assert level=='level4';return a.read(paths(root,level,domain)['dataset']/'identity.json')
    def freeze(root):
        assert Path(root)==python and f.f.held==[8];calls.append(('freeze','python_factors'))
        write(python/'exclusions/freeze.json',{'scratch_full_history':True})
        if 'snapshot' in fail:raise ValueError('native partial snapshot')
        dataset(python,'python_factors')
        if 'freeze' in fail:raise ValueError('native partial dataset')
    def bind(p,r,level,mapping):
        assert p==parent and r==release and level=='level4'
        assert mapping=={'graph_coloring':sources['graph_coloring']['source_root'],'pantry':sources['pantry']['source_root'],'python_factors':str(python)}
        calls.append(('bind','level4'));assert (state/'binding_claim.json').exists()
        if 'bind' in fail:raise ValueError('native binding failure')
        value={'sources':{**sources,'python_factors':{'source_kind':'domain_revision_v1','source_root':str(python)}}}
        write(a.SOURCE_MANIFEST,value);write(a.SOURCE_MANIFEST.with_name('source_manifest.sha256.json'),{'sha256':a.sha(a.SOURCE_MANIFEST)})
        return value
    def cross(root,level,domain):
        assert level=='level4'
        if domain in overlap:raise ValueError('new external overlap')
        calls.append(('cross',domain));return {'files_sha256':{str(f.f.source):a.sha(f.f.source),**new_sources},'current_sources':len(new_sources)}
    def task(root,domain,kind):
        q=paths(root,'level4',domain);labels=[64,65,66,67] if kind=='campaign_v1' else a.read(q['protocol'])['draw_labels']['eval']
        t={'level':'level4','domain':domain,'split':'eval','interface':'native-interface','rows_jsonl':str(q['dataset']/'eval.jsonl'),
            'output':str(q['receipt']),'seeds':labels,'batch_size':8,'row_offset':0,'row_limit':0}
        return {'id':domain+'_eval','level':'level4','domain':domain,'phase':'eval','source_kind':kind,
            'source_root':str(root),'model_label':'7b','tasks':[t]},{str(q['dataset']/'eval.jsonl'):a.sha(q['dataset']/'eval.jsonl')}
    def carried(p,l,d):calls.append(('carry',d));return task(p,d,'campaign_v1')
    def revision_task(root,phase):
        assert phase=='eval';d='python_factors' if Path(root)==python else Path(root).name
        calls.append(('revision_task',d));return task(root,d,'domain_revision_v1')
    core=SimpleNamespace(domain_paths=paths,frozen_identity=frozen,verify_dataset=cross,bind_level=bind,
        source_manifest=lambda *args:a.read(a.SOURCE_MANIFEST),carried_inputs=carried,
        revision=SimpleNamespace(freeze_dataset=freeze,launch_inputs=revision_task),
        original=SimpleNamespace(authenticate=lambda path:a.read(path),labels=lambda *args:[64,65,66,67]),
        launch=SimpleNamespace(evaluator=lambda:SimpleNamespace(INTERFACE='native-interface')))
    def authenticate(path):assert f.f.canonical.read_text()=='neutral';auth.append(path)
    monkeypatch.setattr(a,'load_module',lambda p,n:core if p==a.CORE else SimpleNamespace(authenticate=authenticate))
    monkeypatch.setattr(a,'live_host_guard',lambda *,guest:f.m.first.host_guard(guest=guest))
    f.f.owner['command']=[str(a.PYTHON),'-B',str(a.SOURCE),'freeze-bind'];active=[]
    def identity(pid):
        if active and pid!=f.f.owner['pid']:
            argv=active[0][active[0].index('--')+1:];argv.insert(1,'-B');return {**deepcopy(f.f.owner),'pid':12346,'start_ticks':'guest','command':argv}
        return deepcopy(f.f.owner)
    monkeypatch.setattr(a,'process_identity',identity)
    dispatch={'code':0}
    def run(argv,**kwargs):
        assert kwargs['pass_fds']==(8,) and f.f.held==[8]
        active.append(argv);f.f.canonical.write_text('old')
        try:a.guest_action(state,8,a.sha(a.REVIEW),a.sha(a.FIT_PROVIDER))
        finally:active.clear();f.f.canonical.write_text('neutral')
        return SimpleNamespace(returncode=dispatch['code'])
    monkeypatch.setattr(a,'subprocess',SimpleNamespace(run=run))
    return SimpleNamespace(a=a,f=f,state=state,paths=paths,sources=sources,calls=calls,fail=fail,overlap=overlap,
        new_sources=new_sources,dispatch=dispatch,auth=auth,reviewed_calls=reviewed_calls,original_reviewed=original_reviewed,preparer=prep,
        run=lambda:a.run(state,a.sha(a.REVIEW),a.sha(a.FIT_PROVIDER)))


def test_only_python_freeze_then_one_five_source_bind_and_native_tasks(x):
    before={d:x.a.sha(x.paths(s['source_root'],'level4',d)['dataset']/'identity.json') for d,s in x.sources.items()}
    value=x.run()
    assert [v for v in x.calls if v[0]=='freeze']==[('freeze','python_factors')]
    assert [v for v in x.calls if v[0]=='bind']==[('bind','level4')]
    assert [c['domain'] for c in value['cells']]==list(x.a.DOMAINS)
    assert sum(k=='carry' for k,d in x.calls)==2 and sum(k=='revision_task' for k,d in x.calls)==3
    assert {d:x.a.sha(x.paths(s['source_root'],'level4',d)['dataset']/'identity.json') for d,s in x.sources.items()}==before
    assert value['dependency_ids']==[31261726] and value['models']=={'7b':'synthetic7b'}
    assert x.f.f.calls==['python_factors'] and value['level5_action_performed'] is False
    assert x.auth==[x.a.MANIFEST]


def test_static_real_one_fit_composition_survives_outputs_and_rechecks_new_sources(x,monkeypatch):
    value=x.run();fit_calls=list(x.f.f.calls)
    class Foreign:
        def stat(self):raise AssertionError('no static historical NFS device replay')
    monkeypatch.setattr(x.a,'LOCK',Foreign());monkeypatch.setattr(x.f.m.first,'LOCK',Foreign())
    first=x.a.confirmation_inputs(x.state,guest=False)
    for cell in value['cells']:
        out=Path(cell['tasks'][0]['output']);write(out,{'scratch_future_receipt':True});write(Path(str(out)+'.batches')/'run.json',{})
    extra=x.state.parent/'new_external.json';write(extra,{'scratch_disjoint_rows':True});x.new_sources[str(extra)]=x.a.sha(extra)
    second=x.a.confirmation_inputs(x.state,guest=False)
    assert first['files_sha256']==second['files_sha256'] and first['current_disjointness']!=second['current_disjointness']
    assert x.f.f.calls==fit_calls and len([v for v in x.calls if v[0]=='freeze'])==1
    x.overlap.add('pantry')
    with pytest.raises(ValueError,match='external overlap'):x.a.confirmation_inputs(x.state,guest=False)


@pytest.mark.parametrize('where',['snapshot','freeze','bind'])
def test_partial_native_failure_preserves_claims_and_snapshot_no_retry(x,where):
    x.fail.add(where)
    with pytest.raises(ValueError):x.run()
    failure=x.a.read(x.state/'action/failure.json');snapshot=x.a.PYTHON_ROOT/'exclusions/freeze.json'
    assert failure['files_sha256'][str(snapshot)]==x.a.sha(snapshot)
    assert (x.state/'python_freeze/claim.json').exists() and not (x.state/'confirmation_readiness.json').exists()
    with pytest.raises(ValueError,match='already attempted'):x.run()
    assert len([v for v in x.calls if v[0]=='freeze'])==1


@pytest.mark.parametrize('code',[1,143,-15])
def test_nonzero_outer_exit_keeps_complete_readiness_but_static_refuses(x,code):
    x.dispatch['code']=code
    with pytest.raises(ValueError,match='reconciliation'):x.run()
    assert (x.state/'confirmation_readiness.json').exists()
    assert x.a.read(x.state/'action/exit.json')['returncode']==code
    with pytest.raises(ValueError,match='reconciliation'):x.a.confirmation_inputs(x.state,guest=False)


@pytest.mark.parametrize('kind',['failed_fit','changed_recipe','missing_retained','old_manifest','existing_python','parent_holdout','selected_holdout'])
def test_real_fit_and_all_five_freshness_gates_precede_native_actions(x,kind,monkeypatch):
    a=x.a
    if kind=='failed_fit':change(a.FIT_RESULT,lambda v:v.update(status='needs_new_development_revision',failed_domains=['python_factors']))
    elif kind=='changed_recipe':change(a.PYTHON_RECIPE,lambda v:v.update(development_fit_pass=False))
    elif kind=='missing_retained':change(x.f.prep,lambda v:v['preserved_level4_sources'].pop('pantry'));monkeypatch.setattr(a,'PREPARATION_SHA',a.sha(x.f.prep))
    elif kind=='old_manifest':write(a.SOURCE_MANIFEST,{})
    elif kind=='existing_python':write(a.PYTHON_ROOT/'level4/dataset/python_factors/partial.json',{})
    else:
        root=a.PARENT if kind=='parent_holdout' else Path(x.sources['pantry']['source_root'])
        write(x.paths(root,'level4','pantry')['receipt'],{})
    with pytest.raises((ValueError,FileNotFoundError)):x.run()
    assert not any(k in ('freeze','bind') for k,d in x.calls)


@pytest.mark.parametrize('kind',['source','labels','model','binding','freeze_claim','missing_pin','runtime'])
def test_static_reconstruction_refuses_rehashed_claim_task_or_scope_drift(x,kind):
    x.run();a=x.a;ready=x.state/'confirmation_readiness.json';v=a.read(ready)
    if kind=='source':change(a.SOURCE_MANIFEST,lambda v:v['sources']['python_factors'].update(source_root='old-r2'))
    elif kind=='labels':v['cells'][0]['tasks'][0]['seeds']=[1]
    elif kind=='model':v['models']={'7b':'wrong'}
    elif kind=='binding':change(x.state/'binding_claim.json',lambda v:v.update(level='level5'))
    elif kind=='freeze_claim':change(x.state/'python_freeze/claim.json',lambda v:v.update(source_root='wrong'))
    elif kind=='missing_pin':v['files_sha256'].pop(str(x.state/'binding_claim.json'))
    else:change(x.state/'action/runtime.json',lambda v:v.update(lock_device=0))
    v['files_sha256']={p:a.sha(p) for p in v['files_sha256']};write(ready,v);write(x.state/'action/result.json',v)
    write(x.state/'confirmation_readiness.sha256.json',{'sha256':a.sha(ready)})
    with pytest.raises((ValueError,FileNotFoundError)):a.confirmation_inputs(x.state,guest=False)


def test_missing_future_fit_source_or_hash_cannot_enter_review_contract():
    spec=importlib.util.spec_from_file_location('_unconfigured_l4_release',ROOT/'artifacts/continue_modebench_scale_level4_first_release_r4_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    for digest in (None,'','pending','0'*64):
        with pytest.raises((ValueError,FileNotFoundError)):a.reviewed('0'*64,digest,guest=False)


def test_lost_live_fence_blocks_before_native_freeze(x,monkeypatch):
    def lost(fd):raise ValueError('lost live fence')
    monkeypatch.setattr(x.a,'assert_fence',lost)
    with pytest.raises(ValueError,match='lost live fence'):x.run()
    assert not any(k in ('freeze','bind') for k,d in x.calls)

@pytest.fixture
def review_contract(x,monkeypatch):
    a=x.a;prep=x.preparer
    prep.TESTS=ROOT/'tests/test_modebench_scale_level4_python_r4_registration.py'
    prep.REVIEW=ROOT/'artifacts/modebench_scale_level4_python_r4_registration_independent_review_20260913.json'
    monkeypatch.setattr(x.f.a,'STATE',a.FIT_STATE);monkeypatch.setattr(x.f.a,'REVIEW',a.FIT_REVIEW)
    oldload=a.load_module
    def load(path,name):
        if path==a.PREPARER:return prep
        if path==a.FIT_PROVIDER:return x.f.a
        return oldload(path,name)
    monkeypatch.setattr(a,'load_module',load);monkeypatch.setattr(a,'reviewed',x.original_reviewed)
    required=(a.SOURCE,a.TESTS,a.PREDECESSOR_COPY,a.PREDECESSOR_TESTS_COPY,a.PREDECESSOR_MANIFEST,a.FIRST,a._first.TESTS,a._first.REVIEW,a.PREPARER,prep.TESTS,prep.REVIEW,
        a.FIT_PROVIDER,a.FIT_TESTS,a.FIT_REVIEW,a.FIT_RESULT,a.PYTHON_RECIPE,a.STAGER,*a._first.SEALED)
    pins={str(p):a.sha(p) for p in required}
    for p in (a._first.REVIEW,prep.REVIEW,a.FIT_REVIEW):
        for path,digest in a.read(p)['files_sha256'].items():
            assert path not in pins or pins[path]==digest
            pins[path]=digest
    write(a.REVIEW,{'schema':'modebench_scale_level4_first_release_independent_review_v1','status':'reviewed',
        'fit_provider_sha256':a.sha(a.FIT_PROVIDER),'python_fit_result_sha256':a.sha(a.FIT_RESULT),
        'python_recipe_sha256':a.sha(a.PYTHON_RECIPE),'files_sha256':pins})
    return x


def test_unmocked_final_review_contract_requires_real_fit_interface_and_exact_closure(review_contract):
    x=review_contract;a=x.a
    prep,fit,pins=a.reviewed(a.sha(a.REVIEW),a.sha(a.FIT_PROVIDER),guest=False)
    assert prep==x.preparer and fit==x.f.a and pins[str(a.REVIEW)]==a.sha(a.REVIEW)
    assert a.passing_inputs(x.state,a.sha(a.REVIEW),a.sha(a.FIT_PROVIDER),guest=False)['dependency_ids']==[31261726]
    assert not x.state.exists() and not any(k in ('freeze','bind') for k,d in x.calls)


@pytest.mark.parametrize('kind',['draft','wrong_schema','missing_result_hash','wrong_recipe_hash','missing_recipe_pin',
    'missing_dependency','self_pin','future_pin','wrong_fit_schema','wrong_provider_hash'])
def test_final_review_contract_rejects_placeholders_missing_closure_and_cycles(review_contract,kind):
    x=review_contract;a=x.a;v=a.read(a.REVIEW)
    if kind=='draft':v['status']='draft'
    elif kind=='wrong_schema':v['schema']='wrong'
    elif kind=='missing_result_hash':v.pop('python_fit_result_sha256')
    elif kind=='wrong_recipe_hash':v['python_recipe_sha256']='0'*64
    elif kind=='missing_recipe_pin':v['files_sha256'].pop(str(a.PYTHON_RECIPE))
    elif kind=='missing_dependency':v['files_sha256'].pop(str(x.f.f.source))
    elif kind=='self_pin':v['files_sha256'][str(a.REVIEW)]='0'*64
    elif kind=='future_pin':
        p=a.RELEASE/'level4/future.json';write(p,{});v['files_sha256'][str(p)]=a.sha(p)
    elif kind=='wrong_fit_schema':
        change(a.FIT_REVIEW,lambda v:v.update(schema='wrong'))
        v['files_sha256'][str(a.FIT_REVIEW)]=a.sha(a.FIT_REVIEW)
    else:v['fit_provider_sha256']='0'*64
    write(a.REVIEW,v)
    with pytest.raises((ValueError,FileNotFoundError)):a.reviewed(a.sha(a.REVIEW),a.sha(a.FIT_PROVIDER),guest=False)
    assert not x.state.exists() and not any(k in ('freeze','bind') for k,d in x.calls)


def test_prospective_soak_defaults_preserve_historical_modules_and_fence():
    spec=importlib.util.spec_from_file_location('_soak_defaults_l4_release',ROOT/'artifacts/continue_modebench_scale_level4_first_release_r4_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    assert a.HOST=='soak.cs.princeton.edu' and a.UID==363432
    assert a._first.HOST==a._first._base.HOST=='spin.cs.princeton.edu'
    assert a.assert_fence is a._first.assert_fence and a.lifetime_fence is a._first.lifetime_fence
    assert a.FIT_PROVIDER==ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_observed_cpu_20260913.py'
    assert a.FIT_TESTS==ROOT/'tests/test_modebench_scale_level4_python_r4_fit_observed_cpu.py'
    assert a.FIT_REVIEW==ROOT/'artifacts/modebench_scale_level4_python_r4_fit_independent_review_20260913.json'
    assert a.FIT_STATE==ROOT/'var/artifacts/modebench_scale_level4_python_r4_fit_20260913'
    assert a.FIT_RESULT==a.FIT_STATE/'action/result.json'
    assert a.STATE==ROOT/'var/artifacts/modebench_scale_level4_first_release_20260912'
    assert a.RELEASE==ROOT/'var/data/modebench_scale_release_v2'
    assert a.STAGER==ROOT/'artifacts/stage_modebench_scale_python_harder_evidence_20260912.py'


@pytest.mark.parametrize('guest',[False,True])
@pytest.mark.parametrize('kind',['correct','spin','wash','wrong_socket','wrong_uid','wrong_euid','wrong_source'])
def test_live_soak_identity_and_source_view_gate_precedes_release_inputs(monkeypatch,guest,kind):
    spec=importlib.util.spec_from_file_location('_soak_guard_l4_release',ROOT/'artifacts/continue_modebench_scale_level4_first_release_r4_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    host='spin.cs.princeton.edu' if kind=='spin' else 'wash.cs.princeton.edu' if kind=='wash' else a.HOST
    monkeypatch.setattr(a,'os',SimpleNamespace(uname=lambda:SimpleNamespace(nodename=host),
        getuid=lambda:a.UID+1 if kind=='wrong_uid' else a.UID,
        geteuid=lambda:a.UID+1 if kind=='wrong_euid' else a.UID))
    monkeypatch.setattr(a,'socket',SimpleNamespace(gethostname=lambda:'node202' if kind=='wrong_socket' else host))
    expected=a._first.OLD_SHA if guest else a._first.NEUTRAL_SHA
    monkeypatch.setattr(a,'sha',lambda path:'wrong' if kind=='wrong_source' else expected)
    monkeypatch.setattr(a,'passing_inputs',lambda *args,**kwargs:pytest.fail('invalid host/source must fail before inputs'))
    if kind=='correct':a.live_host_guard(guest=guest)
    else:
        with pytest.raises(ValueError):a.live_host_guard(guest=guest)
        with pytest.raises(ValueError):a.run(a.STATE,'unused','unused')


@pytest.mark.parametrize('kind',['relative_python','relative_source','other_python','other_action','extra_option'])
def test_absolute_outer_command_refused_before_inputs_fence_or_state(monkeypatch,kind):
    spec=importlib.util.spec_from_file_location('_absolute_l4_release',ROOT/'artifacts/continue_modebench_scale_level4_first_release_r4_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    command=[str(a.PYTHON),'-B',str(a.SOURCE),'freeze-bind']
    if kind=='relative_python':command[0]='var/seed_paper_eval/paper310/bin/python'
    elif kind=='relative_source':command[2]='artifacts/continue_modebench_scale_level4_first_release_r4_20260913.py'
    elif kind=='other_python':command[0]='/usr/bin/python'
    elif kind=='other_action':command[3]='verify'
    else:command.insert(2,'-u')
    monkeypatch.setattr(a,'process_identity',lambda pid:{'command':command})
    monkeypatch.setattr(a,'live_host_guard',lambda *,guest:None)
    monkeypatch.setattr(a,'passing_inputs',lambda *args,**kwargs:pytest.fail('relative argv must precede inputs'))
    monkeypatch.setattr(a,'lifetime_fence',lambda:pytest.fail('relative argv must precede fence acquisition'))
    with pytest.raises(ValueError,match='full absolute literal Python'):
        a.run(a.STATE,'unused','unused')


def test_exact_absolute_outer_command_keeps_literal_venv_path(monkeypatch):
    spec=importlib.util.spec_from_file_location('_absolute_l4_release_accept',ROOT/'artifacts/continue_modebench_scale_level4_first_release_r4_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    owner={'command':[str(a.PYTHON),'-B',str(a.SOURCE),'freeze-bind','--root',str(a.STATE)]}
    monkeypatch.setattr(a,'process_identity',lambda pid:owner)
    assert a.outer_command_guard() is owner


def test_absolute_outer_command_rechecked_under_fence_before_action_directory(x,monkeypatch):
    original=x.a.outer_command_guard;calls=[]
    def changed_under_fence():
        calls.append(True)
        if len(calls)>1:x.f.f.owner['command'][2]='artifacts/relative_provider.py'
        return original()
    monkeypatch.setattr(x.a,'outer_command_guard',changed_under_fence)
    with pytest.raises(ValueError,match='full absolute literal Python'):x.run()
    assert len(calls)==2 and not x.state.exists() and not x.f.f.held
    assert not any(k in ('freeze','bind') for k,d in x.calls)
