"""Scratch first fits/freeze and native registration; no production actions."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level4_first import fixture as first_fixture, write, change
from test_modebench_scale_level4_passed_freeze import fixture as freeze_fixture
ROOT=Path(__file__).resolve().parents[1]


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


@pytest.fixture
def fixture(freeze_fixture,monkeypatch):
    z=freeze_fixture;z.execute();f=z.f
    spec=importlib.util.spec_from_file_location('scratch_python_preparation',ROOT/'artifacts/register_modebench_scale_level4_python_r3_20260912.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    tmp=f.state.parent;state=tmp/'preparation';revision_root=tmp/'python_r3';parent=tmp/'parent'
    entries=f.a.read(f.state/'registration.json')['revisions']
    for key,value in {'_first':f.a,'FIT_STATE':f.state,'STATE':state,'REVISION_ROOT':revision_root,'PARENT':parent,
        'MANIFEST':f.a.MANIFEST,'LOCK':f.a.LOCK,'LOCK_INODE':f.a.LOCK_INODE,
        'REVIEW':tmp/'preparation_review.json','CANDIDATE':tmp/'candidate.py','CANDIDATE_TESTS':tmp/'candidate_tests.py',
        'CANDIDATE_REVIEW':tmp/'candidate_review.json','assert_fence':f.a.assert_fence,'lifetime_fence':f.a.lifetime_fence,
        'REGISTRATION_SHA':a.sha(f.state/'registration.json'),'FIT_RESULT_SHA':a.sha(f.state/'actions/fit/result.json'),
        'RECIPES':{e['domain']:a.sha(e['recipe_path']) for e in entries},
        'FREEZE':z.a.SOURCE,'FREEZE_TESTS':z.a.TESTS,'FREEZE_REVIEW':z.a.REVIEW,
        'FREEZE_CERTIFICATE':z.a.area(f.state)/'certificate.json','FREEZE_CERTIFICATE_SHA':a.sha(z.a.area(f.state)/'certificate.json'),
        'FREEZE_RUNTIME':z.a.area(f.state)/'action/runtime.json','FREEZE_RUNTIME_SHA':a.sha(z.a.area(f.state)/'action/runtime.json'),
        'HISTORICAL_SPIN_DEVICE':a.read(z.a.area(f.state)/'action/runtime.json')['lock_device'],
        'L4_MANIFEST':tmp/'complete_level4_sources.json','QUALIFICATION':tmp/'qualification.json',
        'QUALIFICATION_EXECUTION':tmp/'qualification_execution.json'}.items():monkeypatch.setattr(a,key,value)
    a.CANDIDATE.write_text('four fixed candidate laws');a.CANDIDATE_TESTS.write_text('structural tests')
    monkeypatch.setattr(a,'CANDIDATE_SHA',a.sha(a.CANDIDATE))
    write(a.QUALIFICATION,{'candidate_sha256':a.CANDIDATE_SHA,'rows':8,'native_original_witnesses':16,'target_registration_performed':False})
    write(a.QUALIFICATION_EXECUTION,{'returncode':0,'candidate_sha256_before':a.CANDIDATE_SHA,'candidate_sha256_after':a.CANDIDATE_SHA})
    monkeypatch.setattr(a,'QUALIFICATION_SHA',a.sha(a.QUALIFICATION));monkeypatch.setattr(a,'QUALIFICATION_EXECUTION_SHA',a.sha(a.QUALIFICATION_EXECUTION))
    sources={e['domain']:{'source_kind':'domain_revision_v1','source_root':e['source_root']} for e in entries}
    for domain in ('countdown','mathir'):
        root=tmp/('carried_'+domain);dataset=z.paths(root,'level4',domain)['dataset']
        write(dataset/'identity.json',{'splits':{s:{'rows':n} for s,n in [('train',384),('dev',128),('eval',128)]}})
        sources[domain]={'source_kind':'campaign_v1','source_root':str(root)}
    write(a.L4_MANIFEST,{'level':'level4','model_label':'7b','sources':sources});monkeypatch.setattr(a,'L4_MANIFEST_SHA',a.sha(a.L4_MANIFEST))
    retained=[]
    def frozen(root,level,domain,kind):
        retained.append(domain);assert level=='level4'
        return a.read(z.paths(root,level,domain)['dataset']/'identity.json')
    retained_core=SimpleNamespace(domain_paths=z.paths,frozen_identity=frozen)
    profiles=[{'fixed_case_law':t} for t in range(4)]
    parent_value={'models':{'7b':{'path':'literal checkpoint'}},'split_sizes':{'train':384,'dev':128,'eval':128},
        'targets':{'python_factors':{'metrics':{'pass_at_1':.3}},'mathir':{}},
        'histograms':{'python_factors':{s:[{'cell':[4],'rows':2}] for s in ('train','dev','eval')},'mathir':{}},
        'tolerances':{'pass_at_1':.04},'selection_seed':6491701,'sampling':{'unchanged':True},'fit':{'fixed_grid':True}}
    write(parent/'protocol.json',parent_value)
    candidate_dep=tmp/'candidate_dependency.py';candidate_dep.write_text('fixed dependency')
    cpins={str(p):a.sha(p) for p in (a.CANDIDATE,a.CANDIDATE_TESTS,candidate_dep)}
    write(a.CANDIDATE_REVIEW,{'schema':'modebench_scale_python_harder_candidates_independent_review_v1','status':'reviewed','files_sha256':cpins})
    pins=f.a.read(f.a.REVIEW)['files_sha256'];pins.update(cpins);pins.update(z.a.read(z.a.REVIEW)['files_sha256'])
    pins.update({str(p):a.sha(p) for p in (a.SOURCE,a.TESTS,a.FIRST,f.a.TESTS,f.a.REVIEW,a.CANDIDATE_REVIEW,a.FREEZE,a.FREEZE_TESTS,a.FREEZE_REVIEW,a.FREEZE_CERTIFICATE,a.FREEZE_RUNTIME,a.QUALIFICATION,a.QUALIFICATION_EXECUTION,*f.a.SEALED)})
    write(a.REVIEW,{'schema':'modebench_scale_level4_python_r3_registration_independent_review_v1','status':'reviewed','files_sha256':pins})
    f.owner['command']=[str(a.PYTHON),'-B',str(a.SOURCE),'prepare'];active=[];calls=[];errors=set();authentications=[]
    monkeypatch.setattr(a.os,'uname',lambda:SimpleNamespace(nodename=a.HOST))
    def identity(pid):
        if active and pid!=f.owner['pid']:
            argv=active[0][active[0].index('--')+1:];argv.insert(1,'-B')
            return {**deepcopy(f.owner),'pid':12346,'start_ticks':'5679','command':argv}
        return deepcopy(f.owner)
    monkeypatch.setattr(a,'process_identity',identity)
    def host_guard(*,guest):
        a.require(a.os.uname().nodename==a.HOST,'truthful host required')
        a.require(f.canonical.read_text()==('old' if guest else 'neutral'),'correct source view required')
    monkeypatch.setattr(f.a,'host_guard',host_guard)
    def paths(root):
        return {'protocol':root/'protocol.json','pools':root/'level4/pools/python_factors',
            'development':root/'level4/results/development/python_factors'}
    def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
    def generation_seed(p,split,tier):return 9000+tier
    def register(root,level,domain,**kwargs):
        calls.append('register');assert f.held==[8] and f.canonical.read_text()=='old'
        assert root==revision_root and level=='level4' and domain=='python_factors'
        assert kwargs=={'parent':parent,'revision':3,'candidate_module':a.CANDIDATE_MODULE}
        assert a.read(state/'amendment.json')['actual_r2_predecessor']==entries[2]
        if 'register_before' in errors:raise RuntimeError('native register exception')
        value={k:deepcopy(v) for k,v in parent_value.items() if k not in ('targets','histograms')}
        value.update(schema='native_revision_schema',root=str(root),parent_root=str(parent),level=level,domain=domain,revision=3,
            parent_protocol_sha256=a.sha(parent/'protocol.json'),candidate_module=a.CANDIDATE_MODULE,candidate_profiles=profiles,
            draw_labels=deepcopy(a.LABELS),files_sha256={str(f.source):a.sha(f.source),str(a.CANDIDATE):a.sha(a.CANDIDATE)})
        for k in ('targets','histograms'):value[k]={'python_factors':deepcopy(parent_value[k]['python_factors'])}
        write(root/'protocol.json',value);write(root/'protocol.sha256.json',{'sha256':a.sha(root/'protocol.json')})
        if 'register_after' in errors:raise RuntimeError('native register exception')
        return value
    def pools(root):
        calls.append('materialize_pools');assert f.held==[8]
        if 'pools_before' in errors:raise RuntimeError('native pools exception')
        q=paths(root);snapshot=root/'exclusions/development.json'
        write(snapshot,{'schema':'modebench_scale_revision_exclusions_v1','stage':'development','protocol_sha256':a.sha(q['protocol']),
            'files_sha256':{str(f.source):a.sha(f.source)}})
        tiers={}
        for tier in range(4):
            batch=[{'scale_candidate_tier':tier,'answer_mode_count':4,'case':i} for i in range(2)]
            path=q['pools']/f'difficulty_{tier}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('\n'.join(json.dumps(row) for row in batch)+'\n')
            tiers[str(tier)]={'rows':2,'rows_sha256':object_sha(batch),'generation_seed':9000+tier,
                'semantic_disjoint':True,'prompt_disjoint':True,'verification':{'rows_verified':2,'exact_canonical_support':True,'unchanged_original_prompt':True}}
        value={'schema':'modebench_scale_revision_pools_v1','level':'level4','domain':'python_factors','protocol_sha256':a.sha(q['protocol']),
            'exclusions':str(snapshot),'exclusions_sha256':a.sha(snapshot),'tiers':tiers}
        write(q['pools']/'identity.json',value)
        if 'pools_after' in errors:raise RuntimeError('native pools exception')
        return value
    def launch(root,phase):
        calls.append('launch_inputs');assert phase=='dev' and f.held==[8]
        if 'tasks' in errors:raise RuntimeError('native task exception')
        q=paths(root);tasks=[{'level':'level4','domain':'python_factors','split':'dev','interface':'native interface',
            'rows_jsonl':str(q['pools']/f'difficulty_{t}.jsonl'),'seeds':a.LABELS['dev'],'batch_size':8,'row_offset':0,'row_limit':0,
            'output':str(q['development']/f'difficulty_{t}.json')} for t in range(4)]
        return {'id':'level4_python_factors_r3_dev','level':'level4','domain':'python_factors','phase':'dev','model_label':'7b',
            'source_kind':'domain_revision_v1','source_root':str(root),'tasks':tasks},{str(f.source):a.sha(f.source)}
    native=SimpleNamespace(SCHEMA='native_revision_schema',candidate_provider=lambda name:SimpleNamespace(PROFILES={'python_factors':profiles}),
        register=register,materialize_pools=pools,launch_inputs=launch,paths=paths,rows_from_jsonl=rows,sha=object_sha,generation_seed=generation_seed,
        original=SimpleNamespace(deserialize_cells=lambda h:Counter({tuple(v['cell']):v['rows'] for v in h}),union_histogram=lambda h:next(iter(h.values()))),
        cell_histogram=lambda domain,rs:Counter({(4,):len(rs)}),evaluator=SimpleNamespace(INTERFACE='native interface'))
    def authenticate(path):assert f.canonical.read_text()=='neutral';authentications.append(path)
    def load(path,name):
        if path==f.a.REVISION:return native
        if path==a.FREEZE:return z.a
        if path==f.a.CORE:return retained_core
        return SimpleNamespace(authenticate=authenticate)
    monkeypatch.setattr(a,'load_module',load)
    dispatch={'code':0}
    def run(argv,**kwargs):
        assert kwargs['pass_fds']==(8,) and f.held==[8];active.append(argv);f.canonical.write_text('old')
        try:a.guest_action(state,8,a.sha(a.REVIEW))
        finally:f.canonical.write_text('neutral');active.clear()
        return SimpleNamespace(returncode=dispatch['code'])
    monkeypatch.setattr(a,'subprocess',SimpleNamespace(run=run))
    return SimpleNamespace(a=a,f=f,state=state,entries=entries,calls=calls,errors=errors,dispatch=dispatch,paths=paths,native=native,
        candidate_dep=candidate_dep,authentications=authentications,freeze=z,retained=retained,execute=lambda:a.run(state,a.sha(a.REVIEW)))



def test_only_register_four_pools_and_tasks_preserve_all_passing_sources_and_old_targets(fixture):
    x=fixture;before={e['domain']:x.a.sha(e['recipe_path']) for e in x.entries};v=x.execute()
    assert x.calls==['register','materialize_pools','launch_inputs']
    assert v['status']=='ready_for_fresh_python_r3_development'
    assert set(v['preserved_level4_sources'])=={'countdown','graph_coloring','mathir','pantry'}
    assert not v['source_binding_performed'] and not v['new_fit_decisions'] and not v['heldout_performed']
    assert x.freeze.calls==['graph_coloring','pantry'] and x.f.calls==['graph_coloring','pantry','python_factors']
    assert {e['domain']:x.a.sha(e['recipe_path']) for e in x.entries}==before
    p=x.a.read(x.a.REVISION_ROOT/'protocol.json');parent=x.a.read(x.a.PARENT/'protocol.json')
    assert p['targets']['python_factors']==parent['targets']['python_factors']
    assert p['draw_labels']=={'dev':[7203000,7203001,7203002,7203003],'eval':[7203500,7203501,7203502,7203503]}
    assert not (x.state/'source_binding.json').exists()


def test_readonly_provider_has_exact7b_inputs_and_accepts_future_receipt_growth(fixture,monkeypatch):
    x=fixture;x.execute();before=list(x.calls)
    class ForeignLock:
        def stat(self):raise AssertionError('historical proof cannot use worker NFS device')
    monkeypatch.setattr(x.a,'LOCK',ForeignLock());monkeypatch.setattr(x.freeze.a,'LOCK',ForeignLock())
    value=x.a.development_inputs(x.state,guest=False)
    assert value['models']=={'7b':'literal checkpoint'} and value['cell']['id']=='level4_python_factors_r3_dev'
    assert value['readiness_sha256']==x.a.sha(x.state/'certificate.json')
    task=value['cell']['tasks'][0];write(task['output'],{'grows':True});write(Path(task['output']+'.batches')/'batch.json',{})
    assert x.a.development_inputs(x.state,guest=False)==value and x.calls==before
    assert x.freeze.calls==['graph_coloring','pantry'] and x.f.calls==['graph_coloring','pantry','python_factors']


@pytest.mark.parametrize('phase',['register_before','register_after','pools_before','pools_after','tasks'])
def test_partial_native_action_saved_without_retry(fixture,phase):
    x=fixture;x.errors.add(phase)
    with pytest.raises(RuntimeError):x.execute()
    assert (x.state/'amendment.json').is_file() and (x.state/'action/failure.json').is_file()
    before=list(x.calls)
    with pytest.raises(ValueError,match='already attempted'):x.execute()
    assert x.calls==before


@pytest.mark.parametrize('code',[1,143,-15])
def test_nonzero_wrapper_preserves_certificate_without_waiver(fixture,code):
    x=fixture;x.dispatch['code']=code
    with pytest.raises(ValueError,match='explicit reconciliation'):x.execute()
    assert (x.state/'certificate.json').is_file() and x.a.read(x.state/'action/exit.json')['returncode']==code
    with pytest.raises(ValueError):x.a.verify(x.state,guest=False)


@pytest.mark.parametrize('kind',['registration','fit','freeze_certificate','freeze_runtime','failed_recipe','candidate','qualification','qualification_execution',
    'first_failed','freeze_failed','retained_dataset','retained_source','wrong_host','missing_review_pin','candidate_review'])
def test_exact_prior_evidence_retained_sources_and_qualification_required(fixture,monkeypatch,kind):
    x=fixture;a=x.a
    if kind=='registration':monkeypatch.setattr(a,'REGISTRATION_SHA','wrong')
    elif kind=='fit':monkeypatch.setattr(a,'FIT_RESULT_SHA','wrong')
    elif kind=='freeze_certificate':monkeypatch.setattr(a,'FREEZE_CERTIFICATE_SHA','wrong')
    elif kind=='freeze_runtime':monkeypatch.setattr(a,'FREEZE_RUNTIME_SHA','wrong')
    elif kind=='failed_recipe':change(x.entries[2]['recipe_path'],lambda v:v.update(development_fit_pass=True))
    elif kind=='candidate':a.CANDIDATE.write_text('changed')
    elif kind=='qualification':change(a.QUALIFICATION,lambda v:v.update(rows=0))
    elif kind=='qualification_execution':change(a.QUALIFICATION_EXECUTION,lambda v:v.update(returncode=143))
    elif kind=='first_failed':write(x.f.state/'actions/fit/failure.json',{})
    elif kind=='freeze_failed':write(a.FREEZE_RUNTIME.parent/'failure.json',{})
    elif kind=='retained_dataset':(x.freeze.paths(Path(x.entries[0]['source_root']),'level4','graph_coloring')['dataset']/'identity.json').unlink()
    elif kind=='retained_source':change(a.L4_MANIFEST,lambda v:v['sources']['mathir'].update(source_root='wrong'))
    elif kind=='wrong_host':monkeypatch.setattr(a.os,'uname',lambda:SimpleNamespace(nodename='wash.cs.princeton.edu'))
    elif kind=='missing_review_pin':change(a.REVIEW,lambda v:v['files_sha256'].pop(str(a.CANDIDATE_TESTS)))
    else:change(a.CANDIDATE_REVIEW,lambda v:v.update(status='draft'))
    with pytest.raises((ValueError,FileNotFoundError)):x.execute()
    assert not x.calls


@pytest.mark.parametrize('field',['targets','tolerances','sampling','draw_labels','candidate_module','candidate_profiles','revision','domain'])
def test_registered_native_policy_cannot_change_historical_targets_or_other_settings(fixture,monkeypatch,field):
    x=fixture;register=x.native.register
    def changed(*args,**kwargs):
        p=register(*args,**kwargs);p[field]='wrong';write(x.a.REVISION_ROOT/'protocol.json',p)
        write(x.a.REVISION_ROOT/'protocol.sha256.json',{'sha256':x.a.sha(x.a.REVISION_ROOT/'protocol.json')});return p
    monkeypatch.setattr(x.native,'register',changed)
    with pytest.raises(ValueError):x.execute()
    assert x.calls==['register']


@pytest.mark.parametrize('kind',['missing_tier','seed','rows','verification','task_labels','task_range'])
def test_all_four_full_native_tiers_and_tasks_required(fixture,monkeypatch,kind):
    x=fixture
    if kind.startswith('task'):
        launch=x.native.launch_inputs
        def changed(*args):
            cell,pins=launch(*args);cell['tasks'][0]['seeds' if kind=='task_labels' else 'row_limit']=[1] if kind=='task_labels' else 1
            return cell,pins
        monkeypatch.setattr(x.native,'launch_inputs',changed)
    else:
        pools=x.native.materialize_pools
        def changed(root):
            v=pools(root)
            if kind=='missing_tier':v['tiers'].pop('3')
            elif kind=='verification':v['tiers']['0']['verification']['exact_canonical_support']=False
            else:v['tiers']['0']['generation_seed' if kind=='seed' else 'rows']=0
            write(x.paths(root)['pools']/'identity.json',v);return v
        monkeypatch.setattr(x.native,'materialize_pools',changed)
    with pytest.raises(ValueError):x.execute()
    assert not (x.state/'certificate.json').exists()


@pytest.mark.parametrize('kind',['runtime','guest_argv','missing_exit','future_pin','retained','targets'])
def test_static_preparation_proof_rejects_tampering_without_native_calls(fixture,kind):
    x=fixture;x.execute();a=x.a
    if kind=='runtime':change(x.state/'action/runtime.json',lambda v:v.update(lock_device=0))
    elif kind=='guest_argv':change(x.state/'action/guest_runtime.json',lambda v:v['command'].remove('-B'))
    elif kind=='missing_exit':(x.state/'action/exit.json').unlink()
    elif kind=='targets':change(a.REVISION_ROOT/'protocol.json',lambda v:v.update(targets={'changed':True}))
    else:
        v=a.read(x.state/'certificate.json')
        if kind=='retained':v['preserved_level4_sources'].pop('mathir')
        else:
            task=a.read(x.state/'development_tasks.json')[0];write(task['output'],{});v['files_sha256'][task['output']]=a.sha(task['output'])
        write(x.state/'certificate.json',v);write(x.state/'action/result.json',v)
        write(x.state/'certificate.sha256.json',{'sha256':a.sha(x.state/'certificate.json')})
    with pytest.raises((ValueError,FileNotFoundError)):a.development_inputs(x.state,guest=False)
    assert x.calls==['register','materialize_pools','launch_inputs']


def test_source_exposes_no_binding_freezing_fit_or_scheduler_action():
    path=ROOT/'artifacts/register_modebench_scale_level4_python_r3_20260912.py';tree=ast.parse(path.read_text())
    attrs={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not attrs & {'bind_level','freeze_dataset','fit_domain','submit','audit'}
    source=path.read_text();assert 'level4_python_factors_r3' in source and 'level4_python_coloring_r3' not in source


@pytest.mark.parametrize('kind',['owner_start','owner_command','dead_owner','fence','manifest','source','host'])
def test_live_owner_and_fence_guard_before_any_new_native_operation(fixture,kind):
    x=fixture;a=x.a;f=x.f
    runtime={**deepcopy(f.owner),'host':a.HOST,'fence_fd':8,'lock_inode':a.LOCK_INODE,
        'lock_device':a.LOCK.stat().st_dev,'source_sha256':a.sha(a.SOURCE),'view_manifest_sha256':a.sha(a.MANIFEST)}
    if kind=='owner_start':runtime['start_ticks']='wrong'
    elif kind=='owner_command':runtime['command']=['wrong']
    elif kind=='dead_owner':f.owner['state']='Z'
    elif kind=='fence':runtime['fence_fd']=9
    elif kind=='manifest':runtime['view_manifest_sha256']='wrong'
    elif kind=='source':runtime['source_sha256']='wrong'
    else:runtime['host']='wash.cs.princeton.edu'
    write(x.state/'action/runtime.json',runtime);f.held.append(8);f.canonical.write_text('old')
    try:
        with pytest.raises(ValueError):a.guard(x.state,8)
    finally:f.held.clear();f.canonical.write_text('neutral')
    assert not x.calls


def test_lost_fence_blocks_native_preparation_after_host_only_authentication(fixture,monkeypatch):
    x=fixture
    def lost(fd):raise ValueError('lost fence')
    monkeypatch.setattr(x.a,'assert_fence',lost)
    with pytest.raises(ValueError,match='lost fence'):x.execute()
    assert x.authentications==[x.a.MANIFEST] and not x.calls
