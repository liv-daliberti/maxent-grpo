"""Scratch-only native API simulation; no real pools, certifier, model or fit."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_ready_fits import fixture as fit_fixture, write

ROOT=Path(__file__).resolve().parents[1]


def change(path,update):
    value=json.loads(Path(path).read_text());update(value);write(path,value)


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


@pytest.fixture
def fixture(fit_fixture,monkeypatch):
    f=fit_fixture;f.failures.add('graph_coloring');f.run()
    spec=importlib.util.spec_from_file_location('scratch_graph_preparation',ROOT/'artifacts/register_modebench_scale_level5_graph_r3_20260912.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    tmp=f.state.parent;state=tmp/'preparation';revision_root=tmp/'graph_r3';parent=tmp/'parent'
    entries=f.a.read(f.state/'registration.json')['revisions']
    for key,value in {'_first':f.a,'FIT_STATE':f.state,'STATE':state,'REVISION_ROOT':revision_root,'PARENT':parent,
        'MANIFEST':f.a.VIEW,'LOCK':f.a.first.LOCK,'LOCK_INODE':f.a.first.LOCK_INODE,
        'REVIEW':tmp/'preparation_review.json','CANDIDATE':tmp/'candidate.py','CANDIDATE_TESTS':tmp/'candidate_tests.py',
        'CANDIDATE_REVIEW':tmp/'candidate_review.json','assert_fence':f.a.assert_fence,'lifetime_fence':f.a.lifetime_fence,
        'REGISTRATION_SHA':a.sha(f.state/'registration.json'),'FIT_RESULT_SHA':a.sha(f.state/'action/result.json'),
        'RECIPES':{e['domain']:a.sha(e['recipe_path']) for e in entries}}.items():monkeypatch.setattr(a,key,value)
    a.CANDIDATE.write_text('four fixed candidate laws');a.CANDIDATE_TESTS.write_text('structural tests')
    profiles=[{'fixed_topology':t} for t in range(4)]
    parent_value={'models':{'14b':{'path':'literal checkpoint'}},'split_sizes':{'train':384,'dev':128,'eval':128},
        'targets':{'graph_coloring':{'metrics':{'pass_at_1':.3}},'mathir':{}},
        'histograms':{'graph_coloring':{s:[{'cell':[4],'rows':2}] for s in ('train','dev','eval')},'mathir':{}},
        'tolerances':{'pass_at_1':.04},'selection_seed':6491701,'sampling':{'unchanged':True},'fit':{'fixed_grid':True}}
    write(parent/'protocol.json',parent_value)
    candidate_dep=tmp/'candidate_dependency.py';candidate_dep.write_text('fixed dependency')
    cpins={str(p):a.sha(p) for p in (a.CANDIDATE,a.CANDIDATE_TESTS,candidate_dep)}
    write(a.CANDIDATE_REVIEW,{'schema':'modebench_scale_graph_r3_candidates_independent_review_v1','status':'reviewed','files_sha256':cpins})
    pins=f.a.read(f.a.REVIEW)['files_sha256'];pins.update(cpins)
    pins.update({str(p):a.sha(p) for p in (a.SOURCE,a.TESTS,a.FIRST,f.a.TESTS,f.a.REVIEW,a.CANDIDATE_REVIEW,*f.a.SEALED)})
    write(a.REVIEW,{'schema':'modebench_scale_level5_graph_r3_registration_independent_review_v1','status':'reviewed','files_sha256':pins})
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
    monkeypatch.setattr(f.a.first,'host_guard',host_guard)
    def paths(root):
        return {'protocol':root/'protocol.json','pools':root/'level5/pools/graph_coloring',
            'development':root/'level5/results/development/graph_coloring'}
    def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
    def generation_seed(p,split,tier):return 8000+tier
    def register(root,level,domain,**kwargs):
        calls.append('register');assert f.held==[8] and f.canonical.read_text()=='old'
        assert root==revision_root and level=='level5' and domain=='graph_coloring'
        assert kwargs=={'parent':parent,'revision':3,'candidate_module':a.CANDIDATE_MODULE}
        assert a.read(state/'amendment.json')['actual_r2_predecessor']==entries[0]
        if 'register_before' in errors:raise RuntimeError('native register exception')
        value={k:deepcopy(v) for k,v in parent_value.items() if k not in ('targets','histograms')}
        value.update(schema='native_revision_schema',root=str(root),parent_root=str(parent),level=level,domain=domain,revision=3,
            parent_protocol_sha256=a.sha(parent/'protocol.json'),candidate_module=a.CANDIDATE_MODULE,candidate_profiles=profiles,
            draw_labels=deepcopy(a.LABELS),files_sha256={str(f.source):a.sha(f.source),str(a.CANDIDATE):a.sha(a.CANDIDATE)})
        for k in ('targets','histograms'):value[k]={'graph_coloring':deepcopy(parent_value[k]['graph_coloring'])}
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
            tiers[str(tier)]={'rows':2,'rows_sha256':object_sha(batch),'generation_seed':8000+tier,
                'semantic_disjoint':True,'prompt_disjoint':True,'verification':{'rows_verified':2,'exact_canonical_support':True,'unchanged_original_prompt':True}}
        value={'schema':'modebench_scale_revision_pools_v1','level':'level5','domain':'graph_coloring','protocol_sha256':a.sha(q['protocol']),
            'exclusions':str(snapshot),'exclusions_sha256':a.sha(snapshot),'tiers':tiers}
        write(q['pools']/'identity.json',value)
        if 'pools_after' in errors:raise RuntimeError('native pools exception')
        return value
    def launch(root,phase):
        calls.append('launch_inputs');assert phase=='dev' and f.held==[8]
        if 'tasks' in errors:raise RuntimeError('native task exception')
        q=paths(root);tasks=[{'level':'level5','domain':'graph_coloring','split':'dev','interface':'native interface',
            'rows_jsonl':str(q['pools']/f'difficulty_{t}.jsonl'),'seeds':a.LABELS['dev'],'batch_size':8,'row_offset':0,'row_limit':0,
            'output':str(q['development']/f'difficulty_{t}.json')} for t in range(4)]
        return {'id':'level5_graph_coloring_r3_dev','level':'level5','domain':'graph_coloring','phase':'dev','model_label':'14b',
            'source_kind':'domain_revision_v1','source_root':str(root),'tasks':tasks},{str(f.source):a.sha(f.source)}
    native=SimpleNamespace(SCHEMA='native_revision_schema',candidate_provider=lambda name:SimpleNamespace(PROFILES={'graph_coloring':profiles}),
        register=register,materialize_pools=pools,launch_inputs=launch,paths=paths,rows_from_jsonl=rows,sha=object_sha,generation_seed=generation_seed,
        original=SimpleNamespace(deserialize_cells=lambda h:Counter({tuple(v['cell']):v['rows'] for v in h}),union_histogram=lambda h:next(iter(h.values()))),
        cell_histogram=lambda domain,rs:Counter({(4,):len(rs)}),evaluator=SimpleNamespace(INTERFACE='native interface'))
    def authenticate(path):assert f.canonical.read_text()=='neutral';authentications.append(path)
    monkeypatch.setattr(a,'load_module',lambda path,name:native if path==f.a.REVISION else SimpleNamespace(authenticate=authenticate))
    dispatch={'code':0}
    def run(argv,**kwargs):
        assert kwargs['pass_fds']==(8,) and f.held==[8];active.append(argv);f.canonical.write_text('old')
        try:a.guest_action(state,8,a.sha(a.REVIEW))
        finally:f.canonical.write_text('neutral');active.clear()
        return SimpleNamespace(returncode=dispatch['code'])
    monkeypatch.setattr(a,'subprocess',SimpleNamespace(run=run))
    return SimpleNamespace(a=a,f=f,state=state,entries=entries,calls=calls,errors=errors,dispatch=dispatch,paths=paths,native=native,
        candidate_dep=candidate_dep,authentications=authentications,execute=lambda:a.run(state,a.sha(a.REVIEW)))


def test_native_registration_four_full_pools_and_exact_development_tasks_only(fixture):
    x=fixture;before={d:x.a.sha(e['recipe_path']) for d,e in zip(('graph_coloring','mathir'),x.entries)};v=x.execute()
    assert x.calls==['register','materialize_pools','launch_inputs'] and v['status']=='ready_for_fresh_graph_r3_development'
    assert v['native_parent_recipe_reproducibility'] and v['native_row_certification'] and not v['new_fit_decisions']
    assert not v['source_binding_performed'] and not v['submission_performed'] and not v['heldout_performed']
    assert {d:x.a.sha(e['recipe_path']) for d,e in zip(('graph_coloring','mathir'),x.entries)}==before
    tasks=x.a.read(x.state/'development_tasks.json');assert len(tasks)==4
    assert all(t['seeds']==[8103000,8103001,8103002,8103003] and t['row_offset']==t['row_limit']==0 for t in tasks)
    assert not x.f.held and len(x.f.calls)==2


def test_static_verify_never_generates_or_fits_and_allows_future_receipts_to_grow(fixture):
    x=fixture;value=x.execute();before=list(x.calls)
    task=x.a.read(x.state/'development_tasks.json')[0];write(task['output'],{'future_output':True})
    assert x.a.verify(x.state,guest=False)==value and x.calls==before and len(x.f.calls)==2


def test_duplicate_preparation_never_repeats_generation(fixture):
    x=fixture;x.execute()
    with pytest.raises(ValueError,match='already attempted'):x.execute()
    assert x.calls==['register','materialize_pools','launch_inputs']


@pytest.mark.parametrize('phase',['register_before','register_after','pools_before','pools_after','tasks'])
def test_partial_native_failure_preserves_claim_outputs_and_refuses_retry(fixture,phase):
    x=fixture;x.errors.add(phase)
    with pytest.raises(RuntimeError,match='native'):x.execute()
    assert (x.state/'amendment.json').is_file() and (x.state/'action/failure.json').is_file()
    assert not (x.state/'certificate.json').exists()
    before=list(x.calls)
    with pytest.raises(ValueError,match='already attempted'):x.execute()
    assert x.calls==before


@pytest.mark.parametrize('code',[1,143,-15])
def test_actual_nonzero_wrapper_preserves_completed_certificate_without_waiver(fixture,code):
    x=fixture;x.dispatch['code']=code
    with pytest.raises(ValueError,match='explicit reconciliation'):x.execute()
    assert x.a.read(x.state/'action/exit.json')['returncode']==code and (x.state/'certificate.json').is_file()
    assert x.a.read(x.state/'action/failure.json')['files_sha256'][str(x.state/'certificate.json')]==x.a.sha(x.state/'certificate.json')
    with pytest.raises(ValueError,match='reconciliation'):x.a.verify(x.state,guest=False)


@pytest.mark.parametrize('kind',['registration_sha','fit_sha','failed_fit_action','graph_recipe','candidate_changed','candidate_review','candidate_dependency',
    'missing_review_pin','review_cycle','existing_revision','wrong_host','wrong_scope'])
def test_preconditions_block_native_registration(fixture,monkeypatch,kind):
    x=fixture;a=x.a
    if kind=='registration_sha':monkeypatch.setattr(a,'REGISTRATION_SHA','wrong')
    elif kind=='fit_sha':monkeypatch.setattr(a,'FIT_RESULT_SHA','wrong')
    elif kind=='failed_fit_action':write(x.f.state/'action/failure.json',{'failed':True})
    elif kind=='graph_recipe':change(x.entries[0]['recipe_path'],lambda v:v.update(development_fit_pass=True))
    elif kind=='candidate_changed':a.CANDIDATE.write_text('changed')
    elif kind=='candidate_review':change(a.CANDIDATE_REVIEW,lambda v:v.update(status='draft'))
    elif kind=='candidate_dependency':x.candidate_dep.write_text('changed')
    elif kind=='missing_review_pin':change(a.REVIEW,lambda v:v['files_sha256'].pop(str(a.CANDIDATE_TESTS)))
    elif kind=='review_cycle':change(a.REVIEW,lambda v:v['files_sha256'].update({str(a.REVIEW):'x'}))
    elif kind=='existing_revision':a.REVISION_ROOT.mkdir()
    elif kind=='wrong_host':monkeypatch.setattr(a.os,'uname',lambda:SimpleNamespace(nodename='wash.cs.princeton.edu'))
    else:
        value=x.f.a.verify_existing(x.f.state);value['failed_domains']=['mathir']
        monkeypatch.setattr(x.f.a,'verify_existing',lambda *args,**kwargs:value)
    with pytest.raises((ValueError,FileNotFoundError)):x.execute()
    assert not x.calls


@pytest.mark.parametrize('field',['revision','domain','draw_labels','targets','candidate_module','candidate_profiles'])
def test_native_protocol_drift_refuses_generation(fixture,monkeypatch,field):
    x=fixture;register=x.native.register
    def changed(*args,**kwargs):
        value=register(*args,**kwargs);value[field]='wrong';write(x.a.REVISION_ROOT/'protocol.json',value)
        write(x.a.REVISION_ROOT/'protocol.sha256.json',{'sha256':x.a.sha(x.a.REVISION_ROOT/'protocol.json')});return value
    monkeypatch.setattr(x.native,'register',changed)
    with pytest.raises(ValueError):x.execute()
    assert x.calls==['register']


@pytest.mark.parametrize('kind',['missing_tier','rows','seed','certification','task_range','task_seed'])
def test_incomplete_pools_or_changed_native_tasks_refuse_readiness(fixture,monkeypatch,kind):
    x=fixture
    if kind.startswith('task'):
        launch=x.native.launch_inputs
        def changed(*args):
            cell,pins=launch(*args)
            cell['tasks'][0]['row_limit' if kind=='task_range' else 'seeds']=1 if kind=='task_range' else [1]
            return cell,pins
        monkeypatch.setattr(x.native,'launch_inputs',changed)
    else:
        pools=x.native.materialize_pools
        def changed(root):
            value=pools(root)
            if kind=='missing_tier':value['tiers'].pop('3')
            elif kind=='rows':value['tiers']['0']['rows']=1
            elif kind=='seed':value['tiers']['0']['generation_seed']=1
            else:value['tiers']['0']['verification']['exact_canonical_support']=False
            write(x.paths(root)['pools']/'identity.json',value);return value
        monkeypatch.setattr(x.native,'materialize_pools',changed)
    with pytest.raises(ValueError):x.execute()
    assert not (x.state/'certificate.json').exists()


@pytest.mark.parametrize('kind',['owner_start','owner_command','dead_owner','fence','manifest','source','host'])
def test_live_owner_and_inherited_fence_guard_before_registration(fixture,kind):
    x=fixture;a=x.a;f=x.f;runtime={**deepcopy(f.owner),'host':a.HOST,'fence_fd':8,'lock_inode':a.LOCK_INODE,
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


@pytest.mark.parametrize('kind',['tasks','pool','amendment','missing_pin','certificate','missing_exit','guest_argv'])
def test_static_reconciliation_refuses_tampered_or_incomplete_preparation(fixture,kind):
    x=fixture;x.execute();a=x.a
    if kind=='tasks':change(x.state/'development_tasks.json',lambda v:v.pop())
    elif kind=='pool':(x.paths(a.REVISION_ROOT)['pools']/'difficulty_0.jsonl').write_text('{}\n')
    elif kind=='amendment':change(x.state/'amendment.json',lambda v:v.update(revision=2))
    elif kind in ('missing_pin','certificate'):
        value=a.read(x.state/'certificate.json')
        if kind=='missing_pin':value['files_sha256'].pop(str(x.state/'development_tasks.json'))
        else:value['source_binding_performed']=True
        write(x.state/'certificate.json',value);write(x.state/'action/result.json',value)
        write(x.state/'certificate.sha256.json',{'sha256':a.sha(x.state/'certificate.json')})
    elif kind=='missing_exit':(x.state/'action/exit.json').unlink()
    else:change(x.state/'action/guest_runtime.json',lambda v:v['command'].remove('-B'))
    with pytest.raises((ValueError,FileNotFoundError)):a.verify(x.state,guest=False)
    assert x.calls==['register','materialize_pools','launch_inputs']


def test_lost_fence_blocks_science_after_only_host_authentication(fixture,monkeypatch):
    x=fixture
    def lost(fd):raise ValueError('lost fence')
    monkeypatch.setattr(x.a,'assert_fence',lost)
    with pytest.raises(ValueError,match='lost fence'):x.execute()
    assert x.authentications==[x.a.MANIFEST] and not x.calls


def test_development_provider_is_stable_after_new_receipts_and_pins_original_cell(fixture):
    x=fixture;x.execute();before=list(x.calls);first=x.a.development_inputs(x.state,guest=False)
    task=first['cell']['tasks'][0];write(task['output'],{'model_output_grows':True})
    write(Path(task['output']+'.batches')/'batch0.json',{'partial':True})
    assert x.a.development_inputs(x.state,guest=False)==first and x.calls==before
    assert first['models']=={'14b':'literal checkpoint'} and first['readiness_sha256']==x.a.sha(first['readiness_path'])
    assert first['files_sha256'][str(x.state/'action/exit.json')]==x.a.sha(x.state/'action/exit.json')
    assert task['output'] not in first['files_sha256']


def test_failed_preparation_cannot_supply_transport_inputs_and_pins_native_partial_outputs(fixture):
    x=fixture;x.errors.add('pools_after')
    with pytest.raises(RuntimeError):x.execute()
    pins=x.a.read(x.state/'action/failure.json')['files_sha256']
    assert pins[str(x.a.REVISION_ROOT/'protocol.json')]==x.a.sha(x.a.REVISION_ROOT/'protocol.json')
    pool=x.paths(x.a.REVISION_ROOT)['pools']/'difficulty_0.jsonl'
    assert pins[str(pool)]==x.a.sha(pool)
    with pytest.raises(ValueError,match='reconciliation'):x.a.development_inputs(x.state,guest=False)


@pytest.mark.parametrize('field',['review_sha256','view_manifest_sha256','lock_inode','lock_device','command'])
def test_static_owner_view_review_and_fence_cannot_be_relinked(fixture,field):
    x=fixture;x.execute();directory=x.state/'action'
    change(directory/'runtime.json',lambda v:v.update({field:['wrong'] if field=='command' else 'wrong'}))
    digest=x.a.sha(directory/'runtime.json')
    change(directory/'intent.json',lambda v:v.update(runtime_sha256=digest))
    change(directory/'guest_runtime.json',lambda v:v.update(outer_runtime_sha256=digest))
    with pytest.raises(ValueError):x.a.verify(x.state,guest=False)
