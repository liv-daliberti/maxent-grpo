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
    spec=importlib.util.spec_from_file_location('scratch_graph_r4_preparation',ROOT/'artifacts/register_modebench_scale_level5_graph_r4_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    tmp=f.state.parent;state=tmp/'preparation';revision_root=tmp/'graph_r4';parent=tmp/'parent'
    entries=f.a.read(f.state/'registration.json')['revisions']
    for key,value in {'_first':f.a,'FIT_STATE':f.state,'STATE':state,'REVISION_ROOT':revision_root,'PARENT':parent,
        'MANIFEST':f.a.VIEW,'LOCK':f.a.first.LOCK,'LOCK_INODE':f.a.first.LOCK_INODE,
        'REVIEW':tmp/'preparation_review.json','CANDIDATE':tmp/'candidate.py','CANDIDATE_TESTS':tmp/'candidate_tests.py',
        'CANDIDATE_REVIEW':tmp/'candidate_review.json','assert_fence':f.a.assert_fence,'lifetime_fence':f.a.lifetime_fence,
        'REGISTRATION_SHA':a.sha(f.state/'registration.json'),'FIT_RESULT_SHA':a.sha(f.state/'action/result.json'),
        'RECIPES':{e['domain']:a.sha(e['recipe_path']) for e in entries}}.items():monkeypatch.setattr(a,key,value,raising=False)
    a.CANDIDATE.write_text('four fixed candidate laws');a.CANDIDATE_TESTS.write_text('structural tests')
    profiles=[{'fixed_topology':t} for t in range(4)]
    parent_value={'models':{'14b':{'path':'literal checkpoint'}},'split_sizes':{'train':384,'dev':128,'eval':128},
        'targets':{'graph_coloring':{'metrics':{'pass_at_1':.3}},'mathir':{}},
        'histograms':{'graph_coloring':{s:[{'cell':[4],'rows':2}] for s in ('train','dev','eval')},'mathir':{}},
        'tolerances':{'pass_at_1':.04},'selection_seed':6491701,'sampling':{'unchanged':True},'fit':{'fixed_grid':True}}
    write(parent/'protocol.json',parent_value)
    candidate_dep=tmp/'candidate_dependency.py';candidate_dep.write_text('fixed dependency')
    negative_entry={'domain':'graph_coloring','source_root':str(tmp/'old_graph_r3'),
        'recipe_path':entries[0]['recipe_path'],'recipe_sha256':a.FAILED_RECIPE_SHA,'fit_result_sha256':a.FIT_RESULT_SHA}
    retained={d:{'source_root':str(tmp/d)} for d in ('countdown','mathir','python_factors')}
    monkeypatch.setattr(a,'negative_predecessor',lambda guest:(negative_entry,{str(f.source):a.sha(f.source)}))
    monkeypatch.setattr(a,'retained_sources',lambda guest:(retained,{str(f.source):a.sha(f.source)}))
    monkeypatch.setattr(a,'historical_closure',lambda:{str(f.source):a.sha(f.source)})
    for key,name in [('DECISION','decision.json'),('INVENTORY','inventory.json'),('L4_ADMISSION','level4_admission.json'),
        ('PRESERVATION','preserved/manifest.json')]:monkeypatch.setattr(a,key,tmp/name)
    monkeypatch.setattr(a,'CANDIDATE_SHA',a.sha(a.CANDIDATE));monkeypatch.setattr(a,'CANDIDATE_TESTS_SHA',a.sha(a.CANDIDATE_TESTS))
    write(a.DECISION,{'schema':'modebench_scale_level5_graph_pantry_law_decision_v1','status':'prospective_laws_selected_for_preparation_components',
        'level':'level5','graph':{'revision':4,'candidate_module':a.CANDIDATE_MODULE,'candidate_sha256':a.CANDIDATE_SHA,
        'base_choice_development_informed':True,'cpu_only_scratch_overlap_not_claimed_absent':True,
        'incomplete_cancelled_scratch_inventories_disclosed':True,'exclusion_policy':'selected native history and six-base policy'},'files_sha256':{}})
    monkeypatch.setattr(a,'DECISION_SHA',a.sha(a.DECISION))
    write(a.L4_ADMISSION,{'schema':'modebench_scale_composite_level_admission_v1','level':'level4','model_label':'7b',
        'difficulty_matched':True,'test_split':'eval','domains':{d:{'splits':{s:{'rows':n} for s,n in [('train',384),('dev',128),('eval',128)]}}
            for d in ('countdown','graph_coloring','mathir','pantry','python_factors')},'files_sha256':{str(f.source):a.sha(f.source)}})
    history_ids=[['graph_coloring',99]];history_prompts=['known native prompt'];native_pins={str(f.source):a.sha(f.source)}
    write(a.INVENTORY,{'schema':'modebench_scale_level5_graph_r4_fresh_history_v1',
        'status':'reviewed_current_native_and_known_exposed_inventory','candidate_sha256':a.CANDIDATE_SHA,
        'law_decision_sha256':a.DECISION_SHA,'exclusion_policy':'selected native history and six-base policy',
        'level4_admission_sha256':a.sha(a.L4_ADMISSION),'native_discovery_complete':True,'known_exposed_sources_complete':True,
        'all_1771_mixtures_capacity_claimed':False,'current_capacity_claimed':False,'identities':history_ids,
        'prompt_sha256':history_prompts,'native_files_sha256':native_pins,'known_exposed_files_sha256':{},'files_sha256':native_pins})
    preserved=[]
    for path in (a.CANDIDATE,a.CANDIDATE_TESTS):
        copy=a.PRESERVATION.parent/path.name;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(path.read_bytes());copy.chmod(0o444)
        preserved.append({'original_path':str(path),'preserved_path':str(copy),'sha256':a.sha(path)})
    write(a.PRESERVATION,{'schema':'modebench_scale_graph_direct_anchor_code_preservation_v1',
        'status':'preserved_before_scientific_registration','files':preserved})
    monkeypatch.setattr(a,'NEGATIVE_PINS',{str(f.source):a.sha(f.source)})
    monkeypatch.setattr(a,'RETAINED_PINS',{str(f.source):a.sha(f.source)})
    monkeypatch.setattr(a,'_history',SimpleNamespace(TESTS=f.a.TESTS,REVIEW=f.a.REVIEW))
    cpins={str(p):a.sha(p) for p in (a.CANDIDATE,a.CANDIDATE_TESTS,candidate_dep)}
    write(a.CANDIDATE_REVIEW,{'schema':'modebench_scale_graph_direct_anchor_candidates_independent_review_v1',
        'status':'reviewed_structural_component','candidate_sha256':a.CANDIDATE_SHA,'law_decision_sha256':a.DECISION_SHA,
        'all_1771_mixtures_or_current_capacity_claimed':False,'files_sha256':cpins})
    def repin():
        pins=dict(f.a.read(f.a.REVIEW)['files_sha256']);pins.update(cpins)
        pins.update({str(p):a.sha(p) for p in (a.SOURCE,a.TESTS,a.HISTORY,f.a.TESTS,f.a.REVIEW,a.CANDIDATE_REVIEW,
            a.DECISION,a.INVENTORY,a.PRESERVATION,a.L4_ADMISSION,*map(Path,a.NEGATIVE_PINS),*map(Path,a.RETAINED_PINS))})
        pins.update({x['preserved_path']:x['sha256'] for x in preserved})
        write(a.REVIEW,{'schema':'modebench_scale_level5_graph_r4_registration_independent_review_v1',
            'status':'reviewed','production_activation_after_level4_completion':True,'files_sha256':pins})
    repin()
    f.owner['command']=[str(a.PYTHON),'-B',str(a.SOURCE),'prepare'];active=[];calls=[];errors=set();authentications=[]
    monkeypatch.setattr(a.os,'uname',lambda:SimpleNamespace(nodename=a.HOST))
    monkeypatch.setattr(a.socket,'gethostname',lambda:a.HOST)
    monkeypatch.setattr(a.os,'getuid',lambda:a.UID);monkeypatch.setattr(a.os,'geteuid',lambda:a.UID)
    f.owner['uid']=a.UID
    def identity(pid):
        if active and pid!=f.owner['pid']:
            argv=active[0][active[0].index('--')+1:];argv.insert(1,'-B')
            return {**deepcopy(f.owner),'pid':12346,'start_ticks':'5679','command':argv}
        return deepcopy(f.owner)
    monkeypatch.setattr(a,'process_identity',identity)
    def host_guard(*,guest):
        a.require(a.os.uname().nodename==a.HOST,'truthful host required')
        a.require(f.canonical.read_text()==('old' if guest else 'neutral'),'correct source view required')
    monkeypatch.setattr(a,'live_host_guard',host_guard)
    def paths(root):
        return {'protocol':root/'protocol.json','pools':root/'level5/pools/graph_coloring',
            'development':root/'level5/results/development/graph_coloring'}
    def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
    def generation_seed(p,split,tier):return 8000+tier
    def register(root,level,domain,**kwargs):
        calls.append('register');assert f.held==[8] and f.canonical.read_text()=='old'
        assert root==revision_root and level=='level5' and domain=='graph_coloring'
        assert kwargs=={'parent':parent,'revision':4,'candidate_module':a.CANDIDATE_MODULE}
        assert a.read(state/'amendment.json')['actual_r3_predecessor']==negative_entry
        if 'register_before' in errors:raise RuntimeError('native register exception')
        value={k:deepcopy(v) for k,v in parent_value.items() if k not in ('targets','histograms')}
        value.update(schema='native_revision_schema',root=str(root),parent_root=str(parent),level=level,domain=domain,revision=4,
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
            'identities':history_ids,'prompt_sha256':history_prompts,'files_sha256':native_pins})
        tiers={}
        for tier in range(4):
            batch=[{'scale_candidate_tier':tier,'answer_mode_count':4,'case':10*tier+i} for i in range(2)]
            path=q['pools']/f'difficulty_{tier}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('\n'.join(json.dumps(row) for row in batch)+'\n')
            tiers[str(tier)]={'rows':2,'rows_sha256':object_sha(batch),'generation_seed':8000+tier,
                'semantic_disjoint':True,'prompt_disjoint':True,'verification':{'rows_verified':2,'exact_canonical_support':True,'unchanged_original_prompt':True,'graph_direct_anchor_structural_profile':True}}
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
        return {'id':'level5_graph_coloring_r4_dev','level':'level5','domain':'graph_coloring','phase':'dev','model_label':'14b',
            'source_kind':'domain_revision_v1','source_root':str(root),'tasks':tasks},{str(f.source):a.sha(f.source)}
    candidate=SimpleNamespace(PROFILES={'graph_coloring':profiles},blocked_projections=lambda ids:set(ids),
        identity=lambda domain,row:('graph_coloring',row['case']),base_projections=lambda key:{key})
    native=SimpleNamespace(SCHEMA='native_revision_schema',candidate_provider=lambda name:candidate,
        labels=lambda level,domain,revision,phase:list(range(8104000+(500 if phase=='eval' else 0),8104004+(500 if phase=='eval' else 0))),
        register=register,materialize_pools=pools,launch_inputs=launch,paths=paths,rows_from_jsonl=rows,sha=object_sha,generation_seed=generation_seed,
        original=SimpleNamespace(history=lambda domain,root:({a.tuplify(x) for x in history_ids},set(history_prompts),dict(native_pins)),deserialize_cells=lambda h:Counter({tuple(v['cell']):v['rows'] for v in h}),union_histogram=lambda h:next(iter(h.values()))),
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
        candidate_dep=candidate_dep,authentications=authentications,repin=repin,history_ids=history_ids,history_prompts=history_prompts,execute=lambda:a.run(state,a.sha(a.REVIEW)))


def test_native_registration_four_full_pools_and_exact_development_tasks_only(fixture):
    x=fixture;before={d:x.a.sha(e['recipe_path']) for d,e in zip(('graph_coloring','mathir'),x.entries)};v=x.execute()
    assert x.calls==['register','materialize_pools','launch_inputs'] and v['status']=='ready_for_fresh_graph_r4_development'
    assert v['native_parent_recipe_reproducibility'] and v['native_row_certification'] and not v['new_fit_decisions']
    assert not v['source_binding_performed'] and not v['submission_performed'] and not v['heldout_performed']
    assert {d:x.a.sha(e['recipe_path']) for d,e in zip(('graph_coloring','mathir'),x.entries)}==before
    tasks=x.a.read(x.state/'development_tasks.json');assert len(tasks)==4
    assert all(t['seeds']==[8104000,8104001,8104002,8104003] and t['row_offset']==t['row_limit']==0 for t in tasks)
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


@pytest.mark.parametrize('kind',['relative_python','relative_source','wrong_action','changed_before_fence'])
def test_absolute_outer_prefix_blocks_before_any_state_or_native_call(fixture,monkeypatch,kind):
    x=fixture;a=x.a
    if kind=='relative_python':x.f.owner['command'][0]='var/seed_paper_eval/paper310/bin/python'
    elif kind=='relative_source':x.f.owner['command'][2]='artifacts/register_modebench_scale_level5_graph_r4_20260913.py'
    elif kind=='wrong_action':x.f.owner['command'][3]='verify'
    else:
        from contextlib import contextmanager
        original=a.lifetime_fence
        @contextmanager
        def changed():
            with original() as fd:
                x.f.owner['command'][2]='relative.py'
                yield fd
        monkeypatch.setattr(a,'lifetime_fence',changed)
    with pytest.raises(ValueError,match='absolute literal'):x.execute()
    assert not x.state.exists() and not a.REVISION_ROOT.exists() and not x.calls


@pytest.mark.parametrize('kind',['unadmitted','partial_l4','wrong_level','wrong_model','sizes','missing_admission_pin',
    'missing_review_dependency','candidate_changed','tests_changed','policy_changed','capacity_claim','unreviewed_inventory',
    'missing_native_discovery','missing_exposed_inventory','unpreserved','self_preservation','hardlink_preservation','writable_copy',
    'review_cycle','future_pin','activation_missing'])
def test_final_activation_requires_actual_policy_history_preservation_and_completed_l4(fixture,kind):
    x=fixture;a=x.a
    if kind in ('unadmitted','partial_l4','wrong_level','wrong_model','sizes'):
        def mutate(v):
            if kind=='unadmitted':v['difficulty_matched']=False
            elif kind=='partial_l4':v['domains'].pop('pantry')
            elif kind=='wrong_level':v['level']='level5'
            elif kind=='wrong_model':v['model_label']='14b'
            else:v['domains']['countdown']['splits']['eval']['rows']=127
        change(a.L4_ADMISSION,mutate)
        change(a.INVENTORY,lambda v:v.update(level4_admission_sha256=a.sha(a.L4_ADMISSION)));x.repin()
    elif kind=='candidate_changed':a.CANDIDATE.write_text('changed law')
    elif kind=='tests_changed':a.CANDIDATE_TESTS.write_text('changed tests')
    elif kind in ('policy_changed','capacity_claim','unreviewed_inventory','missing_native_discovery','missing_exposed_inventory'):
        key,value={'policy_changed':('exclusion_policy','blanket all CPU scratch burn'),
            'capacity_claim':('current_capacity_claimed',True),'unreviewed_inventory':('status','draft'),
            'missing_native_discovery':('native_discovery_complete',False),'missing_exposed_inventory':('known_exposed_sources_complete',False)}[kind]
        change(a.INVENTORY,lambda v:v.update({key:value}));x.repin()
    elif kind in ('unpreserved','self_preservation','hardlink_preservation','writable_copy'):
        value=a.read(a.PRESERVATION);entry=value['files'][0];copy=Path(entry['preserved_path'])
        if kind=='unpreserved':value['files'].pop()
        elif kind=='self_preservation':entry['preserved_path']=entry['original_path']
        elif kind=='hardlink_preservation':
            copy.unlink();copy.hardlink_to(entry['original_path']);copy.chmod(0o444)
        else:copy.chmod(0o600)
        write(a.PRESERVATION,value);x.repin()
    else:
        def mutate(v):
            if kind=='missing_admission_pin':v['files_sha256'].pop(str(a.L4_ADMISSION))
            elif kind=='missing_review_dependency':v['files_sha256'].pop(str(x.candidate_dep))
            elif kind=='review_cycle':v['files_sha256'][str(a.REVIEW)]='cycle'
            elif kind=='future_pin':v['files_sha256'][str(a.REVISION_ROOT/'future.json')]='future'
            else:v['production_activation_after_level4_completion']=False
        change(a.REVIEW,mutate)
    with pytest.raises((ValueError,FileNotFoundError)):x.execute()
    assert not x.calls and not x.state.exists()


@pytest.mark.parametrize('kind',['changed_native_file','new_native_source','missing_external_identity','missing_external_prompt'])
def test_current_discovery_gap_fails_before_native_claim(fixture,monkeypatch,kind):
    x=fixture;a=x.a
    if kind in ('changed_native_file','new_native_source'):
        history=x.native.original.history
        def changed(*args):
            ids,prompts,pins=history(*args)
            if kind=='changed_native_file':pins[str(x.f.source)]='changed'
            else:pins['/specific/new_native_source']='new'
            return ids,prompts,pins
        monkeypatch.setattr(x.native.original,'history',changed)
    else:
        field='identities' if kind=='missing_external_identity' else 'prompt_sha256'
        change(a.INVENTORY,lambda v:v[field].append(['graph_coloring',77] if field=='identities' else 'specific known exposed prompt'))
        x.repin()
    with pytest.raises(ValueError,match='inventory changed|snapshot must cover'):x.execute()
    assert not (x.state/'amendment.json').exists() and not a.REVISION_ROOT.exists() and not x.calls


@pytest.mark.parametrize('kind',['old_history','cross_tier','missing_prompt'])
def test_native_snapshot_and_all_relevant_projection_checks_are_cumulative(fixture,monkeypatch,kind):
    x=fixture;a=x.a
    if kind=='missing_prompt':
        materialize=x.native.materialize_pools
        def changed(root):
            value=materialize(root);snapshot=root/'exclusions/development.json'
            change(snapshot,lambda v:v.update(prompt_sha256=[]));value['exclusions_sha256']=a.sha(snapshot)
            write(x.paths(root)['pools']/'identity.json',value);return value
        monkeypatch.setattr(x.native,'materialize_pools',changed)
    else:
        candidate=x.native.candidate_provider(a.CANDIDATE_MODULE)
        monkeypatch.setattr(candidate,'base_projections',lambda key:{('graph_coloring',99 if kind=='old_history' else 0)})
    with pytest.raises(ValueError,match='snapshot must cover|six-vertex'):x.execute()
    assert not (x.state/'certificate.json').exists()


def test_only_new_action_host_changes_and_fence_objects_remain_historical():
    spec=importlib.util.spec_from_file_location('graph_r4_object_identity',ROOT/'artifacts/register_modebench_scale_level5_graph_r4_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    assert a.HOST=='soak.cs.princeton.edu' and a._history.HOST=='spin.cs.princeton.edu'
    assert a.assert_fence is a._history.assert_fence and a.lifetime_fence is a._history.lifetime_fence
    assert a._first is a._history._first and a._first.first.HOST=='spin.cs.princeton.edu'


@pytest.mark.parametrize('kind',['hostname','uname','uid','euid','view'])
def test_truthful_soak_and_exact_view_are_not_claimed_through_facades(fixture,monkeypatch,kind):
    x=fixture;a=x.a
    # Restore actual draft guard, but use synthetic source hashes and process facts.
    spec=importlib.util.spec_from_file_location('graph_r4_host_guard_definition',a.SOURCE)
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    guard=type(source.live_host_guard)(source.live_host_guard.__code__,a.__dict__)
    monkeypatch.setattr(a._first.first,'CANONICAL',x.f.canonical)
    monkeypatch.setattr(a._first.first,'NEUTRAL_SHA',a.sha(x.f.canonical))
    if kind=='hostname':monkeypatch.setattr(a.socket,'gethostname',lambda:'spin.cs.princeton.edu')
    elif kind=='uname':monkeypatch.setattr(a.os,'uname',lambda:SimpleNamespace(nodename='wash.cs.princeton.edu'))
    elif kind=='uid':monkeypatch.setattr(a.os,'getuid',lambda:12)
    elif kind=='euid':monkeypatch.setattr(a.os,'geteuid',lambda:12)
    else:x.f.canonical.write_text('wrong')
    with pytest.raises(ValueError):guard(guest=False)


def original_function(a,name):
    spec=importlib.util.spec_from_file_location('graph_r4_static_original_'+name,a.SOURCE)
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    return type(getattr(source,name))(getattr(source,name).__code__,a.__dict__)


@pytest.fixture
def saved_evidence(fixture,monkeypatch):
    x=fixture;a=x.a;tmp=x.state.parent/'actual_history_stubs'
    negative=tmp/'negative';recipe=tmp/'r3/level5/recipes/graph_coloring.json';fit=tmp/'fit.py';fit.parent.mkdir(parents=True);fit.write_text('fixed fit helper')
    for key,value in [('NEGATIVE_STATE',negative),('NEGATIVE_RECIPE',recipe),('NEGATIVE_FIT',fit),('L5_MANIFEST',tmp/'manifest.json')]:monkeypatch.setattr(a,key,value)
    result={'status':'needs_new_development_revision','failed_domains':['graph_coloring'],
        'fits':[{'domain':'graph_coloring','development_fit_pass':False}]}
    base={str(x.f.source):a.sha(x.f.source)}
    write(negative/'action/result.json',result);write(negative/'registration.json',{'files_sha256':base})
    write(recipe,{'development_fit_pass':False,'input_sha256':base})
    monkeypatch.setattr(a,'FIT_RESULT_SHA',a.sha(negative/'action/result.json'));monkeypatch.setattr(a,'FAILED_RECIPE_SHA',a.sha(recipe))
    negative_pins={str(p):a.sha(p) for p in (fit,recipe,negative/'action/result.json',negative/'registration.json')}
    monkeypatch.setattr(a,'NEGATIVE_PINS',negative_pins)
    calls=[]
    def verify(root,guest):calls.append(('static_negative_verify',root,guest));return deepcopy(result)
    monkeypatch.setattr(a,'load_module',lambda path,name:SimpleNamespace(verify_existing=verify))
    retained={};certs=[];rpins={};sources={}
    for domain in ('countdown','mathir','python_factors'):
        root=tmp/domain;dataset=root/'level5/dataset'/domain;p=root/'level5/recipes'/(domain+'.json')
        write(p,{'development_fit_pass':True,'input_sha256':base})
        value={'level':'level5','domain':domain,'recipe_sha256':a.sha(p),
            'splits':{s:{'rows':n} for s,n in [('train',384),('dev',128),('eval',128)]}}
        write(dataset/'identity.json',value)
        kind='campaign_v1' if domain=='countdown' else 'domain_revision_v1'
        retained[domain]={'source_root':str(root),'source_kind':kind,'recipe_sha256':a.sha(p),'identity_sha256':a.sha(dataset/'identity.json')}
        sources[domain]={'source_root':str(root),'source_kind':kind}
        if domain!='countdown':
            cert=tmp/(domain+'_certificate.json');write(cert,{'frozen':[{'domain':domain}],'files_sha256':base});certs.append(str(cert));rpins[str(cert)]=a.sha(cert)
    write(a.L5_MANIFEST,{'sources':sources,'files_sha256':base});rpins[str(a.L5_MANIFEST)]=a.sha(a.L5_MANIFEST)
    monkeypatch.setattr(a,'RETAINED',retained);monkeypatch.setattr(a,'RETAINED_CERTIFICATES',certs);monkeypatch.setattr(a,'RETAINED_PINS',rpins)
    return SimpleNamespace(x=x,a=a,calls=calls,result=result,negative=original_function(a,'negative_predecessor'),
        retained=original_function(a,'retained_sources'),closure=original_function(a,'historical_closure'))


def test_exact_negative_and_all_three_frozen_sources_are_static_only(saved_evidence):
    e=saved_evidence;entry,pins=e.negative(guest=False);retained,rpins=e.retained(guest=False)
    assert entry['recipe_sha256']==e.a.FAILED_RECIPE_SHA and entry['domain']=='graph_coloring'
    assert set(retained)=={'countdown','mathir','python_factors'}
    assert e.calls==[('static_negative_verify',e.a.NEGATIVE_STATE,False)]
    assert e.closure()=={str(e.x.f.source):e.a.sha(e.x.f.source)}
    assert not e.x.calls


@pytest.mark.parametrize('kind',['recipe_changed','result_changed','different_failure','passing','verifier_differs'])
def test_negative_predecessor_cannot_be_replaced_or_bypassed(saved_evidence,kind):
    e=saved_evidence;a=e.a
    if kind=='recipe_changed':change(a.NEGATIVE_RECIPE,lambda v:v.update(development_fit_pass=True))
    elif kind=='result_changed':change(a.NEGATIVE_STATE/'action/result.json',lambda v:v.update(status='passed'))
    elif kind=='different_failure':e.result['failed_domains']=['pantry']
    elif kind=='passing':e.result['fits'][0]['development_fit_pass']=True
    else:e.result['other']='different saved result'
    with pytest.raises(ValueError):e.negative(guest=False)
    assert not e.x.calls


@pytest.mark.parametrize('domain',['countdown','mathir','python_factors'])
@pytest.mark.parametrize('kind',['recipe','dataset_identity','source_manifest','freeze_certificate'])
def test_three_retained_sources_keep_literal_passing_recipes_and_frozen_rows(saved_evidence,domain,kind):
    e=saved_evidence;a=e.a;root=Path(a.RETAINED[domain]['source_root'])
    if kind=='recipe':change(root/'level5/recipes'/(domain+'.json'),lambda v:v.update(development_fit_pass=False))
    elif kind=='dataset_identity':change(root/'level5/dataset'/domain/'identity.json',lambda v:v['splits']['train'].update(rows=383))
    elif kind=='source_manifest':change(a.L5_MANIFEST,lambda v:v['sources'][domain].update(source_root='/wrong'))
    else:change(a.RETAINED_CERTIFICATES[0],lambda v:v.update(frozen=[]))
    with pytest.raises(ValueError):e.retained(guest=False)
    assert not e.x.calls


@pytest.mark.parametrize('root_name',['STATE','REVISION_ROOT'])
def test_dangling_root_symlink_cannot_redirect_native_registration(fixture,root_name):
    x=fixture;path=getattr(x.a,root_name);target=x.state.parent/'redirected_native_root'
    path.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='fixed fresh|already attempted'):x.execute()
    assert not target.exists() and not x.calls


@pytest.mark.parametrize('field,value',[('lock_device',987654),('pid',90210),('start_ticks','90210')])
def test_runtime_record_is_bound_even_when_old_hash_links_are_rewritten(fixture,field,value):
    x=fixture;x.execute();directory=x.state/'action';a=x.a
    assert a.read(x.state/'certificate.json')['files_sha256'][str(directory/'runtime.json')]==a.sha(directory/'runtime.json')
    change(directory/'runtime.json',lambda v:v.update({field:value}));digest=a.sha(directory/'runtime.json')
    change(directory/'intent.json',lambda v:v.update(runtime_sha256=digest))
    change(directory/'guest_runtime.json',lambda v:v.update(outer_runtime_sha256=digest))
    with pytest.raises(ValueError):a.verify(x.state,guest=False)


def test_native_direct_anchor_certificate_flag_is_mandatory(fixture,monkeypatch):
    x=fixture;materialize=x.native.materialize_pools
    def changed(root):
        value=materialize(root);value['tiers']['0']['verification'].pop('graph_direct_anchor_structural_profile')
        write(x.paths(root)['pools']/'identity.json',value);return value
    monkeypatch.setattr(x.native,'materialize_pools',changed)
    with pytest.raises(ValueError,match='structural certification'):x.execute()
    assert not (x.state/'certificate.json').exists()


def test_future_admission_gate_matches_registered_level4_publication_destination():
    import ast
    def constant_path(file,name):
        tree=ast.parse(file.read_text())
        values=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        assert len(values)==1 and isinstance(values[0],ast.BinOp) and isinstance(values[0].op,ast.Div)
        return Path(values[0].right.value)
    gate=constant_path(ROOT/'artifacts/register_modebench_scale_level5_graph_r4_20260913.py','L4_ADMISSION')
    for filename in ('continue_modebench_scale_level4_first_release_r4_20260913.py','complete_modebench_scale_level4_first_r4_20260913.py'):
        release=constant_path(ROOT/'artifacts'/filename,'RELEASE')
        assert gate==release/'level4/admission.json'
