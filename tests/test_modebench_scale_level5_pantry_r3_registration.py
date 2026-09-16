"""Synthetic Pantry preparation boundaries; native production calls never run."""
import ast
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():p.chmod(0o644)
    p.write_text(json.dumps(v,sort_keys=True))


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()


def mutate(p,fn):
    v=json.loads(Path(p).read_text());fn(v);write(p,v)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('scratch_pantry_preparation',ROOT/'artifacts/register_modebench_scale_level5_pantry_r3_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    for name in ('STATE','REVISION_ROOT','PARENT','NEGATIVE_STATE'):
        monkeypatch.setattr(a,name,tmp_path/name)
    for name in ('SOURCE','TESTS','REVIEW','HISTORY','CANDIDATE','CANDIDATE_TESTS','CANDIDATE_REVIEW','QUALIFICATION','BURN','DECISION','INVENTORY','L4_ADMISSION','NEGATIVE_FIT','NEGATIVE_RECIPE','L5_MANIFEST'):
        p=tmp_path/(name+'.json');write(p,{});monkeypatch.setattr(a,name,p)
    monkeypatch.setattr(a,'PRESERVATION',tmp_path/'preserved/manifest.json')
    histreview=tmp_path/'history_review.json';histtests=tmp_path/'history_tests.py';write(histreview,{'files_sha256':{}});histtests.write_text('history tests')
    monkeypatch.setattr(a,'_history',SimpleNamespace(REVIEW=histreview,TESTS=histtests))
    dependency=tmp_path/'dep.py';dependency.write_text('unchanged scientific source')
    dep_pins={str(dependency):a.sha(dependency)}
    real_history=a.historical_closure;real_inputs=a.inputs
    monkeypatch.setattr(a,'historical_closure',lambda:dict(dep_pins))
    monkeypatch.setattr(a,'NEGATIVE_PINS',dict(dep_pins));monkeypatch.setattr(a,'RETAINED_PINS',dict(dep_pins))
    raw=[['pantry',digest(['burn',i])] for i in range(3945)];prompts=[digest(['prompt',i]) for i in range(3945)]
    write(a.BURN,{'schema':'modebench_scale_pantry_menu_scratch_exclusions_v1','status':'all_materialized_qualification_and_reference_identities',
        'production_reuse_allowed':False,'identity_count':3945,'prompt_count':3945,'identities':raw,'prompt_sha256':prompts,'fixtures_sha256':dep_pins})
    monkeypatch.setattr(a,'BURN_SHA',a.sha(a.BURN))
    write(a.QUALIFICATION,{'schema':'modebench_scale_pantry_menu_feasibility_manifest_v1','status':'native_scratch_capacity_qualified_no_production_adoption',
        'full_capacity_rows':3488,'scratch_exclusions_path':str(a.BURN),'scratch_exclusions_sha256':a.BURN_SHA,
        'files_sha256':{**dep_pins,str(a.BURN):a.BURN_SHA}})
    monkeypatch.setattr(a,'QUALIFICATION_SHA',a.sha(a.QUALIFICATION))
    monkeypatch.setattr(a,'CANDIDATE_SHA',a.sha(a.CANDIDATE));monkeypatch.setattr(a,'CANDIDATE_TESTS_SHA',a.sha(a.CANDIDATE_TESTS))
    profiles=[{'menu_size':7+t//2,'minimum_if_used_g':50,'quantity_step_g':25,'available_g_choices':[100] if t%2==0 else [100,125,150],
        'selected_ingredient_count_range':[2,4],'tier':t} for t in range(4)]
    write(a.CANDIDATE_REVIEW,{'schema':'modebench_scale_pantry_menu_r3_candidates_independent_review_v1','status':'reviewed','blocking_findings':[],
        'source_sha256':a.CANDIDATE_SHA,'tests_sha256':a.CANDIDATE_TESTS_SHA,'remaining_fresh_capacity_guaranteed':False,
        'declared_laws':profiles,'files_sha256':dep_pins})
    monkeypatch.setattr(a,'CANDIDATE_REVIEW_SHA',a.sha(a.CANDIDATE_REVIEW))
    write(a.DECISION,{'schema':'modebench_scale_level5_graph_pantry_law_decision_v1','status':'prospective_laws_selected_for_preparation_components','level':'level5',
        'pantry':{'revision':3,'candidate_module':a.CANDIDATE_MODULE,'candidate_sha256':a.CANDIDATE_SHA,'profiles':profiles,
            'capacity_after_mandatory_burn_not_yet_proven':True,'mandatory_scratch_burn':{'sha256':a.BURN_SHA}},'files_sha256':dep_pins})
    monkeypatch.setattr(a,'DECISION_SHA',a.sha(a.DECISION))
    write(a.L4_ADMISSION,{'schema':'modebench_scale_composite_level_admission_v1','level':'level4','model_label':'7b','difficulty_matched':True,'test_split':'eval',
        'domains':{d:{'splits':{s:{'rows':n} for s,n in [('train',384),('dev',128),('eval',128)]}} for d in ('countdown','graph_coloring','mathir','pantry','python_factors')},'files_sha256':dep_pins})
    historical_id=['pantry',digest('historical')];historical_prompt=digest('historical prompt')
    write(a.INVENTORY,{'schema':'modebench_scale_level5_pantry_r3_fresh_history_v1','status':'reviewed_current_native_and_known_exposed_inventory',
        'law_decision_sha256':a.DECISION_SHA,'candidate_sha256':a.CANDIDATE_SHA,'mandatory_scratch_exclusions_sha256':a.BURN_SHA,
        'level4_admission_sha256':a.sha(a.L4_ADMISSION),'native_discovery_complete':True,'known_exposed_sources_complete':True,
        'capacity_after_mandatory_burn_claimed':False,'identities':[historical_id],'prompt_sha256':[historical_prompt],
        'native_files_sha256':dep_pins,'known_exposed_files_sha256':{},'files_sha256':dep_pins})
    copies=[]
    for p in (a.CANDIDATE,a.CANDIDATE_TESTS):
        q=a.PRESERVATION.parent/p.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());q.chmod(0o444)
        copies.append({'original_path':str(p),'preserved_path':str(q),'sha256':a.sha(p)})
    write(a.PRESERVATION,{'schema':'modebench_scale_pantry_menu_r3_code_preservation_v1','status':'preserved_before_scientific_registration','files':copies})
    def repin():
        names=('SOURCE','TESTS','HISTORY','CANDIDATE','CANDIDATE_TESTS','CANDIDATE_REVIEW','QUALIFICATION','BURN','DECISION','PRESERVATION','INVENTORY','L4_ADMISSION')
        pins={str(getattr(a,n)):a.sha(getattr(a,n)) for n in names}
        pins.update(dep_pins);pins.update({str(p):a.sha(p) for p in (histreview,histtests)})
        pins.update({x['preserved_path']:x['sha256'] for x in copies})
        write(a.REVIEW,{'schema':'modebench_scale_level5_pantry_r3_registration_independent_review_v1','status':'reviewed',
            'production_activation_after_level4_completion':True,'capacity_after_mandatory_burn_claimed':False,'files_sha256':pins})
    repin()
    def check(pins,*,guest=False):
        for p,h in pins.items():a.require(a.sha(p)==h,'pin changed: '+p)
    monkeypatch.setattr(a,'check_pins',check)
    retained={d:{'existing':True} for d in ('countdown','mathir','python_factors')}
    entry={'domain':'pantry','recipe_sha256':a.FAILED_RECIPE_SHA}
    monkeypatch.setattr(a,'inputs',lambda root,guest=True:({'pantry':entry,'_preserved':retained},dict(dep_pins)))
    canonical=tmp_path/'canonical.py';canonical.write_text('neutral');manifest=tmp_path/'manifest.json';write(manifest,{})
    lock=tmp_path/'lock';lock.write_text('fixed lock')
    monkeypatch.setattr(a,'MANIFEST',manifest);monkeypatch.setattr(a,'LOCK',lock);monkeypatch.setattr(a,'LOCK_INODE',lock.stat().st_ino)
    revision_source=tmp_path/'revision.py';revision_source.write_text('native methods are fixture-only')
    monkeypatch.setattr(a,'_first',SimpleNamespace(REVISION=revision_source,first=SimpleNamespace(CANONICAL=canonical,
        OLD_SHA=hashlib.sha256(b'old').hexdigest(),NEUTRAL_SHA=hashlib.sha256(b'neutral').hexdigest())))
    monkeypatch.setattr(a,'os',SimpleNamespace(getuid=lambda:a.UID,geteuid=lambda:a.UID,getpid=lambda:123,
        uname=lambda:SimpleNamespace(nodename=a.HOST),environ={}))
    monkeypatch.setattr(a,'socket',SimpleNamespace(gethostname=lambda:a.HOST))
    owner={'pid':123,'start_ticks':'99','uid':a.UID,'state':'S','command':[str(a.PYTHON),'-B',str(a.SOURCE),'prepare']}
    held=[];active=[];calls=[];errors=set()
    @contextmanager
    def fence():
        assert not held;held.append(8)
        try:yield 8
        finally:held.clear()
    def assert_fence(fd):assert held==[fd]==[8]
    monkeypatch.setattr(a,'lifetime_fence',fence);monkeypatch.setattr(a,'assert_fence',assert_fence)
    monkeypatch.setattr(a,'process_identity',lambda pid:deepcopy(owner))
    parent={'models':{'14b':{'path':'literal14b'}},'split_sizes':{'train':384,'dev':128,'eval':128},
        'targets':{'pantry':{'unchanged_level1_target':True}},'histograms':{'pantry':{s:[{'cell':[8,'family'],'rows':2}] for s in ('train','dev','eval')}},
        'tolerances':{'six':True},'selection_seed':6491701,'sampling':{'unchanged':True},'fit':{'original_grid':True}}
    write(a.PARENT/'protocol.json',parent)
    def paths(root):return {'protocol':root/'protocol.json','pools':root/'level5/pools/pantry','development':root/'level5/results/development/pantry'}
    def register(root,level,domain,**kwargs):
        calls.append('register');assert held==[8] and canonical.read_text()=='old';assert (level,domain)==('level5','pantry')
        assert kwargs=={'parent':a.PARENT,'revision':3,'candidate_module':a.CANDIDATE_MODULE}
        assert (a.STATE/'amendment.json').is_file()
        if 'register' in errors:raise RuntimeError('synthetic register failure')
        _,_,burn=a.mandatory_burn();v={**deepcopy(parent),'schema':'native','level':level,'domain':domain,'revision':3,
            'root':str(root),'parent_root':str(a.PARENT),'parent_protocol_sha256':a.sha(a.PARENT/'protocol.json'),
            'candidate_module':a.CANDIDATE_MODULE,'candidate_profiles':profiles,'draw_labels':a.LABELS,'files_sha256':burn}
        write(root/'protocol.json',v);write(root/'protocol.sha256.json',{'sha256':a.sha(root/'protocol.json')});return v
    def pools(root):
        calls.append('materialize_pools');assert held==[8]
        if 'pools' in errors:raise RuntimeError('synthetic quota exhaustion')
        q=paths(root);snapshot=root/'exclusions/development.json'
        write(snapshot,{'schema':'modebench_scale_revision_exclusions_v1','stage':'development','protocol_sha256':a.sha(q['protocol']),
            'identities':[historical_id],'prompt_sha256':[historical_prompt],'files_sha256':dep_pins})
        tiers={}
        for t in range(4):
            rows=[{'scale_candidate_tier':t,'answer_mode_count':8,'answer_mode_family':'family','problem':f'fixture {t} {i}',
                'id':digest(['native',t,i])} for i in range(2)]
            p=q['pools']/f'difficulty_{t}.jsonl';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            tiers[str(t)]={'rows':2,'rows_sha256':digest(rows),'generation_seed':9000+t,'semantic_disjoint':True,'prompt_disjoint':True,
                'verification':{'rows_verified':2,'exact_canonical_support':True,'unchanged_original_prompt':True,'complete_scratch_exclusions':True,'pantry_menu_revision3_profile':True}}
        v={'schema':'modebench_scale_revision_pools_v1','level':'level5','domain':'pantry','protocol_sha256':a.sha(q['protocol']),
            'exclusions':str(snapshot),'exclusions_sha256':a.sha(snapshot),'tiers':tiers};write(q['pools']/'identity.json',v);return v
    def launch(root,phase):
        calls.append('launch_inputs');assert held==[8] and phase=='dev'
        if 'tasks' in errors:raise RuntimeError('synthetic launch inputs failure')
        q=paths(root);tasks=[{'level':'level5','domain':'pantry','split':'dev','interface':'native interface','rows_jsonl':str(q['pools']/f'difficulty_{t}.jsonl'),
            'seeds':a.LABELS['dev'],'batch_size':8,'row_offset':0,'row_limit':0,'output':str(q['development']/f'difficulty_{t}.json')} for t in range(4)]
        return {'id':'level5_pantry_r3_dev','level':'level5','domain':'pantry','phase':'dev','model_label':'14b','source_kind':'domain_revision_v1','source_root':str(root),'tasks':tasks},dict(dep_pins)
    native=SimpleNamespace(SCHEMA='native',register=register,materialize_pools=pools,launch_inputs=launch,paths=paths,
        candidate_provider=lambda name:SimpleNamespace(PROFILES={'pantry':profiles}),labels=lambda l,d,r,p:list(a.LABELS[p]),
        rows_from_jsonl=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()],sha=digest,
        generation_seed=lambda p,s,t:9000+t,cell_histogram=lambda d,rows:Counter({(8,'family'):len(rows)}),
        identity_set=lambda d,rows:{(d,row['id']) for row in rows},evaluator=SimpleNamespace(INTERFACE='native interface'),
        original=SimpleNamespace(union_histogram=lambda h:next(iter(h.values())),deserialize_cells=lambda h:Counter({tuple(x['cell']):x['rows'] for x in h}),
            history=lambda d,r:({tuple(historical_id)},{historical_prompt},dict(dep_pins))))
    monkeypatch.setattr(a,'load_module',lambda p,n:native if p==revision_source else SimpleNamespace(authenticate=lambda p:None))
    real_guest=a.guest_action
    def run(argv,**kwargs):
        assert kwargs['pass_fds']==(8,);canonical.write_text('old');active.append(argv)
        original=a.process_identity
        count={'n':0}
        def identities(pid):
            count['n']+=1
            if count['n']==2:
                actual=argv[argv.index('--')+1:];actual.insert(1,'-B');return {**deepcopy(owner),'pid':124,'command':actual}
            return deepcopy(owner)
        a.process_identity=identities
        try:real_guest(a.STATE,8,a.sha(a.REVIEW))
        finally:a.process_identity=original;canonical.write_text('neutral');active.clear()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(a,'subprocess',SimpleNamespace(run=run))
    return SimpleNamespace(a=a,calls=calls,errors=errors,owner=owner,repin=repin,raw=raw,prompts=prompts,native=native,canonical=canonical,
        dependency=dependency,profiles=profiles,real_history=real_history,real_inputs=real_inputs,execute=lambda:a.run(a.STATE,a.sha(a.REVIEW)))


def test_complete_synthetic_native_preparation_has_no_fit_or_release(fixture):
    x=fixture;v=x.execute();assert x.calls==['register','materialize_pools','launch_inputs']
    assert v['status']=='ready_for_fresh_pantry_r3_development' and v['level']=='level5'
    assert set(v['preserved_level5_sources'])=={'countdown','mathir','python_factors'}
    assert v['mandatory_scratch_exclusions_sha256']==x.a.BURN_SHA and v['capacity_after_mandatory_burn_claimed'] is False
    assert all(v[k] is False for k in ('new_fit_decisions','model_scoring_performed','source_binding_performed','submission_performed','heldout_performed','admission_performed'))
    assert x.a.development_inputs(x.a.STATE,guest=False)['models']=={'14b':'literal14b'}
    assert x.a.read(x.a.REVISION_ROOT/'protocol.json')['draw_labels']=={'dev':[8403000,8403001,8403002,8403003],'eval':[8403500,8403501,8403502,8403503]}


@pytest.mark.parametrize('phase',['register','pools','tasks'])
def test_partial_failures_record_history_and_cannot_retry(fixture,phase):
    x=fixture;x.errors.add(phase)
    with pytest.raises(RuntimeError):x.execute()
    assert (x.a.STATE/'action/failure.json').is_file()
    before=list(x.calls)
    with pytest.raises(ValueError,match='already attempted'):x.execute()
    assert x.calls==before


@pytest.mark.parametrize('prefix',[
    ['relative/python','-B','relative/source','prepare'],['python','-u','source','prepare'],['python','-B','source','guest']])
def test_wrong_outer_argv_fails_before_state(fixture,prefix):
    x=fixture;x.owner['command']=prefix
    with pytest.raises(ValueError,match='absolute literal'):x.execute()
    assert not x.a.STATE.exists() and not x.calls


@pytest.mark.parametrize('field,value',[('production_activation_after_level4_completion',False),('capacity_after_mandatory_burn_claimed',True),('status','draft'),('blocking_findings',['block'])])
def test_final_review_contract_rejects_unqualified_activation(fixture,field,value):
    x=fixture;mutate(x.a.REVIEW,lambda v:v.update({field:value}))
    with pytest.raises(ValueError):x.execute()
    assert not x.a.STATE.exists() and not x.calls


@pytest.mark.parametrize('field,value',[('difficulty_matched',False),('model_label','14b'),('level','level5'),('test_split','dev')])
def test_actual_level4_completion_required_even_with_repin(fixture,field,value):
    x=fixture;mutate(x.a.L4_ADMISSION,lambda v:v.update({field:value}));mutate(x.a.INVENTORY,lambda v:v.update(level4_admission_sha256=x.a.sha(x.a.L4_ADMISSION)));x.repin()
    with pytest.raises(ValueError,match='Level4'):x.execute()
    assert not x.calls


@pytest.mark.parametrize('field',['identities','prompt_sha256'])
def test_burn_exact_count_and_distinctness(fixture,monkeypatch,field):
    x=fixture;mutate(x.a.BURN,lambda v:v[field].__setitem__(1,v[field][0]));monkeypatch.setattr(x.a,'BURN_SHA',x.a.sha(x.a.BURN))
    mutate(x.a.QUALIFICATION,lambda v:v.update(scratch_exclusions_sha256=x.a.BURN_SHA,files_sha256={str(x.dependency):x.a.sha(x.dependency),str(x.a.BURN):x.a.BURN_SHA}));monkeypatch.setattr(x.a,'QUALIFICATION_SHA',x.a.sha(x.a.QUALIFICATION))
    with pytest.raises(ValueError,match='duplicate'):x.a.mandatory_burn()


def test_native_history_snapshot_need_not_contain_external_scratch_package(fixture):
    x=fixture;x.execute();history=x.a.read(x.a.REVISION_ROOT/'exclusions/development.json')
    assert str(x.a.BURN) not in history['files_sha256']
    assert x.a.read(x.a.REVISION_ROOT/'protocol.json')['files_sha256'][str(x.a.BURN)]==x.a.BURN_SHA


@pytest.mark.parametrize('field',['native_discovery_complete','known_exposed_sources_complete'])
def test_incomplete_fresh_inventory_fails_before_claim(fixture,field):
    x=fixture;mutate(x.a.INVENTORY,lambda v:v.update({field:False}));x.repin()
    with pytest.raises(ValueError,match='inventory'):x.execute()
    assert not x.calls


def test_current_history_drift_fails_before_native_amendment(fixture):
    x=fixture;x.native.original.history=lambda d,r:(set(),set(),{})
    with pytest.raises(ValueError,match='inventory changed'):x.execute()
    assert not (x.a.STATE/'amendment.json').exists() and not x.calls


@pytest.mark.parametrize('field,value',[('native_row_certification',False),('new_fit_decisions',True),('heldout_performed',True),('capacity_after_mandatory_burn_claimed',True)])
def test_successful_certificate_scope_cannot_expand(fixture,field,value):
    x=fixture;x.execute();mutate(x.a.STATE/'certificate.json',lambda v:v.update({field:value}))
    with pytest.raises(ValueError):x.a.verify(x.a.STATE,guest=False)


def test_future_development_receipts_do_not_change_readiness(fixture):
    x=fixture;x.execute();v=x.a.development_inputs(x.a.STATE,guest=False);task=v['cell']['tasks'][0]
    write(task['output'],{'growing':True});write(Path(task['output']+'.batches')/'one.json',{})
    assert x.a.development_inputs(x.a.STATE,guest=False)==v and x.calls==['register','materialize_pools','launch_inputs']


@pytest.mark.parametrize('kind',['relative','outside','self','hardlink'])
def test_preservation_must_be_independent_canonical_copy(fixture,kind):
    x=fixture;p=x.a.PRESERVATION;v=x.a.read(p);entry=v['files'][0]
    if kind=='relative':entry['preserved_path']='relative.py'
    elif kind=='outside':entry['preserved_path']=str(x.dependency)
    elif kind=='self':entry['preserved_path']=entry['original_path']
    else:
        target=Path(entry['preserved_path']);target.unlink();os.link(entry['original_path'],target)
    write(p,v);x.repin()
    with pytest.raises(ValueError):x.execute()
    assert not x.calls


def test_production_source_has_only_native_preparation_mutators_and_no_historical_mutations():
    p=ROOT/'artifacts/register_modebench_scale_level5_pantry_r3_20260913.py';tree=ast.parse(p.read_text())
    names={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='revision'}
    assert {'register','materialize_pools','launch_inputs'}<=names
    assert not names&{'fit_domain','freeze','confirm','admit','generate'}
    assert not any(isinstance(n,(ast.Assign,ast.AnnAssign)) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('_first','_history') for t in getattr(n,'targets',[])) for n in ast.walk(tree))


@pytest.mark.parametrize('name',['STATE','REVISION_ROOT'])
def test_dangling_destination_alias_rejected_before_native(fixture,name):
    x=fixture;getattr(x.a,name).symlink_to(getattr(x.a,name).parent/'absent-target')
    with pytest.raises(ValueError):x.execute()
    assert not x.calls


@pytest.mark.parametrize('kind',['uid','euid','socket','uname','view'])
def test_truthful_actual_host_uid_and_source_view_required(fixture,kind):
    x=fixture
    if kind=='uid':x.a.os.getuid=lambda:1
    elif kind=='euid':x.a.os.geteuid=lambda:1
    elif kind=='socket':x.a.socket.gethostname=lambda:'spin.cs.princeton.edu'
    elif kind=='uname':x.a.os.uname=lambda:SimpleNamespace(nodename='spin.cs.princeton.edu')
    else:x.canonical.write_text('changed')
    with pytest.raises(ValueError):x.execute()
    assert not x.a.STATE.exists() and not x.calls


@pytest.mark.parametrize('kind',['identity','prompt','earlier_tier'])
def test_rehashed_native_rows_cannot_reuse_scratch_or_previous_tier(fixture,kind):
    x=fixture;x.execute();q=x.native.paths(x.a.REVISION_ROOT);path=q['pools']/'difficulty_1.jsonl';rows=x.native.rows_from_jsonl(path)
    if kind=='identity':rows[0]['id']=x.raw[0][1]
    elif kind=='prompt':
        x.a.mandatory_burn=lambda:({tuple(v) for v in x.raw},{digest(rows[0]['problem'])},dict(x.a.read(x.a.REVISION_ROOT/'protocol.json')['files_sha256']))
    else:rows[0]['id']=x.native.rows_from_jsonl(q['pools']/'difficulty_0.jsonl')[0]['id']
    path.chmod(0o644);path.write_text('\n'.join(json.dumps(v) for v in rows)+'\n')
    mutate(q['pools']/'identity.json',lambda v:v['tiers']['1'].update(rows_sha256=digest(rows)))
    with pytest.raises(ValueError,match='disjoint'):
        x.a.native_outputs(x.native,x.a.read(x.a.REVISION_ROOT/'protocol.json'),x.a.read(x.a.STATE/'development_cell.json'))


@pytest.mark.parametrize('key',['complete_scratch_exclusions','pantry_menu_revision3_profile','unchanged_original_prompt'])
def test_original_native_certification_flags_cannot_be_waived(fixture,key):
    x=fixture;x.execute();q=x.native.paths(x.a.REVISION_ROOT)
    mutate(q['pools']/'identity.json',lambda v:v['tiers']['0']['verification'].update({key:False}))
    with pytest.raises(ValueError,match='certification'):
        x.a.native_outputs(x.native,x.a.read(x.a.REVISION_ROOT/'protocol.json'),x.a.read(x.a.STATE/'development_cell.json'))


def test_final_review_must_cover_complete_historical_closure(fixture,monkeypatch):
    x=fixture;p=x.a.STATE.parent/'omitted_history.json';write(p,{'exact_history':True})
    monkeypatch.setattr(x.a,'historical_closure',lambda:{str(p):x.a.sha(p)})
    with pytest.raises(ValueError,match='complete actual negative'):x.execute()
    assert not x.calls


@pytest.fixture
def history(fixture,monkeypatch):
    x=fixture;a=x.a;dep={str(x.dependency):a.sha(x.dependency)}
    negative={'status':'needs_new_development_revision','failed_domains':['pantry'],
        'fits':[{'domain':'pantry','development_fit_pass':False},{'domain':'python_factors','development_fit_pass':True}]}
    write(a.NEGATIVE_STATE/'action/result.json',negative);write(a.NEGATIVE_STATE/'registration.json',{'files_sha256':dep})
    write(a.NEGATIVE_RECIPE,{'domain':'pantry','level':'level5','revision':2,'development_fit_pass':False,'input_sha256':dep})
    monkeypatch.setattr(a,'FAILED_RECIPE_SHA',a.sha(a.NEGATIVE_RECIPE));monkeypatch.setattr(a,'FIT_RESULT_SHA',a.sha(a.NEGATIVE_STATE/'action/result.json'))
    monkeypatch.setattr(a,'NEGATIVE_PINS',{str(p):a.sha(p) for p in (a.NEGATIVE_RECIPE,a.NEGATIVE_STATE/'action/result.json',a.NEGATIVE_STATE/'registration.json')})
    calls=[]
    def verify(root,guest):calls.append('readonly_negative');assert root==a.NEGATIVE_STATE;return deepcopy(negative)
    monkeypatch.setattr(a,'load_module',lambda p,n:SimpleNamespace(verify_existing=verify))
    retained={};certificates=[];manifest={'sources':{},'files_sha256':dep}
    for domain in ('countdown','mathir','python_factors'):
        root=a.STATE.parent/('old_'+domain);recipe=root/'level5/recipes'/(domain+'.json');identity=root/'level5/dataset'/domain/'identity.json'
        write(recipe,{'development_fit_pass':True,'input_sha256':dep})
        write(identity,{'level':'level5','domain':domain,'recipe_sha256':a.sha(recipe),'splits':{s:{'rows':n} for s,n in [('train',384),('dev',128),('eval',128)]}})
        source={'source_root':str(root),'source_kind':'campaign_v1' if domain=='countdown' else 'domain_revision_v1',
            'identity_sha256':a.sha(identity),'recipe_sha256':a.sha(recipe)};retained[domain]=source;manifest['sources'][domain]=source
        if domain!='countdown':
            cert=root/'freeze/certificate.json';write(cert,{'frozen':[{'domain':domain}],'files_sha256':{**dep,str(identity):a.sha(identity)}});certificates.append(str(cert))
    write(a.L5_MANIFEST,manifest);monkeypatch.setattr(a,'RETAINED',retained);monkeypatch.setattr(a,'RETAINED_CERTIFICATES',certificates)
    monkeypatch.setattr(a,'RETAINED_PINS',{str(p):a.sha(p) for p in [a.L5_MANIFEST,*map(Path,certificates)]})
    return SimpleNamespace(x=x,a=a,negative=negative,calls=calls)


def test_immediate_negative_and_three_retained_sources_read_only(history):
    h=history;entries,pins=h.x.real_inputs(h.a.STATE,guest=False)
    assert entries['pantry']['recipe_sha256']==h.a.FAILED_RECIPE_SHA
    assert set(entries['_preserved'])=={'countdown','mathir','python_factors'} and h.calls==['readonly_negative']
    closure=h.x.real_history();assert closure and all(p in pins for p in closure)
    assert not h.x.calls and not h.a.REVISION_ROOT.exists()


@pytest.mark.parametrize('kind',['positive_pantry','lost_python_pass','changed_recipe','recipe_revision'])
def test_immediate_failed_pantry_and_literal_python_pass_are_required(history,kind):
    h=history
    if kind=='positive_pantry':h.negative['fits'][0]['development_fit_pass']=True
    elif kind=='lost_python_pass':h.negative['fits'][1]['development_fit_pass']=False
    elif kind=='changed_recipe':mutate(h.a.NEGATIVE_RECIPE,lambda v:v.update(development_fit_pass=True))
    else:mutate(h.a.NEGATIVE_RECIPE,lambda v:v.update(revision=1))
    with pytest.raises(ValueError):h.a.negative_predecessor(guest=False)
    assert not h.x.calls


@pytest.mark.parametrize('domain',['countdown','mathir','python_factors'])
@pytest.mark.parametrize('kind',['identity','recipe','manifest'])
def test_each_retained_source_is_exact(history,domain,kind):
    h=history;root=Path(h.a.RETAINED[domain]['source_root'])
    if kind=='identity':mutate(root/'level5/dataset'/domain/'identity.json',lambda v:v['splits']['train'].update(rows=383))
    elif kind=='recipe':mutate(root/'level5/recipes'/(domain+'.json'),lambda v:v.update(development_fit_pass=False))
    else:mutate(h.a.L5_MANIFEST,lambda v:v['sources'][domain].update(source_root='wrong'))
    with pytest.raises(ValueError):h.a.retained_sources(guest=False)
    assert not h.x.calls


def test_future_admission_gate_matches_registered_level4_publication_destination():
    import ast
    def constant_path(file,name):
        tree=ast.parse(file.read_text())
        values=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        assert len(values)==1 and isinstance(values[0],ast.BinOp) and isinstance(values[0].op,ast.Div)
        return Path(values[0].right.value)
    gate=constant_path(ROOT/'artifacts/register_modebench_scale_level5_pantry_r3_20260913.py','L4_ADMISSION')
    for filename in ('continue_modebench_scale_level4_first_release_r4_20260913.py','complete_modebench_scale_level4_first_r4_20260913.py'):
        release=constant_path(ROOT/'artifacts'/filename,'RELEASE')
        assert gate==release/'level4/admission.json'
