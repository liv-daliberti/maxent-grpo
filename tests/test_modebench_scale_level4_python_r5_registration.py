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
    spec=importlib.util.spec_from_file_location('scratch_python_preparation',ROOT/'artifacts/register_modebench_scale_level4_python_r5_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    tmp=f.state.parent;state=tmp/'preparation';revision_root=tmp/'python_r5';parent=tmp/'parent'
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
        'QUALIFICATION_EXECUTION':tmp/'qualification_execution.json'}.items():monkeypatch.setattr(a,key,value,raising=False)
    a.CANDIDATE.write_text('four fixed candidate laws');a.CANDIDATE_TESTS.write_text('structural tests')
    monkeypatch.setattr(a,'CANDIDATE_SHA',a.sha(a.CANDIDATE),raising=False)
    write(a.QUALIFICATION,{'candidate_sha256':a.CANDIDATE_SHA,'rows':8,'native_original_witnesses':16,'target_registration_performed':False})
    write(a.QUALIFICATION_EXECUTION,{'returncode':0,'candidate_sha256_before':a.CANDIDATE_SHA,'candidate_sha256_after':a.CANDIDATE_SHA})
    monkeypatch.setattr(a,'QUALIFICATION_SHA',a.sha(a.QUALIFICATION),raising=False);monkeypatch.setattr(a,'QUALIFICATION_EXECUTION_SHA',a.sha(a.QUALIFICATION_EXECUTION),raising=False)
    sources={e['domain']:{'source_kind':'domain_revision_v1','source_root':e['source_root']} for e in entries}
    for domain in ('countdown','mathir'):
        root=tmp/('carried_'+domain);dataset=z.paths(root,'level4',domain)['dataset']
        write(dataset/'identity.json',{'splits':{s:{'rows':n} for s,n in [('train',384),('dev',128),('eval',128)]}})
        sources[domain]={'source_kind':'campaign_v1','source_root':str(root)}
    write(a.L4_MANIFEST,{'level':'level4','model_label':'7b','sources':sources});monkeypatch.setattr(a,'L4_MANIFEST_SHA',a.sha(a.L4_MANIFEST),raising=False)
    retained=[]
    def frozen(root,level,domain,kind):
        retained.append(domain);assert level=='level4'
        return a.read(z.paths(root,level,domain)['dataset']/'identity.json')
    retained_core=SimpleNamespace(domain_paths=z.paths,frozen_identity=frozen)
    profiles=[{'cases':4 if t==0 else 6,'base_cases':4,'ordinary_cases':3,'ordinary_minimum':48,'ordinary_maximum':1000,
        'ordinary_maximum_smallest_factor':5,'ordinary_minimum_proper_divisors':2,'semiprime_cases':1,'semiprime_least_prime':(11,11,13,17)[t],
        'semiprime_distinct_primes':True,'semiprime_larger_prime_ratio_maximum':3,'semiprime_maximum':1000,'base_gcd':1,
        'appended_prime_squares':([],[25,49],[25,49],[25,49])[t],'appended_support_multiplier':1,
        'sampling':'uniform_four_case_bases_conditioned_on_exact_support_base_gcd_and_projection_exclusions'} for t in range(4)]
    parent_value={'models':{'7b':{'path':'literal checkpoint'}},'split_sizes':{'train':384,'dev':128,'eval':128},
        'targets':{'python_factors':{'metrics':{'pass_at_1':.3}},'mathir':{}},
        'histograms':{'python_factors':{s:[{'cell':[4],'rows':2}] for s in ('train','dev','eval')},'mathir':{}},
        'tolerances':{'pass_at_1':.04},'selection_seed':6491701,'sampling':{'unchanged':True},'fit':{'fixed_grid':True}}
    write(parent/'protocol.json',parent_value)
    candidate_dep=tmp/'candidate_dependency.py';candidate_dep.write_text('fixed dependency')
    cpins={str(p):a.sha(p) for p in (a.CANDIDATE,a.CANDIDATE_TESTS,candidate_dep)}
    history_state=tmp/'history_r3';retained_records={}
    for d in ('countdown','graph_coloring','mathir','pantry'):
        source=sources[d];dataset=z.paths(Path(source['source_root']),'level4',d)['dataset']
        retained_records[d]={'source':source,'dataset':str(dataset),'identity_sha256':a.sha(dataset/'identity.json'),
            'splits':a.read(dataset/'identity.json')['splits']}
    write(history_state/'certificate.json',{'preserved_level4_sources':retained_records})
    monkeypatch.setattr(a,'_history',SimpleNamespace(STATE=history_state,TESTS=f.a.TESTS,REVIEW=f.a.REVIEW,
        development_inputs=lambda root,guest:{'files_sha256':{str(f.source):a.sha(f.source),str(history_state/'certificate.json'):a.sha(history_state/'certificate.json')}}))
    for key,path in {'CAPACITY':tmp/'capacity.json','CAPACITY_EXECUTION':tmp/'capacity_exit.json','PRESERVATION':tmp/'preserved/manifest.json',
        'NEGATIVE_FIT':tmp/'negative_fit.py','NEGATIVE_STATE':tmp/'negative_fit','NEGATIVE_RECIPE':tmp/'r3/level4/recipes/python_factors.json',
        'READONLY_EXIT':tmp/'negative_readonly.json'}.items():monkeypatch.setattr(a,key,path)
    a.NEGATIVE_FIT.write_text('scratch exact negative verifier')
    negative_result={'schema':'modebench_scale_level4_python_r4_fit_v1','status':'needs_new_development_revision',
        'failed_domains':['python_factors'],'fits':[{'development_fit_pass':False}]}
    write(a.NEGATIVE_STATE/'action/result.json',negative_result)
    write(a.NEGATIVE_STATE/'registration.json',{'files_sha256':{str(f.source):a.sha(f.source)}})
    write(a.NEGATIVE_RECIPE,{'development_fit_pass':False,'input_sha256':{str(f.source):a.sha(f.source)}})
    write(a.READONLY_EXIT,{'returncode':0,'inputs_unchanged':True,'verified_summary':{k:negative_result[k] for k in ('schema','status','failed_domains')}})
    monkeypatch.setattr(a,'FAILED_RECIPE_SHA',a.sha(a.NEGATIVE_RECIPE));monkeypatch.setattr(a,'FIT_RESULT_SHA',a.sha(a.NEGATIVE_STATE/'action/result.json'))
    monkeypatch.setattr(a,'NEGATIVE_PINS',{str(p):a.sha(p) for p in (a.NEGATIVE_FIT,a.NEGATIVE_RECIPE,a.READONLY_EXIT,a.NEGATIVE_STATE/'action/result.json',a.NEGATIVE_STATE/'registration.json')})
    write(a.CAPACITY,{'schema':'modebench_scale_python_r5_fresh_exact_capacity_v1','status':'fresh_structural_capacity_passed',
        'blocking_capacity':[],'all_inputs_and_source_inventory_unchanged':True,'support_cells':60,'primes':[11,11,13,17],
        'extras':[[],[25,49],[25,49],[25,49]],'profiles':a.normalized_capacity_profiles(profiles),'files_sha256':{str(candidate_dep):a.sha(candidate_dep)}})
    write(a.CAPACITY_EXECUTION,{'returncode':0})
    write(a.QUALIFICATION,{'schema':'modebench_scale_python_r5_native_qualification_v2_v1','status':'passed','candidate_sha256':a.sha(a.CANDIDATE),
        'tiers':[0,1,2,3],'native_original_prompt_unchanged':True,'native_row_certification_passed':True,'target_registration_performed':False,'rows':8,'original_row_calls':8,'native_witness_calls':16,
        'all_emitted_rows_native_discoverable':True})
    pilot_dir=tmp/'pilot_scratch/pools/python_factors';pilot_dir.mkdir(parents=True,exist_ok=True)
    pilot_cases=[[31+i,53,67,89] for i in range(6)]
    pilot_files=[]
    for k in range(2):
        pilot_path=pilot_dir/f'pilot{k}.jsonl'
        pilot_path.write_text('\n'.join(json.dumps({'answer':json.dumps({'cases':c})}) for c in pilot_cases[k*3:(k+1)*3])+'\n')
        pilot_files.append({'original_path':str(pilot_path),'burned_path':str(pilot_path),'sha256':a.sha(pilot_path),'rows':3})
    pilot_manifest=tmp/'pilot_burn_manifest.json'
    write(pilot_manifest,{'schema':'modebench_scale_python_r5_pilot_scratch_burn_v1','total_rows':6,'files':pilot_files})
    monkeypatch.setattr(a,'PILOT_BURN',pilot_manifest);monkeypatch.setattr(a,'PILOT_BURN_SHA',a.sha(pilot_manifest))
    monkeypatch.setattr(a,'PILOT_ROWS',6);monkeypatch.setattr(a,'PILOT_FILES',2)
    pilot_ids=[['python_factors',c] for c in pilot_cases]
    pilot_pins={x['burned_path']:x['sha256'] for x in pilot_files}
    burn_paths=[tmp/'scratch_rows/qualification_burn.jsonl',tmp/'scratch_rows/qualification_rows.jsonl']
    for p in burn_paths:p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic qualified rows\n')
    burned_ids=([['python_factors',[101+i,211,307,401]] for i in range(2)]
        +[['python_factors',[101+i,211,307,401,25,49]] for i in range(2,8)])
    burned_prompts=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(8)]
    change(a.QUALIFICATION,lambda v:v.update(identities=burned_ids,prompt_sha256=burned_prompts,
        burn_rows_path=str(burn_paths[0]),qualified_rows_path=str(burn_paths[1]),files_sha256={str(p):a.sha(p) for p in burn_paths}))
    write(a.QUALIFICATION_EXECUTION,{'returncode':0,'qualification_sha256':a.sha(a.QUALIFICATION),
        'candidate_sha256_before':a.sha(a.CANDIDATE),'candidate_sha256_after':a.sha(a.CANDIDATE)})
    preserved=[]
    for p in (a.CANDIDATE,a.CANDIDATE_TESTS):
        q=a.PRESERVATION.parent/p.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());q.chmod(0o444)
        preserved.append({'original_path':str(p),'preserved_path':str(q),'sha256':a.sha(p)})
    write(a.PRESERVATION,{'schema':'modebench_scale_python_r5_code_preservation_v1','status':'preserved_before_scientific_registration','files':preserved})
    write(a.CANDIDATE_REVIEW,{'schema':'modebench_scale_python_r5_candidates_independent_review_v3','status':'reviewed','files_sha256':cpins})
    def repin_review():
        pins=dict(f.a.read(f.a.REVIEW)['files_sha256']);pins.update(cpins);pins.update(a.NEGATIVE_PINS)
        required=(a.SOURCE,a.TESTS,a.HISTORY,a._history.TESTS,a._history.REVIEW,a.CANDIDATE,a.CANDIDATE_TESTS,a.CANDIDATE_REVIEW,
            a.CAPACITY,a.CAPACITY_EXECUTION,a.QUALIFICATION,a.QUALIFICATION_EXECUTION,a.PRESERVATION,a.NEGATIVE_FIT)
        pins.update({str(p):a.sha(p) for p in required});pins.update(a.read(a.QUALIFICATION)['files_sha256']);pins.update({x['preserved_path']:x['sha256'] for x in preserved})
        value={'schema':'modebench_scale_level4_python_r5_registration_independent_review_v1','status':'reviewed','files_sha256':pins}
        value.update({field:a.sha(p) for field,p in [('candidate_sha256',a.CANDIDATE),('candidate_tests_sha256',a.CANDIDATE_TESTS),
            ('capacity_sha256',a.CAPACITY),('capacity_execution_sha256',a.CAPACITY_EXECUTION),('native_qualification_sha256',a.QUALIFICATION),
            ('native_qualification_execution_sha256',a.QUALIFICATION_EXECUTION),('code_preservation_sha256',a.PRESERVATION)]})
        write(a.REVIEW,value)
    repin_review()
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
        return {'protocol':root/'protocol.json','pools':root/'level4/pools/python_factors',
            'development':root/'level4/results/development/python_factors'}
    def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
    def generation_seed(p,split,tier):return 9000+tier
    def register(root,level,domain,**kwargs):
        calls.append('register');assert f.held==[8] and f.canonical.read_text()=='old'
        assert root==revision_root and level=='level4' and domain=='python_factors'
        assert kwargs=={'parent':parent,'revision':5,'candidate_module':a.CANDIDATE_MODULE}
        assert a.read(state/'amendment.json')['actual_r4_predecessor']==a.negative_predecessor(guest=True)[0]
        if 'register_before' in errors:raise RuntimeError('native register exception')
        value={k:deepcopy(v) for k,v in parent_value.items() if k not in ('targets','histograms')}
        value.update(schema='native_revision_schema',root=str(root),parent_root=str(parent),level=level,domain=domain,revision=5,
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
            'files_sha256':{str(f.source):a.sha(f.source),**{str(p):a.sha(p) for p in burn_paths},**pilot_pins},
            'identities':pilot_ids+burned_ids,'prompt_sha256':burned_prompts})
        tiers={}
        for tier in range(4):
            batch=[{'scale_candidate_tier':tier,'answer_mode_count':4,'case':i,'answer':json.dumps({'cases':[6,10,14,77]+profiles[tier]['appended_prime_squares']})} for i in range(2)]
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
        return {'id':'level4_python_factors_r5_dev','level':'level4','domain':'python_factors','phase':'dev','model_label':'7b',
            'source_kind':'domain_revision_v1','source_root':str(root),'tasks':tasks},{str(f.source):a.sha(f.source)}
    native=SimpleNamespace(SCHEMA='native_revision_schema',candidate_provider=lambda name:SimpleNamespace(PROFILES={'python_factors':profiles}),
        register=register,materialize_pools=pools,launch_inputs=launch,paths=paths,rows_from_jsonl=rows,sha=object_sha,generation_seed=generation_seed,labels=lambda level,domain,revision,phase:list(a.LABELS[phase]),
        original=SimpleNamespace(deserialize_cells=lambda h:Counter({tuple(v['cell']):v['rows'] for v in h}),union_histogram=lambda h:next(iter(h.values()))),
        cell_histogram=lambda domain,rs:Counter({(4,):len(rs)}),evaluator=SimpleNamespace(INTERFACE='native interface'))
    def authenticate(path):assert f.canonical.read_text()=='neutral';authentications.append(path)
    def load(path,name):
        if path==a.CANDIDATE:return SimpleNamespace(PRIMES=(11,11,13,17),EXTRAS=((),(25,49),(25,49),(25,49)),PROFILES={'python_factors':profiles})
        if path==a.NEGATIVE_FIT:return SimpleNamespace(verify_existing=lambda root,guest:deepcopy(negative_result))
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
        candidate_dep=candidate_dep,authentications=authentications,freeze=z,retained=retained,repin_review=repin_review,negative_result=negative_result,execute=lambda:a.run(state,a.sha(a.REVIEW)))



def test_only_register_four_pools_and_tasks_preserve_all_passing_sources_and_old_targets(fixture):
    x=fixture;before={e['domain']:x.a.sha(e['recipe_path']) for e in x.entries};v=x.execute()
    assert x.calls==['register','materialize_pools','launch_inputs']
    assert v['status']=='ready_for_fresh_python_r5_development'
    assert set(v['preserved_level4_sources'])=={'countdown','graph_coloring','mathir','pantry'}
    assert not v['source_binding_performed'] and not v['new_fit_decisions'] and not v['heldout_performed']
    assert x.freeze.calls==['graph_coloring','pantry'] and x.f.calls==['graph_coloring','pantry','python_factors']
    assert {e['domain']:x.a.sha(e['recipe_path']) for e in x.entries}==before
    p=x.a.read(x.a.REVISION_ROOT/'protocol.json');parent=x.a.read(x.a.PARENT/'protocol.json')
    assert p['targets']['python_factors']==parent['targets']['python_factors']
    assert p['draw_labels']=={'dev':[7205000,7205001,7205002,7205003],'eval':[7205500,7205501,7205502,7205503]}
    assert not (x.state/'source_binding.json').exists()


def test_readonly_provider_has_exact7b_inputs_and_accepts_future_receipt_growth(fixture,monkeypatch):
    x=fixture;x.execute();before=list(x.calls)
    class ForeignLock:
        def stat(self):raise AssertionError('historical proof cannot use worker NFS device')
    monkeypatch.setattr(x.a,'LOCK',ForeignLock());monkeypatch.setattr(x.freeze.a,'LOCK',ForeignLock())
    value=x.a.development_inputs(x.state,guest=False)
    assert value['models']=={'7b':'literal checkpoint'} and value['cell']['id']=='level4_python_factors_r5_dev'
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
    path=ROOT/'artifacts/register_modebench_scale_level4_python_r5_20260913.py';tree=ast.parse(path.read_text())
    attrs={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not attrs & {'bind_level','freeze_dataset','fit_domain','submit','audit'}
    source=path.read_text();assert 'level4_python_factors_r5' in source and 'level4_python_coloring_r5' not in source


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


@pytest.mark.parametrize('kind',['capacity_status','capacity_inventory','capacity_profile','capacity_exit','native_status','native_row','native_exit','native_registration','preserved_copy','preserved_mode','candidate_review'])
def test_actual_candidate_evidence_required_even_if_review_rehashed(fixture,kind):
    x=fixture;a=x.a
    if kind=='capacity_status':change(a.CAPACITY,lambda v:v.update(status='draft'))
    elif kind=='capacity_inventory':change(a.CAPACITY,lambda v:v.update(all_inputs_and_source_inventory_unchanged=False))
    elif kind=='capacity_profile':change(a.CAPACITY,lambda v:v.update(profiles=[]))
    elif kind=='capacity_exit':change(a.CAPACITY_EXECUTION,lambda v:v.update(returncode=143))
    elif kind=='native_status':change(a.QUALIFICATION,lambda v:v.update(status='synthetic'))
    elif kind=='native_row':change(a.QUALIFICATION,lambda v:v.update(native_row_certification_passed=False))
    elif kind=='native_exit':change(a.QUALIFICATION_EXECUTION,lambda v:v.update(returncode=143))
    elif kind=='native_registration':change(a.QUALIFICATION,lambda v:v.update(target_registration_performed=True))
    elif kind.startswith('preserved'):
        p=Path(a.read(a.PRESERVATION)['files'][0]['preserved_path']);p.chmod(0o644)
        if kind=='preserved_copy':p.write_text('changed');p.chmod(0o444)
    else:change(a.CANDIDATE_REVIEW,lambda v:v.update(status='draft'))
    x.repin_review()
    with pytest.raises(ValueError):x.execute()
    assert not x.calls and not x.state.exists()


@pytest.mark.parametrize('kind',['recipe','result','registration','readonly'])
def test_exact_negative_r3_history_cannot_be_relabelled(fixture,kind):
    x=fixture;a=x.a;p={'recipe':a.NEGATIVE_RECIPE,'result':a.NEGATIVE_STATE/'action/result.json',
        'registration':a.NEGATIVE_STATE/'registration.json','readonly':a.READONLY_EXIT}[kind]
    change(p,lambda v:v.update(changed=True));x.repin_review()
    with pytest.raises(ValueError):x.execute()
    assert not x.calls and not x.state.exists()


@pytest.mark.parametrize('kind',['relative_python','relative_source','wrong_action','extra_option'])
def test_absolute_command_required_before_any_input_or_mutation(fixture,monkeypatch,kind):
    x=fixture;command=x.f.owner['command']
    if kind=='relative_python':command[0]='var/seed_paper_eval/paper310/bin/python'
    elif kind=='relative_source':command[2]='artifacts/prep.py'
    elif kind=='wrong_action':command[3]='verify'
    else:command.insert(2,'-u')
    monkeypatch.setattr(x.a,'reviewed',lambda *args,**kwargs:pytest.fail('must stop before review'))
    with pytest.raises(ValueError,match='absolute literal'):x.execute()
    assert not x.state.exists() and not x.calls


def test_actual_local_identity_guard_retains_literal_historical_module():
    spec=importlib.util.spec_from_file_location('_r5_live_guard',ROOT/'artifacts/register_modebench_scale_level4_python_r5_20260913.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    assert a.HOST=='soak.cs.princeton.edu' and a.UID==363432
    # The r4 predecessor ran on soak; the original first module it chains to remains spin.
    assert a._history.HOST=='soak.cs.princeton.edu'
    assert a._history._first.HOST=='spin.cs.princeton.edu'
    assert a.assert_fence is a._history.assert_fence and a.lifetime_fence is a._history.lifetime_fence


@pytest.mark.parametrize('field',['semiprime_cases','sampling','ordinary_minimum','ordinary_maximum','ordinary_maximum_smallest_factor','base_gcd','appended_support_multiplier'])
def test_capacity_semantics_are_compared_without_schema_aliasing(fixture,field):
    x=fixture;candidate=x.a.load_module(x.a.CANDIDATE,'test')
    candidate.PROFILES['python_factors'][0][field]='wrong'
    with pytest.raises(ValueError):x.execute()
    assert not x.calls and not x.state.exists()


@pytest.mark.parametrize('kind',['original_alias','outside_directory','hardlink'])
def test_preservation_requires_distinct_canonical_copy(fixture,kind):
    x=fixture;a=x.a;value=a.read(a.PRESERVATION);entry=value['files'][0];original=Path(entry['original_path']);copy=Path(entry['preserved_path'])
    if kind=='original_alias':original.chmod(0o444);entry['preserved_path']=str(original)
    elif kind=='outside_directory':
        out=x.state.parent/'outside.py';out.write_bytes(copy.read_bytes());out.chmod(0o444);entry['preserved_path']=str(out)
    else:
        copy.unlink();original.chmod(0o444);copy.hardlink_to(original)
    write(a.PRESERVATION,value);x.repin_review()
    with pytest.raises(ValueError):x.execute()
    assert not x.calls and not x.state.exists()


@pytest.mark.parametrize('kind',['identity','file'])
def test_native_preparation_must_burn_every_model_exposed_pilot_row(fixture,monkeypatch,kind):
    """A pilot prompt escaping the snapshot is re-emittable, and under the problem-aligned
    seed policy the same draw label would reproduce the identical sample block."""
    x=fixture;original=x.native.materialize_pools
    def changed(root):
        value=original(root);snapshot=root/'exclusions/development.json'
        manifest=x.a.read(x.a.PILOT_BURN)
        if kind=='identity':change(snapshot,lambda v:v['identities'].pop(0))
        else:change(snapshot,lambda v:v['files_sha256'].pop(manifest['files'][0]['burned_path']))
        value['exclusions_sha256']=x.a.sha(snapshot);write(x.paths(root)['pools']/'identity.json',value);return value
    monkeypatch.setattr(x.native,'materialize_pools',changed)
    with pytest.raises(ValueError,match='pilot'):x.execute()
    assert not (x.state/'certificate.json').exists()


@pytest.mark.parametrize('kind',['identity','prompt','burn_file','qualified_file','projection'])
def test_native_preparation_must_burn_all_qualified_rows_before_certificate(fixture,monkeypatch,kind):
    x=fixture;original=x.native.materialize_pools
    def changed(root):
        value=original(root);snapshot=root/'exclusions/development.json'
        if kind=='identity':change(snapshot,lambda v:v['identities'].pop())
        elif kind=='prompt':change(snapshot,lambda v:v['prompt_sha256'].pop())
        elif kind.endswith('file'):
            field='burn_rows_path' if kind=='burn_file' else 'qualified_rows_path'
            change(snapshot,lambda v:v['files_sha256'].pop(x.a.read(x.a.QUALIFICATION)[field]))
        else:
            p=x.paths(root)['pools']/'difficulty_0.jsonl';rows=x.native.rows_from_jsonl(p)
            rows[0]['answer']=json.dumps({'cases':x.a.read(x.a.QUALIFICATION)['identities'][0][1]})
            p.write_text('\n'.join(json.dumps(row) for row in rows)+'\n');value['tiers']['0']['rows_sha256']=object_sha(rows)
        value['exclusions_sha256']=x.a.sha(snapshot);write(x.paths(root)['pools']/'identity.json',value);return value
    monkeypatch.setattr(x.native,'materialize_pools',changed)
    with pytest.raises(ValueError,match='qualified'):x.execute()
    assert not (x.state/'certificate.json').exists() and (x.state/'action/failure.json').exists()


# --- Real-prerequisite gates -------------------------------------------------
# Every other test in this file builds SYNTHETIC prerequisites, and a synthetic
# fixture naturally encodes what the code expects rather than what the sealed
# artifact actually holds. That blind spot let two fatal mismatches reach review:
# a stale r3 schema in the read-only summary check, and a capacity record stored
# in the raw profile shape the registration can never match. These tests read the
# real artifacts, so that class of defect fails here instead of at run time.

@pytest.fixture(scope='module')
def live():
    spec=importlib.util.spec_from_file_location('_live_reg',ROOT/'artifacts/register_modebench_scale_level4_python_r5_20260913.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_real_readonly_exit_matches_the_summary_the_registration_demands(live):
    exit_record=live.read(live.READONLY_EXIT)
    assert exit_record['returncode']==0 and exit_record['inputs_unchanged'] is True
    assert exit_record['verified_summary']=={'schema':'modebench_scale_level4_python_r4_fit_v1',
        'status':'needs_new_development_revision','failed_domains':['python_factors']}
    src=Path(live.SOURCE).read_text()
    assert 'modebench_scale_level4_python_r3_fit_v1' not in src


def test_real_capacity_profiles_are_stored_in_the_shape_the_registration_compares(live):
    capacity=live.read(live.CAPACITY)
    candidate=live.load_module(live.CANDIDATE,'_live_candidate_law')
    assert live.normalized_capacity_profiles(candidate.PROFILES['python_factors'])==capacity['profiles']
    assert capacity['primes']==[11,11,13,17] and capacity['extras']==[[],[25,49],[25,49],[25,49]]
    assert capacity['status']=='fresh_structural_capacity_passed' and capacity['blocking_capacity']==[]


def test_real_prerequisite_schemas_match_every_literal_the_registration_requires(live):
    assert live.read(live.CAPACITY)['schema']=='modebench_scale_python_r5_fresh_exact_capacity_v1'
    assert live.read(live.QUALIFICATION)['schema']=='modebench_scale_python_r5_native_qualification_v2_v1'
    assert live.read(live.PRESERVATION)['schema']=='modebench_scale_python_r5_code_preservation_v1'
    assert live.read(live.CANDIDATE_REVIEW)['schema']=='modebench_scale_python_r5_candidates_independent_review_v3'
    assert live.read(live.CAPACITY_EXECUTION)['returncode']==0
    assert live.read(live.QUALIFICATION_EXECUTION)['returncode']==0


def test_real_candidate_bytes_implement_the_registered_ladder(live):
    candidate=live.load_module(live.CANDIDATE,'_live_candidate_ladder')
    assert tuple(candidate.PRIMES)==(11,11,13,17)
    assert tuple(map(tuple,candidate.EXTRAS))==((),(25,49),(25,49),(25,49))


def test_every_scratch_pilot_row_is_natively_discoverable_so_generation_will_burn_it(live):
    """The pilots were scored by the 7b model. If generation cannot see them it can
    re-emit a scored prompt under the same draw label and reuse the exact sample block."""
    import modebench_scale_source_disjointness as discovery
    burn=ROOT/'var/data/modebench_scale_python_r5_pilot_scratch_20260913'
    manifest=live.read(burn/'burn_manifest.json')
    assert manifest['total_rows']==live.PILOT_ROWS==768 and len(manifest['files'])==live.PILOT_FILES==13
    live_sources=discovery.discover_sources(live.REVISION_ROOT,'level4','python_factors',ROOT/'var/data')
    # discover_sources globs **/<domain>/*.jsonl, but production generation globs
    # **/pools/<domain>/*.jsonl. A layout one level higher would satisfy the first and
    # leave generation blind, so assert the pattern generation actually uses.
    production=set((ROOT/'var/data').glob('**/pools/python_factors/*.jsonl'))
    for row in manifest['files']:
        assert Path(row['burned_path']).is_file() and not Path(row['burned_path']).is_symlink()
        assert live.sha(row['burned_path'])==row['sha256']==live.sha(row['original_path'])
        assert Path(row['burned_path']) in live_sources, row['burned_path']
        assert Path(row['burned_path']) in production, row['burned_path']
    assert live.PILOT_ROWS==768 and live.sha(live.PILOT_BURN)==live.PILOT_BURN_SHA
