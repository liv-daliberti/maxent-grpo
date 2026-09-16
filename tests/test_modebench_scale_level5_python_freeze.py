"""Scratch-only Python freeze/action proof; no real science or controller."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_ready_fits import fixture as prior_fixture
from test_modebench_scale_level5_recovered_fits import fixture as fit_fixture, write

ROOT=Path(__file__).resolve().parents[1]


def change(path,update):
    value=json.loads(Path(path).read_text());update(value);write(path,value)


@pytest.fixture
def fixture(fit_fixture,monkeypatch):
    f=fit_fixture.f;f.failures.add('pantry');fit_fixture.run()
    spec=importlib.util.spec_from_file_location('scratch_python_freeze',ROOT/'artifacts/freeze_modebench_scale_level5_python_20260912.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    entries=f.a.read(f.state/'registration.json')['revisions']
    f.roots=[Path(entry['source_root']) for entry in entries]
    for key,value in {'_first':f.a,'STATE':f.state,'MANIFEST':f.a.VIEW,'LOCK':f.a.first.LOCK,'LOCK_INODE':f.a.first.LOCK_INODE,
        'REVIEW':f.state.parent/'freeze_review.json','assert_fence':f.a.assert_fence,'lifetime_fence':f.a.lifetime_fence,
        'REGISTRATION_SHA':a.sha(f.state/'registration.json'),'FIT_RESULT_SHA':a.sha(f.state/'action/result.json'),
        'RECIPES':{e['domain']:a.sha(e['recipe_path']) for e in entries}}.items():monkeypatch.setattr(a,key,value)
    pins=f.a.read(f.a.REVIEW)['files_sha256'];pins.update({str(p):a.sha(p) for p in (a.SOURCE,a.TESTS,a.FIRST,f.a.TESTS,f.a.REVIEW,*f.a.SEALED)})
    write(a.REVIEW,{'schema':'modebench_scale_level5_python_freeze_independent_review_v1','status':'reviewed','files_sha256':pins})
    f.owner['command']=[str(a.PYTHON),'-B',str(a.SOURCE),'freeze'];active=[];calls=[];identities=[];crosses=[];errors=set();authentications=[]
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
    def paths(root,level,domain):
        return {'dataset':root/level/'dataset'/domain,'recipe':root/level/'recipes'/(domain+'.json'),
            'receipt':root/level/'results/confirmation'/(domain+'.json'),'audit':root/level/'confirmation'/(domain+'.json')}
    def freeze(root):
        domain=root.name;calls.append(domain)
        if 'before' in errors:raise RuntimeError('native freeze failed before dataset')
        assert domain=='python_factors' and f.held==[8] and f.canonical.read_text()=='old'
        q=paths(root,'level5',domain)
        write(root/'exclusions/freeze.json',{'complete_history':True})
        write(q['dataset']/'identity.json',{'splits':{s:{'rows':n,'rows_sha256':s} for s,n in [('train',384),('dev',128),('eval',128)]}})
        for split in ('train','dev','eval'):write(q['dataset']/(split+'.jsonl'),{'fixed':split})
        if 'after' in errors:raise RuntimeError('native freeze failed after dataset')
    def frozen(root,level,domain,kind):
        identities.append(domain);assert level=='level5' and domain=='python_factors' and kind=='domain_revision_v1'
        return a.read(paths(root,level,domain)['dataset']/'identity.json')
    def cross(root,level,domain):
        crosses.append(domain);return {'files_sha256':{str(f.source):a.sha(f.source)}}
    def authenticate(path):
        assert f.canonical.read_text()=='neutral';authentications.append(path)
    core=SimpleNamespace(domain_paths=paths,revision=SimpleNamespace(freeze_dataset=freeze),frozen_identity=frozen,verify_dataset=cross)
    monkeypatch.setattr(a,'load_module',lambda path,name:core if path==f.a.CORE else SimpleNamespace(authenticate=authenticate))
    dispatch={'code':0,'drop':False}
    def run(argv,**kwargs):
        assert kwargs['pass_fds']==(8,) and f.held==[8]
        active.append(argv);f.canonical.write_text('old')
        try:a.guest_action(f.state,8,a.sha(a.REVIEW))
        finally:f.canonical.write_text('neutral');active.clear()
        if dispatch['drop']:(a.area(f.state)/'action/result.json').unlink()
        return SimpleNamespace(returncode=dispatch['code'])
    monkeypatch.setattr(a,'subprocess',SimpleNamespace(run=run))
    return SimpleNamespace(a=a,f=f,entries=entries,calls=calls,identities=identities,crosses=crosses,errors=errors,dispatch=dispatch,
        paths=paths,authentications=authentications,execute=lambda:a.run(f.state,a.sha(a.REVIEW)))


def test_exact_python_freeze_preserves_pantry_failed_fit_and_all_source_recipes(fixture):
    x=fixture;before={e['domain']:x.a.sha(e['recipe_path']) for e in x.entries};result=x.execute()
    assert x.calls==x.identities==x.crosses==['python_factors']
    assert result['domains']==['python_factors'] and result['pantry_development_fit_pass'] is False
    assert result['preserved_pantry_recipe_sha256']==before['pantry'] and result['new_fit_publications']==0
    assert {e['domain']:x.a.sha(e['recipe_path']) for e in x.entries}==before
    assert x.f.calls==['pantry','python_factors'] and not x.f.held and x.f.canonical.read_text()=='neutral'
    assert not x.paths(x.f.roots[0],'level5','pantry')['dataset'].exists()
    assert {s:v['rows'] for s,v in result['frozen'][0]['splits'].items()}=={'train':384,'dev':128,'eval':128}


def test_static_verify_reuses_identity_without_new_fit_or_freeze(fixture):
    x=fixture;result=x.execute();assert x.a.verify(x.f.state,guest=False)==result
    assert x.calls==['python_factors'] and x.identities==['python_factors','python_factors'] and len(x.f.calls)==2


def test_duplicate_action_cannot_freeze_again(fixture):
    x=fixture;x.execute()
    with pytest.raises(ValueError,match='already attempted'):x.execute()
    assert x.calls==['python_factors']


@pytest.mark.parametrize('phase',['before','after'])
def test_native_partial_failure_keeps_claim_without_automatic_retry(fixture,phase):
    x=fixture;x.errors.add(phase)
    with pytest.raises(RuntimeError,match='native freeze failed'):x.execute()
    root=x.a.area(x.f.state)
    assert (root/'domains/python_factors/claim.json').exists() and not (root/'domains/python_factors/result.json').exists()
    assert not (root/'certificate.json').exists() and (root/'action/failure.json').exists()
    with pytest.raises(ValueError,match='already attempted'):x.execute()
    assert x.calls==['python_factors']
    assert x.paths(x.f.roots[1],'level5','python_factors')['dataset'].exists() is (phase=='after')


@pytest.mark.parametrize('code',[1,143,-15])
def test_nonzero_wrapper_preserves_completed_science_and_actual_exit(fixture,code):
    x=fixture;x.dispatch['code']=code
    with pytest.raises(ValueError,match='explicit reconciliation'):x.execute()
    directory=x.a.area(x.f.state)/'action'
    assert x.a.read(directory/'exit.json')['returncode']==code
    assert x.a.read(directory/'failure.json')['files_sha256'][str(directory/'result.json')]==x.a.sha(directory/'result.json')
    assert (x.a.area(x.f.state)/'certificate.json').is_file()
    with pytest.raises(ValueError,match='reconciliation'):x.a.verify(x.f.state,guest=False)
    assert x.calls==['python_factors']


@pytest.mark.parametrize('kind',['missing_exit','nonzero_fit','fit_failure','registration_sha','aggregate_sha','recipe_changed','python_failed','pantry_passed',
    'wrong_host','wrong_review','missing_review_dependency','review_cycle','source_changed','existing_dataset','existing_claim','prior_holdout'])
def test_preconditions_block_python_freeze(fixture,monkeypatch,kind):
    x=fixture;a=x.a;f=x.f
    if kind=='missing_exit':(f.state/'action/exit.json').unlink()
    elif kind=='nonzero_fit':change(f.state/'action/exit.json',lambda v:v.update(returncode=143))
    elif kind=='fit_failure':write(f.state/'action/failure.json',{'partial':True})
    elif kind=='registration_sha':monkeypatch.setattr(a,'REGISTRATION_SHA','wrong')
    elif kind=='aggregate_sha':monkeypatch.setattr(a,'FIT_RESULT_SHA','wrong')
    elif kind=='recipe_changed':change(x.entries[1]['recipe_path'],lambda v:v.update(changed=True))
    elif kind in ('python_failed','pantry_passed'):
        result=f.a.verify_existing(f.state,guest=False)
        if kind=='python_failed':result['failed_domains']=['pantry','python_factors']
        else:result['failed_domains']=[];result['status']='two_level5_development_gates_passed'
        monkeypatch.setattr(f.a,'verify_existing',lambda *args,**kwargs:result)
    elif kind=='wrong_host':monkeypatch.setattr(a.os,'uname',lambda:SimpleNamespace(nodename='wash.cs.princeton.edu'))
    elif kind=='wrong_review':change(a.REVIEW,lambda v:v.update(status='draft'))
    elif kind=='missing_review_dependency':change(a.REVIEW,lambda v:v['files_sha256'].pop(str(f.source)))
    elif kind=='review_cycle':change(a.REVIEW,lambda v:v['files_sha256'].update({str(a.REVIEW):'0'*64}))
    elif kind=='source_changed':f.source.write_text('changed')
    elif kind=='existing_dataset':write(x.paths(f.roots[1],'level5','python_factors')['dataset']/'partial.json',{})
    elif kind=='existing_claim':write(a.area(f.state)/'domains/python_factors/claim.json',{})
    else:write(x.paths(f.roots[1],'level5','python_factors')['receipt'],{'partial':True})
    with pytest.raises((ValueError,FileNotFoundError)):x.execute()
    assert not x.calls


@pytest.mark.parametrize('kind',['owner_start','owner_command','dead_owner','fence','manifest','source','host'])
def test_guest_live_owner_source_and_inherited_fence(fixture,kind):
    x=fixture;a=x.a;f=x.f;directory=a.area(f.state)/'action'
    runtime={**deepcopy(f.owner),'host':a.HOST,'fence_fd':8,'lock_inode':a.LOCK_INODE,'lock_device':a.LOCK.stat().st_dev,
        'source_sha256':a.sha(a.SOURCE),'view_manifest_sha256':a.sha(a.MANIFEST)}
    if kind=='owner_start':runtime['start_ticks']='wrong'
    elif kind=='owner_command':runtime['command']=['wrong']
    elif kind=='dead_owner':f.owner['state']='Z'
    elif kind=='fence':runtime['fence_fd']=9
    elif kind=='manifest':runtime['view_manifest_sha256']='wrong'
    elif kind=='source':runtime['source_sha256']='wrong'
    else:runtime['host']='wash.cs.princeton.edu'
    write(directory/'runtime.json',runtime);f.held.append(8);f.canonical.write_text('old')
    try:
        with pytest.raises(ValueError):a.guard(f.state,8)
    finally:f.held.clear();f.canonical.write_text('neutral')
    assert not x.calls


@pytest.mark.parametrize('kind',['dataset','claim','missing_pin','certificate','guest_argv','missing_action_exit','failure'])
def test_static_proof_rejects_changed_or_incomplete_freeze(fixture,kind):
    x=fixture;x.execute();a=x.a;root=a.area(x.f.state);directory=root/'action'
    if kind=='dataset':change(Path(a.read(root/'certificate.json')['frozen'][0]['dataset'])/'identity.json',lambda v:v['splits']['train'].update(rows=383))
    elif kind=='claim':change(root/'domains/python_factors/claim.json',lambda v:v.update(source_root='/other'))
    elif kind in ('missing_pin','certificate'):
        value=a.read(root/'certificate.json')
        if kind=='missing_pin':value['files_sha256'].pop(str(root/'domains/python_factors/claim.json'))
        else:value['pantry_development_fit_pass']=True
        write(root/'certificate.json',value);write(directory/'result.json',value)
        write(root/'certificate.sha256.json',{'sha256':a.sha(root/'certificate.json')})
    elif kind=='guest_argv':change(directory/'guest_runtime.json',lambda v:v['command'].remove('-B'))
    elif kind=='missing_action_exit':(directory/'exit.json').unlink()
    else:write(directory/'failure.json',{})
    with pytest.raises((ValueError,FileNotFoundError)):a.verify(x.f.state,guest=False)
    assert x.calls==['python_factors']


@pytest.mark.parametrize('field',['review_sha256','view_manifest_sha256','lock_inode','lock_device','command'])
def test_static_action_rebinds_fence_view_owner_and_review(fixture,field):
    x=fixture;x.execute();directory=x.a.area(x.f.state)/'action'
    change(directory/'runtime.json',lambda v:v.update({field:['wrong'] if field=='command' else 'wrong'}))
    digest=x.a.sha(directory/'runtime.json')
    change(directory/'intent.json',lambda v:v.update(runtime_sha256=digest))
    change(directory/'guest_runtime.json',lambda v:v.update(outer_runtime_sha256=digest))
    with pytest.raises(ValueError):x.a.verify(x.f.state,guest=False)
    assert x.calls==['python_factors']


def test_host_only_authentication_and_lost_fence_prevent_native_freeze(fixture,monkeypatch):
    x=fixture
    def lost(fd):raise ValueError('lost fence')
    monkeypatch.setattr(x.a,'assert_fence',lost)
    with pytest.raises(ValueError,match='lost fence'):x.execute()
    assert x.authentications==[x.a.MANIFEST] and not x.calls


def test_completed_freeze_proof_is_host_safe_without_live_device_replay(fixture, monkeypatch):
    x=fixture;v=x.execute();original=Path.stat
    def stat(path,*args,**kwargs):
        if path==x.a.LOCK:raise AssertionError('historical verification cannot query current host device')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',stat)
    assert x.a.verify(x.f.state,guest=False)==v
    assert x.calls==['python_factors']
