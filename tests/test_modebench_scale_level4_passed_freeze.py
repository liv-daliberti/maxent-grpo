"""Scratch execution only: no real freezing, model scoring or controller action."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level4_first import fixture as first_fixture, write, change

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(first_fixture,monkeypatch):
    f=first_fixture;f.publish_registration();f.failures.add('python_factors');f.owner['command'][-1]='fit'
    f.a.run(f.state,'fit')
    spec=importlib.util.spec_from_file_location('scratch_passed_freeze',ROOT/'artifacts/freeze_modebench_scale_level4_passed_20260912.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    for key,value in {'_first':f.a,'STATE':f.state,'MANIFEST':f.a.MANIFEST,'LOCK':f.a.LOCK,'LOCK_INODE':f.a.LOCK_INODE,
        'REVIEW':f.state.parent/'freeze_review.json','assert_fence':f.a.assert_fence,'lifetime_fence':f.a.lifetime_fence,
        'RECIPES':{e['domain']:a.sha(e['recipe_path']) for e in f.entries}}.items():monkeypatch.setattr(a,key,value)
    pins=f.a.read(f.a.REVIEW)['files_sha256'];pins.update({str(p):a.sha(p) for p in (a.SOURCE,a.TESTS,a.FIRST,f.a.TESTS,f.a.REVIEW,*f.a.SEALED)})
    write(a.REVIEW,{'schema':'modebench_scale_level4_passed_freeze_independent_review_v1','status':'reviewed','files_sha256':pins})
    f.owner['command']=[str(a.PYTHON),'-B',str(a.SOURCE),'freeze'];active=[];calls=[];identities=[];crosses=[];errors=set()
    def identity(pid):
        if active and pid!=f.owner['pid']:
            argv=active[0][active[0].index('--')+1:];argv.insert(1,'-B')
            return {**deepcopy(f.owner),'pid':12346,'start_ticks':'67891','command':argv}
        return deepcopy(f.owner)
    monkeypatch.setattr(a,'process_identity',identity)
    def paths(root,level,domain):
        return {'dataset':root/level/'dataset'/domain,'recipe':root/level/'recipes'/(domain+'.json'),
            'receipt':root/level/'results/confirmation'/(domain+'.json'),'audit':root/level/'confirmation'/(domain+'.json')}
    def freeze(root):
        domain=root.name;calls.append(domain)
        if domain in errors:raise RuntimeError('native freeze exception')
        assert domain in a.DOMAINS and f.held==[8] and f.canonical.read_text()=='old'
        q=paths(root,'level4',domain)
        write(root/'exclusions/freeze.json',{'complete_history':True})
        write(q['dataset']/'identity.json',{'splits':{s:{'rows':n,'rows_sha256':s} for s,n in [('train',384),('dev',128),('eval',128)]}})
        for split in ('train','dev','eval'):write(q['dataset']/(split+'.jsonl'),{'fixed':split})
    def frozen(root,level,domain,kind):
        identities.append(domain);assert level=='level4' and kind=='domain_revision_v1'
        return a.read(paths(root,level,domain)['dataset']/'identity.json')
    def cross(root,level,domain):
        crosses.append(domain);return {'files_sha256':{str(f.source):a.sha(f.source)}}
    def authenticate(path):
        assert f.canonical.read_text()=='neutral';f.authentications.append(path)
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
    monkeypatch.setattr(a.subprocess,'run',run)
    return SimpleNamespace(a=a,f=f,calls=calls,identities=identities,crosses=crosses,errors=errors,dispatch=dispatch,paths=paths,
        execute=lambda:a.run(f.state,a.sha(a.REVIEW)))


def test_freezes_only_two_actual_passed_domains_preserves_python_and_full_original_identity(fixture):
    x=fixture;before={e['domain']:x.a.sha(e['recipe_path']) for e in x.f.entries};result=x.execute()
    assert x.calls==list(x.a.DOMAINS) and x.identities==list(x.a.DOMAINS) and x.crosses==list(x.a.DOMAINS)
    assert result['domains']==['graph_coloring','pantry'] and result['python_development_fit_pass'] is False
    assert result['preserved_python_recipe_sha256']==before['python_factors'] and result['new_fit_publications']==0
    assert {e['domain']:x.a.sha(e['recipe_path']) for e in x.f.entries}==before
    assert x.f.calls==list(x.f.a.DOMAINS) and not x.f.held and x.f.canonical.read_text()=='neutral'
    assert not x.paths(Path(x.f.entries[2]['source_root']),'level4','python_factors')['dataset'].exists()
    for record in result['frozen']:assert {s:v['rows'] for s,v in record['splits'].items()}=={'train':384,'dev':128,'eval':128}


def test_static_verification_reuses_original_identity_without_freeze_or_fit(fixture):
    x=fixture;result=x.execute();before=list(x.calls)
    assert x.a.verify(x.f.state,guest=False)==result and x.calls==before
    assert x.identities==list(x.a.DOMAINS)*2 and x.f.calls==list(x.f.a.DOMAINS)


def test_duplicate_action_refused_without_duplicate_science(fixture):
    x=fixture;x.execute()
    with pytest.raises(ValueError,match='already attempted'):x.execute()
    assert x.calls==list(x.a.DOMAINS)


def test_native_second_freeze_exception_preserves_first_and_both_claims_without_retry(fixture):
    x=fixture;x.errors.add('pantry')
    with pytest.raises(RuntimeError,match='native freeze exception'):x.execute()
    root=x.a.area(x.f.state)
    assert (root/'domains/graph_coloring/result.json').is_file() and (root/'domains/pantry/claim.json').is_file()
    assert not (root/'domains/pantry/result.json').exists() and not (root/'certificate.json').exists()
    assert (root/'action/failure.json').is_file()
    with pytest.raises(ValueError,match='already attempted'):x.execute()
    assert x.calls==list(x.a.DOMAINS)


@pytest.mark.parametrize('code',[1,143])
def test_nonzero_wrapper_preserves_actual_completed_freezes_and_requires_reconciliation(fixture,code):
    x=fixture;x.dispatch['code']=code
    with pytest.raises(ValueError,match='explicit reconciliation'):x.execute()
    directory=x.a.area(x.f.state)/'action'
    assert x.a.read(directory/'exit.json')['returncode']==code
    assert x.a.read(directory/'failure.json')['files_sha256'][str(directory/'result.json')]==x.a.sha(directory/'result.json')
    assert (x.a.area(x.f.state)/'certificate.json').is_file()
    with pytest.raises(ValueError,match='reconciliation'):x.a.verify(x.f.state,guest=False)
    assert x.calls==list(x.a.DOMAINS)


@pytest.mark.parametrize('kind',['missing_exit','nonzero_fit','fit_failure','recipe_changed','graph_failed','python_passed',
    'wrong_host','wrong_review','missing_review_dependency','review_cycle','source_changed','existing_dataset','existing_claim','prior_holdout'])
def test_preconditions_refuse_freezes(fixture,monkeypatch,kind):
    x=fixture;a=x.a;f=x.f
    if kind=='missing_exit':(f.state/'actions/fit/exit.json').unlink()
    elif kind=='nonzero_fit':change(f.state/'actions/fit/exit.json',lambda v:v.update(returncode=143))
    elif kind=='fit_failure':write(f.state/'actions/fit/failure.json',{'partial':True})
    elif kind=='recipe_changed':change(f.entries[0]['recipe_path'],lambda v:v.update(changed=True))
    elif kind in ('graph_failed','python_passed'):
        result=f.a.verify_fits(f.state,guest=False)
        if kind=='graph_failed':result['failed_domains']=['graph_coloring','python_factors']
        else:result['failed_domains']=[];result['status']='level4_development_gates_passed'
        monkeypatch.setattr(f.a,'verify_fits',lambda *args,**kwargs:result)
    elif kind=='wrong_host':monkeypatch.setattr(f.a.os,'uname',lambda:SimpleNamespace(nodename='wash.cs.princeton.edu'))
    elif kind=='wrong_review':change(a.REVIEW,lambda v:v.update(status='draft'))
    elif kind=='missing_review_dependency':change(a.REVIEW,lambda v:v['files_sha256'].pop(str(f.sealed)))
    elif kind=='review_cycle':change(a.REVIEW,lambda v:v['files_sha256'].update({str(a.REVIEW):'0'*64}))
    elif kind=='source_changed':f.source.write_text('changed')
    elif kind=='existing_dataset':write(x.paths(Path(f.entries[0]['source_root']),'level4','graph_coloring')['dataset']/'partial.json',{})
    elif kind=='existing_claim':write(a.area(f.state)/'domains/graph_coloring/claim.json',{})
    else:write(x.paths(Path(f.entries[0]['source_root']),'level4','graph_coloring')['receipt'],{'partial':True})
    with pytest.raises((ValueError,FileNotFoundError)):x.execute()
    assert not x.calls


@pytest.mark.parametrize('kind',['owner_start','owner_command','dead_owner','fence','manifest','source','host'])
def test_guest_checks_live_owner_source_and_inherited_fence_before_science(fixture,monkeypatch,kind):
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
def test_readonly_proof_rejects_tampering_or_incomplete_action(fixture,kind):
    x=fixture;x.execute();a=x.a;root=a.area(x.f.state);directory=root/'action'
    if kind=='dataset':change(Path(a.read(root/'certificate.json')['frozen'][0]['dataset'])/'identity.json',lambda v:v['splits']['train'].update(rows=383))
    elif kind=='claim':change(root/'domains/graph_coloring/claim.json',lambda v:v.update(source_root='/other'))
    elif kind in ('missing_pin','certificate'):
        value=a.read(root/'certificate.json')
        if kind=='missing_pin':value['files_sha256'].pop(str(root/'domains/graph_coloring/claim.json'))
        else:value['python_development_fit_pass']=True
        write(root/'certificate.json',value);write(directory/'result.json',value)
        write(root/'certificate.sha256.json',{'sha256':a.sha(root/'certificate.json')})
    elif kind=='guest_argv':change(directory/'guest_runtime.json',lambda v:v['command'].remove('-B'))
    elif kind=='missing_action_exit':(directory/'exit.json').unlink()
    else:write(directory/'failure.json',{})
    with pytest.raises((ValueError,FileNotFoundError)):a.verify(x.f.state,guest=False)
    assert x.calls==list(a.DOMAINS)


def test_authentication_is_host_only_and_lost_fence_blocks_freezing(fixture,monkeypatch):
    x=fixture
    def lost(fd):raise ValueError('lost fence')
    monkeypatch.setattr(x.a,'assert_fence',lost)
    before=len(x.f.authentications)
    with pytest.raises(ValueError,match='lost fence'):x.execute()
    assert len(x.f.authentications)==before+1 and not x.calls


@pytest.mark.parametrize('field',['review_sha256','view_manifest_sha256','lock_inode','lock_device','command'])
def test_static_action_rebinds_recorded_owner_fence_view_and_review(fixture,field):
    x=fixture;x.execute();directory=x.a.area(x.f.state)/'action'
    change(directory/'runtime.json',lambda v:v.update({field:['wrong'] if field=='command' else 'wrong'}))
    digest=x.a.sha(directory/'runtime.json')
    change(directory/'intent.json',lambda v:v.update(runtime_sha256=digest))
    change(directory/'guest_runtime.json',lambda v:v.update(outer_runtime_sha256=digest))
    with pytest.raises(ValueError):x.a.verify(x.f.state,guest=False)
    assert x.calls==list(x.a.DOMAINS)
