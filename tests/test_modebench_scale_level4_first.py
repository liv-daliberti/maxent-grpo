"""Temporary first-stage evidence only; no real science, jobs, or controller."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_text(json.dumps(value,sort_keys=True))


def change(path,update):
    value=json.loads(Path(path).read_text());update(value);write(path,value)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('scratch_level4_first',ROOT/'artifacts/continue_modebench_scale_level4_first_20260912.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    state=tmp_path/'state';canonical=tmp_path/'canonical.py';canonical.write_text('neutral')
    lock=tmp_path/'controller.lock';lock.write_text('')
    for key,value in {'STATE':state,'CANONICAL':canonical,'LOCK':lock,'LOCK_INODE':lock.stat().st_ino,
        'NEUTRAL_SHA':a.sha(canonical),'OLD_SHA':'frozen','MANIFEST':tmp_path/'view.json',
        'REVIEW':tmp_path/'review.json','PLAN':tmp_path/'original/plan.json','L4_MANIFEST':tmp_path/'level4_manifest.json'}.items():
        monkeypatch.setattr(a,key,value)
    write(a.MANIFEST,{'fixture':'view'})
    sealed=tmp_path/'sealed.py';sealed.write_text('sealed')
    source=tmp_path/'source.json';write(source,{'complete':True})
    entries=[];cells=[];sources={};certificates={};pins={str(source):a.sha(source)};receipts=[]
    for i,domain in enumerate(a.DOMAINS):
        root=tmp_path/domain;certificate=a.PLAN.parent/'execution_reconciliations'/str(i)/'reconciliation.json'
        entry={'array_index':i,'domain':domain,'source_root':str(root),
            'recipe_path':str(root/'level4/recipes'/(domain+'.json')),'certificate_path':str(certificate)}
        entries.append(entry);tasks_path=tmp_path/f'tasks{i}.json';tasks=[]
        for tier in range(4):
            output=tmp_path/f'receipt{i}_{tier}.json';receipts.append(output)
            write(output,{'status':'complete','level':'level4','domain':domain,'split':'dev'})
            tasks.append({'output':str(output)});pins[str(output)]=a.sha(output)
        write(tasks_path,tasks);pins[str(tasks_path)]=a.sha(tasks_path)
        cells.append({'level':'level4','domain':domain,'source_root':str(root),'tasks':str(tasks_path)})
        sources[domain]={'source_kind':'domain_revision_v1','source_root':str(root)}
        raw,code=a._base.CELL_EXECUTIONS[i]
        proof={'job_id_raw':str(raw),'state':'FAILED','exit_code':code,'scheduler_success':False,
            'scientific_outputs_complete':True,'completed_receipts':4,'files_sha256':{str(source):a.sha(source)}}
        write(certificate,proof);pins[str(certificate)]=a.sha(certificate)
        certificates[str(i)]={'path':str(certificate),'sha256':a.sha(certificate),'state':'FAILED','exit_code':code}
    write(a.PLAN,{'cells':cells});write(a.L4_MANIFEST,{'sources':sources})
    monkeypatch.setattr(a,'SEALED',{p:a.sha(p) for p in (sealed,a.PLAN,a.L4_MANIFEST)})
    pins.update({str(p):digest for p,digest in a.SEALED.items()})
    write(a.REVIEW,{'schema':'modebench_scale_level4_first_independent_review_v1','status':'reviewed',
        'files_sha256':{str(p):a.sha(p) for p in (a.SOURCE,a.TESTS,*a.SEALED)}})
    calls=[];failures=set();exceptions=set();guards=[];held=[];authentications=[]
    def fit(root):
        domain=root.name;calls.append(domain)
        if domain in exceptions:raise ValueError('actual fitter exception')
        value={'level':'level4','domain':domain,'development_fit_pass':domain not in failures,
            'input_sha256':{str(source):a.sha(source)},'all_original_gates':'fixture'}
        write(next(e['recipe_path'] for e in entries if e['domain']==domain),value);return value
    def authenticate(path):
        assert canonical.read_text()=='neutral','runner.authenticate must never run in frozen guest'
        authentications.append(path);return {}
    monkeypatch.setattr(a,'load_module',lambda path,name:SimpleNamespace(fit_domain=fit) if path==a.REVISION else SimpleNamespace(authenticate=authenticate))
    monkeypatch.setattr(a,'collect_inputs',lambda:(entries,certificates,pins))
    monkeypatch.setattr(a.os,'uname',lambda:SimpleNamespace(nodename=a.HOST))
    owner={'pid':12345,'start_ticks':'67890','state':'R','command':[str(a.PYTHON),'-B',str(a.SOURCE),'register'],'uid':a.os.getuid()}
    active=[]
    def identity(pid):
        if active and pid!=12345:
            command=active[0][active[0].index('--')+1:];command.insert(1,'-B')
            return {**deepcopy(owner),'pid':12346,'command':command,'start_ticks':'67891'}
        return deepcopy(owner)
    monkeypatch.setattr(a,'process_identity',identity)
    @contextmanager
    def fence():
        assert not held;held.append(8)
        try:yield 8
        finally:held.pop()
    monkeypatch.setattr(a,'lifetime_fence',fence)
    def assert_fence(fd):assert fd==8 and held==[8];guards.append(fd)
    monkeypatch.setattr(a,'assert_fence',assert_fence)
    def host_guard(*,guest):
        a.require(a.os.uname().nodename==a.HOST,'truthful host')
        a.require(canonical.read_text()==('old' if guest else 'neutral'),'source view')
    monkeypatch.setattr(a,'host_guard',host_guard)
    dispatch={'code':0,'drop':False}
    def run(command,**kwargs):
        assert kwargs['pass_fds']==(8,) and held==[8]
        action=command[command.index('--action')+1];active.append(command)
        canonical.write_text('old')
        try:a.guest_action(state,action,8,a.sha(a.REVIEW) if action=='register' else None)
        finally:canonical.write_text('neutral');active.clear()
        if dispatch['drop']:(state/'actions'/action/'result.json').unlink()
        return SimpleNamespace(returncode=dispatch['code'])
    monkeypatch.setattr(a.subprocess,'run',run)
    def publish_registration():return a.run(state,'register',a.sha(a.REVIEW))
    return SimpleNamespace(a=a,state=state,canonical=canonical,source=source,sealed=sealed,entries=entries,
        publish_registration=publish_registration,calls=calls,failures=failures,
        exceptions=exceptions,guards=guards,held=held,owner=owner,dispatch=dispatch,receipts=receipts,
        authentications=authentications)


def test_register_and_three_original_fits_are_explicit_independent_actions(fixture):
    f=fixture;a=f.a
    registered=a.run(f.state,'register',a.sha(a.REVIEW))
    assert registered['completed_receipts']==12 and registered['level5_completion_required'] is False
    assert f.calls==[] and not f.held
    f.owner['command'][-1]='fit';result=a.run(f.state,'fit')
    assert f.calls==list(a.DOMAINS) and result['status']=='level4_development_gates_passed'
    assert result['new_grader_invocations']==0 and result['publication_performed'] is False
    assert all((f.state/'fits'/d/'claim.json').exists() for d in a.DOMAINS)
    assert not f.held and f.canonical.read_text()=='neutral'


def test_failed_scientific_gates_are_saved_while_other_prespecified_fits_finish(fixture):
    f=fixture;f.publish_registration();f.failures.add('graph_coloring');f.owner['command'][-1]='fit'
    value=f.a.run(f.state,'fit')
    assert f.calls==list(f.a.DOMAINS) and value['status']=='needs_new_development_revision'
    assert value['failed_domains']==['graph_coloring']
    assert f.a.read(f.state/'fits/graph_coloring/result.json')['development_fit_pass'] is False
    assert not value['confirmation_performed']


def test_duplicate_register_or_fit_never_invokes_fitter_again(fixture):
    f=fixture;f.a.run(f.state,'register',f.a.sha(f.a.REVIEW))
    with pytest.raises(ValueError,match='already attempted'):f.a.run(f.state,'register',f.a.sha(f.a.REVIEW))
    f.owner['command'][-1]='fit';f.a.run(f.state,'fit')
    with pytest.raises(ValueError,match='already attempted'):f.a.run(f.state,'fit')
    assert f.calls==list(f.a.DOMAINS)


def test_partial_exception_preserves_claim_and_finished_recipe_without_retry(fixture):
    f=fixture;f.publish_registration();f.exceptions.add('pantry');f.owner['command'][-1]='fit'
    with pytest.raises(ValueError,match='actual fitter exception'):f.a.run(f.state,'fit')
    assert f.calls==['graph_coloring','pantry']
    assert (f.state/'fits/graph_coloring/result.json').exists() and (f.state/'fits/pantry/claim.json').exists()
    assert not (f.state/'fits/pantry/result.json').exists() and (f.state/'actions/fit/failure.json').exists()
    with pytest.raises(ValueError,match='already attempted'):f.a.run(f.state,'fit')
    assert f.calls==['graph_coloring','pantry']


def test_nonzero_outer_exit_retains_all_published_results(fixture):
    f=fixture;f.publish_registration();f.owner['command'][-1]='fit';f.dispatch['code']=143
    with pytest.raises(ValueError,match='explicit reconciliation'):f.a.run(f.state,'fit')
    directory=f.state/'actions/fit';assert f.a.read(directory/'exit.json')['returncode']==143
    failure=f.a.read(directory/'failure.json')
    assert failure['files_sha256'][str(directory/'result.json')]==f.a.sha(directory/'result.json')
    assert len(f.calls)==3


@pytest.mark.parametrize('kind',['prior_recipe','prior_claim','lost_fence','source_drift','wrong_scope','wrong_review','changed_dependency'])
def test_preconditions_block_new_fit_science(fixture,monkeypatch,kind):
    f=fixture;f.publish_registration();f.owner['command'][-1]='fit'
    if kind=='prior_recipe':write(Path(f.entries[0]['recipe_path']),{'existing':True})
    elif kind=='prior_claim':write(f.state/'fits/graph_coloring/claim.json',{'partial':True})
    elif kind=='lost_fence':
        def lost(fd):raise ValueError('lost fence')
        monkeypatch.setattr(f.a,'assert_fence',lost)
    elif kind=='source_drift':f.source.write_text('changed')
    elif kind=='changed_dependency':f.sealed.write_text('changed')
    elif kind=='wrong_review':change(f.a.REVIEW,lambda v:v.update(status='draft'))
    else:
        change(f.state/'registration.json',lambda v:v.update(level='level5'))
        write(f.state/'registration.sha256.json',{'sha256':f.a.sha(f.state/'registration.json')})
    with pytest.raises(ValueError):f.a.run(f.state,'fit')
    assert not f.calls


@pytest.mark.parametrize('kind',['host','owner','fence_fd','source_sha','owner_command'])
def test_guest_rechecks_truthful_owner_and_continuous_fence(fixture,kind):
    f=fixture;f.publish_registration();f.owner['command'][-1]='fit'
    runtime={**f.owner,'host':f.a.HOST,'fence_fd':8,'lock_inode':f.a.LOCK_INODE,
        'lock_device':f.a.LOCK.stat().st_dev,'source_sha256':f.a.sha(f.a.SOURCE)}
    if kind=='host':runtime['host']='wash.cs.princeton.edu'
    elif kind=='owner':runtime['start_ticks']='other'
    elif kind=='fence_fd':runtime['fence_fd']=9
    elif kind=='source_sha':runtime['source_sha256']='other'
    else:runtime['command']=['other']
    write(f.state/'actions/fit/runtime.json',runtime);f.canonical.write_text('old');f.held.append(8)
    try:
        with pytest.raises(ValueError):f.a.guest_guard(f.state,'fit',8)
    finally:f.held.clear();f.canonical.write_text('neutral')
    assert not f.calls


def test_review_closure_requires_exact_sources_and_no_cycle(fixture):
    f=fixture
    for mutation in ('missing','cycle','schema'):
        original=f.a.read(f.a.REVIEW)
        def update(v):
            if mutation=='missing':v['files_sha256'].pop(str(f.sealed))
            elif mutation=='cycle':v['files_sha256'][str(f.a.REVIEW)]='0'*64
            else:v['schema']='wrong'
        change(f.a.REVIEW,update)
        with pytest.raises(ValueError):f.a.reviewed(f.a.sha(f.a.REVIEW))
        write(f.a.REVIEW,original)


@pytest.mark.parametrize('kind',['nonzero','missing_exit','failure','wrong_guest','wrong_result','wrong_intent'])
def test_registered_output_alone_does_not_authorize_fits(fixture,kind):
    f=fixture;f.publish_registration();directory=f.state/'actions/register'
    if kind=='nonzero':change(directory/'exit.json',lambda v:v.update(returncode=143))
    elif kind=='missing_exit':(directory/'exit.json').unlink()
    elif kind=='failure':write(directory/'failure.json',{'preserved':True})
    elif kind=='wrong_guest':change(directory/'guest_runtime.json',lambda v:v.update(outer_runtime_sha256='wrong'))
    elif kind=='wrong_result':change(directory/'result.json',lambda v:v.update(completed_receipts=11))
    else:change(directory/'intent.json',lambda v:v.update(command=['wrong']))
    f.owner['command'][-1]='fit'
    with pytest.raises((ValueError,FileNotFoundError)):f.a.run(f.state,'fit')
    assert not f.calls


@pytest.mark.parametrize('kind',['source_root','recipe_path','certificate_path','missing_certificate','missing_receipt'])
def test_registered_targets_and_all_proof_pins_are_rebound_statically(fixture,kind):
    f=fixture;f.publish_registration()
    def mutate(v):
        if kind.startswith('missing'):
            path=f.entries[0]['certificate_path'] if kind=='missing_certificate' else str(f.receipts[0])
            v['files_sha256'].pop(path)
        else:v['revisions'][0][kind]='/another/source'
    change(f.state/'registration.json',mutate)
    write(f.state/'registration.sha256.json',{'sha256':f.a.sha(f.state/'registration.json')})
    with pytest.raises(ValueError):f.a.verify_registration(f.state)
    assert not f.calls


def test_runner_authenticates_only_outside_frozen_guest(fixture):
    f=fixture;f.publish_registration();f.owner['command'][-1]='fit';f.a.run(f.state,'fit')
    assert f.authentications==[f.a.MANIFEST,f.a.MANIFEST]


def test_completed_fit_verification_reuses_saved_decisions_without_fitting(fixture):
    f=fixture;f.publish_registration();f.owner['command'][-1]='fit';value=f.a.run(f.state,'fit')
    before=list(f.calls)
    assert f.a.verify_fits(f.state)==value and f.calls==before


@pytest.mark.parametrize('kind',['recipe','claim','result','aggregate','nonzero'])
def test_readonly_fit_proof_rejects_changed_or_incomplete_execution(fixture,kind):
    f=fixture;f.publish_registration();f.owner['command'][-1]='fit';f.a.run(f.state,'fit')
    if kind=='recipe':change(f.entries[0]['recipe_path'],lambda v:v.update(development_fit_pass=False))
    elif kind=='claim':change(f.state/'fits/graph_coloring/claim.json',lambda v:v.update(source_root='/changed'))
    elif kind=='result':change(f.state/'fits/graph_coloring/result.json',lambda v:v.update(recipe_sha256='wrong'))
    elif kind=='aggregate':change(f.state/'actions/fit/result.json',lambda v:v.update(status='needs_new_development_revision'))
    else:change(f.state/'actions/fit/exit.json',lambda v:v.update(returncode=143))
    with pytest.raises(ValueError):f.a.verify_fits(f.state)
    assert f.calls==list(f.a.DOMAINS)
