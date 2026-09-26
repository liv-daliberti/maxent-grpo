"""Scratch authority and real transport integration; no scientific/scheduler work."""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/continue_modebench_scale_composite_spin_20260912.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
    return a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o600)
    path.write_text(json.dumps(value, sort_keys=True))


def mutate(path, fn):
    value = json.loads(path.read_text()); fn(value); write(path, value)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    a = load(SOURCE, 'scratch_spin_authority')
    workspace = tmp_path/'workspace'; workspace.mkdir()
    def file(name, content='scratch'):
        p = workspace/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content); return p
    artifacts = workspace/'composite'; artifacts.mkdir()
    authority = workspace/'authority'
    canonical = file('templates.py', 'neutral'); old = file('preserved.py', 'old')
    lock = file('composite/controller.lock', '')
    changes = {'ROOT':workspace, 'AUTHORITY':authority, 'ARTIFACTS':artifacts,
        'CLAIM':artifacts/'spin_claim.json', 'LOCK':lock, 'LOCK_INODE':lock.stat().st_ino,
        'CANONICAL':canonical, 'NEUTRAL_SHA':a.sha(canonical), 'OLD_SHA':a.sha(old),
        'OLD_AUTHORITY':workspace/'old_wash', 'REVISED_PLAN':artifacts/'revision_development/plan.json',
        'REVISED_RECOVERY':workspace/'recovery', 'RECOVERY_CERTIFICATE':workspace/'recovery/execution_reconciliation.json',
        'RECOVERY_VERIFIER':file('recovery_auditor.py'), 'BUNDLE':workspace/'bundle.json',
        'REVIEW':file('driver_review.json','{"status":"reviewed"}'),
        'TRANSPORT_REVIEW':file('transport_v4_review.json','{"status":"reviewed"}')}
    for key,value in changes.items(): monkeypatch.setattr(a,key,value)
    for key in ('LOCK','LOCK_INODE','CANONICAL','NEUTRAL_SHA','OLD_SHA'):
        monkeypatch.setattr(a._base,key,getattr(a,key))
    monkeypatch.setattr(a.os, 'uname', lambda:SimpleNamespace(nodename=a.HOST))
    monkeypatch.setattr(a, 'local_controllers', lambda:[])
    monkeypatch.setattr(a._base, 'no_alternative_authority', lambda:None)
    old_terminal = a.OLD_AUTHORITY/'terminal.json'
    write(old_terminal, {'status':'execution_failed','old_os_state':'unverified'})
    monkeypatch.setattr(a, 'OLD_TERMINAL_SHA', a.sha(old_terminal))
    def predecessor_inputs(*,guest=False):
        a.require(a.sha(old_terminal)==a.OLD_TERMINAL_SHA,'recorded predecessor changed')
        pins={str(old_terminal):a.OLD_TERMINAL_SHA,str(a.BASE_SOURCE):a.BASE_SHA}
        a.check_pins(pins,guest=guest);return pins
    monkeypatch.setattr(a, 'predecessor_inputs', predecessor_inputs)
    manifest = workspace/'view/manifest.json'
    proot=file('view/proot');copied=file('view/copy.py','old')
    view={'proot':{'path':str(proot)},'preserved_source':{'path':str(old)},'mappings':[{'source':str(copied)}]}
    write(manifest,view);file('view/manifest.sha256',a.sha(manifest))
    write(a.OLD_AUTHORITY/'activation.json',{'view_manifest':str(manifest)})
    write(a.REVISED_RECOVERY/'plan.json',{'view_manifest':str(manifest)})
    cells=[];receipts=[]
    for index in range(7):
        tasks=[]
        for tier in range(4):
            receipt=workspace/f'receipts/{index}_{tier}.json';write(receipt,{'status':'complete'})
            tasks.append({'output':str(receipt)});receipts.append(receipt)
        tasks_path=workspace/f'tasks/{index}.json';write(tasks_path,tasks)
        cells.append({'phase':'dev','tasks':str(tasks_path)})
    write(a.REVISED_PLAN,{'cells':cells,'immutable_inputs_sha256':{str(canonical):a.OLD_SHA}})
    monkeypatch.setattr(a,'REVISED_PLAN_SHA',a.sha(a.REVISED_PLAN))
    sourcepins={}
    for name in ('publication','level4_sources','level5_sources','carried_freeze'):
        path=workspace/(name+'.json');write(path,{'files_sha256':{str(canonical):a.OLD_SHA}})
        sourcepins[path]=a.sha(path)
    monkeypatch.setattr(a,'SOURCE_PINS',sourcepins)
    certs=[]
    for index,(raw,code) in enumerate(a.CELL_EXECUTIONS):
        path=a.REVISED_PLAN.parent/'execution_reconciliations'/str(index)/'reconciliation.json'
        write(path,{'schema':'modebench_scale_completed_stage_cell_reconciliation_v1',
            'status':'verified_scientific_outputs_with_failed_execution','array_job_id':31254520,'array_index':index,
            'job_id_raw':str(raw),'job_id':'31254520_'+str(index),'state':'FAILED','exit_code':code,
            'scheduler_success':False,'scientific_outputs_complete':True,'exit_cause':'unknown',
            'completed_receipts':4,'stage_plan_sha256':a.REVISED_PLAN_SHA,
            'files_sha256':{str(receipts[index*4]):a.sha(receipts[index*4])}})
        certs.append(path)
    amendment={key:str(file('amendment_'+key+('.json' if key=='review' else '.py')))
               for key in ('source','tests','review')}
    recovery_tests=file('recovery_tests.py');recovery_review=file('recovery_review.json')
    write(a.REVIEW, {'status':'reviewed','files_sha256':{str(a.SOURCE):a.sha(a.SOURCE),str(a.TEST_SOURCE):a.sha(a.TEST_SOURCE)}})
    write(a.TRANSPORT_REVIEW, {'files_sha256':{str(a.TRANSPORT):a.sha(a.TRANSPORT),str(a.TRANSPORT_TESTS):a.sha(a.TRANSPORT_TESTS),str(a.PREVIOUS_TRANSPORT):a.PREVIOUS_TRANSPORT_SHA}})
    write(recovery_review, {'files_sha256':{str(a.RECOVERY_VERIFIER):a.sha(a.RECOVERY_VERIFIER),str(recovery_tests):a.sha(recovery_tests)}})
    write(Path(amendment['review']), {'files_sha256':{amendment[key]:a.sha(amendment[key]) for key in ('source','tests')}})
    mandatory=[a.SOURCE,a.TEST_SOURCE,a.REVIEW,a.BASE_SOURCE,a.TRANSPORT,a.TRANSPORT_TESTS,a.TRANSPORT_REVIEW,
        a.TRANSPORT_BASE,a.PREVIOUS_TRANSPORT,a.PREVIOUS_TRANSPORT_TESTS,a.PREVIOUS_TRANSPORT_REVIEW,
        a.RECOVERY_VERIFIER,*a.CELL_HELPERS.values(),*a.CELL_REVIEWS,
        recovery_tests,recovery_review,*map(Path,amendment.values())]
    bundle={'schema':'modebench_scale_spin_authority_reviewed_bundle_v1','status':'reviewed',
        'recovery_audit_tests':str(recovery_tests),'recovery_audit_review':str(recovery_review),
        'recovery_operational_amendments':[amendment],
        'files_sha256':{str(path):a.sha(path) for path in mandatory}}
    for review in (a.REVIEW,a.TRANSPORT_REVIEW,a.PREVIOUS_TRANSPORT_REVIEW,*a.CELL_REVIEWS,recovery_review,Path(amendment['review'])):
        bundle['files_sha256'].update(a.read(review)['files_sha256'])
    write(a.BUNDLE,bundle)
    write(a.RECOVERY_CERTIFICATE,{'schema':a.RECOVERY_SCHEMA,'status':a.RECOVERY_STATUS,'array_job_id':39999001,
        'scheduler_success':False,'recovery_scheduler_success':True,
        'files_sha256':{amendment['source']:a.sha(amendment['source']),str(canonical):a.OLD_SHA}})
    calls={'proofs':0,'cell_verifications':[],'recovery_verifications':0,'sweeps':[],'installs':[],'process':[]}
    core=SimpleNamespace(DEFAULT_PARENT=workspace/'parent',DEFAULT_RELEASE=workspace/'release',DEFAULT_ARTIFACTS=artifacts,
        original=SimpleNamespace(authenticate=lambda path:None),original_plan=lambda path:None)
    def sweep(*args,**kwargs):
        calls['sweeps'].append((args,kwargs));return {'status':'admitted','scratch_only':True}
    core._sweep=sweep
    transport=SimpleNamespace(install=lambda *args:calls['installs'].append(args))
    def load_module(path,name):
        if path==a.RUNNER:return SimpleNamespace(authenticate=lambda path:view)
        if path==a.CONTROLLER:return core
        if path==a.TRANSPORT:return transport
        if path==a.RECOVERY_VERIFIER:
            def verify(root):
                assert root==a.REVISED_RECOVERY;calls['recovery_verifications']+=1;return a.read(a.RECOVERY_CERTIFICATE)
            return SimpleNamespace(verify_existing=verify)
        if path in a.CELL_HELPERS.values():
            def verify(stage,index):
                assert stage=='revision_development';calls['cell_verifications'].append(index);return a.read(certs[index])
            return SimpleNamespace(verify_reconciliation=verify)
        raise AssertionError('unexpected module '+str(path))
    monkeypatch.setattr(a,'load_module',load_module)
    def probe(owner_pid=0):
        return {'command':[str(a.PYTHON),'-B',str(a.SOURCE),'host-probe','--owner-pid',str(owner_pid)],
                'returncode':0,'observation':a.host_probe(owner_pid)}
    monkeypatch.setattr(a,'probe_subprocess',probe)
    def proof_subprocess(manifest,bundle_sha):
        calls['proofs']+=1
        before=canonical.read_text();canonical.write_text('old')
        try: observation=a.proof_check(bundle_sha)
        finally:canonical.write_text(before)
        return {'command':a.proof_command(manifest,bundle_sha),'returncode':0,'observation':observation}
    monkeypatch.setattr(a,'proof_subprocess',proof_subprocess)
    def no_process(*args,**kwargs):raise AssertionError('no real process allowed')
    monkeypatch.setattr(a.subprocess,'run',no_process)
    def dispatch(*,code=0,drop_result=False):
        def run(command,**kwargs):
            calls['process'].append((command,kwargs))
            assert command[:3]==[str(a.PYTHON),'-B',str(a.RUNNER)]
            assert kwargs['pass_fds'] and kwargs['check'] is False
            fd=kwargs['pass_fds'][0];number=int(command[command.index('--sweep')+1])
            canonical.write_text('old')
            try:a.guest_sweep(authority,number,fd)
            finally:canonical.write_text('neutral')
            if drop_result:(authority/'sweeps'/f'{number:06d}'/'result.json').unlink()
            return SimpleNamespace(returncode=code)
        monkeypatch.setattr(a.subprocess,'run',run)
    return SimpleNamespace(a=a,root=authority,manifest=manifest,canonical=canonical,old=old,
        bundle=bundle,bundle_sha=a.sha(a.BUNDLE),receipts=receipts,certs=certs,calls=calls,
        core=core,transport=transport,dispatch=dispatch,amendment=amendment,old_terminal=old_terminal)


def prepare(f):
    return f.a.prepare(f.root,f.manifest,f.bundle_sha)


def repin_activation(f):
    write(f.root/'activation.sha256.json',{'sha256':f.a.sha(f.root/'activation.json')})


def test_preparation_requires_complete_proofs_and_preserves_old_state(fixture):
    f=fixture;before=f.old_terminal.read_bytes();lock=f.a.LOCK.read_bytes()
    value=prepare(f)
    assert value['status']=='prepared_only' and f.calls['proofs']==1
    activation=f.a.verify(f.root)
    assert activation['host']==f.a.HOST and activation['predecessor_os_state']=='unverified'
    assert activation['revised_recovery_job_id']==39999001 and f.old_terminal.read_bytes()==before
    assert f.a.LOCK.read_bytes()==lock and not (f.root/'outer_runtime.json').exists()
    assert f.calls['cell_verifications']==list(range(5)) and f.calls['recovery_verifications']==1
    assert f.calls['sweeps']==[] and f.calls['process']==[]


@pytest.mark.parametrize('kind',['missing_receipt','partial_receipt','missing_cell_audit','missing_recovery','failed_recovery'])
def test_missing_or_incomplete_actual_proof_cannot_create_authority(fixture,kind):
    f=fixture
    if kind=='missing_receipt':f.receipts[-1].unlink()
    elif kind=='partial_receipt':mutate(f.receipts[-1],lambda v:v.update(status='partial'))
    elif kind=='missing_cell_audit':f.certs[-1].unlink()
    elif kind=='missing_recovery':f.a.RECOVERY_CERTIFICATE.unlink()
    else:mutate(f.a.RECOVERY_CERTIFICATE,lambda v:v.update(recovery_scheduler_success=False))
    with pytest.raises((ValueError,FileNotFoundError)):prepare(f)
    assert not f.root.exists() and not f.a.CLAIM.exists() and f.calls['sweeps']==[]


def test_failed_frozen_proof_cannot_create_claim_or_wrapper(fixture,monkeypatch):
    f=fixture
    def fail(*args):raise ValueError('actual proof verifier failed')
    monkeypatch.setattr(f.a,'proof_subprocess',fail)
    with pytest.raises(ValueError,match='proof verifier failed'):prepare(f)
    assert not f.root.exists() and not f.a.CLAIM.exists()


@pytest.mark.parametrize('kind',['wrong_digest','unreviewed','missing_driver','missing_transport','missing_amendment_review'])
def test_review_bundle_closure_is_mandatory(fixture,kind):
    f=fixture
    if kind=='wrong_digest':f.bundle_sha='0'*64
    else:
        def change(v):
            if kind=='unreviewed':v['status']='draft'
            else:
                target={'missing_driver':str(f.a.SOURCE),'missing_transport':str(f.a.TRANSPORT),
                        'missing_amendment_review':f.amendment['review']}[kind]
                v['files_sha256'].pop(target)
        mutate(f.a.BUNDLE,change);f.bundle_sha=f.a.sha(f.a.BUNDLE)
    with pytest.raises(ValueError):prepare(f)
    assert not f.a.CLAIM.exists()


def test_duplicate_preparation_preserves_original_claim(fixture):
    f=fixture;prepare(f);before=f.a.CLAIM.read_bytes()
    with pytest.raises(ValueError,match='already claimed'):prepare(f)
    assert f.a.CLAIM.read_bytes()==before and f.calls['proofs']==1


def test_static_verify_accepts_worker_host_but_enforces_source_view(fixture,monkeypatch):
    f=fixture;prepare(f)
    monkeypatch.setattr(f.a.os,'uname',lambda:SimpleNamespace(nodename='node202.ionic.cs.princeton.edu'))
    with pytest.raises(ValueError,match='fence changed'):f.a.verify(f.root,guest=True)
    f.canonical.write_text('old');assert f.a.verify(f.root,guest=True)['host']==f.a.HOST
    with pytest.raises(ValueError,match='fence changed'):f.a.verify(f.root)


@pytest.mark.parametrize('field,value',[('host','wash.cs.princeton.edu'),('predecessor_os_state','dead'),
    ('revised_recovery_job_id',1),('driver_sha256','0'*64),('scientific_entrypoint','other:_sweep')])
def test_activation_contract_cannot_be_relabelled(fixture,field,value):
    f=fixture;prepare(f);mutate(f.root/'activation.json',lambda v:v.update({field:value}));repin_activation(f)
    with pytest.raises(ValueError):f.a.verify(f.root)


def test_pinned_receipt_change_after_preparation_is_rejected(fixture):
    f=fixture;prepare(f);mutate(f.receipts[27],lambda v:v.update(extra='changed'))
    with pytest.raises(ValueError,match='omits mandatory|pinned input changed'):f.a.verify(f.root)


def test_completion_proof_cannot_omit_audit_or_weaken_success(fixture):
    f=fixture;prepare(f);path=f.root/'revised_recovery_completion.json'
    mutate(path,lambda v:v['proof_verification']['observation']['completed_cell_certificates'].pop('4'))
    mutate(f.root/'activation.json',lambda v:v.update(revised_recovery_completion_sha256=f.a.sha(path)))
    repin_activation(f)
    with pytest.raises(ValueError,match='proof linkage changed'):f.a.verify(f.root)


def test_exact_watch_and_guest_call_unchanged_sweep_once(fixture):
    f=fixture;prepare(f);f.dispatch();result=f.a.watch(f.root)
    assert result['status']=='admitted' and len(f.calls['sweeps'])==len(f.calls['installs'])==1
    assert f.calls['proofs']==2 and f.calls['cell_verifications']==list(range(5))*2
    args,kwargs=f.calls['sweeps'][0]
    assert args==(f.core.DEFAULT_PARENT,f.core.DEFAULT_RELEASE,f.core.DEFAULT_ARTIFACTS) and kwargs=={'advance':True}
    runtime=f.a.read(f.root/'outer_runtime.json')
    assert runtime['host']==f.a.HOST and runtime['owner_pid']==os.getpid()
    assert runtime['revised_recovery_job_id']==39999001
    assert f.a.read(f.root/'terminal.json')['result']==result
    assert f.canonical.read_text()=='neutral'
    with pytest.raises(ValueError,match='already attempted'):f.a.watch(f.root)
    assert len(f.calls['sweeps'])==1


@pytest.mark.parametrize('code,drop_result',[(143,False),(1,False),(0,True)])
def test_nonzero_wrapper_preserves_evidence_and_never_retries(fixture,code,drop_result):
    f=fixture;prepare(f);f.dispatch(code=code,drop_result=drop_result)
    with pytest.raises(ValueError,match='guest outcome'):f.a.watch(f.root)
    directory=f.root/'sweeps/000001';failure=f.a.read(f.root/'failure.json')
    assert f.a.read(directory/'exit.json')['returncode']==code
    assert failure['preserved_execution_evidence_sha256'][str(directory/'exit.json')]==f.a.sha(directory/'exit.json')
    if not drop_result:
        assert failure['preserved_execution_evidence_sha256'][str(directory/'result.json')]==f.a.sha(directory/'result.json')
    assert not (f.root/'terminal.json').exists() and len(f.calls['sweeps'])==1
    with pytest.raises(ValueError,match='already attempted'):f.a.watch(f.root)
    assert len(f.calls['sweeps'])==1


def test_failed_scientific_gate_remains_terminal(fixture):
    f=fixture;prepare(f)
    f.core._sweep=lambda *args,**kwargs:{'status':'needs_new_development_revision','failed':['pantry']}
    f.dispatch();result=f.a.watch(f.root)
    assert result['status']=='needs_new_development_revision'
    assert f.a.read(f.root/'terminal.json')['result']==result


def test_lost_lock_cannot_be_reacquired_or_allow_guest(fixture):
    f=fixture;prepare(f)
    with f.a.lifetime_fence() as fd:
        fcntl.flock(fd,fcntl.LOCK_UN)
        with pytest.raises(ValueError,match='no longer owns'):f.a.assert_fence(fd)
        probe=os.open(f.a.LOCK,os.O_RDWR)
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(probe)
    assert f.calls['sweeps']==[]


def test_local_duplicate_and_wrong_actual_host_block_probe(fixture,monkeypatch):
    f=fixture;monkeypatch.setattr(f.a,'local_controllers',lambda:[{'pid':123}])
    with pytest.raises(ValueError,match='another local'):f.a.host_probe()
    assert f.a.host_probe(123)['local_controllers']==[{'pid':123}]
    monkeypatch.setattr(f.a.os,'uname',lambda:SimpleNamespace(nodename='wash.cs.princeton.edu'))
    with pytest.raises(ValueError,match='actual spin'):f.a.host_probe(123)


def test_only_actual_controller_argv_matches(fixture):
    a=fixture.a
    assert not a.controller_invocation({'pid':os.getpid(),'command':['/bin/python3','-c',str(a.SOURCE),'watch']})
    assert a.controller_invocation({'pid':os.getpid(),'command':['/bin/python3','-B',str(a.SOURCE),'watch']})
    assert a.controller_invocation({'pid':os.getpid(),'command':['/bin/python3','-B',str(a.BASE_SOURCE),'watch']})


def runtime_fixture(f,fd):
    a=f.a;activation=a.verify(f.root);owner=a.process_identity(os.getpid())
    runtime={'schema':'modebench_scale_spin_authority_runtime_v1','host':a.HOST,'uid':os.getuid(),
        'owner_pid':owner['pid'],'owner_start_ticks':owner['start_ticks'],'owner_command':owner['command'],
        'authority_path':str(f.root/'activation.json'),'authority_sha256':a.sha(f.root/'activation.json'),
        'lock_path':str(a.LOCK),'lock_inode':a.LOCK_INODE,'lock_device':os.fstat(fd).st_dev,'fence_fd':fd,
        **{k:activation[k] for k in ('revised_recovery_job_id','revised_recovery_completion_path','revised_recovery_completion_sha256')}}
    write(f.root/'outer_runtime.json',runtime)
    return activation,runtime


@pytest.mark.parametrize('kind',['start_ticks','owner_command','dead','host','fd'])
def test_actual_owner_runtime_guard_precedes_advancement(fixture,monkeypatch,kind):
    f=fixture;prepare(f)
    with f.a.lifetime_fence() as fd:
        activation,runtime=runtime_fixture(f,fd)
        if kind=='start_ticks':mutate(f.root/'outer_runtime.json',lambda v:v.update(owner_start_ticks='0'))
        elif kind=='owner_command':mutate(f.root/'outer_runtime.json',lambda v:v.update(owner_command=['other']))
        elif kind=='dead':
            original=f.a.process_identity
            monkeypatch.setattr(f.a,'process_identity',lambda pid:{**original(pid),'state':'Z'})
        elif kind=='host':monkeypatch.setattr(f.a.os,'uname',lambda:SimpleNamespace(nodename='node202'))
        else:mutate(f.root/'outer_runtime.json',lambda v:v.update(fence_fd=fd+99))
        with pytest.raises(ValueError):f.a.actual_runtime(f.root,activation,fd)
    assert f.calls['sweeps']==[]


def test_sealed_real_transport_accepts_actual_driver_contract(fixture,monkeypatch):
    f=fixture;prepare(f)
    transport=load(f.a.TRANSPORT,'actual_transport_integrated_with_spin')
    monkeypatch.setattr(transport,'AUTHORITY_ROOT',f.root)
    monkeypatch.setattr(transport,'ARTIFACTS',f.a.ARTIFACTS)
    monkeypatch.setattr(transport,'module',lambda path,expected,name:f.a if path==f.a.SOURCE and expected==f.a.sha(f.a.SOURCE) else None)
    monkeypatch.setattr(transport.socket,'gethostname',lambda:f.a.HOST)
    with f.a.lifetime_fence() as fd:
        activation,runtime=runtime_fixture(f,fd)
        record=f.a.authority_record(f.root,activation,runtime,fd)
        f.canonical.write_text('old')
        try:
            pins=transport.authority_pins(record,live=True)
            assert pins[str(f.a.SOURCE)]==f.a.sha(f.a.SOURCE)
            assert pins[str(f.a.TRANSPORT_BASE)]==f.a.TRANSPORT_BASE_SHA
            assert pins[str(f.a.PREVIOUS_TRANSPORT)]==f.a.PREVIOUS_TRANSPORT_SHA
            monkeypatch.setattr(transport.socket,'gethostname',lambda:'node202.ionic.cs.princeton.edu')
            assert transport.authority_pins(record,live=False)==pins
            with pytest.raises(ValueError,match='actual spin'):transport.authority_pins(record,live=True)
        finally:f.canonical.write_text('neutral')


@pytest.mark.parametrize('kind',['missing','mismatch'])
def test_reviewed_nested_dependency_is_required(fixture,kind):
    f=fixture
    nested=str(ROOT/'tests/test_modebench_scale_completed_stage_cell_reconciliation.py')
    assert nested in f.a.read(f.a.CELL_REVIEWS[0])['files_sha256']
    def change(v):
        if kind=='missing':v['files_sha256'].pop(nested)
        else:v['files_sha256'][nested]='0'*64
    mutate(f.a.BUNDLE,change);f.bundle_sha=f.a.sha(f.a.BUNDLE)
    with pytest.raises(ValueError,match='reviewed dependency'):prepare(f)
    assert not f.a.CLAIM.exists()


def test_review_cannot_depend_on_final_bundle(fixture):
    f=fixture
    mutate(f.a.REVIEW,lambda v:v['files_sha256'].update({str(f.a.BUNDLE):f.bundle_sha}))
    mutate(f.a.BUNDLE,lambda v:v['files_sha256'].update({str(f.a.REVIEW):f.a.sha(f.a.REVIEW)}))
    f.bundle_sha=f.a.sha(f.a.BUNDLE)
    with pytest.raises(ValueError,match='final bundle'):prepare(f)
    assert not f.a.CLAIM.exists()
