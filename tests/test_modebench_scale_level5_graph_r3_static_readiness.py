"""Scratch historical actions and cross-client NFS identity, no real actions."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_ready_fits import fixture as fit_fixture
from test_modebench_scale_level5_graph_r3_registration import fixture as preparation_fixture, change, write

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(preparation_fixture, monkeypatch):
    x = preparation_fixture
    x.execute()
    path = ROOT/'artifacts/verify_modebench_scale_level5_graph_r3_static_readiness_20260912.py'
    spec = importlib.util.spec_from_file_location('scratch_static_graph_readiness', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    runtime_paths = [x.state/'action/runtime.json', x.f.state/'action/runtime.json']
    monkeypatch.setattr(adapter, 'HISTORICAL_RUNTIME_PINS', {p: adapter.file_sha(p) for p in runtime_paths})
    monkeypatch.setattr(adapter, 'HISTORICAL_SPIN_DEVICE', x.a.read(runtime_paths[0])['lock_device'])
    observed = x.state/'scratch_cause.json'
    write(observed, {'scratch_only': True, 'state': 'FAILED', 'exit_code': '1:0'})
    monkeypatch.setattr(adapter, 'OBSERVED_CAUSE_PINS', {observed: adapter.file_sha(observed)})
    worker_device = adapter.HISTORICAL_SPIN_DEVICE + 1
    stat_calls = []
    def worker_stat():
        stat_calls.append('worker stat')
        return SimpleNamespace(st_dev=worker_device, st_ino=x.a.LOCK_INODE)
    lock = SimpleNamespace(stat=worker_stat)
    monkeypatch.setattr(x.a, 'LOCK', lock)
    monkeypatch.setattr(x.f.a.first, 'LOCK', lock)
    return SimpleNamespace(x=x, a=adapter, stat_calls=stat_calls, observed=observed)


def test_original_static_checks_reproduce_both_cross_host_failures_then_adapter_reads_full_proof(fixture):
    f=fixture; x=f.x; a=f.a
    with pytest.raises(ValueError, match='exact recorded frozen guest'):
        x.a.successful_action(x.state)
    with pytest.raises(ValueError, match='did not complete successfully'):
        x.f.a.verify_existing(x.f.state,guest=False)
    assert f.stat_calls == ['worker stat','worker stat']
    before=(list(x.calls),list(x.f.calls)); f.stat_calls.clear()
    assert a.install_readonly_verifiers(x.a) is x.a
    value=x.a.development_inputs(x.state,guest=False)
    assert value['cell']['id']=='level5_graph_coloring_r3_dev'
    assert value['readiness_sha256']==x.a.sha(x.state/'certificate.json')
    assert (x.calls,x.f.calls)==before and not f.stat_calls


def test_live_guards_and_all_scientific_functions_are_identical_after_install(fixture):
    f=fixture;p=f.x.a
    before=(p.guard,p.assert_fence,p.lifetime_fence,p.process_identity,p.guest_action,p.run,p.prepare_graph,
            p.native_protocol,p.native_outputs,p.verify,p.inputs,p.development_inputs,
            p._first.guest_guard,p._first.assert_fence,p._first.run,p._first.fit_registered)
    f.a.install_readonly_verifiers(p)
    after=(p.guard,p.assert_fence,p.lifetime_fence,p.process_identity,p.guest_action,p.run,p.prepare_graph,
           p.native_protocol,p.native_outputs,p.verify,p.inputs,p.development_inputs,
           p._first.guest_guard,p._first.assert_fence,p._first.run,p._first.fit_registered)
    assert all(a is b for a,b in zip(before,after))


def test_static_proof_remains_valid_after_outputs_grow_without_adding_mutable_pins(fixture):
    f=fixture;p=f.a.install_readonly_verifiers(f.x.a)
    first=p.development_inputs(f.x.state,guest=False)
    output=first['cell']['tasks'][0]['output'];write(output,{'new_model_output': True})
    write(Path(output+'.batches')/'batch0.json',{'grows':True})
    assert p.development_inputs(f.x.state,guest=False)==first
    assert output not in first['files_sha256'] and not f.stat_calls


@pytest.mark.parametrize('which',['preparation','fit'])
@pytest.mark.parametrize('kind',['device','host','inode','source','uid','command','start_ticks'])
def test_exact_historical_runtime_tampering_rejects_even_after_internal_links_rewritten(fixture,which,kind):
    f=fixture;p=f.a.install_readonly_verifiers(f.x.a)
    root=f.x.state if which=='preparation' else f.x.f.state
    path=root/'action/runtime.json'
    field={'device':'lock_device','host':'host','inode':'lock_inode','source':'source_sha256',
           'uid':'uid','command':'command','start_ticks':'start_ticks'}[kind]
    change(path,lambda v:v.update({field:'tampered'}))
    change(root/'action/intent.json',lambda v:v.update(runtime_sha256=p.sha(path)))
    change(root/'action/guest_runtime.json',lambda v:v.update(outer_runtime_sha256=p.sha(path)))
    with pytest.raises(ValueError,match='historical static-readiness evidence|runtime bytes'):
        if which=='preparation':p.successful_action(root)
        else:p._first.verify_existing(root,guest=False)


@pytest.mark.parametrize('which',['preparation','fit'])
@pytest.mark.parametrize('kind',['nonzero','missing_exit','failure','guest_argv','intent_argv','guest_host','guest_uid','guest_fd','result'])
def test_full_original_ledger_predicates_still_refuse_failed_or_incomplete_actions(fixture,which,kind):
    f=fixture;p=f.a.install_readonly_verifiers(f.x.a);root=f.x.state if which=='preparation' else f.x.f.state
    if kind=='nonzero':change(root/'action/exit.json',lambda v:v.update(returncode=143))
    elif kind=='missing_exit':(root/'action/exit.json').unlink()
    elif kind=='failure':write(root/'action/failure.json',{'failed':True})
    elif kind=='guest_argv':change(root/'action/guest_runtime.json',lambda v:v['command'].remove('-B'))
    elif kind=='intent_argv':change(root/'action/intent.json',lambda v:v['command'].append('--wrong'))
    elif kind=='guest_host':change(root/'action/guest_runtime.json',lambda v:v.update(host='node202.ionic.cs.princeton.edu'))
    elif kind=='guest_uid':change(root/'action/guest_runtime.json',lambda v:v.update(uid=-1))
    elif kind=='guest_fd':change(root/'action/guest_runtime.json',lambda v:v.update(fence_fd=-1))
    else:change(root/'action/result.json',lambda v:v.update(status='invented_success'))
    with pytest.raises((ValueError,FileNotFoundError)):
        if which=='preparation':p.successful_action(root)
        else:p._first.verify_existing(root,guest=False)
    assert not f.stat_calls


@pytest.mark.parametrize('kind',['pool','recipe','tasks','missing_pin','certificate','amendment'])
def test_unchanged_native_structure_and_source_closure_still_validate(fixture,kind):
    f=fixture;x=f.x;p=f.a.install_readonly_verifiers(x.a)
    if kind=='pool':(x.paths(p.REVISION_ROOT)['pools']/'difficulty_0.jsonl').write_text('{}\n')
    elif kind=='recipe':change(x.entries[0]['recipe_path'],lambda v:v.update(development_fit_pass=True))
    elif kind=='tasks':change(x.state/'development_tasks.json',lambda v:v.pop())
    elif kind=='amendment':change(x.state/'amendment.json',lambda v:v.update(revision=2))
    else:
        value=p.read(x.state/'certificate.json')
        if kind=='missing_pin':value['files_sha256'].pop(str(x.state/'development_tasks.json'))
        else:value['source_binding_performed']=True
        write(x.state/'certificate.json',value);write(x.state/'action/result.json',value)
        write(x.state/'certificate.sha256.json',{'sha256':p.sha(x.state/'certificate.json')})
    with pytest.raises((ValueError,FileNotFoundError)):p.development_inputs(x.state,guest=False)


def test_reinstall_or_unsealed_method_replacement_rejected(fixture,monkeypatch):
    f=fixture;p=f.x.a
    original=p.successful_action
    monkeypatch.setattr(p,'successful_action',lambda root:None)
    with pytest.raises(ValueError,match='fresh sealed import'):f.a.install_readonly_verifiers(p)
    monkeypatch.setattr(p,'successful_action',original)
    f.a.install_readonly_verifiers(p)
    with pytest.raises(ValueError,match='fresh sealed import'):f.a.install_readonly_verifiers(p)


@pytest.mark.parametrize('kind',['provider_source','fit_source','cause','runtime'])
def test_explicit_source_and_observed_cause_pins_are_mandatory(fixture,monkeypatch,kind):
    f=fixture
    if kind=='provider_source':monkeypatch.setattr(f.a,'PROVIDER_SHA','0'*64)
    elif kind=='fit_source':monkeypatch.setattr(f.a,'FIT_SHA','0'*64)
    elif kind=='cause':write(f.observed,{'altered':True})
    else:monkeypatch.setattr(f.a,'HISTORICAL_RUNTIME_PINS',{next(iter(f.a.HISTORICAL_RUNTIME_PINS)):'0'*64})
    with pytest.raises(ValueError,match='evidence changed'):f.a.install_readonly_verifiers(f.x.a)


def test_static_functions_preserve_original_ast_except_explicit_historical_device_comparison():
    adapter=ast.parse((ROOT/'artifacts/verify_modebench_scale_level5_graph_r3_static_readiness_20260912.py').read_text())
    pairs=[('register_modebench_scale_level5_graph_r3_20260912.py','successful_action','preparation_successful_action','directory'),
           ('continue_modebench_scale_level5_ready_fits_20260912.py','verify_existing','fit_verify_existing','root')]
    class HistoricalDevice(ast.NodeTransformer):
        def visit_Call(self,node):
            return self.generic_visit(node)
        def visit_Attribute(self,node):
            if node.attr=='st_dev' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='stat':
                return ast.copy_location(ast.Name(id='HISTORICAL_SPIN_DEVICE',ctx=ast.Load()),node)
            return self.generic_visit(node)
    for file,old,new,start_name in pairs:
        tree=ast.parse((ROOT/'artifacts'/file).read_text())
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==old)
        revised=next(n for n in adapter.body if isinstance(n,ast.FunctionDef) and n.name==new)
        offset=next(i for i,n in enumerate(revised.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==start_name for t in n.targets))
        normalized=HistoricalDevice().visit(ast.Module(body=deepcopy(original.body),type_ignores=[]))
        assert ast.dump(normalized)==ast.dump(ast.Module(body=revised.body[offset:],type_ignores=[]))
