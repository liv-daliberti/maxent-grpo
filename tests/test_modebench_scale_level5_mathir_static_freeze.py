"""Scratch-only portable MathIR historical proof; never actual verification/actions."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_ready_fits import fixture as fit_fixture
from test_modebench_scale_level5_mathir_freeze import fixture as freeze_fixture, change, write
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(freeze_fixture, monkeypatch):
    x = freeze_fixture
    certificate = x.execute()
    path = ROOT/'artifacts/verify_modebench_scale_level5_mathir_static_freeze_20260912.py'
    spec = importlib.util.spec_from_file_location('scratch_mathir_portable_adapter', path)
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    fit_runtime = x.f.state/'action/runtime.json'
    freeze_runtime = x.a.area(x.f.state)/'action/runtime.json'
    cert_path = x.a.area(x.f.state)/'certificate.json'
    device = x.a.read(freeze_runtime)['lock_device']
    for name, value in {'STATE': x.f.state, 'FIT_RUNTIME': fit_runtime, 'FIT_RUNTIME_SHA': adapter.file_sha(fit_runtime),
            'FREEZE_RUNTIME': freeze_runtime, 'FREEZE_RUNTIME_SHA': adapter.file_sha(freeze_runtime),
            'CERTIFICATE': cert_path, 'CERTIFICATE_SHA': adapter.file_sha(cert_path),
            'HISTORICAL_SPIN_DEVICE': device}.items():
        monkeypatch.setattr(adapter, name, value)
    monkeypatch.setattr(adapter._graph, 'HISTORICAL_SPIN_DEVICE', device)
    monkeypatch.setattr(adapter._graph, 'HISTORICAL_RUNTIME_PINS', {fit_runtime: adapter.file_sha(fit_runtime)})
    calls = []
    def hostile_stat():
        calls.append('current stat')
        raise AssertionError('historical verification may not stat the current lock')
    lock = SimpleNamespace(stat=hostile_stat)
    monkeypatch.setattr(x.a, 'LOCK', lock)
    monkeypatch.setattr(x.f.a.first, 'LOCK', lock)
    return SimpleNamespace(x=x, a=adapter, stat_calls=calls, certificate=certificate)


def test_both_original_predicates_use_current_stat_but_full_adapted_verify_does_not(fixture):
    f=fixture;x=f.x
    with pytest.raises(AssertionError,match='current lock'):x.a.successful_action(x.f.state)
    with pytest.raises(AssertionError,match='current lock'):x.f.a.verify_existing(x.f.state,guest=False)
    assert f.stat_calls == ['current stat','current stat']
    f.stat_calls.clear();before=(list(x.calls),list(x.f.calls))
    assert f.a.install_readonly_verifiers(x.a) is x.a
    assert x.a.verify(x.f.state,guest=False)==f.certificate
    assert (x.calls,x.f.calls)==before and not f.stat_calls
    assert f.certificate['graph_development_fit_pass'] is False
    assert f.certificate['domains']==['mathir']


def test_only_two_historical_functions_change_and_every_live_guard_stays_original(fixture):
    f=fixture;p=f.x.a
    original=(p.guard,p.assert_fence,p.lifetime_fence,p.process_identity,p.guest_action,p.run,p.freeze_passed,
        p.verify,p.inputs,p.command,p._first.guest_guard,p._first.assert_fence,p._first.run,p._first.fit_registered)
    f.a.install_readonly_verifiers(p)
    after=(p.guard,p.assert_fence,p.lifetime_fence,p.process_identity,p.guest_action,p.run,p.freeze_passed,
        p.verify,p.inputs,p.command,p._first.guest_guard,p._first.assert_fence,p._first.run,p._first.fit_registered)
    assert all(a is b for a,b in zip(original,after))


def test_public_verify_loads_a_private_provider_and_returns_original_certificate(fixture,monkeypatch):
    f=fixture;imports=[]
    def private(path,name):
        imports.append((path,name));return f.x.a
    monkeypatch.setattr(f.a,'private_import',private)
    assert f.a.verify(f.x.f.state,guest=False)==f.certificate
    assert imports==[(f.a.PROVIDER,'_mathir_portable_private_freeze')]
    assert not f.stat_calls


def test_direct_sealed_fit_verifier_is_reused_without_copying_or_modifying_it(fixture):
    f=fixture;before=f.a._graph.fit_verify_existing
    f.a.install_readonly_verifiers(f.x.a)
    assert f.x.f.a.verify_existing(f.x.f.state,guest=False)['failed_domains']==['graph_coloring']
    assert f.a._graph.fit_verify_existing is before is f.a._graph_fit_verifier
    assert before.__code__.co_filename==str(f.a.GRAPH_ADAPTER)
    assert not f.stat_calls


@pytest.mark.parametrize('which',['freeze','fit'])
@pytest.mark.parametrize('field',['lock_device','host','lock_inode','source_sha256','uid','command','start_ticks','pid','view_manifest_sha256','fence_fd'])
def test_exact_runtime_bytes_reject_tampering_even_with_rewritten_internal_links(fixture,which,field):
    f=fixture;p=f.a.install_readonly_verifiers(f.x.a)
    directory=(p.area(f.x.f.state) if which=='freeze' else f.x.f.state)/'action'
    path=directory/'runtime.json';change(path,lambda v:v.update({field:'changed'}))
    change(directory/'intent.json',lambda v:v.update(runtime_sha256=p.sha(path)))
    change(directory/'guest_runtime.json',lambda v:v.update(outer_runtime_sha256=p.sha(path)))
    with pytest.raises(ValueError,match='runtime bytes'):
        if which=='freeze':p.successful_action(f.x.f.state)
        else:p._first.verify_existing(f.x.f.state,guest=False)
    assert not f.stat_calls


@pytest.mark.parametrize('which',['freeze','fit'])
@pytest.mark.parametrize('kind',['nonzero','missing_exit','failure','guest_argv','intent_argv','guest_host','guest_uid','guest_fd','result'])
def test_original_ledger_predicates_retain_all_rejections(fixture,which,kind):
    f=fixture;p=f.a.install_readonly_verifiers(f.x.a)
    directory=(p.area(f.x.f.state) if which=='freeze' else f.x.f.state)/'action'
    if kind=='nonzero':change(directory/'exit.json',lambda v:v.update(returncode=143))
    elif kind=='missing_exit':(directory/'exit.json').unlink()
    elif kind=='failure':write(directory/'failure.json',{'failed':True})
    elif kind=='guest_argv':change(directory/'guest_runtime.json',lambda v:v['command'].remove('-B'))
    elif kind=='intent_argv':change(directory/'intent.json',lambda v:v['command'].append('--altered'))
    elif kind=='guest_host':change(directory/'guest_runtime.json',lambda v:v.update(host='node202.ionic.cs.princeton.edu'))
    elif kind=='guest_uid':change(directory/'guest_runtime.json',lambda v:v.update(uid=-1))
    elif kind=='guest_fd':change(directory/'guest_runtime.json',lambda v:v.update(fence_fd=-1))
    else:change(directory/'result.json',lambda v:v.update(status='invented_success'))
    with pytest.raises((ValueError,FileNotFoundError)):
        if which=='freeze':p.successful_action(f.x.f.state)
        else:p._first.verify_existing(f.x.f.state,guest=False)
    assert not f.stat_calls


@pytest.mark.parametrize('kind',['dataset','claim','recipe','missing_certificate_pin','changed_graph_gate'])
def test_full_scientific_proof_stays_original_and_never_calls_native_actions(fixture,kind):
    f=fixture;x=f.x;p=f.a.install_readonly_verifiers(x.a);area=p.area(x.f.state)
    if kind=='dataset':change(Path(f.certificate['frozen'][0]['dataset'])/'identity.json',lambda v:v['splits']['train'].update(rows=383))
    elif kind=='claim':change(area/'domains/mathir/claim.json',lambda v:v.update(source_root='/other'))
    elif kind=='recipe':change(x.entries[0]['recipe_path'],lambda v:v.update(development_fit_pass=True))
    else:
        value=p.read(area/'certificate.json')
        if kind=='missing_certificate_pin':value['files_sha256'].pop(str(area/'domains/mathir/claim.json'))
        else:value['graph_development_fit_pass']=True
        write(area/'certificate.json',value);write(area/'action/result.json',value)
        write(area/'certificate.sha256.json',{'sha256':p.sha(area/'certificate.json')})
    before=(list(x.calls),list(x.f.calls))
    with pytest.raises((ValueError,FileNotFoundError)):p.verify(x.f.state,guest=False)
    assert (x.calls,x.f.calls)==before and not f.stat_calls


@pytest.mark.parametrize('field',['PROVIDER_SHA','FIT_SHA','GRAPH_ADAPTER_SHA','FREEZE_RUNTIME_SHA','FIT_RUNTIME_SHA','CERTIFICATE_SHA'])
def test_exact_source_runtime_certificate_pins_required(fixture,monkeypatch,field):
    f=fixture;monkeypatch.setattr(f.a,field,'0'*64)
    with pytest.raises(ValueError,match='evidence changed'):f.a.install_readonly_verifiers(f.x.a)


def test_modified_graph_delegate_is_rejected(fixture,monkeypatch):
    f=fixture;monkeypatch.setattr(f.a._graph,'fit_verify_existing',lambda *a,**k:None)
    with pytest.raises(ValueError,match='unchanged exact-runtime'):f.a.install_readonly_verifiers(f.x.a)


def test_reinstall_unsealed_replacement_and_wrong_root_are_rejected(fixture,monkeypatch):
    f=fixture;p=f.x.a;original=p.successful_action
    monkeypatch.setattr(p,'successful_action',lambda root:None)
    with pytest.raises(ValueError,match='fresh sealed import'):f.a.install_readonly_verifiers(p)
    monkeypatch.setattr(p,'successful_action',original)
    f.a.install_readonly_verifiers(p)
    with pytest.raises(ValueError,match='fresh sealed import'):f.a.install_readonly_verifiers(p)
    with pytest.raises(ValueError,match='exact historical MathIR'):p.successful_action(f.x.f.state.parent)


def test_action_body_ast_preserves_every_clause_except_current_device_comparison():
    old=ast.parse((ROOT/'artifacts/freeze_modebench_scale_level5_mathir_20260912.py').read_text())
    new=ast.parse((ROOT/'artifacts/verify_modebench_scale_level5_mathir_static_freeze_20260912.py').read_text())
    original=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='successful_action')
    adapted=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='freeze_successful_action')
    class HistoricalDevice(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if node.attr=='st_dev' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='stat':
                return ast.copy_location(ast.Name(id='HISTORICAL_SPIN_DEVICE',ctx=ast.Load()),node)
            return self.generic_visit(node)
    offset=next(i for i,n in enumerate(adapted.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='directory' for t in n.targets))
    normalized=HistoricalDevice().visit(ast.Module(body=deepcopy(original.body),type_ignores=[]))
    assert ast.dump(normalized)==ast.dump(ast.Module(body=adapted.body[offset:],type_ignores=[]))
