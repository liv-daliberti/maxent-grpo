"""Synthetic exact failed-launch continuation; no native science or production."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_ready_fits import fixture as prior_fixture
from test_modebench_scale_level4_python_r3_fit import fixture as python_fixture,write,change
from test_modebench_scale_level4_python_r3_fit_soak import fixture as soak_fixture
ROOT=Path(__file__).resolve().parents[1]


def load():
    path=ROOT/'artifacts/continue_modebench_scale_level4_python_r3_fit_soak_v2_20260913.py'
    spec=importlib.util.spec_from_file_location('_scratch_soak_v2_fit',path)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);return a


@pytest.fixture
def fixture(soak_fixture,monkeypatch):
    x=soak_fixture;a=load();m=x.m
    monkeypatch.setattr(a,'previous',x.a);monkeypatch.setattr(a,'engine',m)
    oldreview=m.read(m.REVIEW)
    previous_review=x.f.state.parent/'previous_observed_review.json';write(previous_review,oldreview)
    monkeypatch.setattr(a,'PREVIOUS_REVIEW',previous_review)
    failed=x.f.state.parent/'preserved_failed_attempt';monkeypatch.setattr(a,'FAILURE_STATE',failed)
    for key in ('SOURCE','TESTS'):monkeypatch.setattr(m,key,getattr(a,key))
    runtime={'at_utc':'2026-09-13T04:10:44.894963+00:00','command':[
        'var/seed_paper_eval/paper310/bin/python','-B',
        'artifacts/continue_modebench_scale_level4_python_r3_fit_soak_20260913.py','fit',
        '--root',str(failed),'--review-sha256',a.PREVIOUS_REVIEW_SHA],
        'pid':1136449,'start_ticks':'100803258','uid':x.a.LIVE_UID,'host':x.a.LIVE_HOST,
        'source_sha256':a.PREVIOUS_SHA,'lock_inode':m.first.LOCK_INODE,'lock_device':66,
        'view_manifest_sha256':m.sha(m.VIEW),'fence_fd':3}
    write(failed/'action/runtime.json',runtime)
    write(failed/'action/intent.json',{'at_utc':'2026-09-13T04:10:44.901806+00:00',
        'command':a.previous_guest_command(3),'runtime_sha256':m.sha(failed/'action/runtime.json')})
    write(failed/'action/exit.json',{'at_utc':'2026-09-13T04:10:45.447507+00:00','returncode':1})
    write(failed/'action/failure.json',{'at_utc':'2026-09-13T04:10:45.460813+00:00',
        'error':'ValueError','detail':'actual guest execution needs explicit reconciliation',
        'status':'explicit_reconciliation_required_no_automatic_retry',
        'files_sha256':{str(p):m.sha(p) for p in (failed/'action').iterdir()}})
    monkeypatch.setattr(a,'FAILURE_PINS',{p:m.sha(p) for p in (failed/'action').iterdir()})
    monkeypatch.setattr(m,'SEALED',{**m.SEALED,**a.PREVIOUS_PINS,**a.FAILURE_PINS})
    review=m.read(m.REVIEW)
    review['files_sha256'].update({str(p):m.sha(p) for p in (m.SOURCE,m.TESTS,*m.SEALED,previous_review)})
    review.update(prefit_failure_policy=a.FAILURE_POLICY,
        prefit_failure_sha256=a.FAILURE_PINS[failed/'action/failure.json'],
        prefit_failure_observation_sha256=a.PREVIOUS_PINS[a.FAILURE_OBSERVATION])
    write(m.REVIEW,review)
    monkeypatch.setattr(m,'reviewed',a.reviewed)
    x.f.owner['command']=[str(m.PYTHON),'-B',str(m.SOURCE),'fit']
    return SimpleNamespace(a=a,m=m,f=x.f,x=x,run=lambda:a.run(x.f.state,m.sha(m.REVIEW)))


def test_only_exact_prefit_failure_continues_once_with_unchanged_fence_and_fit(fixture):
    x=fixture;before={p:p.read_bytes() for p in x.a.FAILURE_PINS}
    result=x.run()
    assert x.f.calls==['python_factors'] and not x.f.held
    assert x.a.verify_existing(x.f.state)==result and x.f.calls==['python_factors']
    assert all(p.read_bytes()==data for p,data in before.items())
    assert x.m.first.HOST=='soak.cs.princeton.edu' and x.m.BASE.HOST=='spin.cs.princeton.edu'
    assert x.m.guest_guard.__code__.co_filename.endswith('continue_modebench_scale_level5_ready_fits_20260912.py')
    with pytest.raises(ValueError):x.run()
    assert x.f.calls==['python_factors']


@pytest.mark.parametrize('part,value',[(0,'var/seed_paper_eval/paper310/bin/python'),
    (1,'-u'),(2,'artifacts/continue_modebench_scale_level4_python_r3_fit_soak_v2_20260913.py'),(3,'verify')])
def test_relative_or_wrong_outer_command_refused_before_state_creation(fixture,part,value):
    x=fixture;x.f.owner['command'][part]=value
    with pytest.raises(ValueError,match='absolute literal Python and source'):
        x.run()
    assert not x.f.state.exists() and not x.f.calls


@pytest.mark.parametrize('name',['guest_runtime.json','registration.json','fits/python_factors/claim.json'])
def test_any_extra_previous_scientific_or_ambiguous_record_refuses_continuation(fixture,name):
    x=fixture
    path=x.a.FAILURE_STATE/('action/'+name if name=='guest_runtime.json' else name)
    write(path,{'unreviewed_prior_work':True})
    with pytest.raises(ValueError,match='only exact preguest failure'):
        x.run()
    assert not x.f.state.exists() and not x.f.calls


@pytest.mark.parametrize('name',['runtime.json','intent.json','exit.json','failure.json'])
def test_exact_failure_record_bytes_cannot_be_replaced_even_if_final_review_rehashed(fixture,name):
    x=fixture;path=x.a.FAILURE_STATE/'action'/name;change(path,ambiguous_change=True)
    review=x.m.read(x.m.REVIEW);review['files_sha256'][str(path)]=x.m.sha(path);write(x.m.REVIEW,review)
    with pytest.raises(ValueError):x.run()
    assert not x.f.state.exists() and not x.f.calls


def test_native_recipe_prevents_new_fit_even_without_new_state(fixture):
    x=fixture;write(x.m.REVISION_ROOT/'level4/recipes/python_factors.json',{'prior_recipe':True})
    with pytest.raises(ValueError,match='recipe already exists'):x.run()
    assert not x.f.state.exists() and not x.f.calls


@pytest.mark.parametrize('field',['prefit_failure_policy','prefit_failure_sha256','prefit_failure_observation_sha256'])
def test_final_review_requires_exact_explicit_failure_reconciliation(fixture,field):
    x=fixture;review=x.m.read(x.m.REVIEW);review.pop(field);write(x.m.REVIEW,review)
    with pytest.raises(ValueError,match='exact prefit launch failure'):x.run()
    assert not x.f.state.exists() and not x.f.calls


def test_negative_native_fit_is_preserved_after_prefit_reconciliation(fixture):
    x=fixture;x.f.failures.add('python_factors');result=x.run()
    assert result['failed_domains']==['python_factors'] and x.f.calls==['python_factors']
    assert x.a.verify_existing(x.f.state)==result


def test_original_live_fence_remains_enforced(fixture,monkeypatch):
    x=fixture
    def lost(fd):raise ValueError('original inherited fence lost')
    monkeypatch.setattr(x.m,'assert_fence',lost)
    with pytest.raises(ValueError,match='original inherited fence lost'):x.run()
    assert not x.f.calls
