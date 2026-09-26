"""Synthetic evidence gates only; no native fit, grader, freeze or binding."""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path('/n/fs/similarity/maxent-grpo')
SOURCE=ROOT/'artifacts/build_modebench_scale_level4_first_release_r4_final_review_20260913.py'
spec=importlib.util.spec_from_file_location('_release_review_builder_tests',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True));return p

@pytest.fixture
def evidence(tmp_path,monkeypatch):
    m=SimpleNamespace()
    for name in ('SOURCE','TESTS','PREDECESSOR_COPY','PREDECESSOR_TESTS_COPY','PREDECESSOR_MANIFEST','FIRST','PREPARER','FIT_PROVIDER','FIT_TESTS','STAGER','CORE'):
        setattr(m,name,put(tmp_path/(name+'.json'),{'fixture':name}))
    m.REVIEW=tmp_path/'final_provider_review.json';m.STATE=tmp_path/'release_state';m.RELEASE=tmp_path/'release';m.SOURCE_MANIFEST=m.RELEASE/'level4/source_manifest.json'
    m.FIT_STATE=tmp_path/'completed_fit';m.FIT_REVIEW=tmp_path/'final_fit_review.json';m.FIT_RESULT=m.FIT_STATE/'action/result.json'
    m.PYTHON_ROOT=tmp_path/'python_r4';m.PYTHON_RECIPE=m.PYTHON_ROOT/'level4/recipes/python_factors.json';m.PREPARATION_STATE=tmp_path/'prepared'
    m.RETAINED=('countdown','graph_coloring','mathir','pantry');calls=[]
    monkeypatch.setattr(b,'PROVIDER_SHA',b.sha(m.SOURCE));monkeypatch.setattr(b,'PROVIDER_TESTS_SHA',b.sha(m.TESTS))
    recipe={'domain':'python_factors','level':'level4','development_fit_pass':True,'input_sha256':{str(m.FIT_PROVIDER):b.sha(m.FIT_PROVIDER)}}
    put(m.PYTHON_RECIPE,recipe)
    result={'schema':'modebench_scale_level4_python_r4_fit_v1','status':'level4_python_r4_development_gates_passed','level':'level4','domains':['python_factors'],
        'failed_domains':[],'fits':[{'recipe_path':str(m.PYTHON_RECIPE),'recipe_sha256':b.sha(m.PYTHON_RECIPE),'development_fit_pass':True}]}
    put(m.FIT_RESULT,result);put(m.FIT_STATE/'action/exit.json',{'returncode':0})
    put(m.FIT_STATE/'registration.json',{'files_sha256':{str(m.FIT_PROVIDER):b.sha(m.FIT_PROVIDER)}})
    put(m.FIT_STATE/'registration.sha256.json',{'sha256':b.sha(m.FIT_STATE/'registration.json')})
    m._first=SimpleNamespace(TESTS=put(tmp_path/'first_tests.json',{}),REVIEW=put(tmp_path/'first_review.json',{'status':'reviewed','files_sha256':{str(m.FIRST):b.sha(m.FIRST)}}),SEALED={m.FIRST:b.sha(m.FIRST)})
    prep=SimpleNamespace(TESTS=put(tmp_path/'prep_tests.json',{}),REVIEW=put(tmp_path/'prep_review.json',{'status':'reviewed','files_sha256':{str(m.PREPARER):b.sha(m.PREPARER)}}),FAILED_RECIPE_SHA='f'*64)
    prep.development_inputs=lambda *a,**kw:{'files_sha256':{str(m.PREPARER):b.sha(m.PREPARER)}}
    fit=SimpleNamespace(SOURCE=m.FIT_PROVIDER,STATE=m.FIT_STATE,REVIEW=m.FIT_REVIEW)
    def verify(*a,**kw):calls.append('readonly_fit_verification');return b.read(m.FIT_RESULT)
    fit.verify_existing=verify
    put(m.FIT_REVIEW,{'schema':'modebench_scale_level4_python_r4_fit_independent_review_v1','status':'reviewed','files_sha256':{str(m.FIT_PROVIDER):b.sha(m.FIT_PROVIDER),str(m.FIT_TESTS):b.sha(m.FIT_TESTS)}})
    retained={};paths={};identities={}
    for d in m.RETAINED:
        root=tmp_path/('retained_'+d);dataset=root/'dataset';identity={'splits':{'train':384,'validation':128,'test':128}}
        put(dataset/'identity.json',identity);put(dataset/'data.json',{'retained':d});paths[str(root)]={'dataset':dataset};identities[str(root)]=identity
        retained[d]={'source':{'source_root':str(root),'source_kind':'campaign_v1' if d in ('countdown','mathir') else 'domain_revision_v1'},'dataset':str(dataset),'identity_sha256':b.sha(dataset/'identity.json'),'splits':identity['splits']}
    core=SimpleNamespace(domain_paths=lambda root,*a:paths[str(root)],frozen_identity=lambda root,*a:identities[str(root)])
    cert={'preserved_python_failure_sha256':prep.FAILED_RECIPE_SHA,'preserved_level4_sources':retained,'files_sha256':{str(m.PREPARER):b.sha(m.PREPARER)}}
    put(m.PREPARATION_STATE/'certificate.json',cert);m.PREPARATION_SHA=b.sha(m.PREPARATION_STATE/'certificate.json')
    m.load_module=lambda path,name:{m.FIT_PROVIDER:fit,m.PREPARER:prep,m.CORE:core}[path]
    m.pin_files=lambda pins,files:b.merge(pins,{str(p):b.sha(p) for p in files})
    def check(pins,**kw):
        for p,h in pins.items():assert b.sha(p)==h
    m.check_pins=check;m.unobserved=lambda *a:calls.append('readonly_unobserved_paths')
    component=put(tmp_path/'component.json',{'status':'reviewed_current_components','blocking_findings':[],'files_sha256':{str(p):b.sha(p) for p in (m.SOURCE,m.TESTS,m.FIT_PROVIDER,m.FIT_TESTS)}})
    args=SimpleNamespace(component_review=component,component_review_sha256=b.sha(component),fit_provider_sha256=b.sha(m.FIT_PROVIDER),fit_tests_sha256=b.sha(m.FIT_TESTS),fit_review_sha256=b.sha(m.FIT_REVIEW),fit_result_sha256=b.sha(m.FIT_RESULT),recipe_sha256=b.sha(m.PYTHON_RECIPE))
    return SimpleNamespace(m=m,args=args,fit=fit,recipe=recipe,result=result,calls=calls,retained=retained)

def test_complete_saved_pass_assembles_without_publication_or_scientific_action(evidence):
    x=evidence;v=b.assemble(x.m,x.args)
    assert v['python_recipe_sha256']==x.args.recipe_sha256 and v['retained_four']==x.retained
    assert x.calls==['readonly_fit_verification','readonly_unobserved_paths']
    assert not x.m.REVIEW.exists() and not x.m.STATE.exists() and not x.m.RELEASE.exists()
    assert all(v[k]==0 for k in ('fit_calls','native_grader_calls','model_calls','freeze_calls','source_bindings','scheduler_calls'))
    assert str(x.m.FIT_RESULT) in v['files_sha256']

@pytest.mark.parametrize('field',['fit_provider_sha256','fit_tests_sha256','fit_review_sha256','fit_result_sha256','recipe_sha256','component_review_sha256'])
def test_explicit_wrong_actual_digest_rejected_before_publication(evidence,field):
    x=evidence;setattr(x.args,field,'0'*64)
    with pytest.raises(ValueError):b.assemble(x.m,x.args)
    assert not x.m.REVIEW.exists()

@pytest.mark.parametrize('kind',['negative_recipe','negative_result','wrong_domain','wrong_recipe_path','wrong_recipe_sha','verification_disagrees','exit1','exit143','failure_record'])
def test_nonpassing_or_partial_fit_never_admits_provider_review(evidence,kind):
    x=evidence;result=deepcopy(x.result);recipe=deepcopy(x.recipe)
    if kind=='negative_recipe':recipe['development_fit_pass']=False
    elif kind=='negative_result':result['fits'][0]['development_fit_pass']=False
    elif kind=='wrong_domain':result['domains']=['pantry']
    elif kind=='wrong_recipe_path':result['fits'][0]['recipe_path']='/tmp/other_recipe.json'
    elif kind=='wrong_recipe_sha':result['fits'][0]['recipe_sha256']='0'*64
    elif kind=='verification_disagrees':x.fit.verify_existing=lambda *a,**kw:{'status':'different'}
    elif kind=='failure_record':put(x.m.FIT_STATE/'action/failure.json',{'error':'saved failure'})
    else:put(x.m.FIT_STATE/'action/exit.json',{'returncode':1 if kind=='exit1' else 143})
    with pytest.raises(ValueError):b.saved_pass(x.m,x.fit,result,recipe)
    assert not x.m.REVIEW.exists()

@pytest.mark.parametrize('field',['REVIEW','STATE','RELEASE'])
def test_future_output_pins_are_rejected(evidence,field):
    x=evidence
    with pytest.raises(ValueError):b.no_future(x.m,{str(getattr(x.m,field)/'future.json'):'0'*64})

@pytest.mark.parametrize('field',['STATE','REVIEW'])
def test_dangling_claim_or_review_path_prevents_assembly(evidence,field):
    x=evidence;getattr(x.m,field).symlink_to(getattr(x.m,field).with_name('absent_target'))
    with pytest.raises(ValueError):b.assemble(x.m,x.args)
    assert not x.calls


def test_retained_identity_changes_block_release_review(evidence):
    x=evidence;path=Path(x.retained['countdown']['dataset'])/'identity.json';put(path,{'changed':True})
    with pytest.raises(ValueError,match='retained'):b.assemble(x.m,x.args)
    assert not x.m.REVIEW.exists()


def test_review_file_publication_is_exclusive_and_does_not_follow_symlinks(tmp_path):
    p=tmp_path/'review.json';b.write_new(p,{'checked':True});before=p.read_bytes()
    with pytest.raises(FileExistsError):b.write_new(p,{'overwrite':True})
    assert p.read_bytes()==before and p.stat().st_mode & 0o777==0o444
    link=tmp_path/'link.json';target=tmp_path/'absent.json';link.symlink_to(target)
    with pytest.raises(FileExistsError):b.write_new(link,{})
    assert not target.exists()


def test_builder_has_no_native_mutation_calls():
    tree=ast.parse(SOURCE.read_text());forbidden={'fit_domain','fit_registered','audit','register','stage','freeze_dataset','freeze_bind','bind_level','run','Popen','guest','submit','materialize_pools'}
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls & forbidden
