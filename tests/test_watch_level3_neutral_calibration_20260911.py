"""The observer reuses completed steps and never retries scientific selection."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops'))
from watch_level3_neutral_calibration_20260911 import advance

@pytest.fixture
def c(tmp_path):
    art=tmp_path/'art';art.mkdir();data=tmp_path/'data'
    calls=[]
    def require(ok,message):
        if not ok:raise ValueError(message)
    def write(p,v):p.write_text(json.dumps(v))
    def task(t=None):return {'output':str(tmp_path/('confirmation.json' if t is None else f'd{t}.json'))}
    recipe={'registration_sha256':'registered','receipts_sha256':{},'development_fit_pass':True,'gates':{}}
    def fit(_):calls.append('fit');write(art/'recipe.json',recipe);return recipe
    def finalize(_):calls.append('finalize');data.mkdir();write(data/'identity.json',{'files_sha256':{}})
    def submit(name,cmd):calls.append('submit');write(art/'confirmation_submission_result.json',{'returncode':0,'stdout':'123'});return '123'
    def audit(_):
        calls.append('audit');result={'registration_sha256':'registered','admitted':True,'status':'observed_approximate_match','neutral_level3':{},'deltas':{},'gates':{}}
        write(art/'confirmation_report.json',result);write(art/'admission.json',result);return result
    obj=SimpleNamespace(ART=art,DATA=data,read=lambda p:json.loads(p.read_text()),require=require,
        common=SimpleNamespace(verify_pins=lambda _:None),validate_plan=lambda _:None,task=task,fit=fit,finalize=finalize,
        submit_once=submit,gpu_command=lambda *_:[],audit=audit,calls=calls,recipe=recipe)
    return obj

def complete_development(c):
    for t in range(4):Path(c.task(t)['output']).write_text('{}')

def test_waits_for_all_development(c):
    assert advance(c,'registered')['status']=='waiting_development'
    assert c.calls==[]

def test_continues_once_and_reuses_confirmation(c):
    complete_development(c)
    assert advance(c,'registered')['job_id']=='123'
    assert advance(c,'registered')['job_id']=='123'
    assert c.calls==['fit','finalize','submit']
    Path(c.task()['output']).write_text('{}')
    assert advance(c,'registered')['admitted']
    assert advance(c,'registered')['admitted']
    assert c.calls==['fit','finalize','submit','audit']

def test_failed_fit_never_builds_or_resamples(c):
    complete_development(c);c.recipe['development_fit_pass']=False
    assert advance(c,'registered')['terminal']
    assert advance(c,'registered')['terminal']
    assert c.calls==['fit']

def test_ambiguous_submission_is_not_repeated(c):
    complete_development(c);(c.ART/'confirmation_submission_intent.json').write_text('{}')
    with pytest.raises(ValueError,match='ambiguous'):advance(c,'registered')
    assert c.calls==['fit','finalize']
