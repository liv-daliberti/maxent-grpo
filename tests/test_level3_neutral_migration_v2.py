from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import migrate_level3_neutral_v2 as migration


def test_historical_source_redirect_requires_exact_archived_bytes(tmp_path,monkeypatch):
    archived=tmp_path/'archived.py'; archived.write_text('original')
    current=tmp_path/'current.py'; current.write_text('new neutral interface')
    monkeypatch.setattr(migration,'ORIGINAL_TEMPLATE',archived)
    monkeypatch.setattr(migration,'TEMPLATE',current)
    old=migration.digest(archived); monkeypatch.setattr(migration,'OLD_TEMPLATE_SHA',old)
    migration.verify_historical_pins({str(current):old})
    with pytest.raises(ValueError):migration.verify_historical_pins({str(current):'0'*64})
    archived.write_text('changed')
    with pytest.raises(ValueError):migration.verify_historical_pins({str(current):old})


def test_admission_rejects_failed_calibration(tmp_path,monkeypatch):
    monkeypatch.setattr(migration.calibration,'ART',tmp_path)
    (tmp_path/'admission.json').write_text(json.dumps({'admitted':False}))
    with pytest.raises(ValueError,match='difficulty'):migration.admission()


def test_new_cell_changes_prompt_data_namespace_only(monkeypatch):
    old={'domain':'python_factors','level':3,'arm':'maxrl','seed':43,'run_stamp':'old','run_dir':'/old',
         'environment':{'OAT_ZERO_PROMPT_DATA':'old_train','OAT_ZERO_EVAL_DATA':'old_eval'},'resources':{'gpus':1}}
    prepared={'environment':{'OAT_ZERO_PROMPT_TEMPLATE':'qwen_level3_python_factors_neutral_v1','SAVE_PATH':'/prepared',
                            'OAT_ZERO_PROMPT_DATA':'old_train','OAT_ZERO_EVAL_DATA':'old_eval','OAT_ZERO_LEARNING_RATE':'2e-7'},
              'held_submission_command':['sbatch','--hold','--job-name=prepared','--export=ALL,X=y','script']}
    monkeypatch.setattr(migration,'digest',lambda _: 'b'*64)
    original=deepcopy(old)
    cell=migration.make_cell(old,prepared,'e122','a'*64)
    assert old==original
    assert cell['resources']==old['resources']
    assert cell['environment']['OAT_ZERO_LEARNING_RATE']=='2e-7'
    assert cell['environment']['OAT_ZERO_PROMPT_DATA'].endswith('modebench_level3_matched_neutral_v2/python_factors/train')
    assert cell['environment']['OAT_ZERO_EVAL_DATA'].endswith('modebench_level3_matched_neutral_v2/python_factors/eval')
    assert cell['run_dir'] not in ('/old','/prepared')
    assert '--hold' in cell['command']


@pytest.mark.parametrize('field,value',[('JobState','RUNNING'),('Restarts','1'),('RunTime','00:00:01'),('StartTime','2026-09-11T12:00:00')])
def test_started_jobs_never_migrate(field,value,monkeypatch):
    import os
    fields={'JobId':'123','JobState':'PENDING','Reason':'JobHeldUser','Priority':'0',
            'RunTime':'00:00:00','Restarts':'0','StartTime':'Unknown','UserId':f'user({os.getuid()})'}
    fields[field]=value
    monkeypatch.setattr(migration,'run',lambda _: ' '.join(k+'='+v for k,v in fields.items()))
    with pytest.raises(ValueError,match='started'):migration.held_without_training('123',{'run_dir':'unused'},'e122')


def test_plan_and_admission_required_before_any_queue_action(tmp_path,monkeypatch):
    plan=tmp_path/'plan.json';plan.write_text('{}');monkeypatch.setattr(migration,'PLAN',plan)
    called=[];monkeypatch.setattr(migration,'run',lambda x:called.append(x))
    with pytest.raises(ValueError,match='hash'):migration.migrate('0'*64)
    assert not called


def test_all_real_prepared_cells_redirect_the_actual_training_loader(monkeypatch):
    prepared=migration.read(migration.PREPARED)
    monkeypatch.setattr(migration,'digest',lambda _:'b'*64)
    for campaign in prepared['campaigns']:
        original=migration.read(migration.E122_PLAN if campaign['campaign']=='e122' else migration.E124_PLAN)
        for item in campaign['cells']:
            matches=[c for c in original['cells'] if c['domain']=='python_factors' and c.get('level',3)==3 and c['arm']==item['arm'] and c['seed']==item['seed']]
            assert len(matches)==1
            source=matches[0]
            cell=migration.make_cell(source,item,campaign['campaign'],'a'*64)
            env=cell['environment']
            assert env['OAT_ZERO_PROMPT_DATA']==str(migration.calibration.DATA/'python_factors/train')
            assert env['OAT_ZERO_EVAL_DATA']==str(migration.calibration.DATA/'python_factors/eval')
            assert 'OAT_ZERO_TRAIN_DATA' not in env
            if 'OAT_ZERO_DATA_ROOT' in source['environment']:
                assert env['OAT_ZERO_DATA_ROOT']==str(migration.calibration.DATA/'python_factors')
            assert cell['resources']==source['resources']
            assert '--hold' in cell['command']
