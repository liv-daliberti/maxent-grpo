from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SOURCE=Path(__file__).resolve().parents[1]/'ops/exp_scaling/register_modebench_level3_v3.py'
spec=importlib.util.spec_from_file_location('v3_registration_test',SOURCE)
register=importlib.util.module_from_spec(spec)
spec.loader.exec_module(register)


def isolate(monkeypatch,tmp_path):
    common=register.common
    campaign=tmp_path/'campaign'
    monkeypatch.setattr(common,'CAMPAIGN',campaign)
    monkeypatch.setattr(common,'REGISTRATION',campaign/'registration.json')
    monkeypatch.setattr(common,'RESULTS',tmp_path/'results')
    monkeypatch.setattr(common,'DATASET',tmp_path/'dataset')
    monkeypatch.setattr(common,'POOL_ROOTS',{'graph_coloring':tmp_path/'graph','python_factors':tmp_path/'python'})
    return common


def test_previews_are_allowed_without_publishing_any_scientific_inputs(monkeypatch,tmp_path):
    common=isolate(monkeypatch,tmp_path)
    preview=common.CAMPAIGN/'graph_v8/preview.stdout'
    preview.parent.mkdir(parents=True)
    preview.write_text('structural preview')
    register.no_new_work()
    assert not common.REGISTRATION.exists()


@pytest.mark.parametrize('kind', ['registration','results','dataset','graph_pool','python_pool','seal','claim','legacy_claim','plan','prepare','worker','task','submission','submission_result','worker_claim'])
def test_new_work_or_existing_registration_refused(monkeypatch,tmp_path,kind):
    common=isolate(monkeypatch,tmp_path)
    paths={'registration':common.REGISTRATION,'results':common.RESULTS,'dataset':common.DATASET,
           'graph_pool':common.POOL_ROOTS['graph_coloring'],'python_pool':common.POOL_ROOTS['python_factors'],
           'seal':common.CAMPAIGN/'development/seal.json','claim':common.CAMPAIGN/'development/development_execution_claim.json',
           'legacy_claim':common.CAMPAIGN/'development/execution_claim.json','plan':common.CAMPAIGN/'development/plan.json',
           'prepare':common.CAMPAIGN/'development/prepare_intent.json','worker':common.CAMPAIGN/'development/worker.slurm',
           'task':common.CAMPAIGN/'development/3b_graph_v8_d0_tasks.json',
           'submission':common.CAMPAIGN/'development/submission_graph_0_intent.json',
           'submission_result':common.CAMPAIGN/'development/submission_00_result.json',
           'worker_claim':common.CAMPAIGN/'development/worker_00_execution_claim.json'}
    path=paths[kind]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.touch()
    with pytest.raises(ValueError):
        register.no_new_work()


def test_missing_launcher_stops_before_historical_outcome_reads(monkeypatch,tmp_path):
    common=isolate(monkeypatch,tmp_path)
    monkeypatch.setattr(register,'ROOT',tmp_path)
    monkeypatch.setattr(register,'REQUIRED',('missing_launcher.py',))
    monkeypatch.setattr(common,'digest',lambda path: register.AMENDMENT_SHA if Path(path)==register.AMENDMENT else register.COMMON_SHA)
    monkeypatch.setattr(common,'read',lambda path: {'contract':common.contract(),'status':'prospective_before_new_candidate_publication_or_model_outcomes'})
    monkeypatch.setattr(common,'authenticate_inherited',lambda:pytest.fail('should not inspect outcomes before missing source failure'))
    with pytest.raises(ValueError,match='required reviewed implementation is absent'):
        register.build_registration()


def test_five_targets_cannot_be_reselected_or_remeasured(monkeypatch,tmp_path):
    common=isolate(monkeypatch,tmp_path)
    monkeypatch.setattr(register,'REQUIRED',())
    monkeypatch.setattr(common,'digest',lambda path:register.AMENDMENT_SHA if Path(path)==register.AMENDMENT else register.COMMON_SHA)
    amendment={'contract':common.contract(),'status':'prospective_before_new_candidate_publication_or_model_outcomes',
               'benchmark_targets':{d:{'fixed_metrics':{'pass1':.2,'pass8':.5}} for d in common.DOMAINS}}
    monkeypatch.setattr(common,'read',lambda path:deepcopy(amendment))
    monkeypatch.setattr(common,'authenticate_inherited',lambda:{})
    targets={d:{'metrics':{'pass1':.2,'pass8':.5}} for d in common.DOMAINS}
    targets['python_factors']['metrics']['pass1']=.21
    monkeypatch.setattr(common,'fixed_references',lambda:targets)
    with pytest.raises(ValueError,match='all five measured reference targets'):
        register.build_registration()


def test_default_command_never_publishes(monkeypatch,tmp_path):
    common=isolate(monkeypatch,tmp_path)
    record={'files_sha256':{'one':'hash'},'directory_files':{},
            'candidate_revisions':{'graph_coloring':{'rows_per_tier':142}}}
    monkeypatch.setattr(register,'build_registration',lambda:record)
    monkeypatch.setattr(common,'atomic_new',lambda *args:pytest.fail('read-only invocation published'))
    monkeypatch.setattr(sys,'argv',['register'])
    register.main()
    assert not common.REGISTRATION.exists()


def test_publish_validates_exact_written_registration(monkeypatch,tmp_path):
    common=isolate(monkeypatch,tmp_path)
    record={'files_sha256':{'one':'hash'},'directory_files':{}}
    monkeypatch.setattr(register,'build_registration',lambda:record)
    seen=[]
    def validate(path,pin):
        assert json.loads(path.read_text())==record and common.digest(path)==pin
        seen.append((path,pin))
    monkeypatch.setattr(common,'validate_registration',validate)
    monkeypatch.setattr(sys,'argv',['register','--publish'])
    register.main()
    assert len(seen)==1 and seen[0][0]==common.REGISTRATION
