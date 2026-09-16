from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops'))
import modebench_current_data as d


def test_neutral_python_refuses_data_without_admission(tmp_path,monkeypatch):
 monkeypatch.setattr(d,'ART',tmp_path)
 with pytest.raises(ValueError,match='awaits passing'):
  d.training_environment(3,'python_factors',{'OAT_ZERO_PROMPT_TEMPLATE':d.prompts.historical.profile_metadata(3,'python_factors')['template_name']})

def test_updates_actual_train_eval_variables_without_mutating_input(monkeypatch,tmp_path):
 root=tmp_path/'python_factors';proof={'dataset_identity_sha256':'data','admission_sha256':'admission'}
 monkeypatch.setattr(d,'neutral_python_dataset',lambda:(root,proof))
 env={'OAT_ZERO_PROMPT_TEMPLATE':d.prompts.historical.profile_metadata(3,'python_factors')['template_name'],'OAT_ZERO_PROMPT_DATA':'old train','OAT_ZERO_EVAL_DATA':'old eval','OAT_ZERO_DATA_ROOT':'old root'}
 result=d.training_environment(3,'python_factors',env)
 assert result['OAT_ZERO_PROMPT_DATA']==str(root/'train')
 assert result['OAT_ZERO_EVAL_DATA']==str(root/'eval')
 assert result['OAT_ZERO_DATA_ROOT']==str(root)
 assert result['OAT_ZERO_PROMPT_TEMPLATE']=='qwen_level3_python_factors_neutral_v1'
 assert env['OAT_ZERO_PROMPT_DATA']=='old train'

def test_historical_condition_preserves_registered_data(monkeypatch):
 def forbidden():raise AssertionError('historical condition must not consume neutral admission')
 monkeypatch.setattr(d,'neutral_python_dataset',forbidden)
 env={'OAT_ZERO_PROMPT_TEMPLATE':d.prompts.historical.profile_metadata(3,'python_factors')['template_name'],'OAT_ZERO_PROMPT_DATA':'old train','OAT_ZERO_EVAL_DATA':'old eval'}
 result=d.training_environment(3,'python_factors',env,condition=d.prompts.HISTORICAL)
 assert result['OAT_ZERO_PROMPT_DATA']=='old train' and result['OAT_ZERO_EVAL_DATA']=='old eval'
