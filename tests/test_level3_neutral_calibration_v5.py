"""Neutral calibration preserves measurement and refuses mismatched evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import evaluate_modebench_level3_neutral as neutral
import evaluate_modebench_level3_independent as historical
import calibrate_modebench_level3_neutral_v5 as campaign
import modebench_level3_python_neutral_v5 as generator
import modebench_level3_python_v7 as old_generator
from modebench_current_contract import make_messages


def test_adapter_does_not_mutate_original():
    assert historical.INTERFACE=='level2_qwen_r5_independent_v2'
    assert historical.frozen_interface('python_factors')['prompt_profile']=='hybrid_solver_v4'
    assert old_generator.CASE_WINDOWS==((60,224),(60,256),(60,224),(60,256))
    old=historical.frozen_interface('python_factors'); new=neutral.frozen_interface('python_factors')
    changed={'name','prompt_profile','prompt_condition'}
    assert {k:v for k,v in old.items() if k not in changed}=={k:v for k,v in new.items() if k not in changed}
    assert neutral.prompt_messages('python_factors','Task',neutral.CURRENT)==make_messages(3,'python_factors',{'problem':'Task'})
    with pytest.raises(ValueError): neutral.frozen_interface('pantry')


def test_task_confirmation_requires_explicit_phase():
    historical.validate_task(dict(campaign.task(0),interface=historical.INTERFACE),False)
    with pytest.raises(ValueError,match='confirmation'): neutral.validate_task(campaign.task(),False)
    neutral.validate_task(campaign.task(),True)
    assert set(campaign.DEV_LABELS).isdisjoint(campaign.CONF_LABELS)
    assert campaign.task()['rows_jsonl']!=campaign.task(0)['rows_jsonl']


@pytest.mark.parametrize('tier',range(4))
def test_capacity_and_uniform_catalog_preserve_all_support_cells(tier):
    histogram=campaign.common.calibration_histogram('python_factors')
    for (support,),count in histogram.items():
        assert generator.available_capacity(support,tier,set()) >= 4*count
    low,high=generator.CASE_WINDOWS[tier]
    catalog,_=generator.catalog(tier)
    assert all(low<=v<=high for v in catalog)


def test_generator_certifies_real_rows_and_excludes_previous_identities():
    quota=campaign.Counter({32:2,200:1,2240:1})
    rows=generator.build_pool('python_factors',quota,set(),123456,'test',0,1)
    assert campaign.materializer.modes(rows)==quota
    previous=campaign.ids(rows)
    again=generator.build_pool('python_factors',quota,previous,123456,'test',0,1)
    assert not campaign.ids(again)&previous
    from oat_drgrpo.python_modebench import python_factor_mode_count
    for row in rows:
        spec=json.loads(row['answer'])
        assert python_factor_mode_count(spec['cases'])==row['answer_mode_count']
        assert len(spec['cases'])==4
        assert spec['num_externally_certified_modes']==2


def test_neutral_receipt_keeps_actual_prompts_and_rejects_metric_tampering(tmp_path):
    e=neutral.evaluator
    path=tmp_path/'rows.jsonl'
    row={'problem':'Example task','answer':json.dumps({'target':0}),'answer_mode_count':4}
    path.write_text(json.dumps(row)+'\n')
    task=dict(campaign.task(0),rows_jsonl=str(path),output=str(tmp_path/'receipt.json'))
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs): return json.dumps(messages)
        def encode(self,prompt,**kwargs): return list(prompt)
    class LLM:
        def generate(self,prompts,params,**kwargs):
            assert json.loads(prompts[0])==make_messages(3,'python_factors',row)
            assert params[0].n==8
            return [SimpleNamespace(prompt=prompt,outputs=[SimpleNamespace(text='yes' if i<2 else 'no',token_ids=[1],finish_reason='stop') for i in range(8)]) for prompt in prompts]
    model={'label':'3b','vllm_version':'0.8.4'}
    result=e.evaluate_task(LLM(),Tokenizer(),task,model=model,code=e.code_identity(),
                          grader=lambda t,_:'mode' if t=='yes' else None,
                          params_factory=lambda a,*_:SimpleNamespace(seed=a.seed,n=8))
    campaign.validate_receipt(task['output'],task,{'model':model})
    assert result['metrics']['pass1']==.25
    assert result['identity']['interface']['prompt_condition']==neutral.CURRENT
    result['prompt_results'][0]['draws'][0]['pass1']=1
    Path(task['output']).write_text(json.dumps(result))
    with pytest.raises(ValueError,match='draw metrics'): campaign.validate_receipt(task['output'],task,{'model':model})
    with pytest.raises(ValueError): historical.validate_seed_receipt(result,[row])


def test_registration_hash_drift_stops_before_submitting(tmp_path,monkeypatch):
    plan=tmp_path/'registration.json';plan.write_text('{}')
    monkeypatch.setattr(campaign,'PLAN',plan)
    with pytest.raises(ValueError,match='registration hash'): campaign.validate_plan('0'*64)


def test_no_training_command_and_bounded_inference():
    cmd=campaign.gpu_command('development','a'*64)
    assert '--array=0-3%4' in cmd
    assert '--gres=gpu:rtx_6000:1' in cmd
    assert all('train_node' not in part and 'scontrol release' not in part for part in cmd)


def test_task_wording_adds_no_solution_example():
    text=generator.prompt([60,70,80,90])
    assert text.startswith("Write a pure Python lambda expression using only")
    assert "next, range" in text
    assert "n % 2" not in text and "if n==" not in text
    assert "[60, 70, 80, 90]" in text


def test_registered_pilots_are_disjoint_from_future_splits():
    # Exercise the real identity inventory rather than just asserting code text.
    if not (campaign.ROOT/'var/artifacts/modebench_level3_neutral_divisor_pilot/receipt.json').exists():
        pytest.skip('development pilot has not finished')
    blocked,pins=campaign.historical_inventory()
    for name in ('common3','finite','divisor'):
        root=campaign.ROOT/f'var/artifacts/modebench_level3_neutral_{name}_pilot'
        assert campaign.ids(campaign.mixture.read_jsonl(root/'rows.jsonl')) <= blocked
        assert str(root/'receipt.json') in pins


def test_original_external_verifier_positive_control():
    campaign.warm_verifier()
