from __future__ import annotations
from collections import Counter
import importlib.util,json
from pathlib import Path
from datasets import load_from_disk
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'var/data/modebench_harder_v2_matched_r5'

def test_materialized_level2_contract():
 report=json.loads((DATA/'admission_fairness_report.json').read_text())
 assert report['schema'] in {
  'modebench-level2-admission-fairness-v1',
  'modebench_harder_v2_support_matched_splits',
  'modebench-level2-final-admission-fairness-v1',
 }
 if report['schema']=='modebench-level2-final-admission-fairness-v1':
  assert report['status']=='pass'
  assert report['decision']=='admit_all_domains_for_treatment_training'
  assert set(report['domain_decisions'])==set(report['domains'])
  assert set(report['domain_decisions'].values())=={'admit'}
  protocol=report['comparable_prompt_and_response_budget']
  assert protocol['shared']['sample_count']==8
  assert protocol['shared']['max_tokens']==192
  assert protocol['shared']['row_limit']==0
  assert sum(len(cells) for cells in protocol['per_cell'].values())==10
  for domain,cells in report['frozen_base_model_viability'].items():
   assert set(cells)=={'qwen-0.5b','falcon-1b'}
   for cell in cells.values():
    assert cell['status']=='pass'
    assert all(cell['checks'].values())
    assert .10 <= cell['level2_pass_at_8'] <= .90
    assert cell['level2_pass_at_8'] < cell['level1_pass_at_8']
    assert cell['level2_rows']==128
    assert cell['level1_rows']==report['domains'][domain]['dev']['level1_reference_rows']
 else:
  assert report['status']=='pass'
  assert report['decision']=='structurally_admitted_pending_frozen_base_model_viability'
 assert report['split_sizes']=={'train':384,'dev':128,'eval':128}
 for domain,parts in report['domains'].items():
  identities=[]
  for split,size in report['split_sizes'].items():
   name='train' if split=='train' else 'multi_answer'
   rows=[dict(x) for x in load_from_disk(str(DATA/domain/split))[name]]
   assert len(rows)==size
   assert all(parts[split]['checks'].values())
   assert Counter(int(x['answer_mode_count']) for x in rows)==Counter({int(k):v for k,v in parts[split]['answer_mode_count_histogram'].items()})
  assert len(load_from_disk(str(DATA/domain/'eval'))['multi_answer'])==128
 assert report['domains']['pantry']['dev']['histogram_scale_factor']==2
 assert report['domains']['pantry']['dev']['level1_reference_rows']==64

def test_paired_evaluator_is_frozen_pass_at_8_and_development_only():
 text=(ROOT/'ops/evaluate_modebench_level2_viability.py').read_text()
 assert "n=8" in text
 assert "'sample_count':8" in text
 assert "'development_only':True" in text
 assert "'evaluation_prompts_loaded':False" in text
 assert ".10<=l2<=.90 and l2<l1" in text
 assert "apply_chat_template" in text
 assert "'calibration_only':bool(a.row_limit)" in text
 assert "'row_limit':a.row_limit" in text
 spec=importlib.util.spec_from_file_location('modebench_viability',ROOT/'ops/evaluate_modebench_level2_viability.py')
 module=importlib.util.module_from_spec(spec)
 assert spec.loader is not None
 spec.loader.exec_module(module)
 system=module.prompt_messages('python_factors','problem','deliberate_domain_v2')[0]['content']
 assert '\\boxed{}' in system
 assert '\b' not in system
 for domain in ('countdown','python_factors','mathir','pantry','graph_coloring'):
  messages=module.prompt_messages(domain,'ROW_SENTINEL','structured_solver_v3')
  assert messages[1]['content']=='ROW_SENTINEL'
  assert (chr(92)+'boxed') in messages[0]['content']
  assert chr(8) not in messages[0]['content']
 three={'answer':json.dumps({'numbers':[1,2,3],'target':6})}
 three_other_target={'answer':json.dumps({'numbers':[1,2,3],'target':999999})}
 four={'answer':json.dumps({'numbers':[1,2,3,4],'target':24})}
 assert module.countdown_legal_choices(three)==module.countdown_legal_choices(three_other_target)
 assert len(module.countdown_legal_choices(three))==192
 assert len(module.countdown_legal_choices(four))==7680
 assert all(choice.startswith('\\boxed{') and choice.endswith('}') for choice in module.countdown_legal_choices(three))
 for profile in ('countdown_fewshot_v5','countdown_shallow_v6'):
  messages=module.prompt_messages('countdown','ROW_SENTINEL',profile)
  assert messages[-1]['content']=='ROW_SENTINEL'
  assert chr(8) not in ''.join(message['content'] for message in messages)
  assert any((chr(92)+'boxed') in message['content'] for message in messages)

def test_slurm_matrix_covers_both_models_and_five_domains():
 text=(ROOT/'ops/slurm/evaluate_modebench_level2_viability.slurm').read_text()
 assert '#SBATCH --array=0-9' in text
 assert 'Qwen2.5-0.5B-Instruct' in text
 assert 'Falcon3-1B-Instruct' in text
 assert '--max-tokens 192' in text
 assert 'DOMAINS=(countdown graph_coloring python_factors mathir pantry)' in text
