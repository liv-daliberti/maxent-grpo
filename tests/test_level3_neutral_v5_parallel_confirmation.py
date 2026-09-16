"""Placement-only union must equal a serial receipt and reject changed streams."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops/exp_scaling'))
import recover_level3_neutral_v5_20260912 as p

class Tokenizer:
 def apply_chat_template(self,messages,**kwargs):return json.dumps(messages)
 def encode(self,prompt,**kwargs):return list(prompt)
class LLM:
 def generate(self,prompts,params,**kwargs):
  return [SimpleNamespace(prompt=prompt,outputs=[SimpleNamespace(text=('a' if i<q.seed%4+1 else 'b' if i==7 else 'wrong'),token_ids=[1],finish_reason='stop') for i in range(8)]) for prompt,q in zip(prompts,params)]

@pytest.fixture
def run(tmp_path,monkeypatch):
 c=p.c;e=c.neutral.evaluator
 monkeypatch.setattr(c,'ART',tmp_path);monkeypatch.setattr(c,'RESULTS',tmp_path)
 monkeypatch.setattr(c,'PLAN',tmp_path/'plan.json');monkeypatch.setattr(p,'PARALLEL',tmp_path/'parallel.json')
 monkeypatch.setattr(p,'check_parallel',lambda:None)
 model={'label':'3b','vllm_version':'0.8.4'}
 c.PLAN.write_text(json.dumps({'model':model}));p.PARALLEL.write_text('{}')
 rows=[{'problem':f'Find proper factors for problem {i}.','answer':json.dumps({'verifier':'python_factor_function','python_version':'factor-v1','cases':[6,10,14,22]}),'answer_mode_count':16} for i in range(2)]
 (tmp_path/'eval.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 def evaluate(task):
  return e.evaluate_task(LLM(),Tokenizer(),task,model=model,code=e.code_identity(),confirm_eval=True,
   grader=lambda text,_:None if text=='wrong' else text,
   params_factory=lambda args,*_:SimpleNamespace(seed=args.seed,n=8))
 for i in range(4):evaluate(p.task(i))
 return tmp_path,e,evaluate

def test_union_equals_serial_native_receipt(run):
 root,e,evaluate=run
 serial=evaluate({**p.c.task(),'output':str(root/'serial.json')})
 p.merge();merged=json.loads(Path(p.c.task()['output']).read_text())
 assert merged['identity']==serial['identity']
 assert merged['prompt_results']==serial['prompt_results']
 assert merged['metrics']==serial['metrics']
 assert e.validate_seed_receipt(merged)['distinct_child_seeds']==64
 assert len(merged['aggregation']['source_sha256'])==4
 p.merge() # A second call validates the same receipt instead of rewriting it.

def test_changed_draw_stream_is_rejected(run):
 root,e,_=run;path=Path(p.task(2)['output']);d=json.loads(path.read_text())
 d['prompt_results'][0]['draws'][0]['child_seeds'][1]+=9;path.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='RNG metadata'):p.merge()
 assert not Path(p.c.task()['output']).exists()

def test_wrong_combined_provenance_is_rejected(run):
 p.merge();path=Path(p.c.task()['output']);d=json.loads(path.read_text());d['aggregation']['source_sha256']={};path.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='source mismatch'):p.merge()
