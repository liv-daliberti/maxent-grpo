from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'ops'),str(Path(__file__).resolve().parents[1]/'src')]
from prepare_pantry_adaptation import perturbations
from run_pantry_adaptation import survivors,recovery_messages
from oat_drgrpo.pantry_plan import validate_pantry_plan


def spec():
    return {'verifier':'pantry_plan','pantry_version':'pantry-v1','min_ingredients':2,'max_ingredients':2,'certified_mode_count':2,'forbidden_tags':[],
            'ingredients':[{'id':x,'available_g':50,'min_if_used_g':50,'step_g':25,'tags':[],'attributes_per_100g':{'protein_g':10 if x=='a' else 0}} for x in 'abc'],
            'targets':{'mass_g':{'min':100,'max':100},'protein_g':{'min':5}}}


def test_exhaustive_feasibility_and_witnesses():
    original=spec();pp=perturbations(original)
    assert [(p['target'],p['feasible']) for p in pp]==[('a',False),('b',True),('c',True)]
    assert original==spec()
    for p in pp:
        if p['feasible']:assert validate_pantry_plan(p['witness'],p['spec']) is not None
        assert all(i['available_g']==50 for i in p['spec']['ingredients'])


def test_saved_plan_never_reallocated_or_rescued():
    pp={p['target']:p for p in perturbations(spec())}
    s=[{'candidate':'a=50;b=50','verified':True},{'candidate':'a=50;c=50','verified':False}]
    assert survivors(s,pp['b']['spec'])==[]
    assert survivors(s,pp['c']['spec'])==[0]


def test_full_history_and_update_preserved():
    m=[{'role':'system','content':'system'},{'role':'user','content':'original'}]
    result=recovery_messages(m,'outage a',[{'text':'plan1'},{'text':'bad'}],[{'text':'failed recovery'}])
    assert result[0]==m[0]
    assert all(x in result[1]['content'] for x in ('original','outage a','plan1','bad','failed recovery','Verifier: invalid'))


def test_worker_stops_on_success_and_counts_censored_calls(tmp_path,monkeypatch):
    import json,types
    from followup_metrics import atomic_new,file_sha
    from run_pantry_adaptation import run
    import evaluate_modebench_level2_viability as decoder
    import frontier_modebench_contract as contract
    import oat_drgrpo.math_grader as grader
    calls=[]
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):return json.dumps(messages)
        def encode(self,text,**kwargs):return [0]*len(text.split())
    class LLM:
        def __init__(self,**kwargs):pass
        def get_tokenizer(self):return Tokenizer()
        def generate(self,prompts,params,**kwargs):
            prompt=prompts[0];p=params[0];calls.append((prompt,p.n))
            text='a=50;c=50' if 'OUTAGE B' in prompt and 'Recovery attempts rejected' in prompt else 'invalid'
            samples=[types.SimpleNamespace(index=i,text=text,token_ids=[1,2],finish_reason='stop') for i in range(p.n)]
            return [types.SimpleNamespace(prompt=prompt,outputs=samples)]
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(LLM=LLM,SamplingParams=lambda **kw:types.SimpleNamespace(**kw)))
    monkeypatch.setattr(decoder,'sampling_params',lambda *a:types.SimpleNamespace(guided_decoding=None))
    monkeypatch.setattr(contract,'make_messages',lambda l,d,r:[{'role':'system','content':'system'},{'role':'user','content':r['problem']}])
    monkeypatch.setattr(grader,'_extract_modebench_candidate',lambda t,a:t)
    monkeypatch.setattr(grader,'validated_modebench_outcome_key',lambda t,a:(v.canonical_key if (v:=validate_pantry_plan(t,json.loads(a))) is not None else None))
    checkpoint={'label':'synthetic','files':[],'model_path':str(tmp_path)}
    original=spec();pp=[p for p in perturbations(original) if p['feasible']]
    for p in pp:p['update']='OUTAGE '+p['target'].upper()
    row={'level':2,'domain':'pantry','row_index':0,'problem':'synthetic plan','answer':json.dumps(original)}
    tasks=[{'id':split+'_0','split':split,'row':row,'perturbations':pp} for split in ('dev','eval')]
    source=tmp_path/'saved'/'synthetic';source.mkdir(parents=True);responses=source/'responses.jsonl';responses.write_text('[]\n')
    saved=source/'result.json';atomic_new(saved,{'status':'complete','identity':{'checkpoint':checkpoint},'responses_path':str(responses),'responses_sha256':file_sha(responses),'prompt_results':[{'level':2,'domain':'pantry','arm':'original','row_index':0,'attempts':[{'text':'invalid','token_count':2,'verified':False,'canonical_key':None} for _ in range(8)]}]})
    inputs=tmp_path/'inputs.json';atomic_new(inputs,{'base':str(tmp_path),'checkpoints':[checkpoint],'tasks':tasks,'temperature_grid':[.7,1,1.3],'saved_results_root':str(tmp_path/'saved')})
    plan=tmp_path/'plan.json';atomic_new(plan,{'code_sha256':{},'inputs':str(inputs),'inputs_sha256':file_sha(inputs),'saved_result_sha256':{'synthetic':file_sha(saved)}})
    run(plan,0);result=json.loads((tmp_path/'results/synthetic/result.json').read_text())
    assert len(result['records'])==6
    for r in result['records']:
        assert not r['zero_call_recovery']
        assert r['recovery_calls']==(2 if r['perturbation']=='outage_b' else 8)
        assert r['recovered']==(r['perturbation']=='outage_b')
        assert r['recovery_output_tokens']==2*r['recovery_calls']
    assert sum(n for _,n in calls)==24+16+30
    assert len(result['request_receipts'])==3+1+8+30
    before=len(calls);run(plan,0);assert len(calls)==before
    # Audit the completed fake-model run through the same full reporting path.
    import analyze_pantry_adaptation as analyzer
    import numpy as np
    monkeypatch.setattr(analyzer,'BASE',tmp_path)
    (tmp_path/'execution_plan.json').write_bytes(plan.read_bytes())
    summary=analyzer.summarize_checkpoint(checkpoint,json.loads(plan.read_text()),json.loads(inputs.read_text()),[tasks[1]],np.zeros((20,1),dtype=int))
    assert summary['validated_new_responses']==70
    assert summary['actual_new_collection_cost']['output_tokens']==140
    for strategy in ('ordinary','temperature','diversity_prompt'):
        metrics=summary['summary'][strategy+'/outage']
        assert metrics['unresolved']['estimate']==.5
        assert metrics['capped_recovery_calls']['estimate']==5
        assert metrics['recovery_at_1']['estimate']==0
        assert metrics['recovery_at_2']['estimate']==.5
