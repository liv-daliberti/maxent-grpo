"""Authenticate completed inference receipts and report adaptation versus cost."""
from pathlib import Path
import argparse,json,sys
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/modebench_inference_followups_20260911/pantry'
sys.path[:0]=[str(BASE/'code/ops'),str(BASE/'code/src')]
from followup_metrics import atomic_new,file_sha,sha
from run_pantry_adaptation import survivors,recovery_messages
from frontier_modebench_contract import make_messages
from oat_drgrpo.math_grader import validated_modebench_outcome_key,_extract_modebench_candidate
STRATEGIES=('ordinary','temperature','diversity_prompt')


def interval(values,indices):
    x=np.asarray(values,float);b=x[indices];den=np.isfinite(b).sum(1);boot=np.divide(np.nansum(b,axis=1),den,out=np.full(len(indices),np.nan),where=den>0);valid=boot[np.isfinite(boot)]
    return {'estimate':float(np.nanmean(x)) if np.isfinite(x).any() else None,'ci95':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,'eligible_problems':int(np.isfinite(x).sum())}


def cost(receipts):
    return {'responses':sum(len(r['samples']) for r in receipts),'http_or_generation_groups':len(receipts),'logical_input_tokens':sum(r['logical_input_tokens'] for r in receipts),'output_tokens':sum(s['output_tokens'] for r in receipts for s in r['samples']),'generation_wall_seconds':sum(r['generation_wall_seconds'] for r in receipts),'verification_seconds':sum(s['verification_seconds'] for r in receipts for s in r['samples'])}


def summarize_checkpoint(cp,plan,data,tasks,indices):
    folder=BASE/'results'/cp['label'];path=folder/'result.json';r=json.loads(path.read_text());identity=sha({'plan':file_sha(BASE/'execution_plan.json'),'checkpoint':cp});assert r['identity']==identity and r['status']=='complete' and r['checkpoint']==cp
    assert file_sha(r['saved_source']['path'])==plan['saved_result_sha256'][cp['label']]==r['saved_source']['sha256']
    requests={};allseeds=set()
    for receipt in r['request_receipts']:
        p=Path(receipt['path']);assert p.parent==folder/'requests' and file_sha(p)==receipt['sha256'];x=json.loads(p.read_text());req=x['request'];assert x['binding']==sha([identity,req]);uid=req['uid'];assert uid not in requests;requests[uid]=x
        assert len(x['samples'])==req['n'] and x['logical_input_tokens']==req['prompt_tokens']*req['n'];assert x['input_tokens_per_response']==req['prompt_tokens']
        assert x['generation_wall_seconds']>=0 and req['max_tokens']==192 and req['prompt_tokens']+192<=8192
        for s in x['samples']:
            assert s['seed'] not in allseeds;allseeds.add(s['seed']);assert 0<=s['output_tokens']<=192 and s['verified']==(s['canonical_key'] is not None)
    assert len(list((folder/'requests').glob('*.json')))==len(requests)
    used=set();taskmap={t['id']:t for t in data['tasks']}
    def get(uid,row,messages,temperature,n):
        x=requests[uid];q=x['request'];assert q['row_sha256']==sha(row) and q['messages']==messages and q['temperature']==temperature and q['n']==n;used.add(uid)
        for s in x['samples']:
            key=validated_modebench_outcome_key(s['text'],row['answer']);assert key==s['canonical_key'] and (key is not None)==s['verified'];assert _extract_modebench_candidate(s['text'],row['answer'])==s['candidate']
        return x
    calibration=[];calc=[]
    for temperature in data['temperature_grid']:
        per=[]
        for task in data['tasks']:
            if task['split']!='dev':continue
            row=task['row'];x=get(f'dev_t{temperature}_{task["id"]}',row,make_messages(2,'pantry',row),temperature,8);calc.append(x)
            pp=[p for p in task['perturbations'] if p['feasible'] and p['kind']=='outage'];per.append(sum(bool(survivors(x['samples'],p['spec'])) for p in pp)/len(pp) if pp else None)
        vals=[v for v in per if v is not None];calibration.append({'temperature':temperature,'survival':sum(vals)/len(vals),'eligible_dev_problems':len(vals),'per_problem':per})
    assert calibration==r['calibration'];chosen=min(calibration,key=lambda x:(-x['survival'],abs(x['temperature']-1),x['temperature']))['temperature'];assert chosen==r['chosen_temperature']
    records={(x['task'],x['strategy'],x['perturbation']):x for x in r['records']};assert len(records)==len(r['records']);derived=[];oldsource=json.loads(Path(r['saved_source']['path']).read_text());old={p['row_index']:p for p in oldsource['prompt_results'] if p['level']==2 and p['domain']=='pantry' and p['arm']=='original'}
    for task in tasks:
        p=folder/'portfolios'/f'{task["id"]}.json';saved=json.loads(p.read_text());assert saved['identity']==identity and saved['chosen_temperature']==chosen;row=task['row'];messages=make_messages(2,'pantry',row);portfolios=saved['portfolios']
        original=portfolios['ordinary'];assert len(original)==8
        for x,y in zip(original,old[row['row_index']]['attempts']):
            assert all(x[k]==v for k,v in y.items());key=validated_modebench_outcome_key(x['text'],row['answer']);assert key==x['canonical_key'] and x['verified']==(key is not None)
        temp=get('eval_temp_'+task['id'],row,messages,chosen,8);assert portfolios['temperature']==temp['samples']
        diversity=[];dcost=[]
        for j in range(8):
            history='\n'.join(f'{i+1}. {s["text"]}' for i,s in enumerate(diversity)) or '(none yet)'
            dm=[messages[0],{'role':'user','content':messages[1]['content']+'\n\nPrevious responses (including invalid attempts):\n'+history+'\n\nGive one valid plan using a different ingredient support from all previous responses, where feasible. Use the required boxed format.'}]
            x=get(f'eval_diverse_{task["id"]}_{j}',row,dm,1.0,1);diversity+=x['samples'];dcost.append(x)
        assert diversity==portfolios['diversity_prompt']
        initial_costs={'ordinary':{'responses':8,'logical_input_tokens':8*saved['ordinary_input_tokens_per_response'],'output_tokens':sum(s['output_tokens'] for s in original),'generation_wall_seconds':None,'verification_seconds':None},'temperature':cost([temp]),'diversity_prompt':cost(dcost)}
        for strategy in STRATEGIES:
            ss=portfolios[strategy];ic=initial_costs[strategy]
            for pi,p in enumerate(task['perturbations']):
                if not p['feasible']:continue
                rec=records.pop((task['id'],strategy,p['id']));survives=bool(survivors(ss,p['spec']));assert rec['zero_call_recovery']==survives and rec['initial_correct']==sum(s['verified'] for s in ss) and rec['initial_distinct']==len({s['canonical_key'] for s in ss if s['verified']})
                attempts=[];rc=[];revised={**row,'answer':json.dumps(p['spec'],sort_keys=True)}
                for j in range(rec['recovery_calls']):
                    assert not survives and not any(s['verified'] for s in attempts)
                    x=get(f'recovery_{task["id"]}_{strategy}_{pi}_{j}',revised,recovery_messages(messages,p['update'],ss,attempts),1.0,1);rc.append(x);attempts+=x['samples']
                recovered=survives or bool(attempts and attempts[-1]['verified']);assert recovered==rec['recovered'] and 0<=len(attempts)<=8 and (recovered or len(attempts)==8)
                cc=cost(rc);assert cc['output_tokens']==rec['recovery_output_tokens']
                derived.append({**rec,'metrics':{'zero_call_survival':float(survives),'unresolved':float(not recovered),'capped_recovery_calls':len(attempts),'initial_correct8':rec['initial_correct'],'initial_distinct8':rec['initial_distinct'],'initial_input_tokens':ic['logical_input_tokens'],'initial_output_tokens':ic['output_tokens'],'recovery_input_tokens':cc['logical_input_tokens'],'recovery_output_tokens':cc['output_tokens'],'total_input_tokens':ic['logical_input_tokens']+cc['logical_input_tokens'],'total_output_tokens':ic['output_tokens']+cc['output_tokens'],'recovery_generation_seconds':cc['generation_wall_seconds'],**{f'recovery_at_{b}':float(recovered and len(attempts)<=b) for b in range(9)}}})
    assert not records and used==set(requests)
    allmetrics=derived[0]['metrics'];per_problem={};summary={}
    for strategy in STRATEGIES:
        for kind in ('outage','diet'):
            values=[]
            for task in tasks:
                xx=[r['metrics'] for r in derived if r['strategy']==strategy and r['kind']==kind and r['task']==task['id']]
                values.append({m:sum(r[m] for r in xx)/len(xx) if xx else None for m in allmetrics})
            group=strategy+'/'+kind;per_problem[group]=values;summary[group]={m:interval([r[m] for r in values],indices) for m in allmetrics}
    return {'checkpoint':cp,'chosen_temperature':chosen,'calibration':calibration,'calibration_cost':cost(calc),'actual_new_collection_cost':cost(list(requests.values())),'summary':summary,'per_problem':per_problem,'records':derived,'source_result_sha256':file_sha(path),'validated_new_responses':len(allseeds)}


def main():
    out=BASE/'analysis_complete';out.mkdir(exist_ok=True);plan=json.loads((BASE/'execution_plan.json').read_text());data=json.loads((BASE/'inputs.json').read_text())
    assert all((BASE/'results'/cp['label']/'result.json').exists() for cp in data['checkpoints']),'All five completed workers required; no partial inferential report.'
    for p,h in plan['code_sha256'].items():assert file_sha(p)==h
    assert file_sha(BASE/'inputs.json')==plan['inputs_sha256']
    frozen=json.loads((BASE/'analysis_freeze.json').read_text());assert file_sha(__file__)==frozen['code_sha256']
    tasks=[t for t in data['tasks'] if t['split']=='eval'];indices=np.random.default_rng(20260911).integers(0,len(tasks),(20000,len(tasks)));models={}
    for cp in data['checkpoints']:
        models[cp['label']]=summarize_checkpoint(cp,plan,data,tasks,indices);print(json.dumps({'event':'checkpoint_authenticated','checkpoint':cp['label']}),flush=True)
    effects={}
    for seed in (43,46):
        a=next(c['label'] for c in data['checkpoints'] if c['training_seed']==seed and c['training_method']=='drgrpo');b=next(c['label'] for c in data['checkpoints'] if c['training_seed']==seed and c['training_method']=='replay_drgrpo');effects[str(seed)]={}
        for g,aa in models[a]['per_problem'].items():
            bb=models[b]['per_problem'][g];effects[str(seed)][g]={m:[y[m]-x[m] if x[m] is not None and y[m] is not None else None for x,y in zip(aa,bb)] for m in aa[0]}
    contrasts={seed:{g:{m:interval(v,indices) for m,v in metrics.items()} for g,metrics in groups.items()} for seed,groups in effects.items()}
    contrasts['fixed_two_seed_average']={g:{m:interval([sum(v)/2 if all(x is not None for x in v) else None for v in zip(effects['43'][g][m],effects['46'][g][m])],indices) for m in metrics} for g,metrics in effects['43'].items()}
    result={'status':'complete','schema':'pantry-adaptation-analysis-v1','models':models,'replay_minus_drgrpo':contrasts,'execution_plan_sha256':file_sha(BASE/'execution_plan.json'),'analysis_freeze_sha256':file_sha(BASE/'analysis_freeze.json'),'bootstrap':{'replicates':20000,'seed':20260911,'unit':'whole problem, all perturbations and strategies together; fixed training seeds'},'cost_note':'Per-perturbation totals charge the initial portfolio once plus its recovery. Actual collection cost counts every generated response once, with calibration separate. Input tokens are logical counts, including reused prefixes; they do not measure actual cached compute. Historical ordinary generation timing is unavailable.','validation':'All receipts, full prompts, calibration selection, strict grades, stopping, censoring and response accounting authenticated.'}
    atomic_new(out/'results.json',result)
    lines=['# Pantry adaptation: controls, recovery and cost','','Eight initial plans; all feasible independently frozen perturbations; up to eight recovery calls. Percentages/calls/tokens average perturbations within each problem, then problems. Seeds 43/46 are fixed checkpoints.','','| Checkpoint | Strategy | Outage survival | Recovered by 8 calls | Capped recovery calls | Total input tokens | Total output tokens |','|---|---|---:|---:|---:|---:|---:|']
    for label,v in models.items():
        for strategy in STRATEGIES:
            m=v['summary'][strategy+'/outage'];f=lambda key,k=1:f"{k*m[key]['estimate']:.2f}"
            lines.append(f"| {label} | {strategy} | {f('zero_call_survival',100)}% | {f('recovery_at_8',100)}% | {f('capped_recovery_calls')} | {f('total_input_tokens')} | {f('total_output_tokens')} |")
    lines+=['','Full JSON contains paired 95% problem-bootstrap intervals, complete recovery curves at 0–8 calls, dietary results, fixed-seed replay contrasts, selected temperatures and calibration cost.','',''+result['cost_note'],'']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps({'event':'analysis_complete','new_responses':sum(m['validated_new_responses'] for m in models.values())}),flush=True)

if __name__=='__main__':main()
