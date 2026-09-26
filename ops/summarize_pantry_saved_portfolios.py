"""Zero-call Pantry adaptation on all original saved n=8 portfolios."""
from pathlib import Path
import json,sys
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/modebench_inference_followups_20260911/pantry'
sys.path[:0]=[str(BASE/'code/ops'),str(BASE/'code/src')]
from followup_metrics import atomic_new,file_sha,sha
from run_pantry_adaptation import survivors
from oat_drgrpo.math_grader import validated_modebench_outcome_key,_extract_modebench_candidate


def interval(values,indices):
    v=np.asarray(values,float);valid=np.isfinite(v)
    if not valid.any():return {'estimate':None,'ci95':None,'eligible_problems':0}
    b=v[indices];den=np.isfinite(b).sum(1);bb=np.divide(np.nansum(b,axis=1),den,out=np.full(len(b),np.nan),where=den>0)
    return {'estimate':float(np.nanmean(v)),'ci95':np.quantile(bb[np.isfinite(bb)],[.025,.975]).tolist(),'eligible_problems':int(valid.sum())}


def main():
    out=BASE/'saved_portfolios';out.mkdir(exist_ok=True)
    plan=json.loads((BASE/'execution_plan.json').read_text());data=json.loads((BASE/'inputs.json').read_text());assert file_sha(BASE/'inputs.json')==plan['inputs_sha256']
    for p,h in plan['code_sha256'].items():assert file_sha(p)==h
    freeze={'code_sha256':file_sha(__file__),'execution_plan_sha256':file_sha(BASE/'execution_plan.json'),'bootstrap_seed':20260911,'replicates':20000,'outcomes_read':False,'created_at_utc':datetime.now(timezone.utc).isoformat()}
    atomic_new(out/'implementation_freeze.json',freeze)
    tasks=[t for t in data['tasks'] if t['split']=='eval'];indices=np.random.default_rng(20260911).integers(0,len(tasks),(20000,len(tasks)));models={};records=[];source={};raw={}
    for c in data['checkpoints']:
        p=Path(data['saved_results_root'])/c['label']/'result.json';assert file_sha(p)==plan['saved_result_sha256'][c['label']];s=json.loads(p.read_text());assert s['status']=='complete' and s['identity']['checkpoint']==c
        assert file_sha(s['responses_path'])==s['responses_sha256'];source[str(p)]=file_sha(p)
        groups={r['row_index']:r for r in s['prompt_results'] if r['domain']=='pantry' and r['level']==2 and r['arm']=='original'};per=[]
        for t in tasks:
            row=t['row'];samples=[]
            for a in groups[row['row_index']]['attempts']:
                k=validated_modebench_outcome_key(a['text'],row['answer']);assert a['verified']==(k is not None) and a['canonical_key']==k
                samples.append({**a,'candidate':_extract_modebench_candidate(a['text'],row['answer'])})
            assert len(samples)==8
            v={'correct8':sum(s['verified'] for s in samples),'distinct8':len({s['canonical_key'] for s in samples if s['verified']}),'pass8':float(any(s['verified'] for s in samples))}
            for kind in ('outage','diet'):
                pp=[p for p in t['perturbations'] if p['feasible'] and p['kind']==kind];vv=[]
                for perturbation in pp:
                    saved=survivors(samples,perturbation['spec']);vv.append(bool(saved));records.append({'checkpoint':c['label'],'task':t['id'],'kind':kind,'perturbation':perturbation['id'],'saved_usable':len(saved),'survival':bool(saved)})
                v[kind+'_survival']=sum(vv)/len(vv) if vv else None
                v[kind+'_eligible']=len(pp)
            per.append(v)
        models[c['label']]={'checkpoint':c,'per_problem':per,'metrics':{k:interval([v[k] for v in per],indices) for k in ('correct8','distinct8','pass8','outage_survival','diet_survival')}};raw[c['label']]=per
    effects={}
    for seed in (43,46):
        a=next(c for c in data['checkpoints'] if c['training_method']=='drgrpo' and c['training_seed']==seed);b=next(c for c in data['checkpoints'] if c['training_method']=='replay_drgrpo' and c['training_seed']==seed)
        effects[str(seed)]={}
        for k in ('correct8','distinct8','pass8','outage_survival','diet_survival'):
            effects[str(seed)][k]=[y[k]-x[k] if x[k] is not None and y[k] is not None else None for x,y in zip(raw[a['label']],raw[b['label']])]
    contrasts={seed:{k:interval(v,indices) for k,v in metrics.items()} for seed,metrics in effects.items()}
    contrasts['fixed_two_seed_average']={k:interval([sum(v)/2 if all(x is not None for x in v) else None for v in zip(effects['43'][k],effects['46'][k])],indices) for k in effects['43']}
    matched={}
    for kind in ('outage','diet'):
        v=[]
        for i in range(len(tasks)):
            eligible=[]
            for seed in (43,46):
                a=next(c['label'] for c in data['checkpoints'] if c['training_method']=='drgrpo' and c['training_seed']==seed);b=next(c['label'] for c in data['checkpoints'] if c['training_method']=='replay_drgrpo' and c['training_seed']==seed)
                if raw[a][i]['correct8']==raw[b][i]['correct8'] and raw[a][i][kind+'_survival'] is not None:eligible.append(raw[b][i][kind+'_survival']-raw[a][i][kind+'_survival'])
            v.append(sum(eligible)/len(eligible) if eligible else None)
        matched[kind]=interval(v,indices)
    result={'schema':'pantry-saved-portfolio-adaptation-v1','models':models,'paired_replay_minus_drgrpo':contrasts,'matched_observed_correctness':matched,'source_sha256':source,'inputs_sha256':file_sha(BASE/'inputs.json'),'freeze_sha256':file_sha(out/'implementation_freeze.json'),'records':records,'feasibility_counts':{kind:{'candidates':sum(p['kind']==kind for t in tasks for p in t['perturbations']),'feasible':sum(p['kind']==kind and p['feasible'] for t in tasks for p in t['perturbations'])} for kind in ('outage','diet')},'new_model_calls':0,'limitations':['Original saved plans revalidated unchanged; infeasible perturbations excluded independently.','Seeds 43 and 46 are fixed checkpoints; intervals resample problems, not training seeds.','Outages and dietary restrictions are averaged within problem before averaging problems.','Recovery calls, temperature controls and diversity prompting are separate ongoing inference jobs.']}
    atomic_new(out/'results.json',result)
    lines=['# Pantry saved-portfolio adaptation','','32 held-out problems; eight unchanged saved plans per checkpoint. All feasible task-derived perturbations were frozen before scoring. Intervals resample whole problems (20,000 draws).','','| Checkpoint | Correct / 8 | Distinct / 8 | Survives outage | Survives dietary restriction |','|---|---:|---:|---:|---:|']
    def f(v,k=1):
        if v['estimate'] is None:return 'undefined'
        return f"{v['estimate']*k:.2f} [{v['ci95'][0]*k:.2f}, {v['ci95'][1]*k:.2f}]"
    for label,d in models.items():
        m=d['metrics'];lines.append(f"| {label} | {f(m['correct8'])} | {f(m['distinct8'])} | {f(m['outage_survival'],100)}% | {f(m['diet_survival'],100)}% |")
    lines+=['','Fixed two-seed average, Re:Dr minus DrGRPO: outage survival '+f(contrasts['fixed_two_seed_average']['outage_survival'],100)+' percentage points; dietary survival '+f(contrasts['fixed_two_seed_average']['diet_survival'],100)+' percentage points.','','This first stage uses no new inference. The running control/recovery study measures additional calls, unresolved cases and token usage.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps({'status':'complete','feasibility':result['feasibility_counts'],'contrasts':contrasts['fixed_two_seed_average']}))

if __name__=='__main__':main()
