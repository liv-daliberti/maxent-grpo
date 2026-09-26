"""Matched coarse-key endpoint probes on the existing 25-checkpoint prompt study."""
from pathlib import Path
import importlib.util,itertools,json,sys
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ops'))
from followup_metrics import atomic_new,file_sha,coarse_key,portfolio,collision
BASE=ROOT/'artifacts/modebench_inference_followups_20260911/local_coarse_v2'
SOURCE=ROOT/'artifacts/modebench_prompt_ablation_20260911'


def measure(keys):
    v=portfolio([keys],[8]);return {'distinct8':v['distinct8'],'pass1':v['correct']/8,'pass8':v['pass8'],'extra8':v['extra8'],'collision':collision(keys)}


def interval(x,indices):
    a=np.asarray(x,float);bb=a[indices];den=np.isfinite(bb).sum(1);boot=np.divide(np.nansum(bb,axis=1),den,out=np.full(len(indices),np.nan),where=den>0);v=boot[np.isfinite(boot)]
    return {'estimate':float(np.nanmean(a)) if np.isfinite(a).any() else None,'ci95':np.quantile(v,[.025,.975]).tolist() if len(v) else None,'eligible_prompts':int(np.isfinite(a).sum())}


def main():
    BASE.mkdir(exist_ok=True);planpath=SOURCE/'local/plan_v2.json';plan=json.loads(planpath.read_text())
    audit=json.loads((Path(plan['output_root'])/plan['checkpoints'][1]['label']/'prompt_ablation_normalization_audit.json').read_text());modulepath=Path(audit['analyzer_source']['path']);assert file_sha(modulepath)==audit['analyzer_source']['sha256']
    atomic_new(BASE/'implementation_freeze.json',{'source_sha256':{str(Path(__file__)):file_sha(__file__),str(ROOT/'ops/followup_metrics.py'):file_sha(ROOT/'ops/followup_metrics.py'),str(modulepath):file_sha(modulepath),str(planpath):file_sha(planpath)},'created_at_utc':datetime.now(timezone.utc).isoformat(),'outcomes_read':False,'scope':'Existing 32-prompt L2/L3 endpoint probes; not the full main-paper training population.'})
    spec=importlib.util.spec_from_file_location('_authenticated_prompt_analyzer',modulepath);loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader)
    design=loader.authenticate_design(SOURCE);models={};sources={}
    for cp in plan['checkpoints']:
        runs=loader.authenticate_local(design,planpath,cp);models[cp['label']]={}
        for run in runs:
            sources[cp['label']]=run['sources'];arm=next(iter(run['strict'].values()))['arm']
            for grading in ('strict','normalized_secondary'):
                grouped=defaultdict(list)
                for (level,domain,row_index,sample_index),record in run[grading].items():grouped[level,domain,row_index].append((sample_index,record['canonical_key']))
                for (level,domain,row_index),draws in sorted(grouped.items()):
                    assert sorted(i for i,k in draws)==list(range(8));keys=[k for i,k in sorted(draws)];row=run['rows'][level,domain,row_index];answer=json.loads(row['answer']);coarse=[coarse_key(k,answer) for k in keys]
                    f,c=measure(keys),measure(coarse);assert f['pass1']==c['pass1'] and f['pass8']==c['pass8'] and c['distinct8']<=f['distinct8']+1e-12
                    if f['collision'] is not None:assert c['collision']>=f['collision']-1e-12
                    models[cp['label']][grading,arm,level,domain,row_index]=(f,c)
        print(cp['label'],flush=True)
    indices=np.random.default_rng(20260911).integers(0,32,(20000,32));contrasts=[]
    for grading,arm,level,domain in itertools.product(('strict','normalized_secondary'),('original','neutral'),(2,3),('python_factors','mathir','pantry')):
        cps=[c for c in plan['checkpoints'] if c['domain']==domain];seeds=sorted({c['training_seed'] for c in cps});paired=[]
        for seed in seeds:
            a=next(c for c in cps if c['training_seed']==seed and c['training_method']=='drgrpo');b=next(c for c in cps if c['training_seed']==seed and c['training_method']=='replay_drgrpo')
            keys=sorted(k for k in models[a['label']] if k[:4]==(grading,arm,level,('pantry_plan' if domain=='pantry' else domain)));assert len(keys)==32
            per=[]
            for k in keys:
                fa,ca=models[a['label']][k];fb,cb=models[b['label']][k]
                per.append({keydef+'_'+metric:(y[metric]-x[metric] if x[metric] is not None and y[metric] is not None else None) for keydef,x,y in [('fine',fa,fb),('coarse',ca,cb)] for metric in fa})
            paired.append(per)
            contrasts.append({'grading':grading,'arm':arm,'level':level,'domain':domain,'seed':seed,'metrics':{m:interval([v[m] for v in per],indices) for m in per[0]}})
        mean=[]
        for i in range(32):
            mean.append({m:(sum(p[i][m] for p in paired)/len(paired) if all(p[i][m] is not None for p in paired) else None) for m in paired[0][i]})
        contrasts.append({'grading':grading,'arm':arm,'level':level,'domain':domain,'seed':'fixed_average','seeds':seeds,'metrics':{m:interval([v[m] for v in mean],indices) for m in mean[0]}})
    r={'schema':'local-coarse-endpoint-probes-v1','contrasts':contrasts,'source_bindings':sources,'freeze_sha256':file_sha(BASE/'implementation_freeze.json'),'scope':'25 frozen checkpoints; original and neutral prompts; L2/L3; Python, MathIR, Pantry; same paired endpoint subsets as prompt ablation. Not a reanalysis of all main-paper training trajectories.','bootstrap':'20,000 paired whole-prompt replicates, seed 20260911; fixed training seeds averaged, no population seed inference.','all_pass_metrics_unchanged':True}
    atomic_new(BASE/'results.json',r)
    lines=['# Matched local coarse-key sensitivity','','Existing 32-prompt endpoint probes, replay minus DrGRPO, fixed seed averages. These use the saved prompt-ablation cohort, not every main-paper training run. MathIR and Pantry keep their outcome keys.','','| Domain | Level | Wording | Seeds | Fine ΔD@8 | Coarse ΔD@8 |','|---|---:|---|---:|---:|---:|']
    for c in contrasts:
        if c['grading']!='strict' or c['seed']!='fixed_average':continue
        def f(metric):
            m=c['metrics'][metric];return f"{m['estimate']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"
        lines.append(f"| {c['domain']} | {c['level']} | {c['arm']} | {len(c['seeds'])} | {f('fine_distinct8')} | {f('coarse_distinct8')} |")
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
