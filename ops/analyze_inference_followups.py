"""Authenticated all-pair portfolios and task-defined coarse outcome analysis."""
from pathlib import Path
import argparse,itertools,json,math,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'src')]
from followup_metrics import atomic_new,file_sha,sha,read_lines,coarse_key,portfolio,collision,cross_metrics,counts
from summarize_frontier_comparison import validate_evidence,sample_key,compact_sha
BASE=ROOT/'artifacts/modebench_inference_followups_20260911'
DOMAINS=('countdown','graph_coloring','mathir','pantry_plan','python_factors')


def request_messages(request):
    msgs=request.get('input',request.get('messages'))
    if not isinstance(msgs,list):raise ValueError('unrecognized request messages')
    if 'system' in request:msgs=[{'role':'system','content':request['system']}]+msgs
    return msgs


def support(row):
    spec=json.loads(row['answer']);domain=row['domain'];m=row['metadata'].get('answer_mode_count')
    if domain=='countdown':return {'fine_status':'unknown','coarse_status':'unknown','fine_count':None,'coarse_count':None,'fine_uniform_collision':None,'coarse_uniform_collision':None,'fine_induced_coarse_collision':None}
    ncoarse=m;induced=1/m
    if domain=='graph_coloring':
        colors=spec['partial_colors'];hidden=[i for i,c in enumerate(colors) if c is None];cc=Counter()
        for assignment in itertools.product((1,2,3),repeat=len(hidden)):
            full=list(colors)
            for i,c in zip(hidden,assignment):full[i]=c
            if any(full[u-1]==full[v-1] for u,v in spec['edges']):continue
            key='graph_coloring:'+''.join(map(str,full));cc[coarse_key(key,spec)]+=1
        assert sum(cc.values())==m
        ncoarse=len(cc);induced=sum(v*v for v in cc.values())/(m*m)
    elif domain=='python_factors':
        fine=1;ncoarse=1;induced=1
        for n in spec['cases']:
            ds=[d for d in range(2,n) if n%d==0];c=Counter(tuple(sorted((d,n//d))) for d in ds)
            fine*=len(ds);ncoarse*=len(c);induced*=sum(v*v for v in c.values())/len(ds)**2
        assert fine==m
    return {'fine_status':'exact_certificate','coarse_status':'exact_quotient' if domain in ('graph_coloring','python_factors') else 'unchanged_exact_certificate','fine_count':m,'coarse_count':ncoarse,'fine_uniform_collision':1/m,'coarse_uniform_collision':1/ncoarse,'fine_induced_coarse_collision':induced}


def load_authenticated():
    comparison_path=ROOT/'artifacts/frontier_models_comparison_20260911/comparison.json'
    comparison=json.loads(comparison_path.read_text());models={};rows=None;reference_messages=None;sources={str(comparison_path):file_sha(comparison_path)}
    auditpath=ROOT/'artifacts/frontier_modebench_gpt56sol_20260911/support_reference_audit.json'
    support_audit=json.loads(auditpath.read_text());assert not support_audit['exact_count_mismatches'];sources[str(auditpath)]=file_sha(auditpath)
    for model,rec in comparison['models'].items():
        run=Path(rec['directory'])
        for name,expected in rec['source_sha256'].items():
            assert file_sha(run/name)==expected;sources[str(run/name)]=expected
        summary=json.loads((run/'summary.json').read_text());assert summary['status']=='complete'
        validate_evidence(run,summary,model)
        rr=read_lines(run/'rows.jsonl')
        if rows is None:rows=sorted(rr,key=lambda r:(r['level'],r['domain'],r['row_index']))
        assert sorted(rr,key=lambda r:(r['level'],r['domain'],r['row_index']))==rows
        messages={};request_ids=set()
        for request in read_lines(run/'requests.jsonl'):
            pid=sample_key(request)[:3];sid=sample_key(request);assert sid not in request_ids;request_ids.add(sid)
            msg=request_messages(request['request']);assert pid not in messages or messages[pid]==msg;messages[pid]=msg
        assert len(request_ids)==15360
        if reference_messages is None:reference_messages=messages
        assert messages==reference_messages,model+' prompt mismatch'
        primary=Path(summary['primary_samples_path']);assert file_sha(primary)==summary['primary_samples_sha256'];sources[str(primary)]=file_sha(primary)
        raw={sample_key(s):s for s in read_lines(run/'samples.jsonl')};strict=read_lines(primary)
        assert len(raw)==len(strict)==15360
        normpath=run/'normalized_samples.jsonl';assert file_sha(normpath)==summary['normalized_secondary']['cache_sha256'];sources[str(normpath)]=file_sha(normpath)
        version=summary['normalized_secondary']['normalization_source_sha256']
        cache={s['strict_receipt_sha256']:s['normalization'] for s in read_lines(normpath) if s['normalization_source_sha256']==version}
        grouped={g:defaultdict(dict) for g in ('strict','normalized_secondary')}
        for s in strict:
            sid=sample_key(s);receipt={**raw[sid],**{k:s[k] for k in ('verified','canonical_key','graded_text')}};n=cache[compact_sha(receipt)]
            if s['verified']:assert n['verified'] and s['canonical_key']==n['canonical_key']
            for name,grade in [('strict',s),('normalized_secondary',n)]:
                assert sid[3] not in grouped[name][sid[:3]]
                assert grade['verified']==(grade['canonical_key'] is not None)
                grouped[name][sid[:3]][sid[3]]=grade['canonical_key']
        models[model]={}
        for name,group in grouped.items():
            assert len(group)==1920
            keys=[]
            for row in rows:
                p=(row['level'],row['domain'],row['row_index']);assert set(group[p])==set(range(8));keys.append([group[p][i] for i in range(8)])
            models[model][name]=keys
        print(json.dumps({'event':'authenticated','model':model,'responses':15360}),flush=True)
    return rows,models,sources


def pair_values(a,b):
    v=cross_metrics(a,b);aa=portfolio([a],[8]);bb=portfolio([b],[8]);mix=portfolio([a,b],[4,4])
    for k in ('distinct8','pass8','extra8','correct'):
        v['a_'+k]=aa[k];v['b_'+k]=bb[k];v['mix_'+k]=mix[k]
        v['mix_minus_a_'+k]=mix[k]-aa[k];v['mix_minus_b_'+k]=mix[k]-bb[k];v['mix_minus_average_'+k]=mix[k]-(aa[k]+bb[k])/2
    v['cross_collision_both_two']=v['cross_collision'] if v['correct_a']>=2 and v['correct_b']>=2 else None
    v['within_mean_both_two']=(v['collision_a']+v['collision_b'])/2 if v['cross_collision_both_two'] is not None else None
    for m in (1,2,4):
        eligible=v['correct_a']>=2*m and v['correct_b']>=2*m
        ac=[k for k in a if k is not None];bc=[k for k in b if k is not None]
        mixed=portfolio([ac,bc],[m,m])['distinct8'] if eligible else None
        ca=portfolio([ac],[2*m])['distinct8'] if eligible else None;cb=portfolio([bc],[2*m])['distinct8'] if eligible else None
        v[f'matched_{m}_mix_distinct']=mixed;v[f'matched_{m}_a_distinct']=ca;v[f'matched_{m}_b_distinct']=cb
        v[f'matched_{m}_mix_minus_average']=mixed-(ca+cb)/2 if eligible else None
        v[f'matched_{m}_mix_minus_a']=mixed-ca if eligible else None;v[f'matched_{m}_mix_minus_b']=mixed-cb if eligible else None
    return v


def model_values(fine,coarse,ref):
    a=portfolio([fine],[8]);b=portfolio([coarse],[8]);ca,cb=collision(fine),collision(coarse)
    assert a['pass8']==b['pass8'] and a['correct']==b['correct'] and b['distinct8']<=a['distinct8']+1e-10
    assert ca is None or cb>=ca-1e-10
    return {'pass1':a['correct']/8,'pass8':a['pass8'],'fine_distinct8':a['distinct8'],'coarse_distinct8':b['distinct8'],'distinct8_change':b['distinct8']-a['distinct8'],'fine_extra8':a['extra8'],'coarse_extra8':b['extra8'],'fine_collision':ca,'coarse_collision':cb,'collision_change':cb-ca if ca is not None else None,'fine_uniform_collision':ref['fine_uniform_collision'] if ca is not None else None,'coarse_uniform_collision':ref['coarse_uniform_collision'] if ca is not None else None,'fine_excess_uniform':ca-ref['fine_uniform_collision'] if ca is not None and ref['fine_uniform_collision'] is not None else None,'coarse_excess_uniform':cb-ref['coarse_uniform_collision'] if cb is not None and ref['coarse_uniform_collision'] is not None else None,'coarse_excess_fine_induced':cb-ref['fine_induced_coarse_collision'] if cb is not None and ref['fine_induced_coarse_collision'] is not None else None}


def describe(values,weights):
    names=list(values[0]);a=np.asarray([[r[k] for k in names] for r in values],float);mask=np.isfinite(a);n=mask.sum(0)
    with np.errstate(invalid='ignore',divide='ignore'):
        point=np.nansum(a,axis=0)/n;boots=(weights@np.nan_to_num(a))/(weights@mask.astype(float))
    result={}
    for j,k in enumerate(names):
        valid=boots[:,j][np.isfinite(boots[:,j])]
        result[k]={'estimate':float(point[j]) if n[j] else None,'ci95':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,'eligible_prompts':int(n[j]),'total_prompts':len(values),'defined_replicates':len(valid)}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=BASE/'offline');ap.add_argument('--replicates',type=int,default=20000);a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    codefiles=[Path(__file__),ROOT/'ops/followup_metrics.py',ROOT/'ops/summarize_frontier_comparison.py']
    protocol=BASE/'protocol_manifest.json'
    frozen={'created_at_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':{str(p):file_sha(p) for p in codefiles},'protocol_manifest_sha256':file_sha(protocol),'replicates':a.replicates,'bootstrap_seed':20260911,'outcomes_read':False}
    if (out/'implementation_freeze.json').exists():
        old=json.loads((out/'implementation_freeze.json').read_text());assert old['code_sha256']==frozen['code_sha256'] and old['replicates']==a.replicates
    else:atomic_new(out/'implementation_freeze.json',frozen)
    cachepath=out/'authenticated_keys.json'
    if cachepath.exists():
        cache=json.loads(cachepath.read_text())
        for p,h in cache['source_sha256'].items():assert file_sha(p)==h
        rows=cache['rows'];models=cache['models'];sources=cache['source_sha256']
    else:
        rows,models,sources=load_authenticated();atomic_new(cachepath,{'rows':rows,'models':models,'source_sha256':sources})
    references=[support(r) for r in rows];refs_path=out/'support_references.json'
    if not refs_path.exists():atomic_new(refs_path,{'records':[{'level':r['level'],'domain':r['domain'],'row_index':r['row_index'],**ref} for r,ref in zip(rows,references)],'note':'Graph/Python exact counts independently recomputed; MathIR/Pantry use the existing audited finite-domain certificates. Countdown has open verifier support and no uniform numeric reference. Coarse-uniform and uniform-fine pushed through coarsening are different references.'})
    rng=np.random.default_rng(20260911);groups={}
    for l in (1,2,3):
        for d in DOMAINS:
            ids=[i for i,r in enumerate(rows) if r['level']==l and r['domain']==d];assert len(ids)==128
            w=np.zeros((a.replicates,128),dtype=float);idx=rng.integers(0,128,size=(a.replicates,128));np.add.at(w,(np.arange(a.replicates)[:,None],idx),1)
            groups[f'level{l}/{d}']=(ids,w)
    # Stratified bootstrap for global prompt-average estimates; same draws for every model/pair/key definition.
    groups['all']=(list(range(len(rows))),np.concatenate([groups[f'level{l}/{d}'][1] for l in (1,2,3) for d in sorted(DOMAINS)],axis=1))
    result={'schema':'inference-followups-offline-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'models':list(models),'responses':107520,'prompt_count':1920,'pairs':21,'source_sha256':sources,'cache_sha256':file_sha(cachepath),'implementation_freeze_sha256':file_sha(out/'implementation_freeze.json'),'support_reference_sha256':file_sha(refs_path),'bootstrap':{'seed':20260911,'replicates':a.replicates,'unit':'whole prompt, stratified by domain/level; shared resamples for all models, pairs and keys','intervals':'pointwise percentile 95%, descriptive; no pair selection or multiplicity correction'},'coarse_models':{},'cross_model':{}}
    perpath=out/'per_prompt.jsonl'
    with perpath.open('w') as recordfile:
        for grading in ('strict','normalized_secondary'):
            fine={m:v[grading] for m,v in models.items()};coarse={m:[[coarse_key(k,json.loads(r['answer'])) for k in ks] for r,ks in zip(rows,v)] for m,v in fine.items()}
            for m in models:
                vals=[model_values(f,c,ref) for f,c,ref in zip(fine[m],coarse[m],references)]
                result['coarse_models'].setdefault(grading,{})[m]={g:describe([vals[i] for i in ids],w) for g,(ids,w) in groups.items()}
            for keys,pools in [('fine',fine),('coarse',coarse)]:
                section=result['cross_model'].setdefault(grading,{}).setdefault(keys,{})
                for ma,mb in itertools.combinations(models,2):
                    vals=[pair_values(ka,kb) for ka,kb in zip(pools[ma],pools[mb])]
                    for row,v in zip(rows,vals):recordfile.write(json.dumps({'grading':grading,'keys':keys,'model_a':ma,'model_b':mb,'level':row['level'],'domain':row['domain'],'row_index':row['row_index'],**v},sort_keys=True,allow_nan=False)+'\n')
                    section[ma+' | '+mb]={g:describe([vals[i] for i in ids],w) for g,(ids,w) in groups.items()}
                    print(json.dumps({'event':'pair_complete','grading':grading,'keys':keys,'pair':[ma,mb]}),flush=True)
    result['per_prompt_sha256']=file_sha(perpath)
    result['validation']={'all_seven_original_cohorts_authenticated':True,'all_requests_same_prompt_text':True,'same_successes_under_coarsening':True,'all_per_prompt_monotonicity_checks_pass':True,'all_21_pairs_included':True,'exact_subset_expectations':True}
    result['limitations']=['Mixed and unmixed portfolios match response counts, not provider token/compute cost.','Observed overlap/Jaccard are finite-budget descriptions, not estimates of full support overlap.','Correctness matching discards ineligible prompts; denominators and defined bootstrap replicates are explicit.','Output divisor pairs do not identify algorithms.','All intervals are conditional on saved response cohorts and resample prompts, not fresh model draws.','Two observed graphs related by renaming merge only when shown colors are fixed.','Inference robustness does not by itself establish robustness of every training-effect estimate.']
    atomic_new(out/'results.json',result)
    write_report(out,result)


def write_report(out,r):
    def fmt(v,percent=False):
        if v['estimate'] is None:return 'undefined'
        k=100 if percent else 1
        ci=v['ci95'];return f"{k*v['estimate']:.3f} [{k*ci[0]:.3f}, {k*ci[1]:.3f}]"
    lines=['# Inference follow-ups: complete offline analyses','','107,520 authenticated responses, seven models, 1,920 matched prompts, all 21 model pairs. Intervals are paired whole-prompt bootstrap 95% intervals (20,000 replicates). Comparisons match eight responses; token and compute budgets differ by provider.','','## Strict grading: 4+4 mixtures','','Each row averages the same 1,920 prompts. ΔD is mixed distinct correct outcomes minus the average of its two eight-response constituents. The last column holds the successful-response budget fixed at four (2+2 versus four from each constituent); only prompts with at least four correct responses from both models qualify.','','| Pair | ΔD (fine) | Δpass@8 (percentage points) | ΔD (coarse) | Correctness matched ΔD | Eligible |','|---|---:|---:|---:|---:|---:|']
    for pair,s in r['cross_model']['strict']['fine'].items():
        v=s['all'];c=r['cross_model']['strict']['coarse'][pair]['all'];x=v['matched_2_mix_minus_average']
        lines.append(f"| {pair} | {fmt(v['mix_minus_average_distinct8'])} | {fmt(v['mix_minus_average_pass8'],True)} | {fmt(c['mix_minus_average_distinct8'])} | {fmt(x)} | {x['eligible_prompts']} |")
    lines+=['','## Coarser outcomes, strict grading','','Collision averages prompts with at least two correct responses. Accuracy and pass@8 are identical under both key definitions.','','| Model | Fine D@8 | Coarse D@8 | Fine collision | Coarse collision |','|---|---:|---:|---:|---:|']
    for m,s in r['coarse_models']['strict'].items():
        v=s['all'];lines.append(f"| {m} | {fmt(v['fine_distinct8'])} | {fmt(v['coarse_distinct8'])} | {fmt(v['fine_collision'])} | {fmt(v['coarse_collision'])} |")
    lines+=['','Full results include every domain/level, both grading rules, mixtures versus each constituent, correctness budgets 2/4/8, cross- and within-model collision, overlap counts, Jaccard, and finite-sample squared-distance estimates (negative estimates retained). Graph/Python coarse supports are recomputed; Countdown has no numeric uniform-support baseline. Pantry supports remain unchanged.','','These are inference analyses. They do not identify distinct Python algorithms or independently validate all training-effect contrasts.','']
    (out/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
