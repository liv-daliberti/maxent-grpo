"""Freeze input-only Pantry perturbations and existing checkpoint identities."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'src')]
from followup_metrics import atomic_new,file_sha,sha,read_lines
from make_pantry_plan_mode_data import enumerate_pantry_supports
from oat_drgrpo.pantry_plan import validate_pantry_plan
BASE=ROOT/'artifacts/modebench_inference_followups_20260911/pantry'


def perturbations(spec):
    supports=enumerate_pantry_supports(spec)
    if not supports:raise ValueError('original task infeasible')
    options=[('outage',i['id']) for i in spec['ingredients']]
    options += [('diet',tag) for tag in sorted({t for i in spec['ingredients'] for t in i['tags']}-set(spec['forbidden_tags']))]
    records=[]
    for kind,target in options:
        revised=deepcopy(spec)
        if kind=='outage':
            for ingredient in revised['ingredients']:
                if ingredient['id']==target:ingredient['tags']=sorted(set(ingredient['tags'])|{'outage'})
            revised['forbidden_tags']=sorted(set(revised['forbidden_tags'])|{'outage'})
            update=f'An ingredient outage has occurred: {target} is unavailable. Do not use it. All other quantities, nutrition targets, and rules stay the same.'
        else:
            revised['forbidden_tags']=sorted(set(revised['forbidden_tags'])|{target})
            update=f'An additional dietary constraint now forbids ingredients tagged {target}. All other quantities, nutrition targets, and rules stay the same.'
        # Exclusions only: the complete original-support census plus a witness per
        # support is sufficient for the revised support census without resampling.
        accepted=[(support,candidate) for support,candidate in sorted(supports.items()) if validate_pantry_plan(candidate,revised) is not None]
        records.append({'id':kind+'_'+target,'kind':kind,'target':target,'spec':revised,'update':update,
                        'feasible':bool(accepted),'feasible_supports':len(accepted),
                        'witness':accepted[0][1] if accepted else None,
                        'original_supports':len(supports),'certificate':'exhaustive_original_support_enumeration_then_exact_exclusion_check'})
    return records


def prepare():
    from datasets import load_from_disk
    source=ROOT/'artifacts/modebench_prompt_ablation_20260911'
    ref=json.loads((ROOT/'artifacts/modebench_inference_followups_20260911/protocol_manifest.json').read_text())
    for name,digest in ref['reference_input_sha256'].items():
        if file_sha(name)!=digest:raise ValueError('protocol input changed: '+name)
    plan=json.loads((source/'local/plan_v2.json').read_text())
    rows=[r for r in read_lines(source/'rows.jsonl') if r['domain']=='pantry_plan' and r['level']==2]
    assert len(rows)==32
    devpath=ROOT/'var/data/modebench_harder_v2_matched_r5/pantry/dev'
    ds=load_from_disk(str(devpath));ds=ds['multi_answer']
    dev=[{'level':2,'domain':'pantry_plan','row_index':i,'answer':r['answer'],'problem':r['problem']} for i,r in enumerate(ds)]
    dev=sorted(dev,key=lambda r:sha(['pantry-adaptation-dev',20260911,r['row_index']]))[:16]
    assert not ({r['problem'] for r in dev}&{r['problem'] for r in rows})
    checkpoints=[c for c in plan['checkpoints'] if c['domain'] in (None,'pantry')]
    assert len(checkpoints)==5
    tasks=[]
    for split,items in [('eval',rows),('dev',dev)]:
        for row in sorted(items,key=lambda r:r['row_index']):
            spec=json.loads(row['answer']) if isinstance(row['answer'],str) else row['answer']
            task={'id':f"{split}_{row['row_index']:03d}",'split':split,'row':row,'spec':spec,'perturbations':perturbations(spec)}
            tasks.append(task)
            print(json.dumps({'task':task['id'],'eligible':sum(p['feasible'] for p in task['perturbations']),'candidates':len(task['perturbations'])}),flush=True)
    payload={'schema':'pantry-adaptation-inputs-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
             'repo_root':str(ROOT),'base':str(BASE),'tasks':tasks,'checkpoints':checkpoints,
             'source_plan':str(source/'local/plan_v2.json'),'source_plan_sha256':file_sha(source/'local/plan_v2.json'),
             'saved_results_root':plan['output_root'],'development_source':str(devpath),
             'development_files':{str(p):file_sha(p) for p in devpath.rglob('*') if p.is_file()},
             'code_sha256':{str(p):file_sha(p) for p in [Path(__file__),ROOT/'ops/make_pantry_plan_mode_data.py',ROOT/'src/oat_drgrpo/pantry_plan.py']},
             'temperature_grid':[0.7,1.0,1.3],'max_recovery_calls':8,'outcomes_read':False}
    atomic_new(BASE/'inputs.json',payload)
    print(json.dumps({'status':'frozen','tasks':len(tasks),'feasible_perturbations':sum(p['feasible'] for t in tasks for p in t['perturbations'])}),flush=True)

if __name__=='__main__':prepare()
