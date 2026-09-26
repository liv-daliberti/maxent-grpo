#!/usr/bin/env python3
"""One user-requested fresh confirmation of a prospectively fixed numeric mixture.

Separate from V2's full development/fitting protocol. This check never creates
an admission record, changes release gates, starts training, or tunes on eval.
All splits and the one 17:3 recipe are sealed before any new model outcomes.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,shlex,shutil,statistics,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import calibrate_modebench_level3_neutral_v2 as c
import modebench_level3_python_neutral_v5 as gen
ART=ROOT/'var/artifacts/modebench_level3_neutral_fixed_confirmation_20260912'
DATA=ROOT/'var/data/modebench_level3_neutral_fixed_confirmation_20260912'
PLAN=ART/'registration.json'
SOURCE=Path(__file__).resolve()
LABELS=[7029000,7029001,7029002,7029003]
WEIGHTS=[17,0,0,3]
GEN_SEEDS={'train':13137100,'dev':13337100,'eval':13537100}
new,read,digest,require=c.new,c.read,c.digest,c.require

def now():return datetime.now(timezone.utc).isoformat()

def task(i):
    return {'domain':c.DOMAIN,'level':'level3','split':'eval','interface':c.neutral.INTERFACE,
            'rows_jsonl':str(ART/'eval.jsonl'),'output':str(ART/f'confirmation_seed{i}.json'),
            'seeds':[LABELS[i]],'batch_size':8,'row_offset':0,'row_limit':0}

def check(expected):
    require(digest(PLAN)==expected,'confirmation registration changed')
    plan=read(PLAN);c.common.verify_pins(plan['files_sha256'])
    actual=c.neutral.evaluator.model_identity(Path(plan['model']['path']),'3b')
    actual['vllm_version']=c.neutral.evaluator.ENGINE_CONTRACT['vllm_version']
    require(actual==plan['model'],'frozen checkpoint changed')
    require(plan['code_identity']==c.neutral.code_identity(),'evaluator changed')
    require(plan['weights']==WEIGHTS and plan['labels']==LABELS,'recipe or draws changed')
    return plan

def prepare():
    require(not ART.exists() and not DATA.exists(),'fresh candidate and confirmation namespaces required')
    blocked,pins=c.historical_inventory()
    # The reused V2 helper omits its own pools; this new check must exclude them.
    for p in sorted(c.POOLS.glob('*.jsonl')):
        blocked |= c.ids(c.mixture.read_jsonl(p));pins[str(p)]=digest(p)
    pilots=sorted((ROOT/'var/artifacts').glob('modebench_level3_neutral*pilot/rows.jsonl'))
    for p in pilots:
        blocked |= c.ids(c.mixture.read_jsonl(p));pins[str(p)]=digest(p)
        receipt=p.with_name('receipt.json')
        if receipt.exists():pins[str(receipt)]=digest(receipt)
    ART.mkdir();DATA.mkdir();original=set(blocked)
    new(ART/'excluded_identities.json',sorted(blocked,key=repr))
    from datasets import Dataset,DatasetDict
    splits={}
    for split in ('train','dev','eval'):
        target=c.common.reference_histograms(c.DOMAIN)[split]
        cells=c.mixture.allocate_cells(target,WEIGHTS,c.fitter.SELECTION_SEED)
        rows=[];exclusion=set(blocked)
        for tier in (0,3):
            quota=Counter({k[0]:n[tier] for k,n in cells.items() if n[tier]})
            if quota:
                fresh=gen.build_pool(c.DOMAIN,quota,blocked,GEN_SEEDS[split]+1000*tier,
                                     'neutral_fixed_confirmation_20260912_'+split,tier,1)
                rows.extend(fresh);blocked |= c.ids(fresh)
        rows.sort(key=lambda r:c.mixture.sha([GEN_SEEDS[split],c.mixture.sha(r)]))
        checks=c.materializer.verify_rows(c.DOMAIN,rows,c.materializer.reference_rows(c.DOMAIN,split),
                                          c.materializer.modes(c.materializer.reference_rows(c.DOMAIN,split)),exclusion)
        require(not c.ids(rows)&original,'historical/pilot identity leakage')
        DatasetDict({c.materializer.SPLITS[split][1]:Dataset.from_list(rows)}).save_to_disk(str(DATA/c.DOMAIN/split))
        splits[split]={'rows':len(rows),'row_sha256':c.mixture.sha(rows),'checks':checks,
                       'tier_counts':dict(Counter(r['level3_difficulty'] for r in rows))}
        if split=='eval':(ART/'eval.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    for domain in c.common.DOMAINS:
        if domain!=c.DOMAIN:
            shutil.copytree(c.common.DATASET/domain,DATA/domain)
            require({str(p.relative_to(DATA/domain)):digest(p) for p in (DATA/domain).rglob('*') if p.is_file()}==
                    {str(p.relative_to(c.common.DATASET/domain)):digest(p) for p in (c.common.DATASET/domain).rglob('*') if p.is_file()},'retained domain changed')
    for p in DATA.rglob('*'):
        if p.is_file():pins[str(p)]=digest(p)
    for i in range(4):new(ART/f'task{i}.json',[task(i)])
    evidence=ROOT/'var/artifacts/modebench_level3_neutral_divisor_pilot/receipt.json'
    require(evidence.exists(),'completed development pilot is required')
    for p in [SOURCE,Path(gen.__file__),ART/'eval.jsonl',ART/'excluded_identities.json',evidence,*ART.glob('task*.json')]:pins[str(p)]=digest(p)
    for name,sha in c.mixture.local_dependency_sources([SOURCE,Path(gen.__file__)]).items():pins[str(ROOT/name)]=sha
    code=c.neutral.code_identity()
    for name,sha in code.items():pins[str(ROOT/name)]=sha
    rows=c.mixture.read_jsonl(ART/'eval.jsonl');schedule=c.neutral.evaluator.schedule_record(c.DOMAIN,rows,LABELS)
    require(len({s for group in schedule['request_seeds'] for s in group})==512,'confirmation request RNG overlap')
    benchmark=c.common.benchmark_metadata()[c.DOMAIN]
    require(digest(benchmark['receipt_path'])==benchmark['receipt_sha256'],'fixed reference changed')
    pins[benchmark['receipt_path']]=benchmark['receipt_sha256']
    model=read(c.PLAN)['model']
    plan={'schema':'neutral_fixed_candidate_fresh_confirmation_v1','created_at':now(),
          'user_request':'okay... do a fresh confirmation?',
          'weights':WEIGHTS,'weight_denominator':20,'labels':LABELS,'generation_seeds':GEN_SEEDS,
          'selection':'One hand-specified 17:3 common3/alleven numeric recipe, fixed before fresh confirmation.',
          'development_evidence':str(evidence),'even_stratum_separately_model_scored':False,
          'full_development_fit_claimed':False,'admission_or_training_automated':False,
          'no_recipe_or_subset_selection_from_confirmation':True,
          'baseline':{'pass1':.2109375,'pass8':.76953125},'tolerances':c.TOLERANCES,
          'baseline_interface':'historical hinted Level1 reference retained; no same-interface causal comparison',
          'model':model,'code_identity':code,'splits':splits,'dataset':str(DATA),
          'other_domains_byte_identical_to':str(c.common.DATASET),'seed_schedule_sha256':c.mixture.sha(schedule),
          'files_sha256':pins,'planned_responses':4096,'confirmation_prompts':128}
    new(PLAN,plan)
    new(DATA/'identity.json',{'schema':'neutral_fixed_candidate_dataset_v1','registration_sha256':digest(PLAN),
                             'splits':splits,'difficulty_admitted':False})
    return {'registration_sha256':digest(PLAN),'dataset':str(DATA),'splits':splits}

def submit(expected):
    check(expected);require(not (ART/'submission_intent.json').exists(),'single submission only')
    free=shutil.disk_usage(ROOT/'var').free/2**30
    # Existing shared workload bound plus four conservative 32-GiB output slots.
    require(free>2513+128,'insufficient space after conservative shared reservations')
    cmd=['sbatch','--parsable','--partition=all','--qos=normal','--array=0-3%3',
         '--gres=gpu:rtx_6000:1','--cpus-per-task=6','--mem=48G','--time=00:50:00',
         '--exclude=node103','--chdir='+str(ROOT),'--job-name=l3-neutral-fresh-confirm',
         '--output='+str(ART/'worker-%A_%a.out'),'--error='+str(ART/'worker-%A_%a.err'),
         '--export=ALL,VLLM_USE_V1=0,VLLM_ATTENTION_BACKEND=XFORMERS,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4',
         '--wrap=source ops/repo_env.sh; exec '+shlex.join([str(c.PYTHON),'-u','-B',str(SOURCE),'worker','--registration-sha256',expected])]
    new(ART/'submission_intent.json',{'command':cmd,'free_gib':free,'reserved_output_gib':128,'registration_sha256':expected})
    p=subprocess.run(cmd,capture_output=True,text=True)
    result={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};new(ART/'submission_result.json',result)
    require(p.returncode==0,'submission failed');return result

def worker(expected):
    plan=check(expected);i=int(os.environ['SLURM_ARRAY_TASK_ID']);require(i in range(4),'invalid shard')
    new(ART/f'claim{i}.json',{'job_id':os.environ['SLURM_JOB_ID'],'registration_sha256':expected,'created_at':now()})
    c.neutral.evaluator.main(['--model',plan['model']['path'],'--model-label','3b',
                              '--tasks-json',str(ART/f'task{i}.json'),'--confirm-eval'])

def combine(receipts):
    require(len(receipts)==4,'four independent receipts required')
    reference=receipts[0]['prompt_results'];labels=[]
    for receipt in receipts:
        labels.extend(receipt['identity']['seeds'])
        require([r['row_sha256'] for r in receipt['prompt_results']]==[r['row_sha256'] for r in reference],'shard row mismatch')
        require(receipt['information_boundary']['confirmation_explicitly_authorized'] is True,'unconfirmed shard')
    require(labels==LABELS,'missing, repeated or reordered confirmation labels')
    results=[]
    for i,row in enumerate(reference):
        draws=[d for receipt in receipts for d in receipt['prompt_results'][i]['draws']]
        require(len(draws)==4 and [d['seed'] for d in draws]==LABELS,'invalid prompt draw union')
        results.append({'row_sha256':row['row_sha256'],**{m:statistics.mean(d[m] for d in draws) for m in ('pass1','pass8','distinct8')}})
    return results

def audit(expected):
    plan=check(expected);require(not (ART/'confirmation_report.json').exists(),'single frozen confirmation report required')
    receipts=[];sources={};all_rows=None
    from oat_drgrpo.math_grader import validated_modebench_outcome_key
    count=0
    for i in range(4):
        p=Path(task(i)['output']);receipt,rows=c.validate_receipt(p,task(i),plan)
        if all_rows is None:all_rows=rows
        require(c.mixture.sha(rows)==plan['splits']['eval']['row_sha256'],'eval split mismatch')
        for row,result in zip(rows,receipt['prompt_results']):
            for draw in result['draws']:
                for attempt in draw['attempts']:
                    require(validated_modebench_outcome_key(attempt['text'],row['answer'])==attempt['canonical_key'],'external regrade mismatch');count+=1
        receipts.append(receipt);sources[str(p)]=digest(p)
    results=combine(receipts);require(count==4096 and len(results)==128,'incomplete confirmation')
    import numpy as np
    rng=np.random.default_rng(7030100);sample=rng.integers(0,128,size=(10000,128))
    metrics={m:statistics.mean(r[m] for r in results) for m in ('pass1','pass8','distinct8')}
    delta={m:metrics[m]-plan['baseline'][m] for m in c.TOLERANCES}
    gates={m:abs(delta[m])<=plan['tolerances'][m] for m in c.TOLERANCES}
    ci={m:np.quantile(np.asarray([r[m] for r in results])[sample].mean(axis=1)-plan['baseline'][m],[.025,.975]).tolist() for m in c.TOLERANCES}
    report={'status':'observed_approximate_match' if all(gates.values()) else 'outside_match_tolerance',
            'registration_sha256':expected,'dataset':str(DATA),'confirmation_receipts_sha256':sources,
            'metrics':metrics,'baseline':plan['baseline'],'deltas':delta,'tolerances':plan['tolerances'],'gates':gates,
            'delta_ci95_fixed_reference':ci,'bootstrap_unit':'prompt, four independent draws averaged',
            'external_verifier_responses_regraded':count,'full_development_fit_claimed':False,
            'automatic_admission':False,'training_started':False,'no_confirmation_tuning':True,'created_at':now()}
    new(ART/'combined_prompt_metrics.json',results);new(ART/'confirmation_report.json',report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','worker','audit']);p.add_argument('--registration-sha256');a=p.parse_args()
    result=prepare() if a.action=='prepare' else globals()[a.action](a.registration_sha256)
    print(json.dumps(result,sort_keys=True,default=str),flush=True)
if __name__=='__main__':main()
