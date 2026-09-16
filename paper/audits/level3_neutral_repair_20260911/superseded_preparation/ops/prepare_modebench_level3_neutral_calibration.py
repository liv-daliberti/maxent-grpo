#!/usr/bin/env python3
"""Freeze fresh neutral-Python development, then submit its fixed four tiers.

A separate dataset revision is admitted only after development fitting and
fresh confirmation. This preparation never changes an existing dataset/run.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'ops/exp_scaling'),str(ROOT/'src')]
from evaluate_modebench_level3 import atomic_new,file_sha,sha
BASE=ROOT/'artifacts/modebench_level3_neutral_calibration_20260911_r2'
SOURCE_DATA=ROOT/'var/data/modebench_level3_matched_v3'
REFERENCE=ROOT/'var/data/modebench_harder_v2_matched_r5'
TARGET=ROOT/'var/results/modebench_level3_v2/confirmation_python_v6/confirmation_05b_python_factors.json'
FINAL=ROOT/'var/data/modebench_level3_neutral_matched_v1'
MODEL=ROOT/'var/cache/huggingface/transformers/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1'
PYTHON=ROOT/'var/seed_paper_eval/paper310/bin/python'
DOMAIN='python_factors'
DEV_LABELS=list(range(6819100,6819104))
CONF_LABELS=list(range(6819200,6819204))
GEN_SEEDS={'dev':120091100,'train':120191100,'eval':120291100}


def read(path):return json.loads(Path(path).read_text())
def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def row_id(row):return tuple(sorted(json.loads(row['answer'])['cases']))
def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as h:
        for row in rows:h.write(json.dumps(row,sort_keys=True)+'\n')
def load_rows(path,split):
    from datasets import load_from_disk
    return [dict(r) for r in load_from_disk(str(path))[split]]
def now():return datetime.now(timezone.utc).isoformat()


def prepare(base):
    import modebench_level3_python_neutral_v2 as candidate
    import evaluate_modebench_level3_neutral as evaluator
    from make_python_factor_mode_data import _row
    from modebench_independent_seeds import seed_schedule
    from evaluate_modebench_base_grid import model_identity
    if (base/'plan.json').exists():raise FileExistsError('preparation already sealed')
    base.mkdir(parents=True,exist_ok=True)
    target=read(TARGET)
    values={m:target['metrics'][m]['mean'] if isinstance(target['metrics'][m],dict) else target['metrics'][m]
            for m in ('pass1','pass8')}
    assert values=={'pass1':0.2109375,'pass8':0.76953125}
    references={s:load_rows(REFERENCE/DOMAIN/s,'train' if s=='train' else 'multi_answer') for s in GEN_SEEDS}
    hist={s:Counter(r['answer_mode_count'] for r in rows) for s,rows in references.items()}
    union=hist['dev']|hist['eval']
    assert sum(union.values())==166
    sources={str(TARGET):file_sha(TARGET)}
    blocked=set()
    prior_roots=set((ROOT/'var/data').glob('modebench_harder*'))|set((ROOT/'var/data').glob('modebench_level3*'))
    prior_roots.add(ROOT/'var/data/python_factor_modebench_v1')
    datasets=[]
    for directory in sorted(prior_roots):
        for split in ('train','dev','eval'):
            location=(directory if directory.name=='python_factor_modebench_v1' else directory/DOMAIN)/split
            if (location/'dataset_dict.json').exists():datasets.append(location)
    for part in ('development','confirmation'):
        location=ROOT/'var/data/e117_evaluation_reserve_v1'/part/DOMAIN/'eval'
        if (location/'dataset_dict.json').exists():datasets.append(location)
    for location in sorted(set(datasets)):
        from datasets import load_from_disk
        for subset in load_from_disk(str(location)).values():blocked.update(row_id(dict(r)) for r in subset)
        for p in sorted(location.rglob('*')):
            if p.is_file():sources[str(p)]=file_sha(p)
    pools=sorted({p for d in prior_roots for p in (d/'pools'/DOMAIN).glob('*.jsonl')})
    for p in pools:
        blocked.update(row_id(r) for r in jsonl(p));sources[str(p)]=file_sha(p)
    atomic_new(base/'excluded_cases.json',sorted(blocked))
    protocol={'schema':'level3-neutral-python-calibration-protocol-v1','registered_at_utc':now(),
        'purpose':'Recalibrate Level3 Python after adopting the neutral prompt; edit numeric task distribution.',
        'domain':DOMAIN,'reference':{'kind':'historical_fixed_empirical_target','path':str(TARGET),'sha256':file_sha(TARGET),'values':values,
            'interface':'historical registered-hints Level1 Qwen0.5B; not a new neutral control'},
        'candidate_model':'frozen Qwen2.5-3B-Instruct','candidate_interface':evaluator.frozen_interface(DOMAIN),
        'prompt':'current python_level3_neutral_v1, identical during calibration and future training/evaluation',
        'interpretation':'Numerical matching to the existing benchmark success rates; different reference wording precludes a pure difficulty-only causal comparison or two-sided equivalence claim.',
        'tolerances':{'pass1':.04,'pass8':.08},'fit_metrics':['pass1','pass8'],'distinct8_is_not_fitted':True,
        'case_windows':candidate.CASE_WINDOWS,'minimum_bands':candidate.MINIMUM_BANDS,'smallest_proper_factor_at_most':5,
        'exact_support_histograms':{s:dict(h) for s,h in hist.items()},'calibration_union_histogram':dict(union),
        'development_rows_per_tier':166,'development_tiers':4,'draws_per_prompt':4,'responses_per_draw':8,
        'development_labels':DEV_LABELS,'confirmation_labels':CONF_LABELS,'generation_seeds':GEN_SEEDS,
        'selection':'All 1771 grid20 weights, min maximum normalized error over dev/eval support-cell forecasts; one hash-fixed development subset. No selection by held-out outcomes.',
        'fitter':'ops/exp_scaling/fit_modebench_level3_fixed_reference.py:choose_fixed_reference_mixture',
        'fitter_sha256':file_sha(ROOT/'ops/exp_scaling/fit_modebench_level3_fixed_reference.py'),
        'selection_seed':6391701,'development_fail_policy':'Preserve failure and design a separate prospective revision; no seed/subset retries.',
        'confirmation':'After a passing development fit, generate fresh 384 train and 128 eval problems; selected dev has128. Freeze all rows before one fresh 128x4x8 confirmation. Require both unrounded gates; report prompt-bootstrap intervals and regrade every saved attempt.',
        'final_data_root':str(FINAL),'prior_data_root':str(SOURCE_DATA),
        'unchanged_domains':['countdown','graph_coloring','mathir','pantry'],
        'historical_excluded_cases':len(blocked),'excluded_cases_sha256':file_sha(base/'excluded_cases.json'),
        'existing_data_and_results_modified':False,'treatment_training':False}
    atomic_new(base/'protocol.json',protocol)
    code=base/'code'
    for subtree in ('ops','src'):
        for p in sorted((ROOT/subtree).rglob('*.py')):
            if '__pycache__' in p.parts:continue
            dest=code/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    p=ROOT/'tests/test_modebench_level3_neutral_calibration.py';dest=code/'tests'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    pins={str(p):file_sha(p) for p in sorted(code.rglob('*.py'))}
    capacities={str(t):{str(s):candidate.available_capacity(s,t,{(DOMAIN,c) for c in blocked}) for s in union} for t in range(4)}
    assert all(capacities[str(t)][str(s)]>=n for t in range(4) for s,n in union.items())
    atomic_new(base/'capacity.json',capacities)
    # Warm the unchanged external verifier before deterministic witness checks.
    _row(cases=(6,8,10,12),split_tag='neutral_calibration_warmup',seed=0,index=0)
    tasks=[];allblocks=set();pool_records={}
    for tier in range(4):
        rows=candidate.build_pool(DOMAIN,union,{(DOMAIN,c) for c in blocked},GEN_SEEDS['dev']+1000*tier,'neutral_l3_development',tier,multiplier=1)
        assert Counter(r['answer_mode_count'] for r in rows)==union
        ids={row_id(r) for r in rows};assert len(ids)==len(rows) and not ids&blocked;blocked|=ids
        path=base/'pools'/f'tier_{tier}.jsonl';write_rows(path,rows)
        schedules=seed_schedule(DOMAIN,[r['problem'] for r in rows],DEV_LABELS)
        blocks={v for row in schedules for v in row};assert not allblocks&blocks;allblocks|=blocks
        task={'domain':DOMAIN,'level':'level3','split':'dev','interface':evaluator.INTERFACE,'batch_size':8,
              'rows_jsonl':str(path),'seeds':DEV_LABELS,'output':str(base/'development'/f'tier_{tier}.json')}
        taskfile=base/'tasks'/f'tier_{tier}.json';atomic_new(taskfile,[task]);tasks.append(str(taskfile))
        pool_records[str(tier)]={'path':str(path),'sha256':file_sha(path),'rows':len(rows),'task_path':str(taskfile),'task_sha256':file_sha(taskfile)}
        print(json.dumps({'event':'pool_ready','tier':tier,'rows':len(rows)}),flush=True)
    atomic_new(base/'all_development_excluded_cases.json',sorted(blocked))
    plan={'schema':'level3-neutral-python-development-plan-v1','created_at_utc':now(),'repo_root':str(ROOT),
          'base':str(base),'protocol_sha256':file_sha(base/'protocol.json'),'protocol':protocol,'source_sha256':sources,
          'code_sha256':pins,'model':model_identity(MODEL,'3b'),'python':str(PYTHON),'tasks':tasks,'pools':pool_records,
          'expected_responses':4*166*32,'independent_request_blocks':len(allblocks),'maximum_concurrent_gpus':4}
    worker=base/'worker.slurm'
    worker.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+str(ROOT)+'\nsource ops/repo_env.sh\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1\nexec '+str(PYTHON)+' -B '+str(code/'ops/evaluate_modebench_level3_neutral.py')+' --model '+str(MODEL)+' --model-label 3b --tasks-json '+str(base/'tasks')+'/tier_${SLURM_ARRAY_TASK_ID:?}.json\n')
    plan['worker_sha256']=file_sha(worker)
    atomic_new(base/'plan.json',plan)
    (base/'logs').mkdir(exist_ok=True)
    print(json.dumps({'event':'prepared','base':str(base),'expected_responses':plan['expected_responses'],'protocol_sha256':plan['protocol_sha256']}),flush=True)


def submit(base):
    plan=read(base/'plan.json')
    assert file_sha(base/'protocol.json')==plan['protocol_sha256']
    for group in ('code_sha256','source_sha256'):
        for path,digest in plan[group].items():assert file_sha(path)==digest,path
    for row in plan['pools'].values():
        assert file_sha(row['path'])==row['sha256'] and file_sha(row['task_path'])==row['task_sha256']
    assert file_sha(base/'worker.slurm')==plan['worker_sha256']
    cmd=['sbatch','--parsable','--job-name=mb-l3-neutral-dev','--partition=lowprio','--account=mltheory','--gres=gpu:a5000:1','--cpus-per-task=6','--mem=48G','--time=02:00:00','--array=0-3%4','--chdir='+plan['repo_root'],'--output='+str(base/'logs/%A_%a.out'),'--error='+str(base/'logs/%A_%a.err'),str(base/'worker.slurm')]
    atomic_new(base/'submission_intent.json',{'created_at_utc':now(),'plan_sha256':file_sha(base/'plan.json'),'command':cmd})
    result=subprocess.run(cmd,text=True,capture_output=True)
    value={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'job_id':result.stdout.strip().split(';')[0] if result.returncode==0 else None}
    atomic_new(base/'submission_result.json',value)
    if result.returncode:raise RuntimeError(result.stderr)
    print(json.dumps(value),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','submit'));parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args();args.base=args.base.resolve()
    args.base.mkdir(parents=True,exist_ok=True)
    with (args.base/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (prepare if args.action=='prepare' else submit)(args.base)
