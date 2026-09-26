#!/usr/bin/env python3
"""Versioned Python-only neutral calibration, fitting and fresh confirmation.

No training is released here. A separate migration must consume admission.json.
The old empirical Level1 target is retained with its original prompt provenance.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'ops/exp_scaling'), str(ROOT/'ops'), str(ROOT/'src')]
import modebench_level3_v3_common as common
import materialize_modebench_level3 as materializer
import modebench_level3_python_neutral_v1 as generator
import fit_modebench_level3 as mixture
import fit_modebench_level3_fixed_reference as fitter
import evaluate_modebench_level3_neutral as neutral

ART = ROOT/'var/artifacts/modebench_level3_neutral_v1'
POOLS = ROOT/'var/data/modebench_level3_calibration_neutral_v1/pools/python_factors'
DATA = ROOT/'var/data/modebench_level3_matched_neutral_v1'
RESULTS = ROOT/'var/results/modebench_level3_neutral_v1'
PLAN = ART/'registration.json'
SOURCE = Path(__file__).resolve()
PYTHON = ROOT/'var/seed_paper_eval/paper310/bin/python'
DOMAIN = 'python_factors'
DEV_LABELS = [6528000, 6528001, 6528002, 6528003]
CONF_LABELS = [6529000, 6529001, 6529002, 6529003]
GEN_SEEDS = {'dev': 10537100, 'train': 10737100, 'eval': 10937100}
TOLERANCES = {'pass1': .04, 'pass8': .08}
require, read, digest, new = common.require, common.read, common.digest, common.atomic_new


def now(): return datetime.now(timezone.utc).isoformat()


def rows_at(path):
    from datasets import load_from_disk
    d = load_from_disk(str(path))
    return [dict(row) for subset in d.values() for row in subset]


def ids(rows): return materializer.identity_set(DOMAIN, rows)


def historical_inventory():
    """Freeze all historical dataset and candidate identities, including v3."""
    blocked = materializer.historical_ids(DOMAIN)
    paths = set()
    for root in sorted((ROOT/'var/data').iterdir()):
        if root.name.startswith(('modebench_harder', 'modebench_level3')) and root != DATA:
            for split in common.SPLITS:
                d = root/DOMAIN/split
                if (d/'dataset_dict.json').is_file(): paths.update(p for p in d.rglob('*') if p.is_file())
            for p in (root/'pools'/DOMAIN).glob('*.jsonl'):
                if p.parent == POOLS: continue
                paths.add(p); blocked |= ids(mixture.read_jsonl(p))
    # Include the Level1 source and reserve files used by historical_ids.
    for path in materializer.LEVEL1[DOMAIN].values():
        paths.update(p for p in path.rglob('*') if p.is_file())
    return blocked, {str(p):digest(p) for p in sorted(paths)}


def task(tier=None):
    confirmation = tier is None
    return {'domain': DOMAIN, 'level':'level3', 'split':'eval' if confirmation else 'dev',
            'interface':neutral.INTERFACE, 'rows_jsonl':str(ART/'eval.jsonl' if confirmation else POOLS/f'difficulty_{tier}.jsonl'),
            'output':str(RESULTS/'confirmation.json' if confirmation else RESULTS/f'development_d{tier}.json'),
            'seeds':CONF_LABELS if confirmation else DEV_LABELS,
            'row_offset':0, 'row_limit':0, 'batch_size':8}


def validate_plan(expected):
    require(digest(PLAN) == expected, 'registration hash mismatch')
    plan = read(PLAN)
    common.verify_pins(plan['files_sha256'])
    require(plan['interface'] == neutral.frozen_interface(DOMAIN)
            and plan['code_identity'] == neutral.code_identity(), 'neutral execution implementation changed')
    actual = neutral.evaluator.model_identity(Path(plan['model']['path']), '3b')
    actual['vllm_version'] = neutral.evaluator.ENGINE_CONTRACT['vllm_version']
    require(actual == plan['model'], 'frozen 3B model identity changed')
    return plan


def prepare():
    require(not PLAN.exists(), 'fresh neutral calibration registration required')
    require(not POOLS.exists() and not DATA.exists() and not RESULTS.exists(), 'fresh calibration namespace required')
    benchmark = common.benchmark_metadata()[DOMAIN]
    require(digest(benchmark['receipt_path']) == benchmark['receipt_sha256'], 'fixed measured reference changed')
    reference = read(benchmark['receipt_path'])
    baseline = {m:statistics.mean(r[m] for r in reference['prompt_results']) for m in TOLERANCES}
    require(baseline == {'pass1':.2109375, 'pass8':.76953125}, 'historical target changed')
    model = read(ROOT/'var/artifacts/modebench_level3_v3/development/seal.json')['models']['3b']
    blocked, pins = historical_inventory()
    ART.mkdir(parents=True, exist_ok=True); POOLS.mkdir(parents=True); RESULTS.mkdir(parents=True)
    new(ART/'excluded_identities.json', sorted(blocked, key=repr))
    calibration = Counter({k[0]:v for k,v in common.calibration_histogram(DOMAIN).items()})
    capacity = {str(t):{str(s):generator.available_capacity(s,t,blocked) for s in calibration} for t in range(4)}
    require(all(capacity[str(t)][str(s)] >= 4*n for t in range(4) for s,n in calibration.items()), 'insufficient fresh capacity')
    original_blocked = set(blocked)
    generated = {}
    for tier in range(4):
        rows = generator.build_pool(DOMAIN, calibration, blocked, GEN_SEEDS['dev']+1000*tier,
                                    'level3_neutral_v1_development', tier, 1)
        checks = materializer.verify_rows(DOMAIN, rows, materializer.reference_rows(DOMAIN,'dev'), calibration, blocked)
        p = POOLS/f'difficulty_{tier}.jsonl'
        p.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
        pins[str(p)] = digest(p); blocked |= ids(rows)
        new(p.with_suffix('.identity.json'), {'rows':len(rows),'row_sha256':mixture.sha(rows),'checks':checks,
                                             'generation_seed':GEN_SEEDS['dev']+1000*tier})
        pins[str(p.with_suffix('.identity.json'))] = digest(p.with_suffix('.identity.json'))
        generated[str(tier)] = {'path':str(p),'rows':len(rows),'sha256':digest(p)}
        new(ART/f'development_d{tier}_tasks.json', [task(tier)])
    for p in [SOURCE, Path(generator.__file__), generator.BASE, ROOT/'tests/test_level3_neutral_calibration.py',
              ROOT/'ops/evaluate_modebench_level3_neutral.py', ART/'excluded_identities.json',
              Path(benchmark['receipt_path']), ROOT/'artifacts/modebench_level3_neutral_default_20260911/registration.json',
              *ART.glob('*_tasks.json')]: pins[str(p)] = digest(p)
    for name, sha in neutral.code_identity().items(): pins[str(ROOT/name)] = sha
    for name, sha in mixture.local_dependency_sources([SOURCE, Path(generator.__file__)]).items(): pins[str(ROOT/name)] = sha
    # Fresh problem identities make effective prompt-seeded streams independent of historical runs.
    requests = set()
    for tier in range(4):
        rows = mixture.read_jsonl(POOLS/f'difficulty_{tier}.jsonl')
        require(not ids(rows)&original_blocked, 'new development overlaps historical data')
        schedule = neutral.evaluator.schedule_record(DOMAIN, rows, DEV_LABELS)
        blocks = {s for group in schedule['request_seeds'] for s in group}
        require(len(blocks) == len(rows)*4 and not requests&blocks, 'development RNG collision')
        requests |= blocks
    plan = {'schema':'python_level3_neutral_calibration_registration_v1', 'created_at':now(),
            'domain':DOMAIN,'model':model,'interface':neutral.frozen_interface(DOMAIN),
            'code_identity':neutral.code_identity(), 'baseline':baseline, 'benchmark':benchmark,
            'baseline_interface':reference['identity']['interface'],
            'reference_semantics':'fixed_measured_Level1_with_historical_hints',
            'same_interface_comparison_claimed':False, 'approximate_match_not_statistical_equivalence':True,
            'tolerances':TOLERANCES, 'split_sizes':common.SPLITS, 'generation_seeds':GEN_SEEDS,
            'development_labels':DEV_LABELS,'confirmation_labels':CONF_LABELS,
            'presets':generator.PRESETS,'case_windows':generator.CASE_WINDOWS,'minimum_bands':generator.MINIMUM_BANDS,
            'calibration_pools':generated, 'capacity':capacity, 'files_sha256':pins,
            'selection':{'algorithm':fitter.ALGORITHM,'seed':fitter.SELECTION_SEED,'grid':20,
                         'selected_development_sets_scored':1,'confirmation_rows_selected_using_outcomes':False},
            'training_release_requires':'fresh confirmation admission.json plus original resource gates',
            'other_domains':'retain admitted v3 files byte-for-byte', 'prior_prompt_choice_informed_by_evaluation':True}
    new(PLAN, plan)
    return {'status':'prepared', 'registration_sha256':digest(PLAN),'development_rows':664,'development_attempts':21248}


def submit_once(name, command):
    intent, result = ART/f'{name}_submission_intent.json', ART/f'{name}_submission_result.json'
    require(not intent.exists() and not result.exists(), 'existing submission ownership; inspect before any retry')
    new(intent, {'created_at':now(),'command':command})
    p = subprocess.run(command, text=True, capture_output=True)
    new(result, {'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'recorded_at':now()})
    require(p.returncode == 0 and re.fullmatch(r'[1-9][0-9]*(?:;[^\s]+)?\s*',p.stdout), 'ambiguous/failed sbatch; preserve submission records')
    return p.stdout.strip().split(';')[0]


def gpu_command(phase, expected):
    args = ['sbatch','--parsable','--partition=all','--qos=normal','--gres=gpu:rtx_6000:1',
            '--cpus-per-task=6','--mem=48G','--time=01:00:00','--exclude=node103',
            '--chdir='+str(ROOT),'--job-name=l3-neutral-'+phase,
            '--output='+str(ART/(phase+'-%A_%a.out')),'--error='+str(ART/(phase+'-%A_%a.err'))]
    if phase == 'development': args += ['--array=0-3%2']
    tmp = ROOT/'var/tmp/modebench_level3_neutral_v1'
    args += ['--export=ALL,TMPDIR='+str(tmp)+',VLLM_USE_V1=0,VLLM_ATTENTION_BACKEND=XFORMERS,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4']
    cmd = [str(PYTHON),'-u','-B',str(SOURCE),'worker','--phase',phase,'--registration-sha256',expected]
    args += ['--wrap=mkdir -p '+shlex.quote(str(tmp))+'; exec '+shlex.join(cmd)]
    return args


def worker(phase, expected):
    plan = validate_plan(expected)
    tier = int(os.environ['SLURM_ARRAY_TASK_ID']) if phase == 'development' else None
    require(tier is None or tier in range(4), 'unexpected development index')
    claim = ART/(f'development_d{tier}_execution.json' if tier is not None else 'confirmation_execution.json')
    new(claim, {'job_id':os.environ['SLURM_JOB_ID'],'array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),
                'tier':tier,'registration_sha256':expected,'phase':phase,'created_at':now()})
    if phase == 'confirmation':
        require(read(ART/'recipe.json')['development_fit_pass'] is True, 'development did not pass')
        common.verify_pins(read(DATA/'identity.json')['files_sha256'])
        task_path = ART/'confirmation_tasks.json'
    else: task_path = ART/f'development_d{tier}_tasks.json'
    args = ['--model',plan['model']['path'],'--model-label','3b','--tasks-json',str(task_path)]
    if phase == 'confirmation': args += ['--confirm-eval']
    neutral.evaluator.main(args)


def validate_receipt(path, expected_task, plan):
    receipt = read(path)
    rows, source = neutral.evaluator.load_rows(expected_task)
    neutral.evaluator.validate_seed_receipt(receipt, rows)
    identity = receipt['identity']
    require(identity['source'] == source and identity['model'] == plan['model']
            and identity['seeds'] == expected_task['seeds'] and identity['split'] == expected_task['split'], 'receipt task/model mismatch')
    for row,result in zip(rows,receipt['prompt_results']):
        require(result['row_sha256'] == mixture.sha(row), 'receipt row mismatch')
        for draw in result['draws']:
            attempts = draw['attempts']; require(len(attempts)==8, 'missing attempts')
            require(all(a['verified'] == (a['canonical_key'] is not None) for a in attempts), 'inconsistent verification')
            count = sum(a['verified'] for a in attempts)
            values = {'pass1':count/8,'pass8':float(count>0),'distinct8':len({mixture.sha(a['canonical_key']) for a in attempts if a['verified']})}
            require(all(draw[m]==v for m,v in values.items()), 'draw metrics inconsistent')
        require(all(result[m]==statistics.mean(d[m] for d in result['draws']) for m in ('pass1','pass8','distinct8')), 'prompt metrics inconsistent')
    require(receipt['metrics'] == neutral.evaluator.summarize(receipt['prompt_results']), 'aggregate metric mismatch')
    return receipt, rows


def fit(expected):
    plan = validate_plan(expected); pools,scores={},{}
    pins = {}
    for tier in range(4):
        t = task(tier); receipt,rows = validate_receipt(t['output'], t, plan)
        pools[tier] = rows
        scores[tier] = {r['row_sha256']:{m:r[m] for m in TOLERANCES} for r in receipt['prompt_results']}
        pins[t['output']] = digest(t['output'])
    targets = {s:common.reference_histograms(DOMAIN)[s] for s in ('dev','eval')}
    result = fitter.choose_fixed_reference_mixture(DOMAIN,pools,scores,targets,plan['baseline'])
    result['registration_sha256'] = expected; result['receipts_sha256'] = pins
    new(ART/'recipe.json',result)
    return result


def finalize(expected):
    plan = validate_plan(expected); recipe = read(ART/'recipe.json')
    require(recipe['development_fit_pass'] is True and recipe['registration_sha256']==expected, 'passing registered recipe required')
    common.verify_pins(recipe['receipts_sha256'])
    require(not DATA.exists(), 'fresh final dataset required')
    blocked = {(d,tuple(cases)) for d,cases in read(ART/'excluded_identities.json')}
    for tier in range(4): blocked |= ids(mixture.read_jsonl(POOLS/f'difficulty_{tier}.jsonl'))
    from datasets import Dataset,DatasetDict
    DATA.mkdir()
    split_records = {}
    for split in ('dev','train','eval'):
        if split == 'dev':
            rows = [r['row'] for r in recipe['selected']]
            exclusion = blocked - ids(rows)
        else:
            exclusion = set(blocked); rows=[]
            cells = mixture.allocate_cells(common.reference_histograms(DOMAIN)[split],recipe['weights'],fitter.SELECTION_SEED)
            for tier in range(4):
                quota = Counter({k[0]:n[tier] for k,n in cells.items() if n[tier]})
                if quota:
                    fresh = generator.build_pool(DOMAIN,quota,blocked,GEN_SEEDS[split]+1000*tier,
                                                 'level3_neutral_v1_'+split,tier,1)
                    rows.extend(fresh); blocked |= ids(fresh)
            rows.sort(key=lambda r:mixture.sha([GEN_SEEDS[split],'split_order',mixture.sha(r)]))
        checks = materializer.verify_rows(DOMAIN,rows,materializer.reference_rows(DOMAIN,split),
                                          materializer.modes(materializer.reference_rows(DOMAIN,split)),exclusion)
        DatasetDict({materializer.SPLITS[split][1]:Dataset.from_list(rows)}).save_to_disk(str(DATA/DOMAIN/split))
        split_records[split] = {'rows':len(rows),'rows_sha256':mixture.sha(rows),'checks':checks}
        if split == 'eval': (ART/'eval.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    import shutil
    for domain in common.DOMAINS:
        if domain == DOMAIN: continue
        shutil.copytree(common.DATASET/domain,DATA/domain)
        require({str(p.relative_to(DATA/domain)):digest(p) for p in (DATA/domain).rglob('*') if p.is_file()}
                == {str(p.relative_to(common.DATASET/domain)):digest(p) for p in (common.DATASET/domain).rglob('*') if p.is_file()}, 'retained domain changed')
    files = {str(p):digest(p) for p in DATA.rglob('*') if p.is_file()}
    for p in (ART/'recipe.json',ART/'eval.jsonl'): files[str(p)]=digest(p)
    new(DATA/'identity.json',{'schema':'modebench_level3_neutral_splits_v1','registration_sha256':expected,
                             'recipe_sha256':digest(ART/'recipe.json'),'python_splits':split_records,
                             'other_domains_byte_identical_to':str(common.DATASET),'files_sha256':files,
                             'difficulty_admitted':False})
    new(ART/'confirmation_tasks.json',[task()])
    return {'status':'fresh_splits_materialized','dataset':str(DATA)}


def audit(expected):
    plan = validate_plan(expected)
    require(read(ART/'recipe.json')['development_fit_pass'] is True, 'passing development required')
    identity = read(DATA/'identity.json'); common.verify_pins(identity['files_sha256'])
    receipt,rows = validate_receipt(task()['output'],task(),plan)
    require(len(rows)==128 and mixture.sha(rows)==identity['python_splits']['eval']['rows_sha256'], 'confirmation is not the frozen full split')
    # Independently rerun the original external verifier on every confirmation attempt.
    from oat_drgrpo.math_grader import validated_modebench_outcome_key
    attempts=0
    for row,result in zip(rows,receipt['prompt_results']):
        for draw in result['draws']:
            for a in draw['attempts']:
                key=validated_modebench_outcome_key(a['text'],row['answer'])
                require(key==a['canonical_key'] and (key is not None)==a['verified'], 'external confirmation regrade mismatch')
                attempts+=1
    metrics={m:receipt['metrics'][m] for m in TOLERANCES}
    deltas={m:metrics[m]-plan['baseline'][m] for m in TOLERANCES}
    gates={m:abs(deltas[m])<=TOLERANCES[m] for m in TOLERANCES}
    import numpy as np
    rng=np.random.default_rng(6529700)
    uncertainty={}
    for m in TOLERANCES:
        values=np.array([r[m] for r in receipt['prompt_results']])
        boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)-plan['baseline'][m]
        uncertainty[m]={'delta_ci95_fixed_reference':np.quantile(boot,[.025,.975]).tolist(),
                        'unit':'prompt; four draws averaged within each prompt'}
    result={'status':'observed_approximate_match' if all(gates.values()) else 'outside_match_tolerance',
            'admitted':all(gates.values()),'registration_sha256':expected,'dataset':str(DATA),
            'dataset_identity_sha256':digest(DATA/'identity.json'),'confirmation_sha256':digest(task()['output']),
            'baseline':plan['baseline'],'neutral_level3':metrics,'deltas':deltas,'gates':gates,'tolerances':TOLERANCES,
            'uncertainty':uncertainty,'original_external_verifier_attempts_regraded':attempts,
            'created_at':now(),'historical_reference_prompt_retained':True,'statistical_equivalence_claimed':False}
    new(ART/'confirmation_report.json',result)
    if result['admitted']: new(ART/'admission.json',result)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','submit','worker','fit','finalize','audit','continue'])
    p.add_argument('--registration-sha256'); p.add_argument('--phase',choices=['development','confirmation'],default='development')
    a=p.parse_args()
    if a.action=='prepare': result=prepare()
    elif a.action=='worker': result=worker(a.phase,a.registration_sha256)
    else:
        ART.mkdir(parents=True,exist_ok=True)
        with (ART/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            validate_plan(a.registration_sha256)
            if a.action=='submit': result={'job_id':submit_once(a.phase,gpu_command(a.phase,a.registration_sha256))}
            elif a.action=='fit': result=fit(a.registration_sha256)
            elif a.action=='finalize': result=finalize(a.registration_sha256)
            elif a.action=='audit': result=audit(a.registration_sha256)
            else:
                if a.phase=='development':
                    recipe=fit(a.registration_sha256)
                    if recipe['development_fit_pass']:
                        finalize(a.registration_sha256)
                        result={'status':'confirmation_submitted','job_id':submit_once('confirmation',gpu_command('confirmation',a.registration_sha256))}
                    else: result={'status':'development_outside_tolerance','gates':recipe['gates']}
                else: result=audit(a.registration_sha256)
    print(json.dumps(result,sort_keys=True,default=str),flush=True)

if __name__=='__main__': main()
