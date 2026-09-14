#!/usr/bin/env python3
"""Bind the complete Graph/Pantry fresh-concentration inference plan."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/'ops',ROOT/'src'):
    sys.path.insert(0,str(path))
import evaluate_modebench_fresh_concentration as collector
from restore_modebench_fresh_concentration import target_path, atomic_new, digest

BASE=ROOT/'artifacts/modebench_fresh_concentration_20260912'
SCALES=('qwen05b','falcon1b','qwen3b')
DOMAINS=('graph_coloring','pantry_plan')
METHODS=('drgrpo','replay_drgrpo','maxrl','replay_maxrl')
SEEDS={'qwen05b':list(range(43,48)),'falcon1b':list(range(55,60)),'qwen3b':list(range(70,75))}


def bound(path):
    path=Path(path)
    return {'path':str(path),'sha256':digest(path)}


def jsonl_new(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n' for r in records)
    if path.exists():
        if path.read_text()!=raw:raise ValueError(f'prepared prompt records differ: {path}')
    else:
        with path.open('x') as f:f.write(raw)


def validate_coverage(tasks):
    actual={(t['model_scale'],t['domain'],t['method'],
             t.get('eval_replica_id') if t['method']=='initial' else t['training_seed']) for t in tasks}
    expected={(scale,domain,method,seed) for scale in SCALES for domain in DOMAINS
              for method in ('initial',)+METHODS for seed in SEEDS[scale]}
    if len(tasks)!=150 or actual!=expected:raise ValueError('full150taskmatrix not represented exactly')
    for t in tasks:
        if (t['method']=='initial')!=(t['checkpoint_stage']=='initial'):
            raise ValueError('initial references and trained endpoints mislabeled')
        if t['method']=='initial' and (t['training_seed'] is not None or t.get('eval_replica_id') not in SEEDS[t['model_scale']]):
            raise ValueError('initial references require a sampling replica and no training seed')
        if t['method']!='initial' and t.get('eval_replica_id') is not None:
            raise ValueError('trained endpoints cannot carry an initial sampling replica')


def prepare(base=BASE, name='plan.json'):
    base=Path(base);plan_path=base/name
    if plan_path.exists():
        plan=json.loads(plan_path.read_text());validate_coverage(plan['tasks']);collector.validate_plan(plan)
        return plan_path,plan
    inventory_path=base/'checkpoint_inventory.json';inventory=json.loads(inventory_path.read_text())
    registration_path=base/'analysis_registration.json';registration=json.loads(registration_path.read_text())
    if digest(base/'ANALYSIS_PLAN.md')!=registration['sha256']:raise ValueError('registered protocol changed')
    local_hashes_path=base/'local_checkpoint_weight_hashes.json'
    local_hashes=json.loads(local_hashes_path.read_text())
    if local_hashes['inventory_sha256']!=digest(inventory_path):raise ValueError('local weight hash inventory differs')
    hashes={(f['cell_id'],f['name']):f['sha256'] for f in local_hashes['files']}
    records={(r['model_scale'],r['domain'],r['method'],r['training_seed']):r for r in inventory['records']}
    expected={(s,d,m,k) for s in SCALES for d in DOMAINS for m in METHODS for k in SEEDS[s]}
    if set(records)!=expected or len(inventory['records'])!=120:raise ValueError('inventory lost registered checkpoints')
    initial={m['model_scale']:m for m in inventory['initial_models']}
    if len(inventory['initial_models'])!=3 or set(initial)!=set(SCALES):
        raise ValueError('initial model inventory must contain exactly three unique scales')
    inputs={str(p):digest(p) for p in [inventory_path,registration_path,base/'ANALYSIS_PLAN.md',local_hashes_path,base/'prompt_population_audit.json']}
    prompt_sources={g['domain']:g['raw_prompts'] for g in inventory['prompt_groups']}
    templates=collector.default_templates()
    configs={};prompts={}
    interface=ROOT/'var/seed_paper_eval/paper310/lib/python3.10/site-packages/oat/interface.py'
    for scale in SCALES:
        for domain in DOMAINS:
            r=records[(scale,domain,'drgrpo',SEEDS[scale][0])]
            for method in METHODS:
                for seed in SEEDS[scale]:
                    other=records[scale,domain,method,seed]
                    for field in ('source_eval_config','prompt_template','syntax_profile','response_decoder','prompt_encoding','effective_sampling'):
                        if other[field]!=r[field]:
                            raise ValueError(f'evaluation configuration differs within {scale}/{domain}: {other["cell_id"]} {field}')
            original=Path(r['source_eval_config']['path'])
            if digest(original)!=r['source_eval_config']['sha256']:raise ValueError('recorded evaluation source changed')
            source=json.loads(original.read_text())
            normalized={'schema':'modebench-fresh-concentration-normalized-evaluation-v1',
                'prompt_template':r['prompt_template'],'syntax_profile':r['syntax_profile'],
                'response_decoder':r['response_decoder'],'prompt_encoding':r['prompt_encoding'],
                'sampling':source['effective_sampling'],'dtype':'bfloat16',
                'source_evaluation_config':bound(original),
                'precision_evidence':{'source':bound(interface),'lines':[78,83],
                    'interpretation':'OAT explicitly selects bfloat16 for actor vLLM; pinned initial configs agree. Historical dependency identity is unavailable for some runs.'},
                'max_model_len':source['max_model_len'],
                'replication_scope':'Same saved policies, prompts, verifier semantics and documented decoding constraints; newly authenticated runtime, not a claim of historical runtime identity.'}
            config_path=base/'prepared'/f'{scale}_{domain}_evaluation.json'
            atomic_new(config_path,normalized);configs[scale,domain]=(normalized,bound(config_path))
            inputs[str(config_path)]=digest(config_path);inputs[str(original)]=digest(original)
            inputs[str(interface)]=digest(interface)
            rawpath=Path(prompt_sources[domain]['path'])
            if digest(rawpath)!=prompt_sources[domain]['sha256']:raise ValueError('raw prompt source changed')
            inputs[str(rawpath)]=digest(rawpath)
            rawrows=[json.loads(l) for l in rawpath.read_text().splitlines() if l.strip()]
            rendered=[]
            for index,row in enumerate(rawrows):
                answer=row['answer']
                ref=json.loads(answer) if isinstance(answer,str) else answer
                problem=row['problem']
                rendered.append({'prompt_id':collector.sha({'prompt':problem,'reference':ref}),
                    'row_index':row.get('row_index',row.get('prompt_index',index)),
                    'problem':problem,'answer':json.dumps(ref,sort_keys=True,ensure_ascii=False,separators=(',',':')),
                    'rendered_prompt':templates[r['prompt_template']](problem)})
            prompt_path=base/'prepared'/f'{scale}_{domain}_prompts.jsonl'
            jsonl_new(prompt_path,rendered);prompts[scale,domain]=str(prompt_path)
            inputs[str(prompt_path)]=digest(prompt_path)
    tasks=[]
    for scale in SCALES:
        for seed in SEEDS[scale]:
            for method in ('initial',)+METHODS:
                for domain in DOMAINS:
                    config,binding=configs[scale,domain]
                    if method=='initial':
                        model=initial[scale];model_path=model['model_path'];files=model['files']
                        origin={'initial_revision':model['revision'],'shared_initial_weights':True,
                                'eval_replica_id':seed,'training_replication':False}
                    else:
                        model=records[scale,domain,method,seed];model_path=str(target_path(model));files=model['files']
                        origin={'cell_id':model['cell_id'],'completion_receipt':model['completion_receipt'],
                                'original_model_path':model['model_path'],'logged_evaluation_step':model['logged_evaluation_step'],
                                'terminal_export_step':model.get('terminal_export_step'),
                                'archive_manifest':model.get('archive_manifest')}
                    normalized_files=[]
                    for entry in files:
                        h=entry.get('sha256')
                        if h is None:h=hashes[(model['cell_id'],entry['name'])]
                        normalized_files.append({'name':entry['name'],'bytes':entry['bytes'],'sha256':h})
                    task={'task_id':f'level1__{scale}__{domain}__{method}__{seed}',
                          'domain':domain,'level':1,'model_scale':scale,'method':method,'training_seed':None if method=='initial' else seed,
                          'checkpoint_stage':'initial' if method=='initial' else 'terminal',
                          'model_path':model_path,'files':normalized_files,'prompts_path':prompts[scale,domain],
                          'source_eval_config':binding,'source_identity':origin,
                          **{k:config[k] for k in ['prompt_template','syntax_profile','response_decoder','prompt_encoding','sampling']},
                          'engine':{'dtype':'bfloat16','max_model_len':config['max_model_len'],'tensor_parallel_size':1,
                                    'gpu_memory_utilization':0.65,'swap_space':4,'enable_prefix_caching':True,'enforce_eager':True}}
                    if method=='initial':task['eval_replica_id']=seed
                    tasks.append(task)
    validate_coverage(tasks)
    plan={'schema':collector.SCHEMA,'campaign_id':'modebench_fresh_concentration_20260912',
          'created_at_utc':datetime.now(timezone.utc).isoformat(),'output_root':str(base/'results'),
          'draws_per_prompt':64,'prompts_per_task':128,'draw_labels':list(range(91264000,91264008)),
          'seed_namespace':'modebench-fresh-concentration-20260912-v1-independent-task-prompt-block',
          'batch_size':16,'input_sha256':inputs,'code_sha256':collector.code_identity(),
          'tasks':tasks,'expected_response_slots':150*128*64,
          'scheduler':{'partition':'lowprio','account':'mltheory','gres':'gpu:a5000:1','cpus':6,
                       'memory':'40G','time':'02:00:00','maximum_concurrent_owned_gpus':8,
                       'attention_backend':'XFORMERS','python':str(ROOT/'var/seed_paper_eval/paper310/bin/python')},
          'initial_reference_interpretation':'Five independent sampling replicas per model/domain; one shared initial set of weights, not five trained-model replications.',
          'analysis_plan':bound(base/'ANALYSIS_PLAN.md')}
    rows=collector.validate_plan(plan)
    all_seeds=[seed for task in tasks for prompt in collector.task_schedule(plan,task,rows[task['task_id']]) for seed in prompt]
    if len(set(all_seeds))*8!=plan['expected_response_slots']:raise ValueError('full child RNG census differs')
    plan['rng_audit']={'request_blocks':len(all_seeds),'unique_child_seeds':len(set(all_seeds))*8,
                       'minimum_child_seed':min(all_seeds),'maximum_child_seed':max(all_seeds)+7}
    atomic_new(plan_path,plan)
    return plan_path,plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--name',default='plan.json');args=parser.parse_args()
    path,plan=prepare(args.base,args.name)
    print(json.dumps({'status':'prepared','plan':str(path),'sha256':digest(path),'tasks':len(plan['tasks']),
                      'response_slots':plan['expected_response_slots'],'rng_audit':plan['rng_audit']},sort_keys=True))
