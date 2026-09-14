#!/usr/bin/env python3
"""Authenticate and analyze the registered Level-1 fresh concentration panel.

This adapter reads committed response and batch receipts only. It never starts
an engine or reruns a verifier. The collector's CPU plan validation authenticates
code, rendered prompts, evaluation interfaces, and all disjoint RNG schedules.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))
import analyze_modebench_fresh_concentration as analysis
import evaluate_modebench_fresh_concentration as collector

METHODS = ('drgrpo', 'replay_drgrpo', 'maxrl', 'replay_maxrl')
PAIRS = tuple(('initial', method) for method in METHODS) + (
    ('drgrpo', 'replay_drgrpo'), ('maxrl', 'replay_maxrl'))
require = analysis.require


def seed_index(task):
    """Use an explicit registered pairing index, never parse a task-id string."""
    if task['checkpoint_stage'] == 'terminal':
        seed = task['training_seed']
    else:
        seed = task.get('eval_replica_id')
    require(type(seed) is int, 'Every initial replica needs an explicit integer eval_replica_id')
    return seed


def validate_panel_inventory(plan):
    tasks = plan['tasks']
    require(len(tasks) == 150, 'Full panel requires all 150 registered tasks')
    index = {}
    for task in tasks:
        method = 'initial' if task['checkpoint_stage'] == 'initial' else task['method']
        require(method == 'initial' or method in METHODS, 'Unexpected panel method')
        key = task['model_scale'], task['domain'], method, seed_index(task)
        require(key not in index, 'Duplicate model/domain/method/paired-seed task')
        index[key] = task
    scales = sorted({key[0] for key in index})
    domains = sorted({key[1] for key in index})
    require(len(scales) == 3 and set(domains) == {'graph_coloring', 'pantry_plan'},
            'Full panel requires exactly three scales and both Graph/Pantry')
    groups = {}
    for scale in scales:
        for domain in domains:
            seeds = sorted(k[3] for k in index if k[:3] == (scale, domain, 'initial'))
            require(len(seeds) == 5, 'Exactly five initial evaluation replicas required per model/domain')
            require({k for k in index if k[:2] == (scale, domain)} ==
                    {(scale, domain, method, seed) for method in ('initial', *METHODS) for seed in seeds},
                    'Missing registered five-seed method block')
            initial = [index[scale, domain, 'initial', seed] for seed in seeds]
            require(len({collector.sha(t['files']) for t in initial}) == 1,
                    'Initial evaluation replicas must use identical initial model weights/tokenizer files')
            require(all(t.get('eval_replica_id') is not None for t in initial) and
                    len({t['eval_replica_id'] for t in initial}) == 5, 'Initial replica IDs must be explicit and distinct')
            groups[scale, domain] = seeds
    return index, groups


def authenticate_task(plan_path, plan, task, rows):
    """Rebuild the collector's exact flat record sequence from bound batches."""
    directory = collector.resolve(plan['output_root']) / task['task_id']
    read = lambda name: json.loads((directory / name).read_text())
    run, runtime, result = (read(name) for name in ('run.json', 'runtime.json', 'result.json'))
    schedule = collector.task_schedule(plan, task, rows)
    identity = {'schema': collector.SCHEMA, 'plan_sha256': collector.file_sha(plan_path), 'task': task,
                'seed_policy': collector.POLICY, 'engine_contract': collector.ENGINE_CONTRACT,
                'seed_namespace': plan['seed_namespace'], 'draw_labels': plan['draw_labels'],
                'request_seeds': schedule, 'rows_sha256': collector.sha(rows), 'batch_size': plan['batch_size']}
    identity_sha = collector.sha(identity)
    require(run == {'identity_sha256': identity_sha, 'identity': identity}, 'Saved task identity differs from plan')
    runtime_sha = collector.runtime_fingerprint(runtime['runtime'])
    require(runtime['identity_sha256'] == identity_sha and runtime['runtime_sha256'] == runtime_sha,
            'Runtime identity or fingerprint changed')
    require(runtime['runtime']['engine_contract'] == collector.ENGINE_CONTRACT and
            runtime['runtime']['versions']['vllm'] == '0.8.4' and
            runtime['runtime']['environment']['VLLM_USE_V1'] == '0', 'Wrong recorded engine runtime')
    expected_backend = plan.get('scheduler', {}).get('attention_backend')
    require(isinstance(expected_backend, str) and runtime['runtime']['environment'].get('VLLM_ATTENTION_BACKEND') == expected_backend,
            'Recorded attention backend differs from frozen plan')
    stable_environment = lambda env: {k: v for k, v in env.items() if k != 'CUDA_VISIBLE_DEVICES'}
    attempts = sorted((directory / 'attempts').glob('*.json'))
    require(attempts, 'A completed task needs an engine-initialization receipt')
    for path in attempts:
        attempt = json.loads(path.read_text())
        require(attempt['identity_sha256'] == identity_sha and attempt['runtime_sha256'] == runtime_sha,
                'Engine initialization is not bound to this task/runtime')
        require(stable_environment(attempt.get('environment', {})) == stable_environment(runtime['runtime']['environment']),
                'Engine initialization stable environment differs from runtime receipt')
        if plan.get('created_at_utc'):
            require(datetime.fromisoformat(attempt['started_at']) >= datetime.fromisoformat(plan['created_at_utc']),
                    'Engine initialization predates frozen plan')
    flattened, batches = [], []
    for block, label in enumerate(plan['draw_labels']):
        for start in range(0, len(rows), plan['batch_size']):
            selected = rows[start:start+plan['batch_size']]
            name = f'batch_b{block:02d}_{start:04d}.json'
            batch = read(name)
            require(batch['identity_sha256'] == identity_sha and batch['runtime_sha256'] == runtime_sha
                    and batch['requests_sha256'] == collector.sha(batch['requests']), 'Committed batch binding changed')
            require(len(batch['requests']) == len(selected), 'Missing prompt request in batch')
            for index, (row, request) in enumerate(zip(selected, batch['requests']), start):
                expected = collector.request_identity(task, row, block, label, schedule[index][block])
                require(all(request.get(k) == v for k, v in expected.items()), 'Batch prompt/RNG metadata changed')
                require(isinstance(request.get('prompt_token_ids'), list)
                        and all(type(t) is int and t >= 0 for t in request['prompt_token_ids']), 'Invalid prompt token IDs')
                require(len(request['attempts']) == 8, 'Missing child output slots')
                for child, response in enumerate(request['attempts']):
                    require(response['child_index'] == child and response['draw_index'] == block*8+child
                            and response['child_sampling_seed'] == expected['request_seed']+child,
                            'Saved child index or RNG differs')
                    require(type(response['verified']) is bool and response['verified'] == (response['canonical_key'] is not None)
                            and (response['canonical_key'] is None or isinstance(response['canonical_key'], str)),
                            'Saved correctness/key consistency differs')
                    require(isinstance(response['text'], str) and isinstance(response['verifier_text'], str),
                            'Missing raw or decoded response')
                    tokens = response['token_ids']
                    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens)
                            and len(tokens) == response['token_count'] and len(tokens) <= task['sampling']['max_tokens'],
                            'Saved token budget/count differs')
                    allowed = task['sampling']['allowed_token_ids']
                    require(allowed is None or set(tokens) <= set(allowed), 'Saved output escaped registered token support')
                    if task['response_decoder'] == 'pantry_support_mask':
                        require(len(tokens) == task['sampling']['max_tokens'] and response['finish_reason'] == 'length',
                                'Saved Pantry output violates fixed horizon')
                    meta = {k: v for k, v in request.items() if k != 'attempts'}
                    flattened.append({'schema': collector.SCHEMA, 'domain': task['domain'], 'level': task['level'],
                                      'model_scale': task['model_scale'], 'method': task['method'],
                                      'training_seed': task['training_seed'], 'checkpoint_stage': task['checkpoint_stage'],
                                      'eval_replica_id': task.get('eval_replica_id'), **meta, **response})
            batches.append({'name': name, 'sha256': collector.file_sha(directory/name), 'requests': len(selected)})
    require(set(p.name for p in directory.glob('batch_b*.json')) == {b['name'] for b in batches},
            'Unexpected committed batch files')
    require(len(flattened) == len({(r['prompt_id'], r['draw_index']) for r in flattened}) == 128*64,
            'Missing or duplicated full-panel response slots')
    require(len({r['child_sampling_seed'] for r in flattened}) == 128*64, 'Repeated child stream within task')
    require(collector.read_jsonl(directory/'responses.jsonl') == flattened,
            'Saved flat responses differ from the authenticated batches')
    expected_result = {'schema': collector.SCHEMA, 'status': 'complete', 'identity_sha256': identity_sha,
                       'runtime_sha256': runtime_sha, 'task_id': task['task_id'], 'prompts': 128,
                       'draws_per_prompt': 64, 'draws': len(flattened), 'batches': batches,
                       'responses_path': str(directory/'responses.jsonl'),
                       'responses_sha256': collector.file_sha(directory/'responses.jsonl'),
                       'interpretation': 'fresh disjoint request/child RNG streams; initial replicas are evaluation replicas; no old draws pooled'}
    require(result == expected_result, 'Completed task result receipt changed')
    sources = [analysis.file_binding(directory/name) for name in ('run.json','runtime.json','result.json','responses.jsonl')]
    sources += [analysis.file_binding(p) for p in attempts]
    by_prompt = {row['prompt_id']: [] for row in rows}
    for response in flattened:
        by_prompt[response['prompt_id']].append(response)
    summaries = []
    for row in rows:
        responses = sorted(by_prompt[row['prompt_id']], key=lambda x: x['draw_index'])
        summaries.append(analysis.prompt_summary([r['canonical_key'] if r['verified'] else None for r in responses],
                                                prompt_id=row['prompt_id'], row_sha256=collector.sha(row)))
    return {'task': task, 'prompts': summaries, 'sources': sources, 'runtime': runtime['runtime'],
            'draws': len(flattened)}


def audit_panel(plan_path, *, templates=None):
    """Report every planned task; fail closed on changed data, retain incomplete tasks."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    rows_by_task = collector.validate_plan(plan, templates=templates)
    index, groups = validate_panel_inventory(plan)
    panels, statuses = {}, []
    output = collector.resolve(plan['output_root'])
    for task in plan['tasks']:
        task_id = task['task_id']; directory = output/task_id
        if not (directory/'result.json').exists():
            statuses.append({'task_id': task_id, 'status': 'partial' if directory.exists() else 'not_started',
                             'committed_batch_files': len(list(directory.glob('batch_b*.json')))})
            continue
        try:
            panel = authenticate_task(plan_path, plan, task, rows_by_task[task_id])
        except (ValueError, KeyError, TypeError, OSError) as error:
            statuses.append({'task_id': task_id, 'status': 'invalid', 'error': str(error)})
            continue
        panels[task_id] = panel
        statuses.append({'task_id': task_id, 'status': 'authenticated', 'draws': panel['draws']})
    unexpected = sorted(p.name for p in output.iterdir() if p.is_dir() and
                        ((p/'run.json').exists() or (p/'result.json').exists()) and p.name not in rows_by_task) if output.exists() else []
    complete = len(panels) == len(plan['tasks']) and not unexpected
    audit = {'schema': analysis.SCHEMA+'-panel-audit', 'status': 'complete' if complete else 'incomplete',
             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'plan_source': analysis.file_binding(plan_path),
             'expected_tasks': len(plan['tasks']), 'authenticated_tasks': len(panels),
             'expected_response_slots': len(plan['tasks'])*128*64,
             'authenticated_response_slots': sum(p['draws'] for p in panels.values()),
             'unexpected_task_directories': unexpected, 'tasks': statuses,
             'generation_calls': 0, 'verifier_calls': 0,
             'grading_scope': 'Cached collector grades authenticated to frozen code and committed batches; no independent regrading performed by this analyzer.'}
    return audit, plan, index, groups, panels


def build_report(plan_path):
    audit, plan, index, groups, panels = audit_panel(plan_path)
    require(audit['status'] == 'complete', 'Registered panel is incomplete or invalid; run --audit-only for every task status')
    contrasts = []
    for (scale, domain), seeds in sorted(groups.items()):
        for left_method, right_method in PAIRS:
            records = []
            for seed in seeds:
                left = panels[index[scale,domain,left_method,seed]['task_id']]
                right = panels[index[scale,domain,right_method,seed]['task_id']]
                expected = [p['prompt_id'] for p in left['prompts']]
                records.append(analysis.seed_contrast(left['prompts'],right['prompts'],training_seed=seed,
                                                       expected_prompt_ids=expected))
            summary = analysis.aggregate_seed_contrasts(records, expected_seeds=seeds)
            if left_method == 'initial':
                summary.update(independent_initial_checkpoints=1, initial_sampling_replicas=5,
                               initial_weights_shared=True, shared_initial_output_pool=False,
                               uncertainty_scope='paired trained-seed and initial Monte Carlo replica variation, conditional on one fixed initial checkpoint and fixed prompts')
            contrasts.append({'grading':'strict','wording':'original','level':1,'model_scale':scale,'domain':domain,
                              'contrast':right_method+'_minus_'+left_method,'left_method':left_method,'right_method':right_method,
                              'summary':summary,'seeds':records})
    return {'schema':analysis.SCHEMA,'status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'generation_calls':0,'verifier_calls':0,'old_artifacts_modified':False,
            'scope':{'panel':'registered_Level1_fresh','domains':['graph_coloring','pantry_plan'],
                     'model_scales':sorted({k[0] for k in groups}),'prompts_per_cell':128,'fresh_draws_per_prompt':64,
                     'collection_tasks':150,'trained_checkpoint_evaluations':120,'initial_sampling_tasks':30,
                     'initial_model_domain_references':6,'initial_weight_sets':3,'unique_response_slots':1228800,'contrast_cells':len(contrasts)},
            'completeness_audit':audit,'checkpoint_sources':[{'task_id':k,'sources':v['sources'],'runtime':v['runtime']} for k,v in panels.items()],
            'analysis_sources':[analysis.file_binding(__file__),analysis.file_binding(analysis.__file__)],
            'contrasts':contrasts,'population_reconciliation':[analysis.population_reconciliation(c) for c in contrasts],
            'limitations':['The protocol was fixed before new collection, informed by earlier paper outcomes and held-out tasks.',
                           'Intervals are nominal pointwise paired Student-t intervals on five defined seeds, conditional on 128 fixed prompts; no unseen-task uncertainty or multiplicity adjustment.',
                           'Initial replicas vary generation randomness for the same initial weights and are shared across method contrasts within a pairing index.',
                           'Joint R>=2 eligibility changes across comparisons and seeds; complete and selected populations are retained separately.',
                           'Collision is conditional concentration and does not establish extinction, full support recovery, or a diversity-caused accuracy gain.',
                           'Saved grader outputs are authenticated to committed batches and frozen collection/verifier code; this offline adapter performs no independent regrading.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.audit_only:
        audit,*_=audit_panel(args.plan)
        print(json.dumps(audit,indent=2,sort_keys=True,allow_nan=False))
        return 0 if audit['status']=='complete' else 2
    require(args.output is not None,'--output is required for immutable publication')
    print(analysis.write_artifacts(build_report(args.plan),args.output))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
