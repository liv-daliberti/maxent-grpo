#!/usr/bin/env python3
"""Read-only source integrity audit. Writes /tmp/paper_initial_auc_integrity.json."""
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import ujson

ROOT=Path('/n/fs/similarity/maxent-grpo')
GRID=set(range(0,3073,192))
LEDGERS=[('qwen05b','e78_verified_replay_only_05b_jobs.json'),('falcon1b','e79_falcon1b_aligned_verified_replay_jobs.json'),('qwen3b','e80r1_qwen3b_aligned_verified_replay_jobs.json')]


def identity(row):
    prompts=row.get('prompts',[])
    projection=[{key:p.get(key) for key in ('answer_mode_count','option_ids','prompt','prompt_index','reference')} for p in prompts]
    requests={'seed':row.get('seed'),'prompts':[{key:p.get(key) for key in ('option_ids','prompt_index','request_seeds_by_option')} for p in prompts]}
    def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(prompt_sha256=digest(projection),request_sha256=digest(requests))


def audit_run(item):
    scope,scale,run=item
    paths=sorted(Path(run['run_dir']).glob('debug_job*/eval_mode_coverage_draws.jsonl'))
    records=collections.defaultdict(list)
    sources=[]
    malformed=0
    for path in paths:
        sha=hashlib.sha256()
        with path.open('rb') as f:
            for lineno,raw in enumerate(f,1):
                sha.update(raw)
                try: row=ujson.loads(raw)
                except (ValueError,UnicodeDecodeError):
                    malformed+=1; continue
                if row.get('evaluation_kind')!='fixed_seed_sampled_k_neutral': continue
                step=row.get('step')
                if step not in (GRID if scope in ('core_auc','rlep_pantry') else {0}): continue
                key=(step,row.get('draw_index'))
                records[key].append({'source':str(path),'line':lineno,'metrics':row.get('metrics'),**identity(row)})
        sources.append({'path':str(path),'sha256':sha.hexdigest()})
    duplicates=[]
    for key,rows in sorted(records.items()):
        if len(rows)<2: continue
        metrics_conflict=any(r['metrics']!=rows[0]['metrics'] for r in rows[1:])
        prompt_conflict=any(r['prompt_sha256']!=rows[0]['prompt_sha256'] for r in rows[1:])
        request_conflict=any(r['request_sha256']!=rows[0]['request_sha256'] for r in rows[1:])
        duplicates.append({'step':key[0],'draw':key[1],'count':len(rows),'metrics_conflict':metrics_conflict,'prompt_conflict':prompt_conflict,'request_conflict':request_conflict,'rows':rows})
    expected={(s,d) for s in (GRID if scope in ('core_auc','rlep_pantry') else {0}) for d in range(4)}
    missing=sorted(expected-set(records))
    return {'scope':scope,'scale':scale,'domain':run['domain'],'arm':run['arm'],'seed':run['seed'],'job_id':run['job_id'],'run_dir':run['run_dir'],'sources':sources,'keys':len(records),'duplicates':duplicates,'missing_keys':missing,'malformed_json_rows':malformed}


if __name__=='__main__':
    jobs=[]
    for scale,name in LEDGERS:
        ledger=json.loads((ROOT/'var/artifacts'/name).read_text())
        for run in ledger['runs']:
            if run.get('domain') in ('graph_coloring','countdown','python_factors','mathir','pantry_plan') and run.get('arm') in ('control','replay'):
                jobs.append(('core_auc' if scale=='qwen05b' else 'core_initial',scale,run))
    rlep=json.loads((ROOT/'var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json').read_text())
    for run in rlep['runs']:
        if run.get('domain')=='pantry_plan': jobs.append(('rlep_pantry','qwen05b',run))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(audit_run,jobs))
    conflict_runs=[r for r in results if any(d['metrics_conflict'] or d['prompt_conflict'] or d['request_conflict'] for d in r['duplicates'])]
    summary={'core_runs':sum(r['scope'].startswith('core') for r in results),'qwen_all_checkpoint_runs':sum(r['scope']=='core_auc' for r in results),'rlep_pantry_runs':sum(r['scope']=='rlep_pantry' for r in results),'any_duplicate_runs':sum(bool(r['duplicates']) for r in results),'conflicting_runs':len(conflict_runs),'missing_grid_runs':sum(bool(r['missing_keys']) for r in results),'malformed_json_rows':sum(r['malformed_json_rows'] for r in results)}
    payload={'summary':summary,'runs':results}
    Path('/tmp/paper_initial_auc_integrity.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(summary,indent=2))
    for r in conflict_runs:
        print(r['scope'],r['scale'],r['domain'],r['arm'],r['seed'],r['job_id'])
        for d in r['duplicates']:
            if d['metrics_conflict'] or d['prompt_conflict'] or d['request_conflict']:
                print('  ',d['step'],d['draw'],'n=',d['count'],'metrics=',d['metrics_conflict'],'prompt=',d['prompt_conflict'],'request=',d['request_conflict'],'lines=',[(Path(row['source']).parent.name,row['line']) for row in d['rows']])
    print('Saved /tmp/paper_initial_auc_integrity.json')
