"""Read-only audit of reported comparator inputs outside the core audit."""
import concurrent.futures
import hashlib
import json
import re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import ujson

ROOT=Path('/n/fs/similarity/maxent-grpo')
OUT=ROOT/'paper/audits'
loaded={}

def load(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    raw=path.read_bytes()
    loaded[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def identity(row):
    prompts=row.get('prompts',[])
    projection=[{key:p.get(key)for key in ('answer_mode_count','option_ids','prompt','prompt_index','reference')}for p in prompts]
    requests={'seed':row.get('seed'),'prompts':[{key:p.get(key)for key in ('option_ids','prompt_index','request_seeds_by_option')}for p in prompts]}
    return {'prompt_sha256':digest(projection),'request_sha256':digest(requests)}

candidates={}
def add(run,scope,scale,steps,authority):
    key=(str(run['run_dir']),int(run['job_id']))
    if key not in candidates:
        candidates[key]={'run':run,'scope':scope,'scale':scale,'steps':set(steps),'authority':[authority]}
    else:
        candidates[key]['steps'].update(steps)
        candidates[key]['authority'].append(authority)

precheck=[('qwen05b','e95_plain_grpo_Qwen25-05B_jobs.json'),('falcon1b','e95_plain_grpo_Falcon3-1B_jobs.json'),('qwen3b','e95_plain_grpo_Qwen25-3B_jobs.json'),('qwen3b','e114_plain_grpo_qwen3b_extension_jobs.json')]
for scale,name in precheck:
    path='var/artifacts/'+name
    for run in load(path)['runs']:add(run,'plain_grpo',scale,[0,3072],path)

figure=load('paper/figures/direct_comparator_endpoint_effects.json')
ledger_index={}
for path in figure['input_sha256']:
    if path.startswith('var/artifacts/') and path.endswith('.json'):
        for run in load(path).get('runs',[]):ledger_index[int(run['job_id'])]=(path,run)
for cell in figure['cells']:
    for method,data in cell['methods'].items():
        for seed,row in data.get('per_seed',{}).items():
            path,run=ledger_index[int(row['comparator_job_id'])]
            add(run,'plain_grpo'if method=='grpo'else method,cell['scale'],[0,3072]if method=='grpo'else[3072],path)

for scale,name in [('qwen05b','e83_semantic_maxent_without_replay_05b_jobs.json'),('falcon1b','e86_falcon_semantic_maxent_without_replay_jobs.json')]:
    path='var/artifacts/'+name
    for run in load(path)['runs']:
        if scale=='qwen05b'and run['domain']=='pantry_plan':continue
        add(run,'fixed_semantic',scale,[3072],path)
path='var/artifacts/e85_pantry_semantic_repair_jobs.json'
for run in load(path)['runs']:
    if run.get('parent')=='e83':add(run,'fixed_semantic','qwen05b',[3072],path)

previous=load('paper/audits/primary_terminal_integrity_20260904.json')
covered={str(ROOT/r['run_dir'])for r in previous['runs']}
e112=load('paper/results/e112r1_two_scale_exploratory_results.json')
for scale,domains in e112['families'].items():
    for domain,data in domains.items():
        for seed,pair in data['per_seed'].items():
            for arm in ['treatment','replay']:
                row=pair[arm]
                if row['run_dir']in covered:continue
                run={'run_dir':row['run_dir'],'job_id':row['registered_job_id'],'seed':int(seed),'domain':domain,'arm':arm,'frozen_source_sha256':row['source_sha256']}
                add(run,'e112_'+arm,scale,[3072],'paper/results/e112r1_two_scale_exploratory_results.json')

print('MEMBERSHIP',json.dumps(Counter(c['scope']for c in candidates.values())), 'unique',len(candidates),flush=True)
pattern=re.compile(rb'"step"\s*:\s*(0|3072)(?:[,}\s])')
def audit(item):
    run=item['run'];run_dir=Path(run['run_dir']);job=int(run['job_id']);current=run_dir/f'debug_job{job}/eval_mode_coverage_draws.jsonl'
    replaced={int(x)for x in run.get('replaced_job_ids',[])}
    if run.get('replaces_job_id')is not None:replaced.add(int(run['replaces_job_id']))
    source_binding={'registered_job_id':job,'policy':'current registered job, or exact frozen E112 registered job; no outcome-based retry selection','superseded_job_ids':sorted(replaced),'recovery_history':[],'excluded_historical_paths':[],'other_unbound_historical_paths':[]}
    issues=[]
    for h in run.get('repair_history',[]):
        p=Path(h['protocol']);p=p if p.is_absolute()else ROOT/p
        source_binding['recovery_history'].append({'old_job_id':h.get('old_job_id'),'new_job_id':h.get('new_job_id'),'protocol':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()if p.exists()else None})
        if not p.exists():issues.append('missing registered recovery amendment')
    for path in sorted(run_dir.glob('debug_job*/eval_mode_coverage_draws.jsonl')):
        if path==current:continue
        old=path.parent.name.removeprefix('debug_job')
        if old.isdigit()and int(old)in replaced:source_binding['excluded_historical_paths'].append(str(path.relative_to(ROOT)))
        elif item['scope'].startswith('e112_'):
            source_binding['other_unbound_historical_paths'].append(str(path.relative_to(ROOT)))
        else:
            source_binding['other_unbound_historical_paths'].append(str(path.relative_to(ROOT)));issues.append('historical source absent from explicit replacement list')
    result={'scope':item['scope'],'scale':item['scale'],'domain':run['domain'],'seed':run['seed'],'arm':run.get('arm'),'job_id':job,'run_dir':str(run_dir.relative_to(ROOT)),'steps':sorted(item['steps']),'authority':item['authority'],'source_binding':source_binding}
    if not current.exists():return result|{'issues':issues+['missing registered source'],'source':str(current),'checkpoints':{},'duplicates':[]}
    records=defaultdict(list);sha=hashlib.sha256();size=0;malformed=0
    with current.open('rb')as handle:
        for line_number,raw in enumerate(handle,1):
            sha.update(raw);size+=len(raw)
            if not pattern.search(raw):continue
            try:row=ujson.loads(raw)
            except (ValueError,UnicodeDecodeError):malformed+=1;continue
            if row.get('step')not in item['steps']or row.get('evaluation_kind')!='fixed_seed_sampled_k_neutral':continue
            draw=row.get('draw_index')
            if draw not in range(4):issues.append('invalid draw index');continue
            records[(row['step'],draw)].append({'line':line_number,'raw_sha256':hashlib.sha256(raw).hexdigest(),'metrics':row.get('metrics'),**identity(row)})
    duplicates=[]
    for (step,draw),rows in sorted(records.items()):
        if len(rows)>1:
            conflicts={name:any(r[name]!=rows[0][name]for r in rows[1:])for name in ['metrics','prompt_sha256','request_sha256']}
            duplicates.append({'step':step,'draw':draw,'count':len(rows),'conflicts':conflicts,'rows':rows})
    checkpoints={}
    for step in sorted(item['steps']):
        missing=[d for d in range(4)if (step,d)not in records]
        conflicting=any(x['step']==step and any(x['conflicts'].values())for x in duplicates)
        checkpoints[str(step)]={'missing_draws':missing,'conflicting':conflicting,'admissible':not missing and not conflicting,'draw_rows':{str(d):records.get((step,d),[])for d in range(4)}}
        if missing:issues.append(f'missing step {step} draws {missing}')
    source={'path':str(current.relative_to(ROOT)),'sha256':sha.hexdigest(),'bytes_read':size}
    frozen=run.get('frozen_source_sha256',{}).get(str(current))
    if frozen is not None:
        source['frozen_sha256']=frozen;source['matches_frozen_hash']=frozen==source['sha256']
        if frozen!=source['sha256']:issues.append('source hash differs from frozen E112 source')
    if malformed:issues.append('malformed matching JSON rows')
    return result|{'source':source,'issues':issues,'checkpoints':checkpoints,'duplicates':duplicates,'malformed_matching_json_rows':malformed}

results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:
    for future in concurrent.futures.as_completed([pool.submit(audit,item)for item in candidates.values()]):
        result=future.result();results.append(result)
        conflict=[x for x in result['duplicates']if any(x['conflicts'].values())]
        if conflict or result['issues']:
            print('FINDING',json.dumps({k:result[k]for k in ['scope','scale','domain','seed','job_id','issues']}|{'conflicts':conflict}),flush=True)
        if len(results)%50==0:print('SCANNED',len(results),flush=True)
results.sort(key=lambda r:(r['scope'],r['scale'],r['domain'],r['seed']))
summary={'run_count':len(results),'scope_counts':dict(Counter(r['scope']for r in results)),'checkpoint_count':sum(len(r['checkpoints'])for r in results),'admissible_checkpoints':sum(v['admissible']for r in results for v in r['checkpoints'].values()),'runs_with_conflicting_duplicate_keys':sum(any(any(x['conflicts'].values())for x in r['duplicates'])for r in results),'runs_with_identical_duplicates_only':sum(bool(r['duplicates'])and not any(any(x['conflicts'].values())for x in r['duplicates'])for r in results),'runs_with_source_or_missing_data_issues':sum(bool(r['issues'])for r in results),'bytes_hashed':sum(r.get('source',{}).get('bytes_read',0)for r in results if isinstance(r.get('source'),dict))}
report={'schema':'reported-comparator-source-integrity-v1','audited_at_utc':datetime.now(timezone.utc).isoformat(),'selection':'reported comparator figure seeds, all 75 plain-GRPO baseline seeds, fixed semantic-only coverage, and reported E112 sources outside previous 445; no new outcome or checkpoint selection','scope_note':'Plain GRPO audited at steps 0 and3072. Other selected comparators audited at3072 only. Fixed semantic replay combinations not displayed or used by the current paper are outside this supplemental scope.','input_sha256':loaded,'summary':summary,'runs':results}
OUT.mkdir(parents=True,exist_ok=True)
path=OUT/'comparator_source_integrity_20260904.json';path.write_text(json.dumps(report,indent=2)+'\n')
print('DONE',json.dumps(summary),flush=True)
