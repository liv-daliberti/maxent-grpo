"""Read-only scheduler/receipt census, deduplicated by registered science cell."""
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
started=datetime.now(timezone.utc).isoformat()
LEDGERS={'e118':ROOT/'var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json','e119':ROOT/'var/artifacts/e119_level2_qwen05b_factorial_jobs.json'}
domains={'graph':'graph_coloring','countd':'countdown','python':'python_factors','mathir':'mathir','pantry':'pantry_plan'}
labels={'graph_coloring':'Graph','countdown':'Countdown','python_factors':'Python','mathir':'MathIR','pantry_plan':'Pantry'}
arms={'m':'maxrl','rm':'replay_maxrl','d':'drgrpo','rd':'replay_drgrpo'}
registered={};sources={}
for campaign,path in LEDGERS.items():
 data=path.read_bytes();sources[str(path.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
 for r in json.loads(data)['runs']:
  key=(campaign,r.get('scale','qwen05b'),r['domain'],r['arm'],r['seed'])
  assert key not in registered
  registered[key]=r
queue=subprocess.run(['squeue','-h','-u','od2961','-o','%i|%T|%j|%N|%R'],check=True,capture_output=True,text=True).stdout
(OUT/'job_status.squeue.txt').write_text(queue)
active=defaultdict(list);jobs=[]
for line in queue.splitlines():
 job,state,name,node,reason=line.split('|',4)
 if not (name.startswith('e118') or name.startswith('e119')):continue
 m=re.fullmatch(r'(e118q3|e119)-(graph|countd|python|mathir|pantry)-(m|rm|d|rd)-s(\d+)',name)
 assert m,(job,name)
 prefix,domain,arm,seed=m.groups()
 key=('e118' if prefix=='e118q3' else 'e119','qwen3b' if prefix=='e118q3' else 'qwen05b',domains[domain],arms[arm],int(seed))
 assert key in registered,key
 row={'job_id':int(job),'state':state,'name':name,'node':node,'reason':reason,'registered_cell':key,'expected_run_dir':registered[key]['run_dir']}
 active[key].append(row);jobs.append(row)
def verify(job):
 record=subprocess.run(['scontrol','show','job','-o',str(job['job_id'])],capture_output=True,text=True,check=True).stdout
 match=re.search(r'(?:^|[\s,])SAVE_PATH=([^,\s]+)',record)
 assert match and Path(match.group(1)).resolve()==Path(job['expected_run_dir']).resolve(),job
 return {**job,'scheduler_record_sha256':hashlib.sha256(record.encode()).hexdigest(),'save_path_verified':True}
with ThreadPoolExecutor(max_workers=8) as pool:verified=list(pool.map(verify,jobs))
verified_by_id={j['job_id']:j for j in verified}
records=[]
for key,run in sorted(registered.items()):
 campaign,scale,domain,arm,seed=key
 receipt=Path(run['run_dir'])/'TRAINING_COMPLETE.json';completed=False;receipt_sha=None
 if receipt.is_file():
  data=receipt.read_bytes();r=json.loads(data)
  assert r['schema']=='oat_zero_training_complete_v1' and r['terminal_step'] in (3072,3073),receipt
  completed=True;receipt_sha=hashlib.sha256(data).hexdigest()
 allocations=[verified_by_id[j['job_id']] for j in active[key]]
 if completed:state='completed'
 elif any(j['state'] in ('RUNNING','CONFIGURING','COMPLETING') for j in allocations):state='running'
 elif any(j['state']=='PENDING' for j in allocations):state='queued'
 else:state='other'
 records.append({'campaign':campaign,'size':scale,'domain':domain,'arm':arm,'seed':seed,'state':state,'run_dir':run['run_dir'],'receipt_sha256':receipt_sha,'allocations':allocations})
result={'schema':'e118-e119-domain-size-job-status-v1','collection_started_utc':started,'collection_finished_utc':datetime.now(timezone.utc).isoformat(),'scope':'Current completion receipts plus live scheduler allocations. Deduplicate continuations by registered run directory/scientific cell; running takes precedence over queued successor allocations. Completed denotes training receipt, not a new efficacy endpoint audit.','sources':sources,'scheduler_source_sha256':hashlib.sha256(queue.encode()).hexdigest(),'rows':records,'groups':{},'totals':{}}
for campaign in LEDGERS:
 result['totals'][campaign]=dict(Counter(r['state'] for r in records if r['campaign']==campaign))
 for scale,domain in sorted({(r['size'],r['domain']) for r in records if r['campaign']==campaign}):
  rows=[r for r in records if (r['campaign'],r['size'],r['domain'])==(campaign,scale,domain)]
  counts={state:sum(r['state']==state for r in rows) for state in ('completed','running','queued','other')}
  terminal={arm:{r['seed'] for r in rows if r['arm']==arm and r['state']=='completed'} for arm in {r['arm'] for r in rows}}
  common=sorted(set.intersection(*terminal.values()))
  result['groups']['/'.join((campaign,scale,domain))]={'registered':len(rows),**counts,'matched_completed_seeds':common,'complete_five_seed_block':len(common)==5}
result['queued_continuations_of_running_cells']=[{'cell':'/'.join(map(str,key)),'running_job_ids':[j['job_id'] for j in jobs if j['state']=='RUNNING'],'queued_successor_job_ids':[j['job_id'] for j in jobs if j['state']=='PENDING']} for key,jobs in active.items() if any(j['state']=='RUNNING' for j in jobs) and any(j['state']=='PENDING' for j in jobs)]
(OUT/'job_status.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
lines=[f"Live read: {started} to {result['collection_finished_utc']}",'','Counts are scientific cells, with completed / running / queued / other. Running cells with queued continuation jobs are counted once.','']
for campaign in LEDGERS:
 lines.extend([campaign.upper(),'','| Size | Domain | Completed | Running | Queued | Other | Matched completed seeds |','|---|---|---:|---:|---:|---:|---|'])
 for name,g in result['groups'].items():
  c,size,domain=name.split('/')
  if c!=campaign:continue
  lines.append(f"| {size} | {labels[domain]} | {g['completed']}/{g['registered']} | {g['running']} | {g['queued']} | {g['other']} | {','.join(map(str,g['matched_completed_seeds'])) or 'none'} |")
 lines.extend(['',str(result['totals'][campaign]),''])
lines.append('E118 matched seeds require both MaxRL and ReplayMaxRL; E119 matched seeds require all four methods. A complete block has all five registered seeds. Completion receipts retain the latest earlier efficacy-audit counts but this operational read does not rescan evaluation outcomes.')
(OUT/'job_status.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
print('Queued successors of running cells:',result['queued_continuations_of_running_cells'])
