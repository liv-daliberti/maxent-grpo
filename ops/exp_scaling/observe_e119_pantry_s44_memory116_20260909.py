#!/usr/bin/env python3
"""Read-only bounded startup verification for repaired Pantry job31048182."""
import datetime,json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
ART=ROOT/'var/artifacts/e119_pantry_s44_memory116_20260909'
RUN=ROOT/'var/data/xdr_qwen25_0p5b_instruct_maxrl_verified_replay_e119_level2_pantry_replay_maxrl_s44'
TARGET=31048182

def atomic(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)

def main():
 report={'job_id':TARGET,'read_only':True,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observations':[]}
 end=time.monotonic()+48*3600
 allocation_deadline=None
 while time.monotonic()<end:
  r=subprocess.run(['scontrol','show','job','-o',str(TARGET)],capture_output=True,text=True,timeout=20)
  fields=dict(x.split('=',1) for x in r.stdout.split() if '=' in x)
  item={k:fields.get(k) for k in ['JobState','Restarts','StartTime','RunTime','NodeList','MinMemoryNode','TimeLimit']}
  item['at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  rows=[];metrics=RUN/f'debug_job{TARGET}/train_metrics.jsonl'
  if metrics.exists():
   with metrics.open('rb') as h:
    h.seek(max(0,metrics.stat().st_size-2000000));raw=h.read().decode(errors='replace')
   for line in raw.splitlines():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:pass
  fresh=[]
  if fields.get('JobState')=='RUNNING' and fields.get('Restarts')=='6':
   if allocation_deadline is None:allocation_deadline=time.monotonic()+5400
   assert fields.get('MinMemoryNode')=='116G' and fields.get('TimeLimit')=='1-12:00:00'
   started=datetime.datetime.fromisoformat(fields['StartTime']).timestamp()
   wall=time.time()-started
   fresh=[x for x in rows if x.get('trainer/step',0)>480 and x.get('misc/elapse',1e9)<wall+120]
  item['fresh_steps']=[x.get('trainer/step') for x in fresh[-5:]]
  item['fresh_sync_seconds']=[x.get('misc/weight_sync_elapse') for x in fresh[-5:]]
  item['fresh_learner_seconds']=[x.get('train/learn_batch_time') for x in fresh[-5:]]
  report['observations'].append(item)
  if len(fresh)>=2:
   code="""import pathlib,json,time
p=pathlib.Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_31048182')
results=[]
for i in range(2):
 d={k:(p/k).read_text().strip() for k in ['memory.current','memory.peak','memory.high','memory.events','memory.stat']}
 s=dict(x.split() for x in d['memory.stat'].splitlines());d['memory.stat']={k:s.get(k) for k in ['anon','shmem','kernel','file']};results.append(d)
 if i==0:time.sleep(5)
print(json.dumps(results))"""
   probe=subprocess.run(['timeout','-k','3s','25s','srun',f'--jobid={TARGET}','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=0','--gres=none','/usr/bin/python3','-c',code],capture_output=True,text=True,timeout=32)
   item['memory_probe']={'returncode':probe.returncode,'stdout':probe.stdout,'stderr':probe.stderr}
   report['status']='fresh_optimizer_progress_verified';report['latest_verified_step']=fresh[-1]['trainer/step']
   atomic(ART/'startup_observer.json',report);print(json.dumps({'status':report['status'],'job_id':TARGET,'step':report['latest_verified_step']}),flush=True);return
  atomic(ART/'startup_observer.json',report)
  if fields.get('JobState') in ['FAILED','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','TIMEOUT']:
   report['status']='terminal_failure_requires_inspection';atomic(ART/'startup_observer.json',report);return
  if allocation_deadline is not None and time.monotonic()>=allocation_deadline:break
  time.sleep(30)
 report['status']='bounded_window_ended_without_fresh_progress_verification';atomic(ART/'startup_observer.json',report)

if __name__=='__main__':main()
