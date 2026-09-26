#!/usr/bin/env python3
"""Bounded read-only verification of E120 Graph's first resumed optimizer work."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo
import backfill_e120_graph_node204_20260908 as placement

ART=placement.ART
OUTPUT=ART/'optimizer_startup_observations.json'
RECEIPT=ART/'first_optimizer_verified.json'
TZ=ZoneInfo('America/New_York')


def observe(expected_start):
    tx=json.loads(placement.TX.read_text());job=tx['new_job_id']
    rec=placement.base.show(job)
    keys=['JobState','NodeList','StartTime','Restarts','MinMemoryNode','NumCPUs','TresPerNode','TimeLimit','Account','Partition','StdOut']
    sched={k:placement.base.field(rec,k) for k in keys}
    row={'observed_at_utc':placement.base.now(),'job_id':job,'scheduler':sched,'checkpoint_step':576,'first_optimizer_verified':False}
    if sched['JobState']!='RUNNING' or sched['StartTime']!=expected_start or sched['Restarts']!='0':
        row['stopped_for_attempt_change']=True
        return row
    start_local=datetime.fromisoformat(sched['StartTime']).replace(tzinfo=TZ)
    start=start_local.astimezone(timezone.utc)
    row['current_start_time_utc']=start.isoformat()
    p=Path(tx['run']['run_dir'])/f'debug_job{job}/train_metrics.jsonl'
    row['metrics_path']=str(p)
    if p.exists():
        rows=[]
        for line in p.read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
        row['metrics_mtime_utc']=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()
        row['metrics_after_start']=p.stat().st_mtime>=start.timestamp()
        row['metrics_age_seconds']=time.time()-p.stat().st_mtime
        if rows:
            last=rows[-1]
            row['fresh_step']=last.get('trainer/global_step',0)
            row['sleep_wake']={k:last.get(k,0) for k in ['misc/vllm_go_sleep_time','misc/vllm_wake_up_time']}
    p=Path(sched['StdOut']);raw=p.read_text(errors='replace') if p.is_file() else ''
    if '[slurm] host=' in raw:raw=raw[raw.rfind('[slurm] host='):]
    lines=[re.sub(r'\x1b\[[0-9;]*m','',x) for x in raw.splitlines()]
    row['fatal_errors']=[x for x in lines if re.search(r'Traceback|Error:|Fatal Python|CUDA out of memory|OutOfMemoryError|oom-kill',x)][-8:]
    completions=[];restores=[]
    for line in lines:
        m=re.search(r'I(\d{2})(\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)',line)
        if not m:continue
        stamp=datetime.fromisoformat(f'{start_local.year}-{m[1]}-{m[2]}T{m[3]}').replace(tzinfo=TZ).astimezone(timezone.utc)
        if stamp<start:continue
        done=re.search(r'post-learning done step=(\d+)',line)
        if done and int(done[1])>576:completions.append({'at_utc':stamp.isoformat(),'step':int(done[1]),'line':line})
        if 'resume actor weight sync done checkpoint_step=576' in line:restores.append({'at_utc':stamp.isoformat(),'line':line})
    row['current_optimizer_completions']=completions[-5:]
    row['restore_completed']=restores
    remote=f'''from pathlib import Path
import json
p=Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{job}')
s={{k:int(v) for k,v in (line.split() for line in (p/'memory.stat').read_text().splitlines())}}
e={{k:int(v) for k,v in (line.split() for line in (p/'memory.events').read_text().splitlines())}}
print(json.dumps({{'noncache_gib':sum(s.get(k,0) for k in ['anon','shmem','kernel'])/2**30,'events':e,'memory_current':int((p/'memory.current').read_text()),'memory_high':int((p/'memory.high').read_text()),'memory_peak':int((p/'memory.peak').read_text())}}))'''
    probe=subprocess.run(['timeout','-k','3s','25s','srun',f'--jobid={job}','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=0','--gres=none','/usr/bin/python3','-c',remote],capture_output=True,text=True)
    if probe.returncode==0:row['cgroup']=json.loads(probe.stdout)
    else:row['probe_error']=probe.stderr[-1200:]
    after=placement.base.show(job)
    stable=all(placement.base.field(after,k)==sched[k] for k in ['JobState','NodeList','StartTime','Restarts'])
    if not stable:row['stopped_for_attempt_change']=True
    events=row.get('cgroup',{}).get('events',{})
    expected={'JobState':'RUNNING','NodeList':'node204','MinMemoryNode':'128G','NumCPUs':'16','TresPerNode':'gres/gpu:a5000:1','TimeLimit':'3-00:00:00','Account':'allcs','Partition':'lowprio'}
    row['checks']={'scheduler_matches':all(sched[k]==v for k,v in expected.items()),
                   'attempt_stable':stable,'checkpoint576_fully_restored':bool(restores),
                   'step_above576':row.get('fresh_step',0)>576,'current_timestamped_optimizer_work':bool(completions),
                   'metrics_current':row.get('metrics_after_start',False) and row.get('metrics_age_seconds',999999)<180,
                   'positive_actor_sleep_wake':all(float(v)>0 for v in row.get('sleep_wake',{}).values()) and len(row.get('sleep_wake',{}))==2,
                   'no_fatal_errors':not row['fatal_errors'],
                   'no_high_oom_events':all(events.get(k,-1)==0 for k in ['high','oom','oom_kill']),
                   'working_memory_fits':row.get('cgroup',{}).get('noncache_gib',999999)<128}
    row['first_optimizer_verified']=all(row['checks'].values())
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch-seconds',type=int,default=0);args=parser.parse_args()
    expected='2026-09-08T13:42:22';deadline=time.monotonic()+args.watch_seconds
    history=[]
    while True:
        row=observe(expected);history.append(row)
        placement.base.atomic(OUTPUT,history)
        if row['first_optimizer_verified']:
            receipt={'schema':'e120-graph-first-current-attempt-optimizer-v1','verified_at_utc':placement.base.now(),'observer_sha256':placement.digest(__file__),'row':row}
            if not RECEIPT.exists():
                with RECEIPT.open('x') as handle:json.dump(receipt,handle,indent=2);handle.write('\n')
        print(json.dumps({'at':row['observed_at_utc'],'state':row['scheduler']['JobState'],'fresh_step':row.get('fresh_step'),'restore_done':bool(row.get('restore_completed')),'verified':row['first_optimizer_verified'],'noncache_gib':row.get('cgroup',{}).get('noncache_gib'),'errors':row.get('fatal_errors'),'attempt_changed':row.get('stopped_for_attempt_change',False)}),flush=True)
        if row['first_optimizer_verified'] or row.get('stopped_for_attempt_change') or row.get('fatal_errors') or time.monotonic()>=deadline:break
        time.sleep(min(30,max(0,deadline-time.monotonic())))
