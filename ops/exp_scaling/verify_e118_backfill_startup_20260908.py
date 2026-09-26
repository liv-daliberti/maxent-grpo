#!/usr/bin/env python3
"""Read-only, bounded verification of the registered E118 backfill attempts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

import prioritize_e118_capacity_20260905 as base

ART = base.ROOT / 'var/artifacts/e118_owner_backfill_20260908'

def metric(path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 80000))
        rows = f.read().splitlines()
    for line in reversed(rows):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None

def observe(old):
    tx = json.loads((ART / f'{old}.json').read_text())
    item = tx['replacements'][0]
    job = item['new_job_id']
    run = Path(item['run_dir'])
    attempt = run / f'debug_job{job}'
    row = dict(old_job_id=old, job_id=job, run_dir=str(run), attempt_dir=str(attempt), profile=item['profile'])
    rec = base.show(job)
    row['scheduler'] = {k:base.field(rec,k) for k in ('JobState','NodeList','MinMemoryNode','Restarts','StdOut','StdErr')}
    if row['scheduler']['JobState'] == 'RUNNING':
        code = f'''from pathlib import Path
import json
p=Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{job}')
s={{k:int(v) for k,v in (line.split() for line in (p/'memory.stat').read_text().splitlines())}}
e={{k:int(v) for k,v in (line.split() for line in (p/'memory.events').read_text().splitlines())}}
print(json.dumps(dict(noncache_gib=(s.get('anon',0)+s.get('shmem',0)+s.get('kernel',0))/2**30,events=e,memory_current=int((p/'memory.current').read_text()),memory_high=(p/'memory.high').read_text().strip())))'''
        probe = subprocess.run(['srun','--overlap',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=256M','--time=00:02:00','python','-c',code],capture_output=True,text=True,timeout=45)
        if probe.returncode == 0:
            row['cgroup'] = json.loads(probe.stdout)
        else:
            row['probe_error'] = probe.stderr[-1000:]
    path = attempt / 'train_metrics.jsonl'
    data = metric(path)
    if data:
        row.update(fresh_step=data.get('trainer/global_step',0),
                   metrics_age_seconds=time.time()-path.stat().st_mtime,
                   sleep_wake_metrics={k:v for k,v in data.items() if 'sleep' in k or 'wake' in k})
    log = Path(row['scheduler']['StdOut'])
    text = log.read_text(errors='replace') if log.exists() else ''
    if '[slurm] host=' in text:
        text = text[text.rfind('[slurm] host='):]
    row['fatal_errors'] = [re.sub(r'\x1b\[[0-9;]*m','',line) for line in text.splitlines() if re.search(r'Traceback|Error:|Fatal Python',line)][-5:]
    row['evaluation_files'] = [str(p) for p in (attempt/'eval_results').glob('*multi_answer.json')]
    timing = row.get('sleep_wake_metrics',{})
    events = row.get('cgroup',{}).get('events',{})
    row['first_optimizer_verified'] = (
        row['scheduler']['JobState']=='RUNNING' and
        row.get('fresh_step',0)>item['checkpoint_step'] and
        row.get('metrics_age_seconds',999999)<300 and
        any('sleep' in k and float(v)>0 for k,v in timing.items()) and
        any('wake' in k and float(v)>0 for k,v in timing.items()) and
        not row['fatal_errors'] and
        all(events.get(k,-1)==0 for k in ('high','oom','oom_kill')))
    return row

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_jobs',type=int,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=3600)
    args=parser.parse_args(); started=time.time()
    while True:
        rows=[observe(old) for old in args.old_jobs]
        result=dict(schema='e118-backfill-startup-verification-v1',checked_at_utc=datetime.now(timezone.utc).isoformat(),rows=rows,all_verified=all(r['first_optimizer_verified'] for r in rows))
        base.atomic(args.output,result)
        print(json.dumps({'at':result['checked_at_utc'],'all_verified':result['all_verified'],'jobs':[(r['job_id'],r.get('fresh_step'),r['first_optimizer_verified']) for r in rows]}),flush=True)
        if result['all_verified'] or not args.watch or time.time()-started>=args.max_seconds:
            break
        time.sleep(45)
