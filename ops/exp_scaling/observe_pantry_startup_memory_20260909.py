#!/usr/bin/env python3
"""Bounded read-only startup and memory observer for two exact Pantry jobs."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import prioritize_e118_capacity_20260905 as base

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/pantry_startup_memory_observer_20260909'
PLAN = ART / 'plan.json'
STATE = ART / 'latest.json'
LOCK = ART / 'observer.lock'
PROTOCOL = ROOT / 'paper/preregistration/pantry_startup_memory_observer_20260909.md'
AMENDMENT = ROOT / 'var/artifacts/e120_pantry_s71_memory116_20260909/root_postapply_review.json'
IDS = (31033711,31048186)
RESOURCE_FIELDS = ('Account','Partition','ReqNodeList','ExcNodeList','MinMemoryNode','NumCPUs','NumTasks','TresPerNode','TimeLimit','Requeue','Nice','Features','WorkDir')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(parts, timeout=10):
    return subprocess.run(parts, capture_output=True, text=True, timeout=timeout)


def prepare():
    assert not PLAN.exists() and not STATE.exists()
    amendment=json.loads(AMENDMENT.read_text())
    assert amendment['job_id']==31033711 and amendment['status']=='passed' and amendment['ledgers_runtime_helpers_unchanged']
    rows=[]
    for job,stamp,floor,memory in [(31033711,'e120r1_qwen3b_pantry_fresh_frequency_s71',0,'116G'),
                                  (31048186,'e119_level2_pantry_replay_drgrpo_s46',96,'96G')]:
        record=base.show(job);tokens=base.submit_tokens(record);env=base.exports(tokens)
        assert env['RUN_STAMP']==stamp and base.field(record,'MinMemoryNode')==memory
        assert env['OAT_ZERO_AUTO_RESUME']=='1'
        rows.append({'job_id':job,'run_dir':env['SAVE_PATH'],'checkpoint_step':floor,'expected_exports':env,
            'launcher':tokens[-1],'launcher_sha256':digest(tokens[-1]),
            'resource_fields':{k:base.field(record,k) for k in RESOURCE_FIELDS},'initial_scheduler_record':record})
    plan={'schema':'pantry-startup-memory-observer-20260909-v1','prepared_at_utc':base.now(),'rows':rows,
          'amendment':str(AMENDMENT),'amendment_sha256':digest(AMENDMENT),'controller_sha256':digest(__file__),
          'helper_sha256':digest(base.__file__),'protocol_sha256':digest(PROTOCOL),'maximum_queue_wait_hours':48,
          'allocation_observation_hours':6,'maximum_total_hours':54,'poll_seconds':60,
          'read_only_science_jobs':True,'scheduler_mutations_by_observer':False}
    base.atomic(PLAN,plan)
    print(json.dumps({'prepared':True,'job_ids':IDS,'plan':str(PLAN)}),flush=True)


def metrics(path):
    if not path.is_file():
        return None
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 512000))
        lines = handle.read().decode('utf8', 'replace').splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if 'trainer/policy_sgd_step' in value or 'misc/global_step' in value:
            return value
    return None


def inspect_log(path, previous, attempt_key):
    info = previous.get('log', {})
    if not path.is_file():
        return info
    st = path.stat()
    if info.get('attempt_key') != attempt_key or info.get('inode') != st.st_ino or info.get('offset', 0) > st.st_size:
        info = {'attempt_key': attempt_key, 'inode': st.st_ino, 'offset': 0,
                'restore_messages': [], 'fatal_messages': [], 'selected_checkpoint_step': None}
    with path.open('r', errors='replace') as handle:
        handle.seek(info['offset'])
        for line in handle:
            line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
            if '[slurm] host=' in line:
                info.update(restore_messages=[], fatal_messages=[], selected_checkpoint_step=None)
            if 'auto_resume=' in line:
                match = re.search(r'step_(\d+)', line)
                if match:
                    info['selected_checkpoint_step'] = int(match.group(1))
            if any(token in line for token in ('auto_resume=', 'Loaded checkpoint', 'Successfully loaded', 'Restored optimizer')):
                info['restore_messages'] = (info['restore_messages'] + [line[:1800]])[-12:]
            if any(token in line for token in ('Fatal Python error', 'Traceback (most recent call last)', 'CUDA out of memory', 'OutOfMemoryError', 'EngineDeadError')):
                info['fatal_messages'] = (info['fatal_messages'] + [line[:1800]])[-12:]
        info['offset'] = handle.tell()
    return info


def cgroup_probe(job):
    code = '''from pathlib import Path
import json
p=Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_JOB')
s={k:int(v) for k,v in (line.split() for line in (p/'memory.stat').read_text().splitlines())}
e={k:int(v) for k,v in (line.split() for line in (p/'memory.events').read_text().splitlines())}
r={'noncache_gib':sum(s.get(k,0) for k in ['anon','shmem','kernel'])/2**30,'events':e}
for k in ['memory.current','memory.high','memory.max','memory.peak']:
 if (p/k).exists():r[k]=(p/k).read_text().strip()
print(json.dumps(r))'''.replace('JOB', str(job))
    try:
        result = execute(['srun', '--overlap', f'--jobid={job}', '--nodes=1', '--ntasks=1',
                          '--cpus-per-task=1', '--mem=64M', '--time=00:01:00', 'python', '-c', code], timeout=8)
        return {'observed_at_utc': base.now(), 'returncode': result.returncode,
                'value': json.loads(result.stdout) if result.returncode == 0 else None,
                'stderr': result.stderr[-1500:]}
    except Exception as exc:
        return {'observed_at_utc': base.now(), 'probe_error': repr(exc), 'optional_probe': True}


def observe(item, previous):
    job = item['job_id']
    result = execute(['scontrol', 'show', 'job', '-dd', '-o', str(job)])
    if result.returncode or not result.stdout.strip():
        return {'job_id': job, 'status': 'controller_unavailable', 'error': result.stderr[-1500:],
                'previous_verified': previous.get('fresh_optimizer_verified', False)}
    record = result.stdout; tokens = base.submit_tokens(record)
    row = {'job_id': job, 'observed_at_utc': base.now(), 'scheduler_record': record,
           'scheduler': {key: base.field(record, key) for key in ('JobState', 'Reason', 'NodeList', 'RunTime', 'Restarts', 'StartTime', 'TimeLimit')},
           'fresh_optimizer_verified': False}
    if base.exports(tokens) != item['expected_exports'] or tokens[-1] != item['launcher'] or digest(item['launcher']) != item['launcher_sha256']:
        row['status'] = 'identity_or_source_changed'; return row
    changed={k:{'before':v,'after':base.field(record,k)} for k,v in item['resource_fields'].items() if base.field(record,k)!=v}
    if changed:
        row.update(status='resource_drift',resource_changes=changed);return row
    state = row['scheduler']['JobState']
    attempt_key = row['scheduler']['StartTime'] + '/' + row['scheduler']['Restarts']
    row['log'] = inspect_log(Path(base.field(record, 'StdOut')), previous, attempt_key)
    row['status'] = state.lower()
    run = Path(item['run_dir']); receipt = run / 'TRAINING_COMPLETE.json'
    if receipt.is_file():
        data = json.loads(receipt.read_text())
        if data.get('terminal_step', 0) >= 3072:
            row.update(status='completed', fresh_optimizer_verified=True, terminal_receipt=data)
            return row
    path = run / f'debug_job{job}' / 'train_metrics.jsonl'
    latest = metrics(path)
    if latest:
        threshold = row['log'].get('selected_checkpoint_step') or item['checkpoint_step']
        stamp = row['scheduler']['StartTime']
        start_epoch = datetime.fromisoformat(stamp).timestamp() if stamp not in ('Unknown', 'N/A') else float('inf')
        row.update(latest_record_step=latest.get('trainer/policy_sgd_step', latest.get('misc/global_step')),
                   selected_checkpoint_step=threshold, metrics_path=str(path),
                   metrics_mtime=path.stat().st_mtime, metrics_age_seconds=time.time()-path.stat().st_mtime,
                   training_times={k: latest.get(k) for k in ('train/total_time', 'actor/total_time', 'misc/vllm_go_sleep_time', 'misc/vllm_wake_up_time')})
        row['fresh_optimizer_verified'] = (state == 'RUNNING' and row['latest_record_step'] > threshold
            and row['metrics_mtime'] >= start_epoch and row['metrics_age_seconds'] < 300
            and float(latest.get('train/total_time', 0)) > 0 and not row['log'].get('fatal_messages'))
        if row['fresh_optimizer_verified']:
            row['status'] = 'fresh_optimizer_verified'
            old_probe = previous.get('cgroup') if previous.get('log', {}).get('attempt_key') == attempt_key else None
            row['cgroup'] = old_probe or cgroup_probe(job)
    if row['log'].get('fatal_messages'):
        row.update(status='current_attempt_fatal', fresh_optimizer_verified=False)
    return row


def checkpoint_progress(item):
    import sys
    sys.path.insert(0,str(ROOT/'ops'))
    from validate_deepspeed_checkpoint import validate_checkpoint
    rows=[]
    for p in (Path(item['run_dir'])/f"debug_job{item['job_id']}"/'checkpoints').glob('step_*'):
        try:step=int(p.name.removeprefix('step_'))
        except ValueError:continue
        if step>item['checkpoint_step']:
            errors=validate_checkpoint(p)
            rows.append({'step':step,'path':str(p),'errors':errors,'mtime':p.stat().st_mtime})
    return sorted(rows,key=lambda r:r['step'])


def enrich(item,row,previous):
    record=row.get('scheduler_record')
    if not record:return row
    state=row['scheduler']['JobState']
    now=datetime.now(timezone.utc)
    if state=='RUNNING':
        probe=previous.get('cgroup')
        age=(now-datetime.fromisoformat(probe['observed_at_utc'])).total_seconds() if probe else float('inf')
        same_attempt=previous.get('scheduler',{}).get('StartTime')==row['scheduler']['StartTime'] and previous.get('scheduler',{}).get('Restarts')==row['scheduler']['Restarts']
        row['cgroup']=probe if probe and age<900 and same_attempt else row.get('cgroup') or cgroup_probe(item['job_id'])
    row['log_activity']={}
    for field in ['StdOut','StdErr']:
        path=Path(base.field(record,field))
        if not path.is_file():continue
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-48000));lines=handle.read().decode('utf8','replace').splitlines()
        matches=[re.sub(r'\x1b\[[0-9;]*m','',line)[:1000] for line in lines if any(token in line for token in
            ('actor start','actor finished','Loaded checkpoint','auto_resume=','Evaluating','evaluation progress','Fatal Python','Traceback','CUDA out of memory'))]
        row['log_activity'][field]={'path':str(path),'mtime':path.stat().st_mtime,'age_seconds':time.time()-path.stat().st_mtime,'activity_lines':matches[-8:]}
    row['new_checkpoint_candidates']=checkpoint_progress(item)
    stamp=row['scheduler']['StartTime']
    allocation_start=datetime.fromisoformat(stamp).timestamp() if stamp not in ('Unknown','N/A') else float('inf')
    fresh_checkpoint=any(not c['errors'] and c['mtime']>=allocation_start for c in row['new_checkpoint_candidates'])
    row['startup_and_checkpoint_verified']=bool(row.get('fresh_optimizer_verified') and fresh_checkpoint) or row.get('status')=='completed'
    if row['startup_and_checkpoint_verified']:row['status']='startup_and_checkpoint_verified'
    return row


def run(watch):
    plan=json.loads(PLAN.read_text())
    assert digest(__file__)==plan['controller_sha256'] and digest(base.__file__)==plan['helper_sha256']
    assert digest(PROTOCOL)==plan['protocol_sha256']
    now=datetime.now(timezone.utc)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'schema':plan['schema'],'plan_sha256':digest(PLAN),
        'started_at_utc':now.isoformat(),'queue_deadline_utc':(now+timedelta(hours=48)).isoformat(),
        'deadline_utc':(now+timedelta(hours=54)).isoformat(),'jobs':{}}
    assert state['plan_sha256']==digest(PLAN)
    terminal_statuses={'startup_and_checkpoint_verified','queue_deadline_reached','allocation_observation_deadline_reached','identity_or_source_changed','resource_drift'}
    while True:
        if datetime.now(timezone.utc)>=datetime.fromisoformat(state['deadline_utc']):
            state['status']='deadline_reached';base.atomic(STATE,state);return
        for item in plan['rows']:
            key=str(item['job_id']);previous=state['jobs'].get(key,{})
            if previous.get('status') in terminal_statuses:continue
            try:
                row=observe(item,previous)
                first=previous.get('allocation_first_observed_epoch')
                if first is None and row.get('scheduler',{}).get('JobState')=='RUNNING':first=time.time()
                if first is not None:
                    row['allocation_first_observed_epoch']=first
                    row['allocation_observation_deadline_utc']=datetime.fromtimestamp(first+6*3600,timezone.utc).isoformat()
                row=enrich(item,row,previous)
                if not row.get('startup_and_checkpoint_verified'):
                    if first is not None and time.time()>=first+6*3600:row['status']='allocation_observation_deadline_reached'
                    elif first is None and datetime.now(timezone.utc)>=datetime.fromisoformat(state['queue_deadline_utc']):row['status']='queue_deadline_reached'
            except Exception as exc:
                row={'job_id':item['job_id'],'status':'observation_error','error':repr(exc)}
            state['jobs'][key]=row
            with (ART/'observations.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        state['updated_at_utc']=base.now()
        state['status']='all_verified' if all(r.get('startup_and_checkpoint_verified') for r in state['jobs'].values()) and len(state['jobs'])==2 else 'watching'
        base.atomic(STATE,state)
        print(json.dumps({'at':state['updated_at_utc'],'status':state['status'],'jobs':[{'job_id':r['job_id'],'status':r['status'],'step':r.get('latest_record_step'),'checkpoint_verified':r.get('startup_and_checkpoint_verified',False)} for r in state['jobs'].values()]}),flush=True)
        if not watch or all(r['status'] in terminal_statuses for r in state['jobs'].values()):return
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','once','watch'))
    args = parser.parse_args(); ART.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if args.phase == 'prepare' else run(args.phase == 'watch')
