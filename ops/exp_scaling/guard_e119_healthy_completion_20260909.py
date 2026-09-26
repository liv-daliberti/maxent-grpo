#!/usr/bin/env python3
"""Seven-day, three-retry TIMEOUT guard for ten new E119 healthy public GPU routes."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import time

import os
import hashlib
import copy
import subprocess
import recover_e119_health_20260905 as checkpoints
import accelerate_a5000_completion_20260909 as runtime
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/e119_healthy_completion_guard_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e119_healthy_completion_guard_20260909.md'
LOCK = ART / 'singleton.lock'
LEDGER_LOCK = ROOT / 'var/artifacts/e118_ledger_promotion.lock'
PRIMARY = ROOT / 'var/artifacts/e119_level2_qwen05b_factorial_jobs.json'
CONTINUATIONS = ROOT / 'var/artifacts/e119_level2_continuation_jobs.json'
REGISTRATION = ART / 'supervisor.json'
ROUTES = (ROOT / 'var/artifacts/e119_node208_completion_20260909/healthy_route_amendment/transaction.json',
          ROOT / 'var/artifacts/e119_healthy_a6000_completion_20260909/transaction.json')
OLD_IDS = {31124279,31048181,31048187,31048191,31048180,31037836,31048185,31048188,31037843,31037846}
PRESERVE = ('UserId', 'Account', 'Partition', 'ReqNodeList', 'ExcNodeList',
            'MinMemoryNode', 'NumCPUs', 'NumNodes', 'NumTasks', 'TresPerNode',
            'Requeue', 'Nice', 'QOS', 'Dependency', 'Features', 'WorkDir', 'TimeLimit',
            'JobName', 'CPUs/Task', 'Command', 'Comment', 'StdOut', 'StdErr')
ACTIVE = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


def bounded_command(parts, *, check=True):
    """Private process override; no shared helper source file is modified."""
    return subprocess.run(parts, capture_output=True, text=True, check=check, timeout=120)


def install_bounded_scheduler():
    # All reused primitives refer to this same imported module object inside this
    # one CPU process. Existing supervisors live in separate processes.
    base.command = bounded_command


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as handle:
        handle.write(base.encoded(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def save(tx, message):
    tx['updated_at_utc'] = base.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'event': message})
    atomic(TX, tx)
    print(json.dumps({'at': tx['updated_at_utc'], 'event': message}), flush=True)


def ledger_mapping():
    primary = json.loads(PRIMARY.read_text()); data = json.loads(CONTINUATIONS.read_text())
    expected = {'schema': 'e119_level2_continuation_jobs_v1', 'original_ledger': str(PRIMARY),
        'original_ledger_sha256': recovery.digest(PRIMARY), 'same_scientific_cells': True,
        'same_run_directories': True, 'optimizer_update_changed': False, 'treatment_changed': False,
        'released': True, 'installed': True, 'outcomes_inspected': False}
    require(all(data.get(k) == v for k,v in expected.items()), 'E119 continuation contract changed')
    originals = {r['job_id']: r for r in primary['runs']}; result = {}
    require(len(originals) == 100 and len(data['continuations']) == 75, 'E119 cardinality changed')
    for row in data['continuations']:
        original = originals[row['original_job_id']]
        require(all(row[k] == original[k] for k in base.IDENTITY), 'E119 scientific identity changed')
        job = row['continuation_job_id']
        require(job not in result, 'duplicate authoritative successor')
        result[job] = {'original_job_id': row['original_job_id'],
                       'identity': {k: row[k] for k in base.IDENTITY}, 'row': row}
    return result


def mapping():
    plan = json.loads(PLAN.read_text())
    require(recovery.digest(PRIMARY) == plan['primary_sha256'], 'Original scientific ledger changed')
    scope = plan['rows']
    all_rows = ledger_mapping(); result = {}
    for item in scope:
        current = all_rows.get(item['job_id'])
        require(current and current['original_job_id'] == item['original_job_id']
                and current['identity'] == item['identity'], 'registered successor mapping changed')
        require(current['row'].get('dormant_fallback_job_id') == item['old_job_id'], 'fallback lineage changed')
        result[item['job_id']] = current
    return result


def predecessor_resource(record, key):
    try:
        return base.field(record, key)
    except RuntimeError:
        # Slurm omits an unset historical Comment entirely. Its absence is
        # frozen explicitly; all other missing fields remain validation errors.
        if key == 'Comment':
            return None
        raise


def dormant(item):
    record = show(item['old_job_id'])
    require(base.field(record,'JobState') == 'PENDING' and base.field(record,'Reason') == 'JobHeldUser'
            and base.field(record,'Priority') == '0', 'dormant predecessor is not an exact user hold')
    require(base.submit_tokens(record) == item['old_submit_tokens'], 'predecessor submission changed')
    require(base.field(record,'Restarts') == item['old_restarts'], 'predecessor restart changed')
    for key,value in item['old_resources'].items():
        actual = predecessor_resource(record,key)
        if key == 'NumNodes':
            require(actual in {'1','1-1'} and value in {'1','1-1'}, 'predecessor node count changed')
        else: require(actual == value, 'predecessor resource changed: ' + key)
    return record


def writer_check(item):
    dormant(item)
    actual = recovery.active_writers().get(str(Path(item['identity']['run_dir']).resolve()), set())
    require(actual <= {item['job_id'], item['old_job_id']}, 'unexpected same-cell writer')


def before_deadline(tx):
    require(datetime.now(timezone.utc) < datetime.fromisoformat(tx['deadline_utc']),
            'absolute seven-day deadline reached; retain owned hold for review')


def stable(item, record):
    require(base.field(record,'JobId') == str(item['job_id']), 'Guarded job ID changed')
    require(base.submit_tokens(record) == item['submit_tokens'], 'Frozen SubmitLine changed')
    require(recovery.digest(item['submit_tokens'][-1]) == item['launcher_sha256'], 'Frozen launcher changed')
    for field in PRESERVE:
        current, frozen = base.field(record, field), item['resources'][field]
        if field == 'NumNodes':
            require(current in {'1', '1-1'} and frozen in {'1', '1-1'}, 'Guarded allocation is not exactly one node')
        else:
            require(current == frozen, f'Guarded resource changed: {field}')


def show(job):
    result = base.command(['scontrol', 'show', 'job', '-dd', '-o', str(job)], check=False)
    require(result.returncode == 0 and result.stdout.strip(), f'Job {job} no longer has a controller record; manual continuation required')
    return result.stdout


def checkpoint_and_writer(item):
    run = Path(item['identity']['run_dir'])
    require(not recovery.complete(run), 'terminal receipt exists; do not restart')
    detail = checkpoints.checkpoint(item['identity'])
    step = detail['step']
    require(0 < step < 3072, 'checkpoint outside unfinished training range')
    require(not any(int(Path(p).name[5:]) >= step for p in detail['rejected']),
            'newer incomplete checkpoint needs review')
    writer_check(item)
    return detail


def own_hold(item, record, action):
    stable(item, record)
    require(base.field(record, 'JobState') == 'PENDING', 'Owned hold is not pending')
    require(base.field(record, 'Reason') == 'job_requeued_in_held_state' and base.field(record,'Priority') == '0', 'Expected requeue-owned hold missing')
    require(int(base.field(record, 'Restarts')) == action['before_restarts'] + 1, 'Expected exactly one requeue restart increment')


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'existing immutable guard preparation')
    source = [item for path in ROUTES for item in json.loads(path.read_text())['items']]
    require(len(source) == 10 and {i['old_job_id'] for i in source} == OLD_IDS, 'route scope differs')
    current = ledger_mapping(); rows = []
    for route in source:
        job = route.get('new_job_id')
        require(job and route.get('released'), 'all ten final successor routes must be released before freezing')
        ident = current.get(job)
        require(ident and ident['row'].get('dormant_fallback_job_id') == route['old_job_id'], 'unproven successor lineage')
        record = show(job); tokens = base.submit_tokens(record); env = base.exports(tokens)
        require(base.field(record,'JobState') in ACTIVE | {'PENDING'}, 'successor is no longer live or queued')
        require(base.field(record,'Priority') != '0', 'successor is still held')
        require(base.field(record,'Account') == 'mltheory' and base.field(record,'Partition') == 'lowprio', 'unexpected route')
        nodes = set(base.command(['scontrol','show','hostnames',base.field(record,'ReqNodeList')]).stdout.split())
        require(nodes in ({'node205','node207'}, {'node205','node207','node302'}), 'unreviewed node pool')
        require(base.field(record,'TresPerNode') in {'gres/gpu:a6000:1','gres/gpu:1'}, 'unreviewed GPU request')
        require(base.field(record,'MinMemoryNode') == str(route['memory_gib'])+'G'
                and base.field(record,'TimeLimit') == route['time_limit'], 'registered resources changed')
        require(tokens == route['command'] and env == base.exports(route['original_command']), 'science submission changed')
        require(env['SAVE_PATH'] == ident['identity']['run_dir'] and env['RUN_STAMP'] == ident['identity']['run_stamp']
                and env['OAT_ZERO_AUTO_RESUME'] == '1', 'run/resume exports changed')
        item = {'job_id': job, 'old_job_id': route['old_job_id'], 'original_job_id': ident['original_job_id'],
            'identity': ident['identity'], 'submit_tokens': tokens, 'original_command': route['original_command'],
            'launcher_sha256': recovery.digest(tokens[-1]), 'resources': {k:base.field(record,k) for k in PRESERVE},
            'old_submit_tokens': route['original_command'], 'old_restarts':base.field(route['before'],'Restarts'), 'old_resources': {k:predecessor_resource(route['before'],k) for k in PRESERVE},
            'initial_resume_step': route['checkpoint']['step'], 'initial_restarts': int(base.field(record,'Restarts')),
            'max_requeues': 3}
        dormant(item); writer_check(item); rows.append(item)
    files = [Path(__file__), PROTOCOL, Path(base.__file__), Path(recovery.__file__),
             Path(checkpoints.__file__), Path(runtime.__file__)]
    plan = {'schema':'e119-healthy-completion-guard-v1','created_at_utc':base.now(),
        'deadline_utc':(datetime.now(timezone.utc)+timedelta(days=7)).isoformat(),
        'rows':rows,'helper_sha256':{str(p):recovery.digest(p) for p in files},
        'primary_sha256':recovery.digest(PRIMARY), 'route_receipts_sha256':{str(p):recovery.digest(p) for p in ROUTES},
        'runtime_fingerprints':runtime.runtime_fingerprints(rows), 'poll_seconds':60,
        'max_requeues_per_cell':3,'terminal_retry_states':['TIMEOUT'], 'cpu_requeue_cap':7}
    atomic(PLAN,plan)
    print(json.dumps({'prepared':True,'job_ids':[r['job_id'] for r in rows], 'deadline_utc':plan['deadline_utc'],
                      'scheduler_mutations':False}))


def finish_release(tx, item, record, action, job_state):
    expected = action['before_restarts'] + 1
    require(int(base.field(record, 'Restarts')) == expected, 'Restart count changed during release')
    action.update(after=record, released=True)
    job_state.update(status='monitoring', last_resume_step=action['checkpoint_before_release']['step'])
    save(tx, f"{item['job_id']}: released same-ID checkpoint retry {len(job_state['attempts'])}/{item['max_requeues']}; original allocation limit retained")


def reconcile_action(tx, item, action, job_state):
    job = item['job_id']
    record = show(job)
    stable(item, record)
    state, reason = base.field(record, 'JobState'), base.field(record, 'Reason')
    if action.get('release_requested') and state in {'RUNNING', 'CONFIGURING', 'COMPLETING', 'PENDING', 'COMPLETED'}:
        if reason not in {'JobHeldUser', 'JobHeldAdmin', 'job_requeued_in_held_state'}:
            writer_check(item)
            finish_release(tx, item, record, action, job_state)
            return
    own_hold(item, record, action)
    action['own_hold'] = True
    detail = checkpoint_and_writer(item)
    require(detail['step'] >= action['checkpoint_before']['step'], 'Checkpoint regressed after holding')
    if job_state.get('last_resume_step') is not None:
        require(detail['step'] > job_state['last_resume_step'], 'Checkpoint has not advanced since prior retry')
    action['checkpoint_before_release'] = detail
    action['held_before_release'] = record
    before_deadline(tx)
    action['release_requested'] = True
    save(tx, f'{job}: audited owned hold, unchanged recipe/resources and valid advancing checkpoint; releasing')
    base.command(['scontrol', 'release', str(job)])
    record = show(job)
    stable(item, record)
    require(base.field(record, 'JobState') in {'PENDING', 'RUNNING', 'CONFIGURING'}, 'Unexpected post-release state')
    require(base.field(record, 'Reason') not in {'JobHeldUser', 'JobHeldAdmin', 'job_requeued_in_held_state'}, 'Hold remained after release')
    finish_release(tx, item, record, action, job_state)


def observe_one(tx, item, *, apply):
    job = item['job_id']
    key = str(job)
    job_state = tx['jobs'].get(key, dict(status='monitoring', attempts=[], last_resume_step=item['initial_resume_step']))
    if job_state['status'] in {'completed', 'manual_stop'}:
        return dict(job_id=job, status=job_state['status'])
    identities = mapping()
    require(job in identities and identities[job]['identity'] == item['identity'], 'Guarded cell ID/identity changed')
    unfinished = [a for a in job_state['attempts'] if not a.get('released')]
    require(len(unfinished) <= 1, 'Multiple unfinished retry transactions')
    if unfinished:
        if apply:
            reconcile_action(tx, item, unfinished[0], job_state)
            return dict(job_id=job, status=job_state['status'], retries=len(job_state['attempts']))
        return dict(job_id=job, status='action_requires_reconciliation', scheduler_mutations=False)
    run = Path(item['identity']['run_dir'])
    if recovery.complete(run):
        if apply:
            job_state['status'] = 'completed'
            tx['jobs'][key] = job_state
            save(tx, f'{job}: terminal receipt exists; no continuation needed')
        return dict(job_id=job, status='completed')
    record = show(job)
    stable(item, record)
    state, reason = base.field(record, 'JobState'), base.field(record, 'Reason')
    if state in ACTIVE or state == 'PENDING':
        # Queue priority and deliberate holds remain entirely under their existing owner.
        return dict(job_id=job, status='monitoring', state=state, reason=reason,
                    retries=len(job_state['attempts']), runtime=base.field(record, 'RunTime'))
    require(state == 'TIMEOUT', f'{job} entered {state}; manual inspection required, no blind restart')
    require(len(job_state['attempts']) < item['max_requeues'], 'Bounded retry allowance exhausted')
    require(job not in base.queue() and recovery.state(job) == 'TIMEOUT', 'TIMEOUT is not inactive and accounted')
    detail = checkpoint_and_writer(item)
    if job_state.get('last_resume_step') is not None:
        require(detail['step'] > job_state['last_resume_step'], 'No newer valid checkpoint since prior retry; stop to prevent a loop')
    if not apply:
        return dict(job_id=job, status='would_requeue_same_id', checkpoint=detail,
                    retry_number=len(job_state['attempts']) + 1, scheduler_mutations=False)
    # Repeat immediately before mutation; never requeue an active allocation.
    current = show(job)
    stable(item, current)
    require(base.field(current, 'JobState') == 'TIMEOUT' and base.field(current, 'Restarts') == base.field(record, 'Restarts'), 'Attempt/state changed before requeuehold')
    require(job not in base.queue() and recovery.state(job) == 'TIMEOUT', 'Timed-out job became active')
    action = dict(before=current, before_restarts=int(base.field(current, 'Restarts')),
                  checkpoint_before=detail, hold_intent=True, hold_command_returned=False, released=False)
    job_state['attempts'].append(action)
    job_state['status'] = 'holding'
    tx['jobs'][key] = job_state
    save(tx, f"{job}: durable requeuehold intent for bounded retry {len(job_state['attempts'])}/{item['max_requeues']}; never repeat an uncertain call")
    before_deadline(tx)
    base.command(['scontrol', 'requeuehold', str(job)])
    action['hold_command_returned'] = True
    save(tx, f'{job}: requeuehold returned; verify expected owned hold and restart increment')
    reconcile_action(tx, item, action, job_state)
    return dict(job_id=job, status=job_state['status'], retries=len(job_state['attempts']))


def observation_error(tx, item, error):
    state = tx['jobs'].setdefault(str(item['job_id']),
        {'status':'monitoring','attempts':[],'last_resume_step':item['initial_resume_step']})
    if isinstance(error, (subprocess.TimeoutExpired, subprocess.CalledProcessError)):
        state['consecutive_query_errors'] = state.get('consecutive_query_errors',0) + 1
        state['last_query_error'] = repr(error)
        if state['consecutive_query_errors'] < 3:
            # Keep a pending mutation intent exactly as written. The next pass
            # reconciles it; an uncertain requeue command is never repeated.
            save(tx,f"{item['job_id']}: transient scheduler observation error {state['consecutive_query_errors']}/3; retaining intent")
            return {'job_id':item['job_id'],'status':'transient_query_error','error':repr(error),
                    'consecutive_errors':state['consecutive_query_errors']}
    state.update(status='manual_stop',error=repr(error))
    save(tx,f"{item['job_id']}: stopped for review: {error}")
    return {'job_id':item['job_id'],'status':'manual_stop','error':repr(error)}


def cpu_plan():
    script = ART / 'supervisor.slurm'
    contents = '#!/bin/bash\nset -euo pipefail\n' + \
        'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n' + \
        f'cd {ROOT}\nexec /usr/bin/python3 -u {Path(__file__).resolve()} watch --apply\n'
    if script.exists(): require(script.read_text() == contents, 'existing CPU entrypoint differs')
    else: script.parent.mkdir(parents=True,exist_ok=True); script.write_text(contents)
    command = ['sbatch','--parsable','--hold','--job-name=e119-healthy-timeout-guard',
        '--account=mltheory','--partition=lowprio','--nodelist=node915,node917',
        '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=2G','--gres=none',
        '--time=1-01:10:00','--requeue','--export=NONE',f'--chdir={ROOT}',
        f'--output={ART}/supervisor-%j.out',f'--error={ART}/supervisor-%j.err',
        '--comment=e119-healthy-timeout-guard-20260909',str(script)]
    proof=base.command([command[0],'--test-only',*[x for x in command[1:] if x!='--hold']])
    value={'command':command,'script_sha256':recovery.digest(script), 'at':base.now(),
           'forecast_stdout':proof.stdout,'forecast_stderr':proof.stderr,'scheduler_mutations':False}
    atomic(ART/'cpu_submission.json',value)
    print(json.dumps(value))


def register(job):
    require(PLAN.exists(),'freeze the scientific guard plan before CPU registration')
    if REGISTRATION.exists():
        old=json.loads(REGISTRATION.read_text())
        require(old['job_id']==job and old['plan_sha256']==recovery.digest(PLAN),'existing registration differs')
        print(json.dumps({'registered':job,'already_registered':True}));return
    proposal=json.loads((ART/'cpu_submission.json').read_text());record=show(job)
    require(base.field(record,'JobState')=='PENDING' and base.field(record,'Reason')=='JobHeldUser'
            and base.field(record,'Priority')=='0','CPU job must still be held before registration')
    require(base.submit_tokens(record)==proposal['command'],'CPU submission differs from reviewed proposal')
    require(base.field(record,'UserId').endswith('('+str(os.getuid())+')'),'CPU job is not owned by this user')
    require('gres/gpu' not in base.field(record,'ReqTRES'),'CPU registration unexpectedly includes GPUs')
    require(recovery.digest(proposal['command'][-1])==proposal['script_sha256'],'CPU entrypoint changed')
    fields=('UserId','Account','Partition','ReqNodeList','MinMemoryNode','TimeLimit','Requeue','JobName','Command','Comment')
    value={'job_id':job,'plan_sha256':recovery.digest(PLAN),'submit_tokens':proposal['command'],
           'script_sha256':proposal['script_sha256'],'resources':{k:base.field(record,k) for k in fields},'at':base.now()}
    atomic(REGISTRATION,value)
    print(json.dumps({'registered':job,'scheduler_mutations':False}))


def run(*, watch, apply):
    plan = json.loads(PLAN.read_text())
    require(all(recovery.digest(p) == h for p,h in plan['helper_sha256'].items()), 'frozen guard/core helper drift')
    require(recovery.digest(PRIMARY) == plan['primary_sha256'], 'original scientific ledger changed')
    require(runtime.runtime_fingerprints(plan['rows']) == plan['runtime_fingerprints'], 'scientific runtime drift')
    tx = json.loads(TX.read_text()) if TX.exists() else {'schema':plan['schema'], 'plan_sha256':recovery.digest(PLAN),
        'deadline_utc':plan['deadline_utc'], 'jobs':{}, 'events':[]}
    require(tx['plan_sha256'] == recovery.digest(PLAN) and tx['deadline_utc'] == plan['deadline_utc'], 'plan/deadline changed')
    if apply:
        registration = json.loads(REGISTRATION.read_text()); own = os.environ.get('SLURM_JOB_ID')
        require(own and int(own) == registration['job_id'] and registration['plan_sha256'] == recovery.digest(PLAN),
                'applying monitor requires its registered CPU allocation')
        cpu = show(int(own)); require(base.field(cpu,'JobState') == 'RUNNING', 'registered CPU supervisor inactive')
        require('gres/gpu' not in base.field(cpu,'ReqTRES'), 'CPU supervisor unexpectedly requests GPUs')
        require(base.submit_tokens(cpu)==registration['submit_tokens'], 'registered CPU submission changed')
        require(recovery.digest(registration['submit_tokens'][-1])==registration['script_sha256'], 'CPU entrypoint drift')
        require(all(base.field(cpu,k)==v for k,v in registration['resources'].items()), 'registered CPU resource drift')
        require(int(base.field(cpu,'Restarts')) <= 7, 'CPU supervisor requeue cap exhausted')
    started = time.monotonic()
    while True:
        if datetime.now(timezone.utc) >= datetime.fromisoformat(tx['deadline_utc']):
            if apply:
                tx['status']='deadline_reached';save(tx,'Seven-day deadline reached; no new retries or held releases')
                atomic(ART/'status.json',{'at':base.now(),'status':'deadline_reached','deadline_utc':tx['deadline_utc']})
            return
        results = []
        for item in plan['rows']:
            try:
                with LEDGER_LOCK.open('a+') as ledger_lock:
                    fcntl.flock(ledger_lock,fcntl.LOCK_EX)
                    result=observe_one(tx,item,apply=apply)
                    results.append(result)
                    if apply and str(item['job_id']) in tx['jobs']:
                        tx['jobs'][str(item['job_id'])]['consecutive_query_errors']=0
            except Exception as error:
                if apply: results.append(observation_error(tx,item,error))
                else: results.append({'job_id':item['job_id'],'status':'observation_error','error':repr(error)})
        status={'at':base.now(),'read_only':not apply,'jobs':results,'deadline_utc':tx['deadline_utc']}
        print(json.dumps(status),flush=True)
        if apply:
            tx['status']='monitoring';tx['last_pass_at']=status['at'];atomic(TX,tx);atomic(ART/'status.json',status)
            atomic(ART/'ready.json',{'at':status['at'],'job_id':int(os.environ['SLURM_JOB_ID']),
                'plan_sha256':recovery.digest(PLAN),'singleton':True})
        if not watch or all(r['status'] in {'completed','manual_stop'} for r in results): return
        # A routine CPU-only renewal never changes the absolute science deadline.
        if apply and time.monotonic()-started >= 23*3600:
            before_deadline(tx);cpu=show(int(os.environ['SLURM_JOB_ID']))
            require(int(base.field(cpu,'Restarts')) < 7,'CPU renewal cap exhausted')
            save(tx,'Registered CPU supervisor renewal intent; science retries/absolute deadline unchanged')
            base.command(['scontrol','requeue',os.environ['SLURM_JOB_ID']]);return
        time.sleep(60)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','once','watch','cpu-plan','register'));parser.add_argument('--apply',action='store_true');parser.add_argument('--job-id',type=int)
    args=parser.parse_args()
    install_bounded_scheduler()
    require(args.phase != 'prepare' or not args.apply,'prepare never mutates Slurm')
    require(args.phase != 'watch' or args.apply,'watch requires registered CPU supervisor and --apply')
    ART.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='cpu-plan': cpu_plan()
        elif args.phase=='register':
            require(args.job_id is not None, '--job-id is required');register(args.job_id)
        elif args.phase=='prepare':
            with LEDGER_LOCK.open('a+') as ledger_lock:
                fcntl.flock(ledger_lock,fcntl.LOCK_EX);prepare()
        else:run(watch=args.phase=='watch',apply=args.apply)
