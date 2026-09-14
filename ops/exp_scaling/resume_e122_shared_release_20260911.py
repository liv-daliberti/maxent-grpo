#!/usr/bin/env python3
"""Resume the frozen E122 release controller with serialized shared admission.

Scientific inputs and the original once-only release journals remain unchanged.
Only this separately registered CPU supervisor is new. A queue/accounting race
is observed again without issuing any release while the snapshot is unknown.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import control_e122_level3_release as c
import e122_slurm_2511_compat as compat
import launch_e122_level3_factorial as launcher

ART = ROOT / 'var/artifacts/e122_shared_controller_recovery_20260911'
PLAN = ART / 'plan.json'
REG = ART / 'supervisor.json'
SOURCE = Path(__file__).resolve()
PROTOCOL = ROOT / 'paper/preregistration/e122_shared_controller_recovery_20260911.md'
TEST = ROOT / 'tests/test_resume_e122_shared_release_20260911.py'
PYTHON = ROOT / 'var/seed_paper_eval/paper310/bin/python'
LOCKS = (
    ROOT / 'var/artifacts/shared_storage_admission_20260911.lock',
    ROOT / 'var/artifacts/e124_qwen7b_three_level/controller.lock',
    ROOT / 'var/artifacts/e123_level3_factorial/release_controller/.lock',
)
COMPAT_SHA = '86b6ca285c03ef4aa54292245264aefe98beef2f61fa6c8fd2a81319a7e161a3'
COMPAT_TEST_SHA = '34268c70b28a6e754c4afd762e5f03f61140e0d90c091dd7ad7461b794c33efb'
LEDGER_SHA = '0647de37c4f09c5c59e8c6ee8b59f1b634eb9d2870d68534c353abc2ccc5b16d'
ORIGINAL_STATUS = c.status


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fresh_args():
    return c.parse_args(['--plan', str(compat.PLAN), '--plan-sha256', compat.PLAN_SHA256,
        '--held-ledger', str(ROOT/'var/artifacts/e122_level3_factorial_jobs.json'),
        '--held-ledger-sha256', LEDGER_SHA, '--model-choice', '05b', '--status'])


def authenticate():
    compat.install_compat(launcher, COMPAT_SHA, COMPAT_TEST_SHA)
    args = fresh_args()
    return args, c.load_campaign(args, launcher)


@contextmanager
def admission_locks():
    # These are per-admission transaction locks, never daemon lifetime locks.
    # Acquire nonblocking; a busy peer owns this cycle and is left undisturbed.
    with ExitStack() as stack:
        for path in LOCKS:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_plan(path=PLAN, expected=None):
    expected = expected or (ART/'plan.sha256').read_text().strip()
    plan = c.pinned_json(Path(path), expected)
    require(plan['schema'] == 'e122_shared_controller_recovery_v1', 'Wrong supervisor plan schema')
    require(plan['persistent_cap'] == c.MAX_ACTIVE == 4, 'Persistent concurrency changed')
    require(datetime.now(timezone.utc) < datetime.fromisoformat(plan['deadline_utc']), 'Supervisor deadline reached')
    for name, digest in plan['source_pins'].items():
        require(c.digest(Path(name)) == digest, 'Supervisor source/evidence changed: ' + name)
    return plan


def gated_status(campaign, root, storage):
    status = ORIGINAL_STATUS(campaign, root)
    # Preserve all original unknown/ambiguous/terminal/peak gates verbatim.
    if status['next_job_id'] is not None:
        budget = storage.storage_report(include_held_job_ids=[status['next_job_id']])
        status['shared_storage'] = budget
        if not budget['allowed']:
            status['next_job_id'] = None
            status['blocked_reason'] = 'waiting_shared_storage'
    return status


def observe_once(args, storage):
    with admission_locks():
        prior = c.status
        c.status = lambda campaign, root: gated_status(campaign, root, storage)
        try:
            return c.advance_once(args, launcher)
        finally:
            c.status = prior


def retryable_observation(status):
    # Unknown snapshots do not permit release; durable intent errors never retry.
    return status['blocked_reason'] == 'unknown_scheduler_state' and not status['issues'] and not status['needs_operator_review_job_ids']


def prepare(campaign, args, storage_module):
    require(not PLAN.exists(), 'Immutable recovery plan already exists')
    latest = max((c.JOURNAL_ROOT/'status').glob('*.json'))
    old = read(latest)
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(old['observed_at'])).total_seconds()
    require(age > 180 and old['blocked_reason'] == 'unknown_scheduler_state', 'Original watcher has not stopped on its reviewed transient state')
    require(old['unknown_job_ids'] == ['31158680'] and not old['issues'], 'Unexpected original stopping condition')
    current = ORIGINAL_STATUS(campaign, c.JOURNAL_ROOT)
    require(not current['unknown_job_ids'] and not current['issues'] and not current['needs_operator_review_job_ids'], 'Current E122 state requires reconciliation')
    require('31158680' in current['endpoint_evidence'], 'Previously uncertain job lacks positive completed endpoint')
    storage = importlib.import_module(storage_module)
    with admission_locks(), c.locked(c.JOURNAL_ROOT):
        budget = storage.storage_report()
        require(budget['allowed'], 'Shared storage not yet admitted')
    source_paths = [SOURCE, PROTOCOL, TEST, Path(c.__file__), Path(launcher.__file__),
        Path(compat.__file__), compat.TEST, Path(storage.__file__),
        ROOT/'ops/exp_scaling/e122_shared_storage_admission.py',
        ROOT/'ops/exp_scaling/e124_storage_admission.py',
        ROOT/'ops/exp_scaling/bounded_checkpoint_zip_metadata_20260911.py']
    for item in budget.get('static_evidence', []):
        source_paths.append(Path(item['path']))
    plan = {'schema':'e122_shared_controller_recovery_v1', 'created_at_utc':c.now(),
        'deadline_utc':(datetime.now(timezone.utc)+timedelta(days=21)).isoformat(),
        'binding':campaign['binding'], 'persistent_cap':4, 'interval_seconds':60,
        'storage_module':storage_module, 'source_pins':{str(p):c.digest(p) for p in source_paths},
        'original_stop_evidence':{'path':str(latest),'sha256':c.digest(latest),'payload':old},
        'positive_reconciliation':current, 'shared_storage_at_prepare':budget,
        'locks':[str(p) for p in LOCKS], 'original_journal_root':str(c.JOURNAL_ROOT),
        'cpu':{'account':'mltheory','partition':'lowprio','node':'node917','mem':'8G','cpus':2,'time':'3-00:00:00'},
        'no_changes_to_existing_gpu_resources_or_science':True}
    c.immutable_json(PLAN, plan)
    (ART/'plan.sha256').write_text(c.digest(PLAN)+'\n')
    return plan


def cpu_record(job_id, held=None):
    result = c.command(['scontrol','show','job','-dd','-o',str(job_id)])
    require(result.returncode == 0, 'Cannot inspect exact registered CPU')
    rec = result.stdout.strip()
    expected = {'JobId':str(job_id),'JobName':'e122-shared-release','Account':'mltheory',
        'Partition':'lowprio','ReqNodeList':'node917','MinMemoryNode':'8G',
        'NumCPUs':'2','TimeLimit':'3-00:00:00','Comment':'e122-recovery-'+c.digest(PLAN)}
    for key, value in expected.items():
        require(c.field(rec,key)==value, 'CPU resource/binding changed: '+key)
    if held:
        require(c.field(rec,'JobState')=='PENDING' and c.field(rec,'Reason')=='JobHeldUser'
            and c.field(rec,'Priority')=='0', 'Exact new CPU is not held')
    return rec


def submit():
    plan = verify_plan()
    require(not (ART/'cpu_submit.intent.json').exists(), 'CPU submission intent already consumed')
    argv = [str(PYTHON),'-u','-B',str(SOURCE),'watch','--plan-sha256',c.digest(PLAN)]
    command = ['sbatch','--parsable','--hold','--job-name=e122-shared-release',
        '--account=mltheory','--partition=lowprio','--nodelist=node917','--mem=8G',
        '--cpus-per-task=2','--nodes=1','--ntasks=1','--time=3-00:00:00','--requeue',
        '--chdir='+str(ROOT),'--output='+str(ART/'cpu-%j.out'),'--error='+str(ART/'cpu-%j.err'),
        '--comment=e122-recovery-'+c.digest(PLAN),'--wrap=exec '+shlex.join(argv)]
    c.immutable_json(ART/'cpu_submit.intent.json', {'at_utc':c.now(),'command':command,'plan_sha256':c.digest(PLAN)})
    result = c.command(command)
    c.immutable_json(ART/'cpu_submit.result.json',{'at_utc':c.now(),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    require(result.returncode==0 and result.stdout.strip().isdigit(),'CPU submission uncertain; never resubmit')
    jid = result.stdout.strip()
    record = cpu_record(jid, held=True)
    c.immutable_json(REG, {'job_id':jid,'plan_sha256':c.digest(PLAN),'held_record':record,'command':command})
    return jid


def activate():
    verify_plan(); reg = read(REG); cpu_record(reg['job_id'],held=True)
    c.immutable_json(ART/'cpu_release.intent.json',{'at_utc':c.now(),'job_id':reg['job_id'],'plan_sha256':c.digest(PLAN)})
    result = c.command(['scontrol','release',reg['job_id']])
    c.immutable_json(ART/'cpu_release.result.json',{'at_utc':c.now(),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    require(result.returncode==0,'CPU release uncertain; do not repeat')


def initial_refill(args, campaign):
    """Bounded first fill from retained authentication while the CPU is held."""
    plan=verify_plan(); reg=read(REG)
    require(campaign['binding']==plan['binding'],'Retained campaign binding changed')
    require(reg['plan_sha256']==c.digest(PLAN),'CPU registration plan differs')
    cpu_record(reg['job_id'],held=True)
    storage=importlib.import_module(plan['storage_module'])
    observations=[]
    with (ART/'singleton.lock').open('a') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _ in range(3):
            verify_plan(); cpu_record(reg['job_id'],held=True)
            result=observe_once(args,storage)
            observations.append(result)
            write(ART/'initial_refill.json',{'at_utc':c.now(),'cpu_job_id':reg['job_id'],
                'cpu_remained_held':True,'plan_sha256':c.digest(PLAN),'observations':observations})
            if result['blocked_reason'] is not None:break
    return observations


def watch(expected):
    plan = verify_plan(expected=expected); reg = read(REG)
    require(reg['plan_sha256']==expected and reg['job_id']==os.environ.get('SLURM_JOB_ID'),'Exact registered CPU required')
    cpu_record(reg['job_id'])
    with (ART/'singleton.lock').open('a') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(ART/'progress.json',{'at_utc':c.now(),'phase':'authenticating','job_id':reg['job_id']})
        args,campaign = authenticate()
        require(campaign['binding']==plan['binding'],'Frozen campaign binding changed')
        storage = importlib.import_module(plan['storage_module'])
        unknowns = 0; started = time.monotonic()
        while True:
            verify_plan(expected=expected)
            try:
                status = observe_once(args,storage)
            except BlockingIOError:
                write(ART/'progress.json',{'at_utc':c.now(),'phase':'waiting_admission_lock','job_id':reg['job_id']})
                time.sleep(10); continue
            write(ART/'status.json',status)
            write(ART/'ready.json',{'at_utc':c.now(),'job_id':reg['job_id'],'plan_sha256':expected,'singleton':True,'blocked_reason':status['blocked_reason']})
            print(json.dumps({'at_utc':c.now(),'released':status.get('last_release_job_id'),
                'running':status['running'],'held':status['staged_held'],'blocked_reason':status['blocked_reason']}),flush=True)
            if status['blocked_reason']=='complete':return 0
            unknowns = unknowns+1 if retryable_observation(status) else 0
            if unknowns>=5 or status['issues'] or status['needs_operator_review_job_ids']:return 2
            if time.monotonic()-started>36*3600:
                write(ART/'cpu_requeue.json',{'at_utc':c.now(),'job_id':reg['job_id'],'plan_sha256':expected})
                result=c.command(['scontrol','requeue',reg['job_id']])
                require(result.returncode==0,'CPU self-requeue uncertain');return 0
            time.sleep(plan['interval_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['submit','activate','watch'])
    parser.add_argument('--plan-sha256')
    parsed=parser.parse_args()
    if parsed.phase=='submit':print(submit())
    elif parsed.phase=='activate':activate()
    else:raise SystemExit(watch(parsed.plan_sha256))
