#!/usr/bin/env python3
"""Gated CPU-only E124 handoff to the additive array storage reader.

Reuse audited CPU staging/readiness/promotion and the original E124 watcher.
Only the read-only storage callable and shared-lock scope change in memory.
All GPU qualification, release journals, retry rules and scientific pins remain.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import fcntl

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_e124_cpu_controller_20260909 as h
import e124_storage_with_modebench_array_20260911 as storage

ROOT = h.ROOT
SOURCE = Path(__file__).resolve()
TEST = ROOT / 'tests/test_handoff_e124_array_storage_20260911.py'
ART = ROOT / 'var/artifacts/e124_array_storage_handoff_20260911'
SHARED_LOCK = ROOT / 'var/artifacts/shared_storage_admission_20260911.lock'
OLD = 31164037
BASE_PIN = 'e36f1d8c92de68a4e4d5329e5debc1313a74d5a7668a9e4601a2834a1c037881'
STORAGE_PIN = '63370fc4e7d7727bef99b9c3775ead8b06ef069144e59a27d4e6dfcb024883d2'

# Keep the existing helper mechanics but use an isolated transaction and gate.
h.ART = ART
h.PLAN = ART / 'plan.json'
h.TX = ART / 'transaction.json'
h.READY = ART / 'ready.json'
h.GATE = ART / 'gate.json'
h.OLD = OLD
h.SOURCE = SOURCE


@contextmanager
def shared_lock():
    # The inode is pre-existing and shared with the E122/E118 release owners.
    with SHARED_LOCK.open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def old_identity(plan):
    x = h.v2(); raw = x.show(OLD); current = x.fields(raw)
    h.require(x.submit_tokens(raw) == plan['old_submit_tokens'], 'old CPU submission changed')
    for name in ('JobId', 'JobName', 'UserId', 'Account', 'Partition', 'MinMemoryNode',
                 'Command', 'WorkDir', 'StartTime', 'Restarts'):
        h.require(current.get(name) == plan['old_fields'].get(name), 'old CPU identity changed: ' + name)
    h.require(current.get('JobName') == 'e124-cpu-waiter'
        and current.get('UserId', '').endswith(f'({os.getuid()})')
        and 'gres/gpu' not in current.get('ReqTRES', ''), 'old target is not exact owned CPU waiter')
    return current


def audit_gpu_rows(plan):
    """Permit systems to start naturally; preserve every GPU identity and intent."""
    x = h.v2(); canonical, tx = h.read(h.CANONICAL_PLAN), h.read(h.CANONICAL_TX)
    h.require(canonical['plan_sha256'] == tx['plan_sha256'] == plan['science_plan_sha256'], 'science plan differs')
    h.require(set(tx['rows']) == set(plan['gpu_ids']), 'GPU cell inventory changed')
    h.require(tx.get('status') != 'needs_review', 'existing controller requires review')
    for row in [x.systems_row(canonical), *canonical['cells']]:
        item = tx['rows'][row['cell_id']]
        h.require(item['job_id'] == plan['gpu_ids'][row['cell_id']], 'canonical GPU identity changed')
        # This handoff is specific to the current pre-science state. If systems
        # completes, pause for review instead of assuming old held conditions.
        expected = 'released' if row['cell_id'] == 'systems' else 'held'
        h.require(item['status'] == expected, 'GPU transaction advanced during prepared handoff')
        x.audit_job(canonical, row, item, held=(expected == 'held'))
    return tx


def audit_cpu(plan, tx, *, held=False, running=False):
    # Avoid the e124- prefix until canonical promotion: the old watcher rejects
    # unregistered e124-* jobs, including a newly staged replacement CPU.
    x = h.v2(); raw = x.show(tx['job_id']); fields = x.fields(raw)
    expected = {'JobId':str(tx['job_id']), 'JobName':'array-storage-cpu', 'Account':'mltheory',
        'Partition':'lowprio', 'MinMemoryNode':'8G', 'TimeLimit':'1-01:10:00', 'Requeue':'1',
        'Comment':plan['comment'], 'WorkDir':str(ROOT), 'CPUs/Task':'1'}
    for key,value in expected.items():
        h.require(fields.get(key)==value, 'replacement CPU field differs: '+key)
    h.require(fields.get('UserId','').endswith(f'({os.getuid()})'), 'replacement ownership differs')
    h.require(fields.get('NumCPUs') in {'1','2'} and fields.get('NumNodes') in {'1','1-1'}, 'CPU count differs')
    h.require('gres/gpu' not in fields.get('ReqTRES','') and 'gres/gpu' not in fields.get('AllocTRES',''), 'replacement requested GPU')
    h.require(set(h.command(['scontrol','show','hostnames',fields['ReqNodeList']]).stdout.split())==h.POOL, 'CPU pool differs')
    h.require(x.submit_tokens(raw)==plan['command'], 'replacement CPU full submission differs')
    if held:
        h.require(fields.get('JobState')=='PENDING' and fields.get('Reason')=='JobHeldUser'
            and fields.get('Priority')=='0', 'owned held CPU required')
    if running:
        h.require(fields.get('JobState')=='RUNNING' and fields.get('NodeList') in h.POOL, 'allocated CPU required')
    return fields


h.old_identity = old_identity
h.all_gpu_holds = audit_gpu_rows
h.audit_cpu = audit_cpu


def prepare():
    h.require(not h.PLAN.exists() and not h.TX.exists(), 'handoff already prepared')
    h.require(h.digest(h.V2) == BASE_PIN and h.digest(Path(storage.__file__)) == STORAGE_PIN,
              'reviewed controller/storage source changed')
    storage.pins()
    x = h.v2(); canonical = h.read(h.CANONICAL_PLAN)
    with shared_lock(), x.lock():
        oldtx = h.read(h.CANONICAL_TX)
        h.require(oldtx['controller']['job_id'] == OLD, 'old controller identity changed')
        x.verify(canonical, oldtx)
        raw = x.show(OLD)
        plan = dict(schema='e124_array_storage_cpu_handoff_v1', at=h.now().isoformat(), old_job_id=OLD,
            science_plan_sha256=canonical['plan_sha256'], deadline_utc=canonical['deadline_utc'],
            old_controller=copy.deepcopy(oldtx['controller']), old_fields=x.fields(raw),
            old_submit_tokens=x.submit_tokens(raw), gpu_ids={k:v['job_id'] for k,v in oldtx['rows'].items()},
            comment='e124-array-storage-20260911-old31164037', max_wait_seconds=h.MAX_WAIT_SECONDS)
        old_identity(plan); audit_gpu_rows(plan)
        report = storage.controller_storage_report(canonical, oldtx)
        h.require(not report['errors'], 'corrected reader is not healthy')
        h.require(report['own_live_count'] == 1 and report['blocked_reason'] == 'own_concurrency_cap',
                  'existing systems concurrency gate differs')
    ART.mkdir(parents=True, exist_ok=True); (ART / 'logs').mkdir(exist_ok=True)
    script = ART / 'waiter.slurm'
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport PATH=/usr/bin:/bin\nexport PYTHONDONTWRITEBYTECODE=1\nexport OAT_ZERO_REPO_ROOT=' + str(ROOT) + '\ncd ' + str(ROOT) + '\nexec ' + h.PYTHON + ' -B ' + str(SOURCE) + ' waiter\n')
    script.chmod(0o444)
    plan['command'] = ['sbatch', '--parsable', '--hold', '--job-name=array-storage-cpu',
        '--account=mltheory', '--partition=lowprio', '--nodelist=' + ','.join(sorted(h.POOL)),
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=8G', '--gres=none',
        '--time=1-01:10:00', '--requeue', '--comment=' + plan['comment'], '--chdir=' + str(ROOT),
        '--output=' + str(ART / 'logs/waiter-%j.out'), '--error=' + str(ART / 'logs/waiter-%j.err'), str(script)]
    paths = (SOURCE, TEST, Path(h.__file__), h.V2, Path(storage.__file__),
        ROOT / 'tests/test_e124_storage_with_modebench_array_20260911.py',
        Path(storage.throttle.__file__), ROOT / 'tests/test_modebench_array_throttle2_storage_20260911.py',
        Path(storage.arrays.__file__), Path(storage.arrays.original.__file__), Path(storage.base.__file__),
        Path(storage.arrays.bounded_zip.__file__), h.CANONICAL_PLAN, script)
    plan['pins'] = {str(p):h.digest(p) for p in paths}
    plan['storage_at_prepare'] = report
    plan['shared_lock'] = str(SHARED_LOCK)
    plan['original_cap'] = canonical['max_active']
    h.require(plan['original_cap'] == 1, 'original E124 concurrency differs')
    plan['sha256'] = h.seal(plan); h.write(h.PLAN, plan, new=True)
    return {'status':'prepared', 'plan':str(h.PLAN), 'sha256':plan['sha256'],
            'old_job_id':OLD, 'gpu_jobs_changed':False}


def release():
    # Do not repeat an uncertain release while it still appears held.
    plan, tx = h.load()
    if tx.get('release_intent'):
        fields = h.audit_cpu(plan, tx)
        if fields.get('Priority') == '0':
            return {'status':'uncertain_CPU_release_requires_reconciliation', 'job_id':tx['job_id']}
    return h.release()


def handoff():
    plan, tx = h.load()
    if h.GATE.exists():
        h.gate_valid(plan, tx, h.read(h.GATE))
        return {'status':'gate_open', 'job_id':tx['job_id']}
    h.ready_waiter(plan, tx)
    with shared_lock():
        inactive = h.inactive_old()
        if inactive is None:
            x = h.v2()
            with x.lock():
                old_identity(plan); audit_gpu_rows(plan); h.ready_waiter(plan, tx)
                if not tx.get('cancel_intent'):
                    tx['cancel_intent'] = True; h.event(tx, 'cancel_exact_old_CPU_after_new_waiter_readiness')
                    h.command(['scancel', str(OLD)])
            return {'status':'waiting_old_CPU_inactive', 'job_id':tx['job_id'], 'old_job_id':OLD}
        with h.original_locks():
            h.ready_waiter(plan, tx); audit_gpu_rows(plan); h.pins(plan)
            h.require(h.inactive_old() is not None, 'old CPU returned before gate')
            h.promote_controller(plan, tx)
            gate = dict(schema='e124_cpu_waiter_gate_v1', at=h.now().isoformat(), job_id=tx['job_id'],
                old_job_id=OLD, plan_sha256=plan['sha256'], source_sha256=plan['pins'][str(h.V2)], old_inactive=inactive)
            h.write(h.GATE, gate, new=True)
            tx['status']='gate_open'; h.event(tx, 'array_storage_CPU_gate_open_GPU_state_preserved')
    return {'status':'gate_open', 'job_id':tx['job_id']}


def configure_controller(plan, controller):
    """Process-local changes only; the original watcher retains its singleton."""
    original_lock = controller.lock
    @contextmanager
    def coordinated_lock():
        with shared_lock(), original_lock():
            h.pins(plan)
            yield
    controller.lock = coordinated_lock
    controller.storage_report = storage.controller_storage_report
    return controller


def waiter():
    plan, tx = h.load()
    h.require(str(tx.get('job_id')) == os.environ.get('SLURM_JOB_ID'), 'waiter requires exact registered CPU')
    started = time.monotonic(); expires = h.now() + timedelta(seconds=plan['max_wait_seconds'])
    def expired(*args): raise h.WaiterExpired('new CPU exceeded gate window')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, plan['max_wait_seconds'])
    try:
        h.audit_cpu(plan, tx, running=True)
        while time.monotonic()-started < plan['max_wait_seconds']:
            if h.GATE.exists():
                h.pins(plan)
                try:
                    h.gate_valid(plan, tx, h.read(h.GATE))
                except BlockingIOError:
                    time.sleep(1)
                    continue
                h.write(h.READY, dict(at=h.now().isoformat(), job_id=tx['job_id'], plan_sha256=plan['sha256'], phase='starting_array_storage_v2'))
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
                controller = configure_controller(plan, h.v2())
                return {'status':'watcher_exited', 'returncode':controller.watch()}
            h.write(h.READY, dict(at=h.now().isoformat(), job_id=tx['job_id'], plan_sha256=plan['sha256'],
                phase='waiting_gate', expires_at=expires.isoformat(), pid=os.getpid()))
            time.sleep(10)
    except h.WaiterExpired:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    h.write(h.READY, dict(at=h.now().isoformat(), job_id=tx['job_id'], phase='expired_without_gate'))
    return {'status':'waiter_expired_without_gate'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','stage','release','handoff','waiter','status','adopt'))
    parser.add_argument('--job-id', type=int); args=parser.parse_args()
    if args.action in ('prepare','waiter','status'):
        result = h.status() if args.action=='status' else globals()[args.action]()
    else:
        with h.recovery_lock():
            if args.action=='stage': result=h.stage()
            elif args.action=='adopt':
                h.require(args.job_id is not None, '--job-id required'); result=h.adopt(args.job_id)
            else: result=globals()[args.action]()
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
