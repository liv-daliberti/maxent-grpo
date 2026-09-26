#!/usr/bin/env python3
"""Stage a gated CPU-only E124 replacement; retire the old CPU only after readiness.

prepare/status are scheduler-read-only. stage/release/handoff/adopt are explicit
root actions. waiter publishes a heartbeat and waits at most60minutes for a gate.
Neither this helper nor its waiter releases, cancels, or requeues a GPU job.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(os.environ.get('OAT_ZERO_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
BASE = ROOT / 'var/artifacts/e124_qwen7b_three_level'
ART = BASE / 'controller_recovery_20260909_1940/cpu_waiter_v2'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
READY = ART / 'ready.json'
GATE = ART / 'gate.json'
SOURCE = Path(__file__).resolve()
V2 = BASE / 'controller_v2.py'
CANONICAL_TX = BASE / 'transaction.json'
CANONICAL_PLAN = BASE / 'plan.json'
PROTOCOL = ROOT / 'paper/preregistration/e124_cpu_controller_recovery_20260909.md'
OLD = 31162084
POOL = frozenset({'node009', 'node010', 'node012', 'node013', 'node014', 'node015', 'node016', 'node915', 'node917'})
PYTHON = '/usr/local/anaconda3/2024.02/bin/python3'
MAX_WAIT_SECONDS = 3600
READY_MAX_AGE = 45
TERMINAL = {'CANCELLED', 'COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value, *, new=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    target = path if new else path.with_name(path.name + f'.{os.getpid()}.tmp')
    with target.open('x' if new else 'w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    if not new:
        os.replace(target, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def event(tx, label, **values):
    tx.setdefault('events', []).append({'at': now().isoformat(), 'event': label, **values})
    write(TX, tx)


def clean_environment():
    return {k: v for k, v in os.environ.items()
            if not k.startswith(('OAT_ZERO_', 'SBATCH_', 'SLURM_'))
            and k not in {'SAVE_PATH', 'RUN_STAMP', 'ROOT_DIR', 'PYTHONPATH',
                          'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'}}


def command(args):
    kwargs = {'env': clean_environment()} if Path(args[0]).name == 'sbatch' else {}
    return subprocess.run(args, text=True, capture_output=True, timeout=120, check=True, **kwargs)


def v2():
    os.environ['OAT_ZERO_REPO_ROOT'] = str(ROOT)
    spec = importlib.util.spec_from_file_location('e124_unchanged_v2', V2)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pins(plan):
    require(seal({k: v for k, v in plan.items() if k != 'sha256'}) == plan['sha256'], 'recovery plan changed')
    for path, expected in plan['pins'].items():
        require(digest(path) == expected, 'recovery input changed: ' + path)
    canonical = read(CANONICAL_PLAN)
    require(canonical['plan_sha256'] == plan['science_plan_sha256'], 'scientific plan changed')
    require(now() < datetime.fromisoformat(plan['deadline_utc']), 'original E124 deadline reached')


def load():
    plan = read(PLAN); pins(plan)
    tx = read(TX) if TX.exists() else {'schema': 'e124_cpu_waiter_transaction_v1', 'plan_sha256': plan['sha256'], 'status': 'prepared', 'events': []}
    require(tx['plan_sha256'] == plan['sha256'], 'transaction/plan identity differs')
    return plan, tx


@contextmanager
def recovery_lock():
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'mutation.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def original_locks():
    # Never unlink a lock: a stuck old process may still own its inode.
    with (BASE / 'supervisor.lock').open('r') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (BASE / 'controller.lock').open('r') as controller:
            fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def all_gpu_holds(plan):
    x = v2(); canonical = read(CANONICAL_PLAN); tx = read(CANONICAL_TX)
    require(canonical['plan_sha256'] == tx['plan_sha256'] == plan['science_plan_sha256'], 'science identity changed')
    require(set(tx['rows']) == set(plan['gpu_ids']), 'GPU cell inventory changed')
    for row in [x.systems_row(canonical), *canonical['cells']]:
        item = tx['rows'][row['cell_id']]
        require(item['job_id'] == plan['gpu_ids'][row['cell_id']] and item['status'] == 'held', 'GPU is no longer the audited held cell')
        x.audit_job(canonical, row, item, held=True)
    return tx


def old_identity(plan):
    x = v2(); record = x.show(OLD); current = x.fields(record); before = plan['old_fields']
    require(x.submit_tokens(record) == plan['old_submit_tokens'], 'old CPU submission changed')
    for name in ('JobId', 'JobName', 'UserId', 'Account', 'Partition', 'MinMemoryNode', 'Command', 'WorkDir', 'StartTime', 'Restarts'):
        require(current.get(name) == before.get(name), 'old CPU identity changed: ' + name)
    require(current.get('UserId', '').endswith(f'({os.getuid()})') and current.get('JobName') == 'e124-controller', 'old CPU ownership/name differs')
    require('gres/gpu' not in current.get('ReqTRES', ''), 'old allocation is no longer CPU-only')
    return current


def inactive_old():
    queued = command(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T']).stdout
    if any(row.split('|')[0] == str(OLD) for row in queued.splitlines()):
        return None
    result = command(['sacct', '-X', '-n', '-P', '-j', str(OLD), '--format=JobID,State,ExitCode']).stdout
    rows = [r.split('|') for r in result.splitlines() if r.split('|')[0] == str(OLD)]
    require(len(rows) == 1, 'old CPU accounting is missing or ambiguous')
    state = rows[0][1].split()[0].rstrip('+')
    require(state in TERMINAL, 'old CPU is not accounted inactive')
    return {'job_id': OLD, 'state': state, 'exit_code': rows[0][2], 'at': now().isoformat()}


def audit_cpu(plan, tx, *, held=False, running=False):
    x = v2(); jid = tx['job_id']; record = x.show(jid); fields = x.fields(record)
    expected = {'JobId': str(jid), 'JobName': 'e124-cpu-waiter', 'Account': 'mltheory',
                'Partition': 'lowprio', 'MinMemoryNode': '8G', 'TimeLimit': '1-01:10:00',
                'Requeue': '1', 'Comment': plan['comment'], 'WorkDir': str(ROOT)}
    for key, value in expected.items():
        require(fields.get(key) == value, 'new CPU field differs: ' + key)
    require(fields.get('UserId', '').endswith(f'({os.getuid()})'), 'new CPU ownership differs')
    require(fields.get('NumCPUs') in {'1', '2'} and fields.get('CPUs/Task') == '1', 'new CPU count differs')
    require(fields.get('NumNodes') in {'1', '1-1'}, 'new CPU node count differs')
    require('gres/gpu' not in fields.get('ReqTRES', '') and 'gres/gpu' not in fields.get('AllocTRES', ''), 'new waiter acquired a GPU')
    require(set(command(['scontrol', 'show', 'hostnames', fields['ReqNodeList']]).stdout.split()) == POOL, 'CPU pool differs')
    require(x.submit_tokens(record) == plan['command'], 'new CPU full submission differs')
    if held:
        require(fields.get('JobState') == 'PENDING' and fields.get('Reason') == 'JobHeldUser' and fields.get('Priority') == '0', 'owned held CPU required')
    if running:
        require(fields.get('JobState') == 'RUNNING' and fields.get('NodeList') in POOL, 'allocated CPU waiter required')
    return fields


def ready_waiter(plan, tx):
    audit_cpu(plan, tx, running=True)
    ready = read(READY)
    require(ready.get('job_id') == tx['job_id'] and ready.get('plan_sha256') == plan['sha256'], 'wrong waiter readiness')
    require(ready.get('phase') == 'waiting_gate', 'waiter is not waiting for handoff')
    require(0 <= (now() - datetime.fromisoformat(ready['at'])).total_seconds() <= READY_MAX_AGE, 'waiter readiness is stale')
    require((datetime.fromisoformat(ready['expires_at']) - now()).total_seconds() > 120, 'waiter expiry is too close for handoff')
    return ready


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'immutable waiter preparation exists')
    ART.mkdir(parents=True, exist_ok=True); (ART / 'logs').mkdir(exist_ok=True)
    x = v2(); canonical = read(CANONICAL_PLAN); oldtx = read(CANONICAL_TX)
    require(oldtx['controller']['job_id'] == OLD, 'old canonical CPU identity differs')
    prior = read(BASE / 'controller_recovery_20260909_1940/plan.json')
    require(prior['full_v2_pin_verification'] == 'passed' and prior['original_plan_sha256'] == canonical['plan_sha256'], 'completed full pin audit required')
    require(prior['source_sha256'] == digest(V2), 'original verified v2 source changed')
    frozen = ART / 'waiter_control.py'; frozen.write_bytes(SOURCE.read_bytes()); frozen.chmod(0o444)
    script = ART / 'waiter.slurm'
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport PATH=/usr/bin:/bin\nexport PYTHONDONTWRITEBYTECODE=1\nexport OAT_ZERO_REPO_ROOT=' + str(ROOT) + '\ncd ' + str(ROOT) + '\nexec ' + PYTHON + ' ' + str(frozen) + ' waiter\n')
    script.chmod(0o444)
    raw = x.show(OLD)
    plan = {'schema': 'e124_cpu_waiter_plan_v1', 'at': now().isoformat(), 'old_job_id': OLD,
            'science_plan_sha256': canonical['plan_sha256'], 'deadline_utc': canonical['deadline_utc'],
            'old_controller': oldtx['controller'], 'old_fields': x.fields(raw), 'old_submit_tokens': x.submit_tokens(raw),
            'gpu_ids': {key: item['job_id'] for key, item in oldtx['rows'].items()},
            'comment': 'e124-cpu-recovery-20260909-old31162084', 'max_wait_seconds': MAX_WAIT_SECONDS,
            'pins': {str(p): digest(p) for p in (SOURCE, frozen, script, V2, BASE / 'controller.slurm', PROTOCOL, CANONICAL_PLAN)}}
    plan['command'] = ['sbatch', '--parsable', '--hold', '--job-name=e124-cpu-waiter',
        '--account=mltheory', '--partition=lowprio', '--nodelist=' + ','.join(sorted(POOL)),
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=8G', '--gres=none',
        '--time=1-01:10:00', '--requeue', '--comment=' + plan['comment'], '--chdir=' + str(ROOT),
        '--output=' + str(ART / 'logs/waiter-%j.out'), '--error=' + str(ART / 'logs/waiter-%j.err'), str(script)]
    old_identity(plan); require(len(plan['gpu_ids']) == 31, 'expected all 31 held GPU cells')
    all_gpu_holds(plan)
    result = command([plan['command'][0], '--test-only', *[a for a in plan['command'][1:] if a != '--hold']])
    plan['test_only'] = {'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode}
    plan['sha256'] = seal(plan); write(PLAN, plan, new=True)
    return {'status': 'prepared', 'gpu_jobs_held': 31, 'forecast': plan['test_only'], 'plan': str(PLAN)}


def stage():
    plan, tx = load()
    if tx.get('job_id'):
        held = tx.get('status') in {'held', 'held_unverified'}
        audit_cpu(plan, tx, held=held)
        if held and tx.get('status') != 'held':
            tx['status'] = 'held'; event(tx, 'held_waiter_audit_reconciled')
        return {'status': tx['status'], 'job_id': tx['job_id']}
    require(not tx.get('submit_intent'), 'uncertain held submission; locate and adopt exact waiter')
    old_identity(plan); all_gpu_holds(plan)
    tx['submit_intent'] = True; event(tx, 'held_waiter_submit_intent')
    result = command(plan['command']).stdout.strip()
    require(re.fullmatch(r'\d+(;[^\s]+)?', result) is not None, 'ambiguous waiter submission')
    tx.update(job_id=int(result.split(';')[0]), status='held_unverified'); event(tx, 'held_waiter_identity_recorded')
    audit_cpu(plan, tx, held=True)
    tx['status'] = 'held'; event(tx, 'held_waiter_audited')
    return {'status': 'held', 'job_id': tx['job_id']}


def adopt(job_id):
    plan, tx = load()
    require(tx.get('submit_intent') and not tx.get('job_id'), 'adopt only resolves ambiguous submission')
    proposed = copy.deepcopy(tx); proposed['job_id'] = job_id
    audit_cpu(plan, proposed, held=True)
    tx.update(job_id=job_id, status='held'); event(tx, 'exact_held_waiter_adopted')
    return {'status': 'held', 'job_id': job_id}


def release():
    plan, tx = load(); require(tx.get('job_id'), 'stage first')
    fields = audit_cpu(plan, tx)
    if tx.get('release_intent') and fields.get('Priority') != '0':
        tx['status'] = 'released'; event(tx, 'waiter_release_reconciled')
        return {'status': 'released', 'job_id': tx['job_id']}
    audit_cpu(plan, tx, held=True); old_identity(plan); all_gpu_holds(plan)
    tx['release_intent'] = True; event(tx, 'CPU_waiter_release_intent_old_CPU_preserved')
    command(['scontrol', 'release', str(tx['job_id'])])
    require(audit_cpu(plan, tx).get('Priority') != '0', 'waiter release not confirmed')
    tx['status'] = 'released'; event(tx, 'CPU_waiter_released_old_CPU_preserved')
    return {'status': 'released', 'job_id': tx['job_id']}


def promote_controller(plan, tx):
    canonical = read(CANONICAL_TX); old = canonical['controller']
    require(canonical['plan_sha256'] == plan['science_plan_sha256'], 'canonical plan differs')
    before_rows = copy.deepcopy(canonical['rows'])
    if old['job_id'] == tx['job_id']:
        require(old.get('recovery_plan') == str(PLAN) and old['command'] == plan['command'], 'unowned CPU promotion')
        return
    require(old['job_id'] == OLD, 'canonical controller changed')
    replacement = copy.deepcopy(old)
    replacement.update(job_id=tx['job_id'], command=plan['command'], status='released',
                       recovery_plan=str(PLAN), predecessor_job_id=OLD)
    canonical.setdefault('controller_history', []).append(copy.deepcopy(old))
    canonical['controller'] = replacement
    require(canonical['rows'] == before_rows, 'CPU handoff changed GPU rows')
    tx['promotion_intent'] = True; event(tx, 'canonical_CPU_promotion_intent')
    write(CANONICAL_TX, canonical)
    require(read(CANONICAL_TX) == canonical, 'canonical CPU promotion readback differs')


def old_monitoring_state():
    canonical = read(CANONICAL_TX)
    require(canonical['controller']['job_id'] == OLD, 'old CPU identity changed before automatic decision')
    ready = read(BASE / 'controller_ready.json')
    require(ready.get('job_id') == OLD, 'old readiness identity differs')
    age = (now() - datetime.fromisoformat(ready['at'])).total_seconds()
    require(age >= 0, 'old readiness is dated in the future')
    if age <= 300 and canonical.get('status') != 'needs_review':
        return 'recovered'
    return 'stale' if age >= 900 else 'waiting'


def enable_auto():
    plan, tx = load()
    require(not GATE.exists() and not tx.get('cancel_intent'), 'handoff already began')
    require(old_monitoring_state() == 'stale', 'automatic recovery requires a stale old controller')
    old_identity(plan); all_gpu_holds(plan)
    tx['auto_handoff'] = True; event(tx, 'root_authorized_automatic_CPU_handoff_armed')
    return {'status': 'auto_handoff_enabled', 'job_id': tx.get('job_id')}


def handoff(*, automatic=False):
    plan, tx = load()
    if GATE.exists():
        require(read(GATE).get('job_id') == tx['job_id'] and read(GATE).get('plan_sha256') == plan['sha256'], 'unowned existing gate')
        return {'status': 'gate_open', 'job_id': tx['job_id']}
    ready_waiter(plan, tx)
    if automatic:
        require(tx.get('auto_handoff'), 'automatic CPU handoff was not enabled by root')
        if not tx.get('cancel_intent') and read(CANONICAL_TX)['controller']['job_id'] == OLD:
            state = old_monitoring_state()
            if state != 'stale':
                return {'status': 'old_controller_recovered' if state == 'recovered' else 'waiting_old_controller_stale'}
    all_gpu_holds(plan)
    inactive = inactive_old()
    if inactive is None:
        old_identity(plan)
        if not tx.get('cancel_intent'):
            ready_waiter(plan, tx)
            if automatic:
                state = old_monitoring_state()
                if state != 'stale':
                    return {'status': 'old_controller_recovered' if state == 'recovered' else 'waiting_old_controller_stale'}
            tx['cancel_intent'] = True; event(tx, 'cancel_exact_old_CPU_after_allocated_waiter_ready')
            command(['scancel', str(OLD)])
        return {'status': 'waiting_old_CPU_inactive', 'old_job_id': OLD, 'job_id': tx['job_id']}
    try:
        with original_locks():
            ready_waiter(plan, tx); all_gpu_holds(plan); pins(plan)
            require(inactive_old() is not None, 'old CPU reappeared before gate')
            promote_controller(plan, tx)
            gate = {'schema': 'e124_cpu_waiter_gate_v1', 'at': now().isoformat(), 'job_id': tx['job_id'],
                    'old_job_id': OLD, 'plan_sha256': plan['sha256'], 'source_sha256': plan['pins'][str(V2)],
                    'old_inactive': inactive}
            write(GATE, gate, new=True)
            tx['status'] = 'gate_open'; event(tx, 'CPU_gate_open_GPU_checks_unchanged')
    except BlockingIOError:
        return {'status': 'waiting_old_locks', 'job_id': tx['job_id']}
    return {'status': 'gate_open', 'job_id': tx['job_id']}


def gate_valid(plan, tx, gate):
    require(gate.get('job_id') == tx['job_id'] and gate.get('old_job_id') == OLD
            and gate.get('plan_sha256') == plan['sha256'] and gate.get('source_sha256') == plan['pins'][str(V2)], 'gate identity differs')
    canonical = read(CANONICAL_TX)
    require(canonical['controller']['job_id'] == tx['job_id'] and canonical['controller'].get('recovery_plan') == str(PLAN), 'canonical CPU promotion missing')
    require(inactive_old() is not None, 'old CPU is still active')
    with original_locks():
        pass


class WaiterExpired(RuntimeError):
    pass


def waiter():
    started = time.monotonic()
    plan, tx = load(); require(str(tx.get('job_id')) == os.environ.get('SLURM_JOB_ID'), 'waiter must run in its exact recorded allocation')
    expires = now() + timedelta(seconds=plan['max_wait_seconds'])
    def expire(signum, frame):
        raise WaiterExpired('allocated CPU waiter exceeded its 60-minute gate window')
    previous_handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, plan['max_wait_seconds'] - (time.monotonic() - started)))
    try:
        audit_cpu(plan, tx, running=True)
        while time.monotonic() - started < plan['max_wait_seconds']:
            if GATE.exists():
                pins(plan); gate_valid(plan, tx, read(GATE))
                write(READY, {'at': now().isoformat(), 'job_id': tx['job_id'], 'plan_sha256': plan['sha256'], 'phase': 'starting_unchanged_v2'})
                os.environ.update(OAT_ZERO_REPO_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1', PATH='/usr/bin:/bin')
                signal.setitimer(signal.ITIMER_REAL, 0)
                os.execv(PYTHON, [PYTHON, str(V2), 'watch'])
            write(READY, {'at': now().isoformat(), 'job_id': tx['job_id'], 'plan_sha256': plan['sha256'],
                          'phase': 'waiting_gate', 'expires_at': expires.isoformat(), 'pid': os.getpid()})
            current = read(TX)
            if current.get('auto_handoff'):
                try:
                    with recovery_lock():
                        outcome = handoff(automatic=True)
                except BlockingIOError:
                    outcome = {'status': 'waiting_existing_locks'}
                if outcome['status'] == 'old_controller_recovered':
                    write(READY, {'at': now().isoformat(), 'job_id': tx['job_id'], 'plan_sha256': plan['sha256'], 'phase': 'exited_old_controller_recovered'})
                    return outcome
            time.sleep(10)
    except WaiterExpired:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    write(READY, {'at': now().isoformat(), 'job_id': tx['job_id'], 'plan_sha256': plan['sha256'], 'phase': 'expired_without_gate'})
    return {'status': 'waiter_expired_without_gate'}


def status():
    plan, tx = load()
    return {'status': tx.get('status'), 'job_id': tx.get('job_id'), 'old_job_id': OLD,
            'ready': read(READY) if READY.exists() else None, 'gate_open': GATE.exists()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'stage', 'release', 'handoff', 'enable-auto', 'adopt', 'waiter', 'status'))
    parser.add_argument('--job-id', type=int); args = parser.parse_args(argv)
    if args.action in ('prepare', 'waiter', 'status'):
        result = globals()[args.action]()
    else:
        with recovery_lock():
            if args.action == 'enable-auto':
                result = enable_auto()
            elif args.action == 'adopt':
                require(args.job_id is not None, '--job-id is required'); result = adopt(args.job_id)
            else:
                result = globals()[args.action]()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
