#!/usr/bin/env python3
"""Prepared-only by default: inherit three Pantry cells after their old guard expires."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import guard_campaign_timeouts_20260908 as prior
import guard_mathir_hourly_timeouts_20260909 as checkpoints
import accelerate_a5000_completion_20260909 as runtime

base, recovery = prior.base, prior.recovery
ROOT = base.ROOT
ART = ROOT / 'var/artifacts/e118_pantry_successor_guard_20260910'
PLAN, TX = ART / 'plan.json', ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e118_pantry_successor_guard_20260910.md'
OLD_PLAN, OLD_TX, OLD_LOCK = prior.PLAN, prior.TX, prior.LOCK
LOCK, REGISTRATION = ART / 'singleton.lock', ART / 'supervisor.json'
IDS, OLD_CPU = (31048143, 31048144, 31048145), 31159945
AGGREGATE = ROOT / 'var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json'
CPU_FIELDS = ('UserId', 'Account', 'Partition', 'ReqNodeList', 'MinMemoryNode',
              'NumCPUs', 'NumTasks', 'Requeue', 'TimeLimit', 'Command', 'Comment', 'JobName')
require = prior.require


def utcnow():
    return datetime.now(timezone.utc)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as handle:
        handle.write(base.encoded(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(tx, message):
    tx['updated_at_utc'] = base.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'event': message})
    atomic(TX, tx)
    print(json.dumps({'at': tx['updated_at_utc'], 'event': message}), flush=True)


def command(parts, *, check=True):
    return subprocess.run(parts, capture_output=True, text=True, check=check, timeout=45)


def mapping():
    source = json.loads(base.LEDGER.read_text())['runs']
    aggregate = json.loads(AGGREGATE.read_text())['runs']
    require(len(aggregate) == 150, 'E118 scientific denominator changed')
    result = {}
    for job in IDS:
        rows = [r for r in source if r['job_id'] == job]
        matches = [r for r in aggregate if r['job_id'] == job]
        require(len(rows) == len(matches) == 1 and matches[0] == dict(rows[0], scale='qwen3b'),
                f'{job}: authoritative E118 mapping changed')
        result[job] = {'cohort': 'e118', 'identity': {k: rows[0][k] for k in base.IDENTITY}}
    return result


def checkpoint_and_writer(item):
    require(not recovery.complete(Path(item['identity']['run_dir'])), 'Terminal receipt exists')
    selected, rejected = recovery.select_latest_checkpoint(Path(item['identity']['run_dir']))
    require(selected is not None, 'No complete model/optimizer checkpoint')
    detail = dict(checkpoints.checked_checkpoint(selected), rejected=rejected)
    require(0 < detail['step'] < 3072, 'Checkpoint outside unfinished training range')
    require(not any(int(Path(p).name[5:]) >= detail['step'] for p in detail['rejected']),
            'Newer incomplete checkpoint requires manual review')
    recovery.no_other_writer({'old_job_id': item['job_id'], 'new_job_id': item['job_id'],
                              'identity': item['identity']}, recovery.active_writers())
    return detail


def window(plan, now=None):
    now = now or utcnow()
    if now < datetime.fromisoformat(plan['not_before_utc']):
        return 'waiting_for_predecessor_deadline'
    if now >= datetime.fromisoformat(plan['deadline_utc']):
        return 'deadline_reached'
    return 'eligible'


def cpu_command():
    return ['sbatch', '--parsable', '--hold', '--job-name=e118-pantry-successor-guard',
            '--account=mltheory', '--partition=lowprio', '--nodelist=node915,node917',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=2G', '--gres=none',
            '--time=1-01:10:00', '--requeue', '--export=NONE', f'--chdir={ROOT}',
            f'--output={ART}/supervisor-%j.out', f'--error={ART}/supervisor-%j.err',
            '--comment=e118-pantry-successor-20260910', str(ART / 'supervisor.slurm')]


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Existing preparation must not be overwritten')
    source = json.loads(OLD_PLAN.read_text()); old = json.loads(OLD_TX.read_text())
    require(old['plan_sha256'] == recovery.digest(OLD_PLAN), 'Predecessor plan binding changed')
    identities = mapping(); rows = []
    for job in IDS:
        item = copy.deepcopy(next(r for r in source['rows'] if r['job_id'] == job))
        require(item['identity'] == identities[job]['identity'], 'Predecessor science differs')
        record = prior.show(job); prior.stable(item, record)
        require(base.field(record, 'JobState') in prior.ACTIVE | {'PENDING'}, 'Job no longer live or queued')
        detail = checkpoint_and_writer(item)
        item.update(max_requeues=2, initial_resume_step=detail['step'],
                    original_command=item['submit_tokens'], checkpoint_at_preparation=detail)
        rows.append(item)
    script = ('#!/bin/bash\nset -euo pipefail\n'
              'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
              f'cd {ROOT}\nexec /usr/bin/python3 -u {Path(__file__).resolve()} watch --apply\n')
    ART.mkdir(parents=True, exist_ok=True)
    (ART / 'supervisor.slurm').write_text(script)
    helpers = [Path(__file__), PROTOCOL, Path(prior.__file__), Path(base.__file__),
               Path(recovery.__file__), Path(checkpoints.__file__), Path(runtime.__file__),
               ROOT / 'ops/validate_deepspeed_checkpoint.py']
    plan = {'schema': 'e118-pantry-successor-guard-v1', 'created_at_utc': base.now(),
            'not_before_utc': old['deadline_utc'],
            'deadline_utc': (datetime.fromisoformat(old['deadline_utc']) + timedelta(days=7)).isoformat(),
            'old_cpu_job_id': OLD_CPU, 'old_plan_sha256': recovery.digest(OLD_PLAN),
            'rows': rows, 'helper_sha256': {str(p.resolve()): recovery.digest(p) for p in helpers},
            'runtime_fingerprints': runtime.runtime_fingerprints(rows),
            'cpu_submit_tokens': cpu_command(), 'cpu_script_sha256': recovery.digest(ART / 'supervisor.slurm'),
            'poll_seconds': 60, 'max_additional_retries_per_job': 2, 'max_cpu_requeues': 8,
            'scheduler_mutations': False, 'source_jobs_preserved': True}
    atomic(PLAN, plan)
    print(json.dumps({'prepared': str(PLAN), 'job_ids': list(IDS), 'scheduler_mutations': False}))


def load():
    plan = json.loads(PLAN.read_text())
    require(all(recovery.digest(p) == sha for p, sha in plan['helper_sha256'].items()),
            'Prepared code/protocol/helper changed')
    require(recovery.digest(OLD_PLAN) == plan['old_plan_sha256'], 'Predecessor plan changed')
    require(recovery.digest(ART / 'supervisor.slurm') == plan['cpu_script_sha256'], 'CPU script changed')
    require(runtime.runtime_fingerprints(plan['rows']) == plan['runtime_fingerprints'], 'Frozen runtime changed')
    return plan


def register(job):
    plan = load()
    require(not REGISTRATION.exists(), 'CPU registration already exists')
    record = prior.show(job)
    require(base.field(record, 'JobState') == 'PENDING' and base.field(record, 'Reason') == 'JobHeldUser'
            and base.field(record, 'Priority') == '0' and base.field(record, 'Restarts') == '0',
            'Registration requires a never-started held CPU job')
    require(base.submit_tokens(record) == plan['cpu_submit_tokens'], 'CPU submission differs from reviewed command')
    require('gres/gpu' not in base.field(record, 'ReqTRES'), 'Supervisor requests a GPU')
    expected = {'Account': 'mltheory', 'Partition': 'lowprio', 'MinMemoryNode': '2G',
                'NumCPUs': '2', 'NumTasks': '1', 'Requeue': '1', 'TimeLimit': '1-01:10:00',
                'Command': str(ART / 'supervisor.slurm'), 'Comment': 'e118-pantry-successor-20260910',
                'JobName': 'e118-pantry-successor-guard', 'UserId': plan['rows'][0]['resources']['UserId']}
    require(all(base.field(record, k) == v for k, v in expected.items()), 'CPU resources differ')
    require(base.field(record, 'NumNodes') in {'1', '1-1'}, 'CPU node count differs')
    nodes = command(['scontrol', 'show', 'hostnames', base.field(record, 'ReqNodeList')]).stdout.split()
    require(set(nodes) == {'node915', 'node917'}, 'CPU requested node pool differs')
    atomic(REGISTRATION, {'job_id': job, 'plan_sha256': recovery.digest(PLAN),
                          'resources': {k: base.field(record, k) for k in CPU_FIELDS}})
    print(json.dumps({'registered': job, 'scheduler_mutations': False}))


def check_cpu(plan):
    registration = json.loads(REGISTRATION.read_text()); own = os.environ.get('SLURM_JOB_ID')
    require(own and int(own) == registration['job_id'] and registration['plan_sha256'] == recovery.digest(PLAN),
            'Applying monitor requires its registered CPU allocation')
    record = prior.show(int(own))
    require(base.field(record, 'JobState') == 'RUNNING' and 'gres/gpu' not in base.field(record, 'ReqTRES'),
            'Registered CPU allocation is inactive or requests a GPU')
    require(base.submit_tokens(record) == plan['cpu_submit_tokens'], 'Registered CPU submission changed')
    require(all(base.field(record, k) == v for k, v in registration['resources'].items()), 'CPU resources changed')
    require(int(base.field(record, 'Restarts')) <= plan['max_cpu_requeues'], 'CPU renewal cap exhausted')
    return record


def takeover(plan, tx):
    """Caller owns both singleton locks and the ledger lock; old CPU must be inactive."""
    require(window(plan) == 'eligible', 'Predecessor deadline not reached or successor expired')
    require(plan['old_cpu_job_id'] not in base.queue(), 'Predecessor CPU remains active')
    old = json.loads(OLD_TX.read_text())
    require(old['plan_sha256'] == plan['old_plan_sha256'] and old['deadline_utc'] == plan['not_before_utc'],
            'Predecessor transaction binding/deadline changed')
    if tx.get('takeover_at_utc'):
        require(tx['old_final_tx_sha256'] == recovery.digest(OLD_TX), 'Predecessor changed after takeover')
        return
    for item in plan['rows']:
        key = str(item['job_id']); state = old['jobs'].get(key, {})
        require(not any(not a.get('released') for a in state.get('attempts', [])),
                f'{key}: predecessor has an unresolved mutation; retain its hold for review')
        require(state.get('status') in {'monitoring', 'completed'}, f'{key}: predecessor stopped with an unresolved error')
        tx['jobs'][key] = {'status': 'monitoring', 'attempts': [],
                           'last_resume_step': max(item['initial_resume_step'], state.get('last_resume_step', 0)),
                           'predecessor_retries': len(state.get('attempts', []))}
    tx.update(takeover_at_utc=base.now(), old_final_tx_sha256=recovery.digest(OLD_TX))
    save(tx, 'Predecessor expired and inactive; inherited final floors under both singleton locks')


def install_primitives(plan, tx, *, apply):
    # Overrides affect this new CPU process only; no existing helper or guard file changes.
    prior.mapping = mapping; prior.checkpoint_and_writer = checkpoint_and_writer; prior.save = save
    def bounded(parts, *, check=True):
        if parts[:2] in (['scontrol', 'requeuehold'], ['scontrol', 'release']):
            require(apply and tx.get('takeover_at_utc') and window(plan) == 'eligible',
                    'Retry/release outside owned successor window')
            require(int(parts[2]) in IDS, 'Mutation outside three-cell scope')
            require(recovery.digest(OLD_TX) == tx['old_final_tx_sha256'], 'Predecessor changed after takeover')
            require(runtime.runtime_fingerprints(plan['rows']) == plan['runtime_fingerprints'], 'Frozen runtime changed')
            require(window(plan) == 'eligible', 'Retry/release deadline passed during validation')
        return command(parts, check=check)
    base.command = bounded


def acquire(lock):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def observe(plan, tx, *, apply):
    results = []
    for item in plan['rows']:
        try:
            results.append(prior.observe_one(tx, item, apply=apply))
        except Exception as error:
            result = {'job_id': item['job_id'], 'status': 'manual_stop', 'error': repr(error)}
            results.append(result)
            if apply:
                state = tx['jobs'].setdefault(str(item['job_id']), {'attempts': []})
                state.update(status='manual_stop', error=repr(error))
                save(tx, f"{item['job_id']}: visible manual stop; no blind retry or new job: {error}")
    return results


def run(*, watch, apply):
    plan = load()
    tx = json.loads(TX.read_text()) if TX.exists() else {
        'schema': plan['schema'], 'plan_sha256': recovery.digest(PLAN),
        'deadline_utc': plan['deadline_utc'], 'jobs': {}, 'events': []}
    require(tx['plan_sha256'] == recovery.digest(PLAN) and tx['deadline_utc'] == plan['deadline_utc'], 'Successor state changed')
    if apply:
        check_cpu(plan)
    install_primitives(plan, tx, apply=apply)
    started = time.monotonic()
    with OLD_LOCK.open('a+') as old_lock:
        owns_old = False
        while True:
            phase = window(plan)
            report = {'at': base.now(), 'status': phase, 'read_only': not apply,
                      'not_before_utc': plan['not_before_utc'], 'deadline_utc': plan['deadline_utc']}
            if phase == 'deadline_reached':
                report['unresolved_owned_actions'] = [int(job) for job, state in tx['jobs'].items()
                    if any(not a.get('released') for a in state.get('attempts', []))]
            if phase == 'eligible':
                owns_old = owns_old or acquire(old_lock)
                if not owns_old:
                    report['status'] = 'waiting_for_predecessor_singleton'
                elif plan['old_cpu_job_id'] in base.queue():
                    report['status'] = 'waiting_for_predecessor_cpu_exit'
                elif apply:
                    with prior.LEDGER_LOCK.open('a+') as ledger:
                        fcntl.flock(ledger, fcntl.LOCK_EX)
                        takeover(plan, tx)
                        report['jobs'] = observe(plan, tx, apply=True)
                        report['status'] = 'monitoring'
                else:
                    report['status'] = 'would_take_over_after_final_predecessor_validation'
            if not apply:
                # Science inspection is read-only even before takeover; no successor TX is created.
                probe = copy.deepcopy(tx)
                for row in plan['rows']:
                    probe['jobs'].setdefault(str(row['job_id']), {'status': 'monitoring', 'attempts': [],
                                                                 'last_resume_step': row['initial_resume_step']})
                report['jobs'] = observe(plan, probe, apply=False)
                report['scheduler_mutations'] = False
                atomic(ART / 'dry_run.json', report)
            else:
                atomic(ART / 'status.json', report)
                atomic(ART / 'ready.json', {'at': report['at'], 'job_id': int(os.environ['SLURM_JOB_ID']),
                                           'plan_sha256': recovery.digest(PLAN), 'status': report['status']})
            print(json.dumps(report), flush=True)
            if not watch or phase == 'deadline_reached' or (report.get('jobs') and
                    all(r['status'] in {'completed', 'manual_stop'} for r in report['jobs'])):
                return
            if apply and time.monotonic() - started >= 23 * 3600:
                cpu = check_cpu(plan)
                require(int(base.field(cpu, 'Restarts')) < plan['max_cpu_requeues'], 'CPU renewal cap exhausted')
                require(window(plan) != 'deadline_reached', 'Expired successor cannot renew')
                save(tx, 'Registered CPU-only renewal intent; absolute window and science retry counts retained')
                command(['scontrol', 'requeue', os.environ['SLURM_JOB_ID']])
                return
            time.sleep(plan['poll_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'once', 'register', 'watch'))
    parser.add_argument('--apply', action='store_true'); parser.add_argument('--job-id', type=int)
    args = parser.parse_args()
    require(not args.apply or args.phase == 'watch', 'Only registered watch permits scheduler mutations')
    require(args.phase != 'watch' or args.apply, 'watch requires --apply; once is read-only')
    base.command = command
    ART.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as lock:
        require(acquire(lock), 'Another successor instance owns its singleton')
        if args.phase == 'prepare':
            with prior.LEDGER_LOCK.open('a+') as ledger:
                fcntl.flock(ledger, fcntl.LOCK_EX); prepare()
        elif args.phase == 'register':
            require(args.job_id is not None, 'register requires --job-id'); register(args.job_id)
        else:
            try:
                run(watch=args.phase == 'watch', apply=args.apply)
            except Exception as error:
                if args.apply:
                    atomic(ART / 'status.json', {'at': base.now(), 'status': 'manual_stop',
                                                'error': repr(error), 'no_blind_retry': True})
                raise


if __name__ == '__main__':
    main()
