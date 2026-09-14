#!/usr/bin/env python3
"""Separately pinned CPU handoff; retain all original E122 release decisions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import importlib
import json
import os
import re
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resume_e122_shared_release_20260911 as old

c = old.c
require = old.require
ROOT = old.ROOT
SOURCE = Path(__file__).resolve()
TEST = ROOT / 'tests/test_handoff_e122_throttle2_supervisor_20260911.py'
ART = ROOT / 'var/artifacts/e122_throttle2_supervisor_handoff_20260911'
PLAN = ART / 'plan.json'
REG = ART / 'supervisor.json'
OLD_ID = '31245514'
OLD_SHA = '0da4cafb8b8aeee51d8165928b4b5629d5b1603d0febeb69f0a355266091fc4c'


def submission(record):
    match = re.search(r' SubmitLine=(.*?) WorkDir=', record)
    require(match is not None, 'Missing full submission identity')
    return match.group(1)


def verify(expected=None):
    expected = expected or (ART / 'plan.sha256').read_text().strip()
    plan = c.pinned_json(PLAN, expected)
    previous = old.verify_plan(expected=OLD_SHA)
    require(plan['schema'] == 'e122_throttle2_supervisor_handoff_v1', 'Wrong handoff schema')
    require(plan['old_job_id'] == OLD_ID and plan['old_plan_sha256'] == OLD_SHA, 'Wrong predecessor')
    for key in ('binding', 'deadline_utc', 'interval_seconds', 'locks', 'original_journal_root', 'persistent_cap', 'cpu'):
        require(plan[key] == previous[key], 'Original controller contract changed: ' + key)
    require(plan['persistent_cap'] == c.MAX_ACTIVE == 4, 'Cap changed')
    require(datetime.now(timezone.utc) < datetime.fromisoformat(plan['deadline_utc']), 'Original deadline reached')
    for name, digest in plan['source_pins'].items():
        require(c.digest(Path(name)) == digest, 'Handoff source changed: ' + name)
    return plan


def old_cpu(plan):
    record = old.cpu_record(OLD_ID)
    require(submission(record) == submission(plan['old_cpu_record']), 'Old CPU command changed')
    require(c.field(record, 'UserId') == c.field(plan['old_cpu_record'], 'UserId'), 'Old CPU owner changed')
    require('gres/gpu' not in c.field(record, 'ReqTRES'), 'Old target is a GPU allocation')
    require(c.field(record, 'JobState') == 'RUNNING', 'Old CPU is no longer running; reconcile before handoff')
    return record


def cpu_record(jid, expected, held=False, dependency=False):
    result = c.command(['scontrol', 'show', 'job', '-dd', '-o', str(jid)])
    require(result.returncode == 0, 'Cannot inspect exact new CPU')
    record = result.stdout.strip()
    fields = {'JobId': str(jid), 'JobName': 'e122-throttle2-release', 'Account': 'mltheory',
              'Partition': 'lowprio', 'ReqNodeList': 'node917', 'MinMemoryNode': '8G',
              'NumCPUs': '2', 'NumTasks': '1', 'TimeLimit': '3-00:00:00',
              'WorkDir': str(ROOT), 'Requeue': '1', 'Comment': 'e122-throttle2-' + expected}
    for key, value in fields.items():
        require(c.field(record, key) == value, 'New CPU binding/resource changed: ' + key)
    require('gres/gpu' not in c.field(record, 'ReqTRES'), 'CPU handoff requested GPUs')
    require(c.field(record, 'NumNodes') in ('1', '1-1'), 'CPU node count changed')
    wrap = ' '.join([str(old.PYTHON), '-u', '-B', str(SOURCE), 'watch', '--plan-sha256', expected])
    require('--wrap=exec ' + wrap in submission(record), 'New CPU command differs')
    if held:
        require(c.field(record, 'JobState') == 'PENDING' and c.field(record, 'Priority') == '0'
                and c.field(record, 'Reason') == 'JobHeldUser', 'New CPU must remain held')
    if dependency:
        require(c.field(record, 'Dependency') == 'afterany:' + OLD_ID + '(unfulfilled)', 'Exact predecessor gate absent')
    return record


def prepare(storage_module):
    require(not PLAN.exists(), 'Handoff plan already exists')
    previous = old.verify_plan(expected=OLD_SHA)
    reg = old.read(old.REG)
    require(reg['job_id'] == OLD_ID and reg['plan_sha256'] == OLD_SHA, 'Old CPU registration differs')
    record = old.cpu_record(OLD_ID)
    require(submission(record) == submission(reg['held_record']), 'Old registered command changed')
    require(c.field(record, 'JobState') == 'RUNNING', 'Expected old CPU running')
    args, campaign = old.authenticate()
    require(campaign['binding'] == previous['binding'], 'Campaign changed')
    storage = importlib.import_module(storage_module)
    with old.admission_locks(), c.locked(c.JOURNAL_ROOT):
        status = old.ORIGINAL_STATUS(campaign, c.JOURNAL_ROOT)
        require(not status['issues'] and not status['unknown_job_ids'] and not status['needs_operator_review_job_ids'], 'Unresolved original controller status')
        budget = storage.storage_report()
        require(not budget['errors'] and not budget.get('unknown_writers'), 'Corrected storage still has classification errors')
    pins = {str(p): c.digest(p) for p in (SOURCE, TEST, Path(storage.__file__), old.PLAN, old.REG)}
    storage_test = ROOT / 'tests' / ('test_' + storage_module + '.py')
    require(storage_test.is_file(), 'Corrected adapter tests missing')
    pins[str(storage_test)] = c.digest(storage_test)
    for name, digest in storage.EVIDENCE.items():
        path = ROOT / name
        require(c.digest(path) == digest, 'Adapter amendment evidence changed')
        pins[str(path)] = digest
    for item in budget.get('static_evidence', []):
        path = Path(item['path']); pins[str(path)] = c.digest(path)
    plan = {key: previous[key] for key in ('binding', 'deadline_utc', 'interval_seconds', 'locks', 'original_journal_root', 'persistent_cap', 'cpu')}
    plan.update({'schema': 'e122_throttle2_supervisor_handoff_v1', 'at_utc': c.now(),
                 'old_job_id': OLD_ID, 'old_plan_sha256': OLD_SHA, 'old_cpu_record': record,
                 'storage_module': storage_module, 'source_pins': pins,
                 'controller_status_at_prepare': status, 'storage_at_prepare': budget,
                 'authorization': 'Repair E122 storage classification after authorized array throttle amendment; preserve all GPU jobs and scientific inputs.'})
    c.immutable_json(PLAN, plan)
    (ART / 'plan.sha256').write_text(c.digest(PLAN) + '\n')
    print(json.dumps({'plan_sha256': c.digest(PLAN), 'storage_allowed': budget['allowed'], 'cap': 4}))


def submit(expected):
    plan = verify(expected); old_cpu(plan)
    argv = [str(old.PYTHON), '-u', '-B', str(SOURCE), 'watch', '--plan-sha256', expected]
    command = ['sbatch', '--parsable', '--hold', '--job-name=e122-throttle2-release',
               '--account=mltheory', '--partition=lowprio', '--nodelist=node917', '--mem=8G',
               '--cpus-per-task=2', '--nodes=1', '--ntasks=1', '--time=3-00:00:00', '--requeue',
               '--dependency=afterany:' + OLD_ID, '--chdir=' + str(ROOT),
               '--output=' + str(ART / 'cpu-%j.out'), '--error=' + str(ART / 'cpu-%j.err'),
               '--comment=e122-throttle2-' + expected, '--wrap=exec ' + shlex.join(argv)]
    c.immutable_json(ART / 'submit.intent.json', {'at_utc': c.now(), 'command': command, 'plan_sha256': expected})
    result = c.command(command)
    c.immutable_json(ART / 'submit.result.json', {'at_utc': c.now(), 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    require(result.returncode == 0 and result.stdout.strip().isdigit(), 'Submission uncertain; never repeat')
    jid = result.stdout.strip(); record = cpu_record(jid, expected, held=True, dependency=True)
    c.immutable_json(REG, {'job_id': jid, 'plan_sha256': expected, 'held_record': record, 'command': command})
    print(jid)


def activate(expected):
    plan = verify(expected); reg = old.read(REG)
    require(reg['plan_sha256'] == expected, 'New registration differs')
    with old.admission_locks(), c.locked(c.JOURNAL_ROOT):
        old_before = old_cpu(plan)
        cpu_record(reg['job_id'], expected, held=True, dependency=True)
        c.immutable_json(ART / 'release.intent.json', {'at_utc': c.now(), 'job_id': reg['job_id'], 'old_record': old_before})
        released = c.command(['scontrol', 'release', reg['job_id']])
        c.immutable_json(ART / 'release.result.json', {'at_utc': c.now(), 'returncode': released.returncode, 'stdout': released.stdout, 'stderr': released.stderr})
        require(released.returncode == 0, 'Release uncertain; preserve old CPU and inspect')
        gated = cpu_record(reg['job_id'], expected, dependency=True)
        require(c.field(gated, 'JobState') == 'PENDING' and c.field(gated, 'Priority') != '0', 'Replacement is not eligible behind old CPU')
        old_cpu(plan)
        c.immutable_json(ART / 'old_stop.intent.json', {'at_utc': c.now(), 'job_id': OLD_ID, 'successor_record': gated})
        result = c.command(['scancel', OLD_ID])
        c.immutable_json(ART / 'old_stop.result.json', {'at_utc': c.now(), 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        require(result.returncode == 0, 'Old CPU stop uncertain; do not repeat automatically')
    print(json.dumps({'new_cpu': reg['job_id'], 'old_cpu_stop_requested': OLD_ID}))


def watch(expected):
    plan = verify(expected); reg = old.read(REG)
    require(reg['plan_sha256'] == expected and reg['job_id'] == os.environ.get('SLURM_JOB_ID'), 'Exact registered CPU required')
    cpu_record(reg['job_id'], expected)
    with (ART / 'singleton.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old.write(ART / 'progress.json', {'at_utc': c.now(), 'phase': 'authenticating', 'job_id': reg['job_id']})
        args, campaign = old.authenticate()
        require(campaign['binding'] == plan['binding'], 'Campaign binding changed')
        storage = importlib.import_module(plan['storage_module'])
        unknowns = 0; started = time.monotonic()
        while True:
            verify(expected)
            try:
                status = old.observe_once(args, storage)
            except BlockingIOError:
                old.write(ART / 'progress.json', {'at_utc': c.now(), 'phase': 'waiting_admission_lock', 'job_id': reg['job_id']})
                time.sleep(10); continue
            old.write(ART / 'status.json', status)
            old.write(ART / 'ready.json', {'at_utc': c.now(), 'job_id': reg['job_id'], 'plan_sha256': expected, 'singleton': True, 'storage_module': plan['storage_module'], 'blocked_reason': status['blocked_reason']})
            print(json.dumps({'at_utc': c.now(), 'released': status.get('last_release_job_id'), 'running': status['running'], 'held': status['staged_held'], 'blocked_reason': status['blocked_reason']}), flush=True)
            if status['blocked_reason'] == 'complete': return 0
            unknowns = unknowns + 1 if old.retryable_observation(status) else 0
            if unknowns >= 5 or status['issues'] or status['needs_operator_review_job_ids']: return 2
            if time.monotonic() - started > 36 * 3600:
                result = c.command(['scontrol', 'requeue', reg['job_id']])
                require(result.returncode == 0, 'CPU self-requeue uncertain'); return 0
            time.sleep(plan['interval_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'submit', 'activate', 'watch'])
    parser.add_argument('--storage-module')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.phase == 'prepare': prepare(args.storage_module)
    elif args.phase == 'submit': submit(args.plan_sha256)
    elif args.phase == 'activate': activate(args.plan_sha256)
    else: raise SystemExit(watch(args.plan_sha256))
