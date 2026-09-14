#!/usr/bin/env python3
"""Bounded watcher for one reviewed E119 normal-priority hourly continuation.

prepare is read-only with respect to Slurm. apply only reconciles this controller's
held continuation, retries inactive advancing TIMEOUTs, and invokes the reviewed
capacity fallback/retirement APIs. A memory failure never activates the 96G old job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import time

import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery
import recover_e119_health_20260905 as checkpoints
import observe_pantry_s43_startup_memory_20260909 as observer

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/e119_lowprio_completion_20260909/guard'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
CAPACITY_ART = ROOT / 'var/artifacts/e119_lowprio_completion_20260909'
CAPACITY_PLAN = CAPACITY_ART / 'plan.json'
CAPACITY_TX = CAPACITY_ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e119_lowprio_guard_20260909.md'
LOCK = ART / 'watch.lock'
LEDGER_LOCK = ROOT / 'var/artifacts/e118_ledger_promotion.lock'
ACTIVE = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}
HOLD_REASONS = {'JobHeldUser', 'JobHeldAdmin', 'job_requeued_in_held_state'}
ROUTING_FAILURES = {'NODE_FAIL', 'BOOT_FAIL'}
MAX_REQUEUES = 24
WATCH_HOURS = 24
CLEANUP_SECONDS = 65 * 60
PRESERVE = ('UserId', 'Account', 'Partition', 'ReqNodeList', 'ExcNodeList',
            'MinMemoryNode', 'NumCPUs', 'NumNodes', 'NumTasks', 'TresPerNode',
            'Requeue', 'Nice', 'QOS', 'Dependency', 'Features', 'WorkDir', 'TimeLimit')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def capacity():
    import accelerate_e119_lowprio_20260909
    return accelerate_e119_lowprio_20260909


def utcnow():
    return datetime.now(timezone.utc)


def save(tx, message):
    tx['updated_at_utc'] = base.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'event': message})
    base.atomic(TX, tx)
    print(json.dumps({'at': tx['updated_at_utc'], 'event': message}), flush=True)


def show(job):
    result = base.command(['scontrol', 'show', 'job', '-dd', '-o', str(job)], check=False)
    require(result.returncode == 0 and result.stdout.strip(), f'Missing controller record {job}')
    return result.stdout


def stable(plan, record):
    require(base.submit_tokens(record) == plan['submit_tokens'], 'Frozen continuation SubmitLine changed')
    for key, frozen in plan['resources'].items():
        actual = base.field(record, key)
        if key == 'NumNodes':
            require(actual in {'1', '1-1'} and frozen in {'1', '1-1'}, 'Allocation must remain one node')
        else:
            require(actual == frozen, f'Frozen resource changed: {key}')


def identity(plan):
    runs = json.loads(campaign.E119_LEDGER.read_text())['runs']
    rows = [r for r in runs if r['run_stamp'] == plan['identity']['run_stamp']]
    require(len(rows) == 1, 'Original E119 cell missing or duplicated')
    require(all(rows[0][k] == v for k, v in plan['identity'].items()), 'Original E119 cell changed')
    mapping = campaign.e119_continuation_jobs(campaign.E119_LEDGER)
    require(mapping.get(int(rows[0]['job_id']), int(rows[0]['job_id'])) == plan['new_job_id'],
            'Continuation is no longer the authoritative cell writer')


def old_held(plan):
    capacity().old_guard(capacity().load_transaction()['item'])
    record = show(plan['old_job_id'])
    require(base.field(record, 'JobState') == 'PENDING'
            and base.field(record, 'Reason') == 'JobHeldUser'
            and base.field(record, 'Priority') == '0', 'Original fallback is not our exact user hold')
    require(base.submit_tokens(record) == plan['original_command'], 'Original fallback command changed')
    return record


def no_other_writer(plan):
    # An exact verified held predecessor is dormant, and is the only exemption.
    old_held(plan)
    actual = recovery.active_writers().get(str(Path(plan['identity']['run_dir']).resolve()), set())
    require(not actual - {plan['old_job_id'], plan['new_job_id']}, f'Unexpected same-cell writer: {actual}')


def checkpoint(plan):
    detail = checkpoints.checkpoint(plan['identity'])
    require(0 < detail['step'] < 3072, 'Checkpoint is outside unfinished training range')
    return detail


def inactive_timeout(plan, record):
    require(base.field(record, 'JobState') == 'TIMEOUT', 'Only TIMEOUT may be retried')
    require(plan['new_job_id'] not in base.queue(), 'Timed-out job still active in queue')
    require(recovery.state(plan['new_job_id']) == 'TIMEOUT', 'TIMEOUT not yet confirmed by accounting')


def may_retry(tx, step, now=None):
    now = now or utcnow()
    return (not tx.get('memory_failure') and len(tx['attempts']) < MAX_REQUEUES
            and step > tx['last_resume_step']
            and now < datetime.fromisoformat(tx['deadline_utc']))


def memory_failure(value, previous=None):
    if not value:
        return False
    events = value.get('events', {})
    if events.get('oom', 0) or events.get('oom_kill', 0):
        return True
    high = value.get('memory.high', '')
    if not str(high).isdigit():
        return False
    prior = (previous or {}).get('events', {}).get('high', 0)
    return (events.get('high', 0) > prior
            and value.get('noncache_gib', 0) * 2**30 >= int(high))


def observations(plan, tx, record):
    """Only observes; no cancellation, requeue or allocation update."""
    job = plan['new_job_id']
    state = base.field(record, 'JobState')
    attempt = base.field(record, 'StartTime') + '/' + base.field(record, 'Restarts')
    now = time.time()
    row = tx.setdefault('observation', {})
    if row.get('attempt') != attempt:
        row.update(attempt=attempt, log_offsets={}, last_memory_epoch=0)
    for key in ('StdOut', 'StdErr'):
        raw = base.field(record, key)
        if not raw or raw == '(null)':
            continue
        path = Path(raw)
        if not path.is_file():
            continue
        stat = path.stat()
        offset = row['log_offsets'].get(str(path), 0)
        if offset > stat.st_size:
            offset = 0
        with path.open(errors='replace') as handle:
            handle.seek(offset)
            text = handle.read()
            row['log_offsets'][str(path)] = handle.tell()
        matches = re.findall(r'[^\n]*(?:CUDA out of memory|OutOfMemoryError|oom-kill|oom_kill|Detected .*oom_kill)[^\n]*', text, re.I)
        if matches:
            tx['memory_failure'] = True
            row.setdefault('memory_log_evidence', []).extend(x[-1800:] for x in matches[-5:])
    if state == 'OUT_OF_MEMORY':
        tx['memory_failure'] = True
    if state == 'RUNNING' and now - row.get('last_memory_epoch', 0) >= 300:
        observed = observer.cgroup_probe(job)
        value = observed.get('value')
        previous = row.get('cgroup', {}).get('value')
        if memory_failure(value, previous):
            tx['memory_failure'] = True
        row.update(cgroup=observed, last_memory_epoch=now)
    if now - row.get('last_progress_epoch', 0) >= 300:
        try:
            row['checkpoint'] = checkpoint(plan)
            row['durable_progress'] = row['checkpoint']['step'] > tx['last_resume_step']
        except Exception as exc:
            row['checkpoint_observation_error'] = str(exc)
        row['last_progress_epoch'] = now
    row.update(observed_at_utc=base.now(), state=state, runtime=base.field(record, 'RunTime'))
    return row


def frozen(plan):
    require(plan['controller_sha256'] == recovery.digest(__file__), 'Watcher changed after prepare')
    require(plan['protocol_sha256'] == recovery.digest(PROTOCOL), 'Watcher protocol changed')
    require(plan['capacity_plan_sha256'] == recovery.digest(CAPACITY_PLAN), 'Capacity plan changed')
    for path, digest in plan['helper_sha256'].items():
        require(recovery.digest(path) == digest, f'Frozen helper changed: {path}')
    capacity().load_transaction()  # Includes frozen runtime fingerprints.


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Never overwrite an existing watcher plan')
    ctl = capacity()
    item = json.loads(CAPACITY_TX.read_text())['item']
    require(item['old_job_id'] == 31037832 and item.get('new_job_id'), 'Wrong or unstaged predecessor')
    record = show(item['new_job_id'])
    require(base.field(record, 'JobState') == 'PENDING'
            and base.field(record, 'Reason') == 'JobHeldUser', 'Prepare requires staged held continuation')
    require(base.field(record, 'MinMemoryNode') == '116G', 'Reviewed116GiB memory margin missing')
    require(base.field(record, 'TimeLimit') == '01:00:00', 'Reviewed1h allocation missing')
    require(base.field(record, 'Account') == 'mltheory' and base.field(record, 'Partition') == 'all', 'Wrong continuation route')
    tokens = base.submit_tokens(record)
    require(tokens == item['command'], 'Staged command differs from controller transaction')
    ident = {k: item[k] for k in ('domain', 'arm', 'seed', 'run_stamp', 'run_dir')}
    require(ident['run_stamp'] == 'e119_level2_pantry_replay_drgrpo_s44', 'Wrong scientific cell')
    plan = {'schema': 'e119-lowprio-guard-20260909-v1', 'created_at_utc': base.now(),
            'new_job_id': item['new_job_id'], 'old_job_id': item['old_job_id'], 'identity': ident,
            'initial_checkpoint_step': item['checkpoint']['step'], 'submit_tokens': tokens,
            'original_command': item['original_command'],
            'resources': {k: base.field(record, k) for k in PRESERVE},
            'controller_sha256': recovery.digest(__file__), 'protocol_sha256': recovery.digest(PROTOCOL),
            'capacity_controller_sha256': recovery.digest(ctl.__file__),
            'capacity_plan_sha256': recovery.digest(CAPACITY_PLAN),
            'helper_sha256': {str(Path(m.__file__).resolve()): recovery.digest(m.__file__)
                              for m in (ctl, base, recovery, checkpoints, observer, campaign)},
            'maximum_watch_hours': WATCH_HOURS, 'cleanup_seconds': CLEANUP_SECONDS,
            'max_requeues': MAX_REQUEUES, 'poll_seconds': 60}
    old_held(plan)
    detail = checkpoint(plan)
    require(detail['step'] == plan['initial_checkpoint_step'], 'Staged checkpoint moved before watcher preparation')
    ART.mkdir(parents=True, exist_ok=True)
    base.atomic(PLAN, plan)
    print(json.dumps({'plan': str(PLAN), 'new_job_id': item['new_job_id'], 'scheduler_mutations': False}))


def mark_stop(tx, reason):
    tx.update(status='manual_stop', error=reason)
    save(tx, reason + '; inactive continuation and original hold are retained for review')


def routing_fallback(plan, tx, reason):
    require(not tx.get('memory_failure'), 'Memory evidence forbids96GiB fallback')
    # The capacity helper owns exact-identity, inactivity, hold, mapping and release checks.
    tx['fallback_pending_reason'] = reason
    save(tx, 'Persisted reviewed fallback intent: ' + reason)
    try:
        capacity().fallback(plan['new_job_id'], reason)
    except RuntimeError as exc:
        # The helper releases a deadline hold if a pending job races into RUNNING.
        if 'continuation allocated; defer fallback until inactive' in str(exc):
            require(base.field(show(plan['new_job_id']), 'JobState') in ACTIVE,
                    'Fallback race exception did not preserve an active allocation')
            tx.pop('fallback_pending_reason', None)
            save(tx, 'Pending deadline raced allocation; current attempt preserved')
            return
        raise
    tx.pop('fallback_pending_reason', None)
    tx.update(status='fallback', fallback_reason=reason)
    save(tx, 'Reviewed routing fallback activated')


def handoff_deadline_hold(plan, tx, action, record):
    # Capacity explicitly permits this exact guard-owned transition handoff.
    stable(plan, record)
    identity(plan)
    require(not tx.get('memory_failure'), 'Memory failure forbids deadline fallback')
    require(base.field(record, 'JobState') == 'PENDING', 'Handoff requires inactive held successor')
    expected = action['before_restarts'] + 1
    require(int(base.field(record, 'Restarts')) == expected, 'Handoff restart ownership missing')
    reason = base.field(record, 'Reason')
    require(reason == 'job_requeued_in_held_state'
            or (action.get('deadline_handoff_requested') and reason == 'JobHeldUser'),
            'Handoff cannot claim an unrelated user/admin hold')
    require(datetime.fromisoformat(action['requested_at_utc']) < datetime.fromisoformat(tx['deadline_utc']),
            'Handoff requires a persisted pre-deadline action')
    no_other_writer(plan)
    action['deadline_handoff_requested'] = True
    save(tx, 'Persisted exact owned requeue-hold deadline handoff')
    if reason != 'JobHeldUser':
        base.command(['scontrol', 'hold', str(plan['new_job_id'])])
    normalized = show(plan['new_job_id'])
    stable(plan, normalized)
    require(base.field(normalized, 'JobState') == 'PENDING'
            and base.field(normalized, 'Reason') == 'JobHeldUser'
            and base.field(normalized, 'Priority') == '0'
            and int(base.field(normalized, 'Restarts')) == expected,
            'Exact normalized deadline hold missing')
    ctl = capacity()
    ctx = ctl.load_transaction()
    require(ctx['item']['new_job_id'] == plan['new_job_id'], 'Capacity target changed during handoff')
    ctx['item']['deadline_hold_requested'] = True
    ctx['item']['guard_deadline_owned_hold'] = {
        'guard_transaction': str(TX), 'before_restarts': action['before_restarts'],
        'expected_restarts': expected, 'requested_at_utc': action['requested_at_utc'],
        'normalized_record': normalized, 'at_utc': base.now()}
    ctl.save(ctl.TX, ctx)
    routing_fallback(plan, tx, 'deadline')


def reconcile(plan, tx):
    action = tx['attempts'][-1]
    record = show(plan['new_job_id'])
    stable(plan, record)
    reason = base.field(record, 'Reason')
    restarts = int(base.field(record, 'Restarts'))
    require(restarts == action['before_restarts'] + 1, 'Owned requeue restart increment missing; do not repeat')
    if action.get('release_requested') and reason not in HOLD_REASONS:
        require(base.field(record, 'JobState') in ACTIVE | {'PENDING', 'COMPLETED', 'TIMEOUT'}, 'Unexpected release reconciliation state')
        action['released'] = True
        tx['last_resume_step'] = action['checkpoint']['step']
        save(tx, 'Same-ID retry release reconciled')
        return
    if action.get('deadline_handoff_requested') or utcnow() >= datetime.fromisoformat(tx['deadline_utc']) + timedelta(minutes=5):
        handoff_deadline_hold(plan, tx, action, record)
        return
    require(base.field(record, 'JobState') == 'PENDING'
            and reason == 'job_requeued_in_held_state', 'Exact requeue-owned hold missing')
    deadline = datetime.fromisoformat(tx['deadline_utc'])
    # Finish one persisted pre-deadline transition during the first5min of
    # cleanup; a full1h attempt then fits inside the finite65min cleanup period.
    grace = (datetime.fromisoformat(action['requested_at_utc']) < deadline
             and utcnow() < deadline + timedelta(minutes=5))
    require(utcnow() < deadline or grace, 'Deadline reached during owned requeue transition; keep held')
    no_other_writer(plan)
    detail = checkpoint(plan)
    require(detail['step'] >= action['checkpoint']['step'] and detail['step'] > tx['last_resume_step'], 'Checkpoint not advancing at release')
    action['checkpoint'] = detail
    require(not tx.get('memory_failure'), 'Memory evidence forbids retry')
    if not (utcnow() < deadline or (grace and utcnow() < deadline + timedelta(minutes=5))):
        handoff_deadline_hold(plan, tx, action, show(plan['new_job_id']))
        return
    action['release_requested'] = True
    save(tx, 'Persisted exact-owned-hold release intent')
    base.command(['scontrol', 'release', str(plan['new_job_id'])])
    reconcile(plan, tx)


def observe_one(plan, tx, *, apply):
    frozen(plan)
    if tx.get('fallback_pending_reason'):
        if apply:
            routing_fallback(plan, tx, tx['fallback_pending_reason'])
        return 'fallback_reconciliation'
    identity(plan)
    record = show(plan['new_job_id'])
    stable(plan, record)
    if apply:
        observations(plan, tx, record)
    unfinished = [a for a in tx['attempts'] if not a.get('released')]
    require(len(unfinished) <= 1, 'Multiple incomplete retry transactions')
    if unfinished:
        if apply:
            reconcile(plan, tx)
        return 'retry_reconciliation'
    state = base.field(record, 'JobState')
    if recovery.complete(Path(plan['identity']['run_dir'])):
        if state in ACTIVE:
            return 'terminal_receipt_cleanup_running'
        if apply:
            tx['retirement_pending'] = True
            save(tx, 'Persisted dormant predecessor retirement intent')
            capacity().retire_completed(plan['new_job_id'])
            tx.pop('retirement_pending', None)
            tx['status'] = 'completed'
            save(tx, 'Terminal receipt verified; dormant predecessor retired')
        return 'completed'
    expired = utcnow() >= datetime.fromisoformat(tx['deadline_utc'])
    if state in ACTIVE:
        cleanup_expired = utcnow() >= datetime.fromisoformat(tx['cleanup_deadline_utc'])
        if cleanup_expired and apply:
            tx['status'] = 'deadline_running_preserved'
            save(tx, '24h deadline plus65min cleanup reached; active allocation and original hold preserved')
        return 'running'
    if tx.get('memory_failure'):
        if apply:
            mark_stop(tx, 'Observed memory failure/pressure; conservative128GiB repair requires review')
        return 'memory_failure'
    if state == 'PENDING':
        if expired:
            if apply:
                routing_fallback(plan, tx, 'deadline')
            return 'route_deadline'
        return 'pending'
    if state in ROUTING_FAILURES:
        require(plan['new_job_id'] not in base.queue() and recovery.state(plan['new_job_id']) == state,
                'Routing failure not yet inactive and accounted')
        if apply:
            routing_fallback(plan, tx, 'node_failure')
        return 'routing_failure'
    if state != 'TIMEOUT':
        if apply:
            mark_stop(tx, 'Unclassified terminal state ' + state)
        return 'manual_review'
    inactive_timeout(plan, record)
    no_other_writer(plan)
    detail = checkpoint(plan)
    if not may_retry(tx, detail['step']):
        if apply:
            reason = ('deadline' if expired else 'requeue_cap' if len(tx['attempts']) >= MAX_REQUEUES
                      else 'no_progress')
            routing_fallback(plan, tx, reason)
        return 'retry_not_qualified'
    if not apply:
        return 'would_retry'
    # Verify all mutable prerequisites again immediately before requeuehold.
    current = show(plan['new_job_id'])
    stable(plan, current)
    inactive_timeout(plan, current)
    require(base.field(current, 'Restarts') == base.field(record, 'Restarts'), 'Attempt changed')
    require(may_retry(tx, detail['step']), 'Retry permission expired during checkpoint validation')
    action = {'requested_at_utc': base.now(), 'before_restarts': int(base.field(current, 'Restarts')),
              'checkpoint': detail, 'released': False}
    tx['attempts'].append(action)
    save(tx, 'Persisted advancing inactive-TIMEOUT requeuehold intent; uncertain calls are never repeated')
    base.command(['scontrol', 'requeuehold', str(plan['new_job_id'])])
    reconcile(plan, tx)
    return 'retried'


def run(*, watch, apply):
    plan = json.loads(PLAN.read_text())
    frozen(plan)
    tx = json.loads(TX.read_text()) if TX.exists() else {
        'schema': plan['schema'], 'plan_sha256': recovery.digest(PLAN), 'attempts': [],
        'status': 'watching', 'last_resume_step': plan['initial_checkpoint_step'],
        'started_at_utc': base.now(), 'deadline_utc': (utcnow() + timedelta(hours=WATCH_HOURS)).isoformat()}
    tx.setdefault('cleanup_deadline_utc', (datetime.fromisoformat(tx['deadline_utc'])
                                          + timedelta(seconds=CLEANUP_SECONDS)).isoformat())
    require(tx['plan_sha256'] == recovery.digest(PLAN), 'Watcher plan changed')
    ART.mkdir(parents=True, exist_ok=True)
    if apply:
        require(tx['status'] == 'watching', 'Stopped watcher requires explicit review; do not publish fresh readiness')
        save(tx, 'Scoped116GiB continuation watcher started/reconciled')
        guard_id = os.environ.get('SLURM_JOB_ID')
        require(guard_id and guard_id.isdigit(), 'Applying watcher must run in a recorded CPU supervisor')
        base.atomic(CAPACITY_ART / 'guard_ready.json', {
            'guard_job_id': int(guard_id), 'new_job_id': plan['new_job_id'],
            'controller_sha256': plan['capacity_controller_sha256'],
            'plan_sha256': plan['capacity_plan_sha256'], 'watcher_plan_sha256': recovery.digest(PLAN),
            'at_utc': base.now()})
    while tx['status'] == 'watching':
        try:
            with LEDGER_LOCK.open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = observe_one(plan, tx, apply=apply)
                if apply:
                    tx['last_result'] = result
                    base.atomic(TX, tx)
                print(json.dumps({'at': base.now(), 'result': result}), flush=True)
        except Exception as exc:
            if apply:
                retryable = (tx.get('fallback_pending_reason') or tx.get('retirement_pending')
                             or any(not a.get('released') for a in tx['attempts']))
                tx['transition_errors'] = tx.get('transition_errors', 0) + 1
                if retryable and tx['transition_errors'] < 8 and utcnow() < datetime.fromisoformat(tx['cleanup_deadline_utc']):
                    tx['last_transition_error'] = repr(exc)
                    save(tx, 'Durable controller transition needs reconciliation: ' + repr(exc))
                else:
                    mark_stop(tx, repr(exc))
            else:
                raise
        if not watch or not apply or tx['status'] != 'watching':
            break
        time.sleep(plan['poll_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        require(not args.apply and not args.watch, 'Preparation cannot apply or watch')
        prepare()
        return
    ART.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(watch=args.watch, apply=args.apply)


if __name__ == '__main__':
    main()
