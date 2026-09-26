#!/usr/bin/env python3
"""Bounded same-ID timeout retries for the repaired existing Pantry job31048182."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import time

import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/campaign_timeout_guard_31048182_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/campaign_timeout_guard_31048182_20260909.md'
LOCK = ROOT / 'var/artifacts/campaign_timeout_guard_31048182_20260909.lock'
LEDGER_LOCK = ROOT / 'var/artifacts/e118_ledger_promotion.lock'
IDS = (31048182,)
PRESERVE = ('UserId', 'Account', 'Partition', 'ReqNodeList', 'ExcNodeList',
            'MinMemoryNode', 'NumCPUs', 'NumNodes', 'NumTasks', 'TresPerNode',
            'Requeue', 'Nice', 'QOS', 'Dependency', 'Features', 'WorkDir', 'TimeLimit')
ACTIVE = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(tx, message):
    tx['updated_at_utc'] = base.now()
    tx.setdefault('events', []).append(dict(at=tx['updated_at_utc'], event=message))
    base.atomic(TX, tx)
    print(json.dumps(dict(at=tx['updated_at_utc'], event=message)), flush=True)


def mapping():
    result = {}
    source = json.loads(base.LEDGER.read_text())
    aggregate = json.loads(campaign.E118_LEDGER.read_text())
    require(len(aggregate['runs']) == 150, 'E118 aggregate cardinality changed')
    agg = {r['job_id']: r for r in aggregate['runs']}
    for row in source['runs']:
        if row['job_id'] in IDS:
            require(agg.get(row['job_id']) == dict(row, scale='qwen3b'), 'E118 authoritative/aggregate mismatch')
            result[row['job_id']] = dict(cohort='e118', identity={k: row[k] for k in base.IDENTITY})
    continuations = json.loads(campaign.E119_CONTINUATIONS.read_text())
    effective = campaign.e119_continuation_jobs(campaign.E119_LEDGER)
    for row in continuations['continuations']:
        job = row['continuation_job_id']
        if job in IDS:
            require(effective.get(row['original_job_id']) == job, 'E119 original-cell mapping changed')
            result[job] = dict(cohort='e119', original_job_id=row['original_job_id'], identity={k: row[k] for k in base.IDENTITY})
    return result


def stable(item, record):
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
    require(not recovery.complete(run), 'Training already has a terminal receipt; do not restart')
    checkpoint, rejected = recovery.select_latest_checkpoint(run)
    require(checkpoint is not None and not recovery.validate_checkpoint(checkpoint), 'No valid model/optimizer checkpoint')
    step = int(checkpoint.name[5:])
    require(0 < step < 3072, 'Checkpoint is outside unfinished registered training range')
    recovery.no_other_writer(dict(old_job_id=item['job_id'], new_job_id=item['job_id'], identity=item['identity']), recovery.active_writers())
    return dict(path=str(checkpoint), step=step, rejected=rejected)


def own_hold(item, record, action):
    stable(item, record)
    require(base.field(record, 'JobState') == 'PENDING', 'Owned hold is not pending')
    require(base.field(record, 'Reason') == 'job_requeued_in_held_state', 'Expected requeue-owned hold missing')
    require(int(base.field(record, 'Restarts')) == action['before_restarts'] + 1, 'Expected exactly one requeue restart increment')


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Existing plan/transaction must not be overwritten')
    ART.mkdir(parents=True, exist_ok=True)
    old_tx = json.loads((ROOT / 'var/artifacts/campaign_timeout_guard_20260908/transaction.json').read_text())
    require(old_tx['jobs']['31048182']['status'] == 'manual_stop', 'Old immutable guard still owns this target')
    require('MinMemoryNode' in old_tx['jobs']['31048182'].get('error', ''), 'Unexpected old-guard stop reason')
    identities = mapping()
    require(set(identities) == set(IDS), 'Initial guarded IDs are not all authoritative')
    rows = []
    for job in IDS:
        record = show(job)
        require(base.field(record, 'JobState') in {'RUNNING', 'PENDING'}, 'Repaired target must be running or queued')
        ident = identities[job]
        require(base.field(record, 'MinMemoryNode') == '116G', 'Memory amendment missing')
        require(ident['identity']['run_stamp'] == 'e119_level2_pantry_replay_maxrl_s44', 'Wrong repaired cell')
        require(ident['identity']['domain'] == 'pantry_plan', 'Unexpected guarded domain')
        expected_limit = '12:00:00' if ident['cohort'] == 'e118' else '1-12:00:00'
        require(base.field(record, 'TimeLimit') == expected_limit, 'Original allocation limit changed')
        tokens = base.submit_tokens(record)
        env = base.exports(tokens)
        require(env['SAVE_PATH'] == ident['identity']['run_dir'] and env['RUN_STAMP'] == ident['identity']['run_stamp'], 'Frozen run exports differ')
        require(env['OAT_ZERO_AUTO_RESUME'] == '1' and base.field(record, 'Requeue') == '1', 'Existing resume/requeue contract missing')
        rows.append(dict(ident, job_id=job, before=record, submit_tokens=tokens,
                         launcher_sha256=recovery.digest(tokens[-1]),
                         resources={field: base.field(record, field) for field in PRESERVE},
                         initial_restarts=int(base.field(record, 'Restarts')),
                         max_requeues=3 if ident['cohort'] == 'e118' else 1))
    plan = dict(schema='campaign-timeout-guard-31048182-20260909-v1', created_at_utc=base.now(),
                authorization='User requested recovery of broken E118/E119/E120 jobs and accelerated completion.',
                protocol=str(PROTOCOL), protocol_sha256=recovery.digest(PROTOCOL),
                controller_sha256=recovery.digest(__file__),
                helper_sha256={str(Path(m.__file__).resolve()): recovery.digest(m.__file__) for m in (base, recovery, campaign)},
                rows=rows, poll_seconds=30, maximum_watch_hours=48,
                site_policy='Walltime may not be modified after submission; original allocation limits remain unchanged.',
                existing_cells_only=True, scheduler_only=True, outcomes_inspected=False)
    base.atomic(PLAN, plan)
    print(json.dumps(dict(plan=str(PLAN), job_ids=list(IDS), scheduler_mutations=False,
                          retry_caps={r['job_id']:r['max_requeues'] for r in rows}), indent=2))


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
    job_state = tx['jobs'].get(key, dict(status='monitoring', attempts=[]))
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
    base.command(['scontrol', 'requeuehold', str(job)])
    action['hold_command_returned'] = True
    save(tx, f'{job}: requeuehold returned; verify expected owned hold and restart increment')
    reconcile_action(tx, item, action, job_state)
    return dict(job_id=job, status=job_state['status'], retries=len(job_state['attempts']))


def run(*, watch, apply):
    plan = json.loads(PLAN.read_text())
    require(plan['controller_sha256'] == recovery.digest(__file__), 'Guard changed after preparation')
    require(plan['protocol_sha256'] == recovery.digest(PROTOCOL), 'Guard protocol changed')
    require(all(recovery.digest(p) == h for p, h in plan['helper_sha256'].items()), 'Guard helper changed')
    tx = json.loads(TX.read_text()) if TX.exists() else dict(schema=plan['schema'], plan_sha256=recovery.digest(PLAN), jobs={}, events=[])
    require(tx['plan_sha256'] == recovery.digest(PLAN), 'Guard plan changed')
    if apply and not tx.get('deadline_utc'):
        tx['started_at_utc'] = base.now()
        tx['deadline_utc'] = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat()
        save(tx, 'Started bounded 48-hour timeout guard; live and pending allocations remain untouched')
    while True:
        if apply and datetime.now(timezone.utc) >= datetime.fromisoformat(tx['deadline_utc']):
            save(tx, 'Guard reached its 48-hour deadline; no further scheduler mutation')
            return
        results = []
        for item in plan['rows']:
            try:
                with LEDGER_LOCK.open('a+') as ledger_lock:
                    fcntl.flock(ledger_lock, fcntl.LOCK_EX)
                    results.append(observe_one(tx, item, apply=apply))
            except Exception as error:
                result = dict(job_id=item['job_id'], status='manual_stop', error=repr(error))
                results.append(result)
                if apply:
                    job_state = tx['jobs'].setdefault(str(item['job_id']), dict(attempts=[]))
                    job_state.update(status='manual_stop', error=repr(error))
                    save(tx, f"{item['job_id']}: manual stop: {error}; no blind retry/release")
        print(json.dumps(dict(at=base.now(), read_only=not apply, jobs=results)), flush=True)
        if not watch or all(r['status'] in {'completed', 'manual_stop'} for r in results):
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'once', 'watch'))
    parser.add_argument('--apply', action='store_true', help='Permit bounded same-ID timeout retries; default is read-only')
    args = parser.parse_args()
    require(args.phase != 'prepare' or not args.apply, 'prepare never applies scheduler changes')
    require(args.phase != 'watch' or args.apply, 'watch requires explicit --apply; use once for read-only inspection')
    with LOCK.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another guard instance holds the singleton lock')
        if args.phase == 'prepare':
            with LEDGER_LOCK.open('a+') as ledger_lock:
                fcntl.flock(ledger_lock, fcntl.LOCK_EX)
                prepare()
        else:
            run(watch=args.phase == 'watch', apply=args.apply)
