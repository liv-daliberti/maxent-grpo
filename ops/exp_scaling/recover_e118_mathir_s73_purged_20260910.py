#!/usr/bin/env python3
"""Restore E118 MathIR Re:MaxRL s73's owned long fallback after Slurm purged its hourly ID."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path

import backfill_preempted_hourly_20260909 as capacity
import guard_mathir_hourly_timeouts_20260909 as validation

b = capacity.b
ROOT = capacity.ROOT
ART = ROOT / 'var/artifacts/e118_mathir_s73_purged_20260910'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e118_mathir_s73_purged_20260910.md'
GUARD_TX = ROOT / 'var/artifacts/preemption_hourly_timeout_guard_20260909/transaction.json'
OLD, LONG, STEP = 31159778, 31158506, 1920
REASON = 'controller_record_purged'
FILES = (capacity.SOURCE, capacity.AGGREGATE, capacity.TX, GUARD_TX)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(tx, message):
    tx['updated_at_utc'] = b.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'event': message})
    b.atomic(TX, tx)


def item_from(deployment):
    item = next(i for i in deployment['items'] if i['new_job_id'] == OLD)
    require((item['old_job_id'], item['cohort'], item['domain'], item['arm'], item['seed']) ==
            (LONG, 'e118', 'mathir', 'replay_maxrl', 73), 'Exact scientific identity changed')
    return item


def inactive_checkpoint(item):
    require(OLD not in b.queue(), 'Hourly allocation is active')
    require(capacity.prior.recovery.state(OLD) == 'TIMEOUT', 'Accounting is not TIMEOUT')
    controller = b.command(['scontrol', 'show', 'job', '-o', str(OLD)], check=False)
    require(controller.returncode != 0 and 'Invalid job id specified' in controller.stderr,
            'Expected purged Slurm controller record is not confirmed')
    run = Path(item['run_dir'])
    require(not capacity.prior.recovery.complete(run), 'Scientific cell already complete')
    writers = capacity.prior.recovery.active_writers().get(str(run.resolve()), set())
    require(writers <= {LONG}, f'Unexpected live writer: {writers}')
    cp = capacity.prior.checkpoint(item['run_dir'])
    require(cp['step'] == STEP, 'Latest durable checkpoint changed from approved step 1920')
    checked = validation.checked_checkpoint(cp['path'])
    require(checked['step'] == STEP, 'Checkpoint counters changed')
    return checked


def audited_long(item):
    record = b.show(LONG)
    require(b.submit_tokens(record) == item['original_command'], 'Frozen fallback SubmitLine changed')
    require(capacity.nodes(record) == capacity.LONG_NODES, 'Long fallback node pool changed')
    for key in capacity.PRESERVE:
        require(b.field(record, key) == b.field(item['before'], key), f'Fallback resource drift: {key}')
    require(b.field(record, 'Partition') == 'lowprio' and b.field(record, 'TimeLimit') == '3-00:00:00',
            'Fallback must retain its original 72-hour lowprio allocation')
    require(b.field(record, 'JobState') in {'PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING', 'COMPLETED'},
            'Fallback entered an unexpected state')
    return record


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Existing plan/transaction: inspect instead of overwriting')
    deployment = capacity.load_transaction()
    item = item_from(deployment)
    guard = json.loads(GUARD_TX.read_text())['jobs'][str(OLD)]
    require(guard['status'] == 'manual_stop' and 'no longer has a controller record' in guard['error'],
            'Expected purged-controller manual stop changed')
    require(item.get('old_fallback_held') and item.get('released') and not item.get('fallback'),
            'Fallback ownership or prior recovery changed')
    capacity.current_identity(item, OLD)
    held = capacity.old_guard(item, held=True)
    checkpoint = inactive_checkpoint(item)
    helpers = (Path(__file__), PROTOCOL, Path(capacity.__file__), Path(validation.__file__),
               Path(capacity.prior.__file__), Path(b.__file__))
    plan = dict(schema='e118-mathir-s73-purged-controller-recovery-v1', created_at_utc=b.now(),
                authorization='User requested restarting the failed existing workload and accelerating completion.',
                reason=REASON, hourly_job_id=OLD, restored_job_id=LONG, checkpoint=checkpoint,
                treatment_unchanged=True, existing_job_only=True, scheduler_action=f'scontrol release {LONG}',
                resume_interval={'hourly':48, 'original_long_fallback':192},
                helper_sha256={str(p.resolve()):capacity.prior.digest(p) for p in helpers},
                before_sha256={str(p):capacity.prior.digest(p) for p in FILES},
                item=copy.deepcopy(item), held_record=held, guard_manual_stop=guard['error'],
                status='prepared', released=False, events=[])
    ART.mkdir(parents=True, exist_ok=True)
    for p in FILES:
        (ART / (p.parent.name + '__' + p.name + '.before')).write_bytes(p.read_bytes())
    b.atomic(PLAN, plan)
    print(json.dumps({k:plan[k] for k in ('status','reason','hourly_job_id','restored_job_id','scheduler_action','checkpoint')}, indent=2))


def stage_ledgers(tx):
    item = tx['item']
    values = {p:json.loads(p.read_text()) for p in (capacity.SOURCE, capacity.AGGREGATE)}
    capacity.remap_item(values, item, LONG, tx['held_record'], fallback=True)
    row = next(r for r in values[capacity.SOURCE]['runs'] if r['run_dir'] == item['run_dir'])
    row.update(repair_audit=str(TX), manual_recovery_reason=REASON,
               manual_recovery_protocol=str(PROTOCOL), resume_checkpoint=tx['checkpoint']['path'])
    target = next(r for r in values[capacity.AGGREGATE]['runs'] if r['run_dir'] == item['run_dir'])
    target.clear(); target.update(row, scale='qwen3b')
    values[capacity.SOURCE].setdefault('repair_history', []).append(dict(at=b.now(), audit=str(TX),
        reason=REASON, hourly_job_id=OLD, restored_long_job_id=LONG, checkpoint=tx['checkpoint']['path']))
    capacity.checked_images(values)
    tx['staged_ledgers'] = {}
    for p, value in values.items():
        out = ART / (p.name + '.after')
        b.atomic(out, value)
        tx['staged_ledgers'][str(p)] = dict(path=str(out), sha256=capacity.prior.digest(out))
    save(tx, 'Staged exact-cell source and 150-cell aggregate mapping before any scheduler action')


def promote(tx, images):
    for path, image in images.items():
        current = capacity.prior.digest(path)
        require(current in {tx['before_sha256'][path], image['sha256']}, f'Concurrent modification: {path}')
        require(capacity.prior.digest(image['path']) == image['sha256'], f'Staged image changed: {path}')
        if current != image['sha256']:
            b.atomic(Path(path), json.loads(Path(image['path']).read_text()))
            save(tx, f'Promoted {path}')


def apply():
    tx = json.loads((TX if TX.exists() else PLAN).read_text())
    require(all(capacity.prior.digest(p) == h for p, h in tx['helper_sha256'].items()), 'Recovery code/protocol changed after preparation')
    if tx['status'] == 'complete':
        capacity.current_identity(tx['item'], LONG)
        print(json.dumps(dict(already_complete=True, restored_job_id=LONG)))
        return
    item = tx['item']
    if not tx.get('release_requested'):
        for p in (capacity.TX, GUARD_TX):
            require(capacity.prior.digest(p) == tx['before_sha256'][str(p)], f'Coordination changed: {p}')
        checkpoint = inactive_checkpoint(item)
        require(checkpoint == tx['checkpoint'], 'Validated checkpoint metadata changed')
        capacity.old_guard(item, held=True)
        if not tx.get('staged_ledgers'):
            require(all(capacity.prior.digest(p) == h for p, h in tx['before_sha256'].items()), 'Prepared input changed')
            capacity.current_identity(item, OLD)
            stage_ledgers(tx)
        promote(tx, tx['staged_ledgers'])
        capacity.current_identity(item, LONG)
        capacity.old_guard(item, held=True)
        require(OLD not in b.queue(), 'Hourly job unexpectedly active')
        require(capacity.prior.recovery.active_writers().get(str(Path(item['run_dir']).resolve()), set()) <= {LONG}, 'Unexpected writer before release')
        tx['release_requested'] = True
        save(tx, 'Committed canonical mappings and durable intent to release exact owned long fallback')
    record = audited_long(item)
    if b.field(record, 'Reason') == 'JobHeldUser':
        capacity.old_guard(item, held=True)
        inactive_checkpoint(item)
        b.command(['scontrol', 'release', str(LONG)])
        record = audited_long(item)
    require(b.field(record, 'Priority') != '0' and b.field(record, 'Reason') != 'JobHeldUser', 'Fallback remains held')
    tx['released'] = True
    tx['after_release'] = record
    save(tx, 'Verified original 72-hour fallback released with frozen command and resources intact')
    if not tx.get('staged_coordination'):
        require(all(capacity.prior.digest(p) == tx['before_sha256'][str(p)] for p in (capacity.TX, GUARD_TX)), 'Coordination changed after release')
        deployment = json.loads(capacity.TX.read_text())
        deployed = item_from(deployment)
        deployed['old_fallback_held'] = False
        deployed['fallback'] = dict(reason=REASON, status='complete', completed_at=b.now(),
            manual_recovery_audit=str(TX), checkpoint_at_fallback=tx['checkpoint'],
            ledgers_committed=True, release_requested=True, after_release=record)
        deployment.setdefault('events', []).append(dict(at=b.now(), message=f'Manual purged-controller recovery restored long job {LONG}; audit {TX}'))
        guard = json.loads(GUARD_TX.read_text())
        state = guard['jobs'][str(OLD)]
        require(state['status'] == 'manual_stop', 'Guard no longer manually stopped')
        state.update(status='fallback_complete', fallback_reason=REASON, manual_recovery_audit=str(TX))
        guard['updated_at_utc'] = b.now()
        guard.setdefault('events', []).append(dict(at=b.now(), event=f'Manual recovery completed for {OLD}; restored {LONG}; audit {TX}'))
        tx['staged_coordination'] = {}
        for p, value in ((capacity.TX, deployment), (GUARD_TX, guard)):
            out = ART / (p.parent.name + '.after.json')
            b.atomic(out, value)
            tx['staged_coordination'][str(p)] = dict(path=str(out), sha256=capacity.prior.digest(out))
        save(tx, 'Staged coordinated fallback-complete records for existing deployment and stopped guard')
    promote(tx, tx['staged_coordination'])
    capacity.current_identity(item, LONG)
    tx['status'] = 'complete'
    save(tx, 'Recovery complete; checkpoint restoration and fresh optimizer progress require live observation')
    print(json.dumps(dict(status='complete', restored_job_id=LONG, checkpoint_step=STEP, reason=REASON)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply the already prepared exact recovery')
    args = parser.parse_args()
    with (ROOT / 'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (apply if args.apply else prepare)()
