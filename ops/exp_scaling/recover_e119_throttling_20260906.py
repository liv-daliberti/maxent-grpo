#!/usr/bin/env python3
"""Authorized, journaled memory-only recovery of five throttled E119 jobs."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'ops/exp_scaling')]
import recover_e119_memory_pressure_20260905 as base
from recover_e119_health_20260905 import atomic, call, field, show

ART = ROOT / 'var/artifacts/e119_throttling_recovery_20260906'
SOURCE = ROOT / 'var/artifacts/e119_health_20260906_1438/summary.json'
TARGETS = {
    31045873: ('countdown', 'maxrl', 44, 96, 128),
    31045874: ('countdown', 'drgrpo', 43, 96, 128),
    31048150: ('countdown', 'maxrl', 43, 96, 128),
    31048176: ('mathir', 'maxrl', 47, 64, 96),
    31075341: ('pantry_plan', 'replay_drgrpo', 43, 96, 128),
}
PRESERVE = tuple(dict.fromkeys((*base.PRESERVE, 'QOS', 'Nice', 'ReqNodeList', 'Features', 'UserId')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(jid):
    run = base.identities()[jid]
    spec = TARGETS[jid]
    assert (run['domain'], run['arm'], int(run['seed'])) == spec[:3]
    record = base.live_identity(jid, run)
    assert field(record, 'UserId').startswith(getpass.getuser() + '(')
    assert field(record, 'Dependency') in ('(null)', '')
    return record, run


def stable(before, after, memory, same_attempt=False):
    assert field(after, 'MinMemoryNode') == f'{memory}G'
    assert base.submitline(before) == base.submitline(after)
    for key in PRESERVE:
        assert field(before, key) == field(after, key), key
    assert field(after, 'NumNodes') in ('1', '1-1')
    if same_attempt:
        for key in ('JobState', 'NodeList', 'StartTime', 'Restarts'):
            assert field(before, key) == field(after, key), key


def checkpoint(jid, run, require_ready):
    detail = base.timing_and_checkpoint(jid, run)
    assert detail['checkpoint'] and not detail['fresh_restart']
    step = detail['checkpoint_step']
    assert detail['saved_counter_validation']['saved_counters'] == dict.fromkeys(
        ('global_steps', 'global_step', 'prompt_batches_consumed_total'), step)
    assert detail['unsaved_steps'] is not None and 0 <= detail['unsaved_steps'] <= 192
    newer_partial = {path: errors for path, errors in detail['rejected_checkpoints'].items()
                     if int(Path(path).name.split('_')[1]) >= step}
    detail['newer_partial_checkpoints'] = newer_partial
    detail['ready_to_stop'] = not newer_partial
    if require_ready:
        assert not newer_partial, 'Protect the actively writing/newer partial checkpoint'
    with zipfile.ZipFile(Path(detail['checkpoint']) / 'mp_rank_00_model_states.pt') as z:
        raw = z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))
    detail['model_metadata_sha256'] = hashlib.sha256(raw).hexdigest()
    return detail


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'plan.json').exists(), 'Existing plan requires inspection'
    evidence = json.loads(SOURCE.read_text())
    entries = []
    for jid, spec in TARGETS.items():
        record, run = identity(jid)
        assert field(record, 'JobState') == 'RUNNING'
        assert field(record, 'MinMemoryNode') == f'{spec[3]}G'
        pressure = next(r['memory_probe'] for r in evidence['running'] if r['job_id'] == jid)
        assert pressure['noncache_exceeds_high'] and pressure['events.high'] > 0
        d = checkpoint(jid, run, False)
        entries.append(dict(job_id=jid, run=run, before=record, target_memory_gib=spec[4],
                            checkpoint=d, pressure=pressure,
                            routing_clearance_required=jid == 31075341))
    plan = dict(created_at_utc=base.now(), authorization='User explicitly authorized fixing E119 throttling, with checkpoint protection and Pantry routing coordination.',
                method='Same-ID requeuehold, wait for stopped writer, validate model+optimizer ZIP and saved counters, change only MinMemoryNode, audit, release.',
                maximum_repeated_updates=192, entries=entries,
                script_sha256=sha(__file__), base_sha256=sha(base.__file__),
                ledger_sha256=sha(base.campaign.E119_LEDGER), continuation_sha256=sha(base.campaign.E119_CONTINUATIONS),
                evidence_path=str(SOURCE), evidence_sha256=sha(SOURCE))
    atomic(ART / 'plan.json', plan)
    (ART / 'preregistration.md').write_text('''# E119 host-memory throttling recovery — September 6, 2026

The user explicitly authorized fixing the five E119 jobs identified in the
14:38 UTC operational audit. Four exceed 96 GiB non-cache working sets and
MathIR MaxRL seed 47 exceeds 64 GiB. Millions of memory.high events accompany
weight synchronization lasting tens to hundreds of seconds.

Increase only MinMemoryNode: Countdown 31045873, 31045874, 31048150 and Pantry
31075341 from 96 to 128 GiB; MathIR 31048176 from 64 to 96 GiB. Use the same IDs
and preserve full submitted exports/arguments, models, cells, seeds, objectives,
data, optimizer, evaluation, endpoint budget, CPU/GPU counts, accounts, partitions,
placement, QoS, time limits, exclusions, dependencies, retry policy and ledgers.
The expected restart count increases by one as a consequence of same-ID requeue.
31045873 already exceeds the runtime's 12-restart automatic watchdog allowance;
this repair does not change that allowance.

Validate model/optimizer ZIP directories and all three saved step counters.
Never restart from initialization. At most 192 logged updates may repeat from
the latest valid checkpoint, consistent with the existing save cadence and
the authorized recovery. Historical partial checkpoints older than the selected
valid save are retained and excluded by the existing validator. Newer partial
checkpoints block stopping; specifically protect the active step-2688 writer
for 31045874. Archive stdout, stderr, metrics and full scheduler commands before
stopping and again after cleanup. Requeue into a transaction-owned hold, verify
the stopped writer and checkpoint, resize, audit and release only that hold.

Pantry 31075341 requires the root coordinator's node302 routing clearance before
mutation. Retain safe Pantry GPU placement. Larger memory can cause ordinary
queueing; do not lower requests to fit. The five increases total 160 GiB extra
requested memory. Node105's two changes add 64 GiB; node202, MathIR's existing
route and node302 each add 32 GiB. These requests cover observed working sets
with headroom, not a guarantee against all subsequent memory growth.
''')
    print(json.dumps({'prepared': True, 'entries': [{'job_id': e['job_id'], 'ready': e['checkpoint']['ready_to_stop'], 'resume': e['checkpoint']['checkpoint_step'], 'repeat': e['checkpoint']['unsaved_steps']} for e in entries]}), flush=True)


def load(jid):
    plan = json.loads((ART / 'plan.json').read_text())
    for path, key in ((Path(__file__), 'script_sha256'), (Path(base.__file__), 'base_sha256'),
                      (base.campaign.E119_LEDGER, 'ledger_sha256'), (base.campaign.E119_CONTINUATIONS, 'continuation_sha256')):
        assert sha(path) == plan[key], key
    entry = next(e for e in plan['entries'] if e['job_id'] == jid)
    path = ART / str(jid) / 'transaction.json'
    tx = json.loads(path.read_text()) if path.exists() else None
    return plan, entry, path, tx


def stop(jid, pantry_clearance):
    plan, entry, path, tx = load(jid)
    assert tx is None, 'Never repeat an existing stop transaction'
    assert jid != 31075341 or pantry_clearance, 'Root must coordinate node302 routing first'
    record, run = identity(jid)
    stable(entry['before'], record, TARGETS[jid][3], same_attempt=True)
    d = checkpoint(jid, run, True)
    directory = path.parent / 'before_stop'
    directory.mkdir(parents=True, exist_ok=True)
    archives = base.archive(jid, record, d, directory)
    latest, _ = identity(jid)
    stable(record, latest, TARGETS[jid][3], same_attempt=True)
    d = checkpoint(jid, run, True)
    tx = dict(job_id=jid, created_at_utc=base.now(), plan_sha256=sha(ART / 'plan.json'),
              before=latest, run=run, checkpoint_before=d, archives_before=archives,
              hold_intent=True, own_hold=False, resized=False, released=False, applied=False)
    atomic(path, tx)
    call('scontrol', 'requeuehold', str(jid))
    tx['own_hold'] = True
    atomic(path, tx)
    print(json.dumps({'job_id': jid, 'stop_requested': True, 'resume_step': d['checkpoint_step'], 'repeat_updates': d['unsaved_steps']}), flush=True)


def finish(jid):
    plan, entry, path, tx = load(jid)
    assert tx and tx['own_hold'] and tx['hold_intent'] and not tx['released']
    record, run = identity(jid)
    memory = TARGETS[jid][4] if tx['resized'] else TARGETS[jid][3]
    stable(tx['before'], record, memory)
    assert field(record, 'Priority') == '0'
    assert field(record, 'Reason') == 'job_requeued_in_held_state'
    assert int(field(record, 'Restarts')) == int(field(tx['before'], 'Restarts')) + 1
    if field(record, 'JobState') in ('RUNNING', 'COMPLETING'):
        print(json.dumps({'job_id': jid, 'cleanup_pending': True}), flush=True)
        return
    base.assert_own_hold(record)
    d = checkpoint(jid, run, True)
    assert d['checkpoint_step'] >= tx['checkpoint_before']['checkpoint_step']
    if d['checkpoint_step'] == tx['checkpoint_before']['checkpoint_step']:
        assert d['model_metadata_sha256'] == tx['checkpoint_before']['model_metadata_sha256']
    tx['checkpoint_after_writer_stopped'] = d
    if not tx['resized']:
        directory = path.parent / 'after_stop'
        directory.mkdir(exist_ok=True)
        tx['archives_after_stop'] = base.archive(jid, record, d, directory)
        atomic(path, tx)
        call('scontrol', 'update', f'JobId={jid}', f'MinMemoryNode={TARGETS[jid][4] * 1024}')
        record, _ = identity(jid)
        stable(tx['before'], record, TARGETS[jid][4])
        base.assert_own_hold(record)
        tx.update(resized=True, held_after_update=record)
        atomic(path, tx)
    # Revalidate the ledgers and exact job identity immediately before release.
    load(jid)
    record, _ = identity(jid)
    stable(tx['before'], record, TARGETS[jid][4])
    base.assert_own_hold(record)
    call('scontrol', 'release', str(jid))
    after, _ = identity(jid)
    stable(tx['before'], after, TARGETS[jid][4])
    assert field(after, 'JobState') in ('PENDING', 'RUNNING') and field(after, 'Priority') != '0'
    tx.update(released=True, applied=True, after=after, completed_at_utc=base.now())
    atomic(path, tx)
    load(jid)
    print(json.dumps({'job_id': jid, 'applied': True, 'memory_gib': TARGETS[jid][4], 'state': field(after, 'JobState'), 'resume_step': d['checkpoint_step'], 'repeat_updates': d['unsaved_steps']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'stop', 'finish'))
    p.add_argument('--job', type=int, choices=TARGETS)
    p.add_argument('--pantry-routing-cleared', action='store_true')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        assert args.job
        if args.action == 'stop': stop(args.job, args.pantry_routing_cleared)
        else: finish(args.job)
