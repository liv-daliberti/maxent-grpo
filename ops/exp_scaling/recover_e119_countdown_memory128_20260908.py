#!/usr/bin/env python3
"""Journaled same-ID recovery of independently confirmed E119 memory throttling."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'ops/exp_scaling')]
import recover_e119_memory_pressure_20260905 as base
import recover_terminal_timeouts_20260908 as recovery
from recover_e119_health_20260905 import atomic, call, field

ART = ROOT / 'var/artifacts/e119_countdown_memory128_20260908'
PLAN = ART / 'plan.json'
EVIDENCE = ART / 'evidence.json'
PROTOCOL = ROOT / 'paper/preregistration/e119_countdown_memory128_20260908.md'
TARGETS = {31048154: ('countdown', 'drgrpo', 45),
           31048161: ('countdown', 'drgrpo', 47),
           31048157: ('countdown', 'replay_maxrl', 45)}
PRESERVE = tuple(k for k in dict.fromkeys((*base.PRESERVE, 'QOS', 'Nice', 'ReqNodeList',
                                         'Features', 'UserId')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete(run):
    return recovery.complete(Path(run['run_dir']))


def identity(jid):
    run = base.identities()[jid]
    assert (run['domain'], run['arm'], int(run['seed'])) == TARGETS[jid]
    assert not complete(run), 'Completed cells must never be requeued'
    record = base.live_identity(jid, run)
    assert field(record, 'UserId').startswith(getpass.getuser() + '(')
    assert field(record, 'Dependency') in ('(null)', '')
    exp = base.exports(record)
    assert int(exp['OAT_ZERO_SEED']) == int(run['seed'])
    assert exp['OAT_ZERO_AUTO_RESUME'] == '1'
    return record, run


def stable(before, after, memory, limit, attempt=False):
    assert field(after, 'MinMemoryNode') == f'{memory}G'
    assert field(after, 'TimeLimit') == limit
    assert base.submitline(before) == base.submitline(after)
    for key in PRESERVE:
        assert field(before, key) == field(after, key), key
    assert field(after, 'NumNodes') in ('1', '1-1')
    if attempt:
        for key in ('JobState', 'NodeList', 'StartTime', 'Restarts'):
            assert field(before, key) == field(after, key), key


def checkpoint(jid, run):
    detail = base.timing_and_checkpoint(jid, run)
    assert detail['checkpoint'] and not detail['fresh_restart']
    step = detail['checkpoint_step']
    assert detail['saved_counter_validation']['saved_counters'] == dict.fromkeys(
        ('global_steps', 'global_step', 'prompt_batches_consumed_total'), step)
    assert detail['unsaved_steps'] is not None and 0 <= detail['unsaved_steps'] <= 192
    with zipfile.ZipFile(Path(detail['checkpoint']) / 'mp_rank_00_model_states.pt') as archive:
        raw = archive.read(next(n for n in archive.namelist() if n.endswith('data.pkl')))
    detail['model_metadata_sha256'] = hashlib.sha256(raw).hexdigest()
    # A stalled newer partial checkpoint does not supersede the validated save.
    # Preserve it for the existing runtime selector; never delete or rewrite it.
    detail['newer_partial_checkpoints'] = {p: e for p, e in detail['rejected_checkpoints'].items()
                                         if int(Path(p).name.split('_')[1]) >= step}
    return detail


def no_other_writer(jid, run):
    writers = recovery.active_writers()
    assert writers.get(str(Path(run['run_dir']).resolve()), set()) <= {jid}, writers.get(str(Path(run['run_dir']).resolve()))


def frozen_paths(record):
    exp = base.exports(record)
    root = Path(exp['OAT_ZERO_SOURCE_ROOT']).parent
    candidates = [Path(field(record, 'Command')), Path(exp['OAT_ZERO_OPS_SNAPSHOT_ROOT']) / 'run_experiment.sh',
                  root / 'SNAPSHOT_IDENTITY.json']
    candidates.extend(Path(exp['OAT_ZERO_SOURCE_ROOT']).rglob('run.py'))
    return {str(p): sha(p) for p in candidates if p.is_file()}


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    assert not PLAN.exists(), 'Inspect existing plan; never overwrite a transaction'
    evidence = json.loads(EVIDENCE.read_text())
    entries = []
    for key, pressure in evidence['jobs'].items():
        jid = int(key)
        assert jid in TARGETS
        if pressure.get('kind') == 'scheduler_rss_with_confirmed_peer_pressure':
            assert jid == 31048157
            assert pressure['sstat_ave_rss_kib'] > 96 * 2**20
            assert pressure['sstat_max_rss_kib'] > 96 * 2**20
            assert pressure['median_weight_sync_seconds'] > 1000
            assert set(pressure['confirmed_peer_jobs']) == {31048154, 31048161}
        else:
            assert pressure['noncache_gib'] > 96
            assert int(pressure['memory_high']) == 96 * 2**30 and pressure['events']['high'] > 0
        record, run = identity(jid)
        assert field(record, 'JobState') == 'RUNNING'
        assert field(record, 'MinMemoryNode') == '96G'
        assert field(record, 'TimeLimit') == '1-12:00:00'
        for k in ('NodeList', 'StartTime', 'Restarts'):
            assert field(record, k) == pressure['scheduler'][k], k
        no_other_writer(jid, run)
        entries.append(dict(job_id=jid, run=run, before=record, checkpoint=checkpoint(jid, run),
                            pressure=pressure, frozen_sha256=frozen_paths(record)))
    assert entries
    plan = dict(schema='e119-countdown-memory128-v1', created_at_utc=base.now(), entries=entries,
                script_sha256=sha(__file__), base_sha256=sha(base.__file__),
                evidence_sha256=sha(EVIDENCE), protocol_sha256=sha(PROTOCOL),
                ledger_sha256=sha(base.campaign.E119_LEDGER),
                continuation_sha256=sha(base.campaign.E119_CONTINUATIONS),
                selection='Independent memory counters only; user authorized recovery and acceleration',
                change='Same-ID requeuehold then 96->128GiB only; existing 36h walltime and all other resources and training settings preserved',
                maximum_repeated_logged_updates=192)
    atomic(PLAN, plan)
    print(json.dumps({'prepared': True, 'jobs': [{'job': e['job_id'], 'checkpoint': e['checkpoint']['checkpoint_step'],
                                                'repeat': e['checkpoint']['unsaved_steps']} for e in entries]}), flush=True)


def load(jid):
    plan = json.loads(PLAN.read_text())
    for path, key in ((Path(__file__), 'script_sha256'), (Path(base.__file__), 'base_sha256'),
                      (EVIDENCE, 'evidence_sha256'), (PROTOCOL, 'protocol_sha256'),
                      (base.campaign.E119_LEDGER, 'ledger_sha256'),
                      (base.campaign.E119_CONTINUATIONS, 'continuation_sha256')):
        assert sha(path) == plan[key], key
    entry = next(e for e in plan['entries'] if e['job_id'] == jid)
    for path, digest in entry['frozen_sha256'].items():
        assert sha(path) == digest, path
    path = ART / str(jid) / 'transaction.json'
    return entry, path, json.loads(path.read_text()) if path.exists() else None


def stop(jid):
    entry, path, tx = load(jid)
    assert tx is None, 'Existing stop intent requires reconciliation; never repeat requeue'
    record, run = identity(jid)
    stable(entry['before'], record, 96, '1-12:00:00', attempt=True)
    no_other_writer(jid, run)
    detail = checkpoint(jid, run)
    directory = path.parent / 'before_stop'
    directory.mkdir(parents=True, exist_ok=True)
    archives = base.archive(jid, record, detail, directory)
    latest, _ = identity(jid)
    stable(record, latest, 96, '1-12:00:00', attempt=True)
    detail = checkpoint(jid, run)
    tx = dict(job_id=jid, created_at_utc=base.now(), plan_sha256=sha(PLAN), before=latest, run=run,
              checkpoint_before=detail, archives_before=archives,
              hold_intent=True, own_hold=False, resized=False, release_intent=False, released=False)
    atomic(path, tx)
    call('scontrol', 'requeuehold', str(jid))
    tx['own_hold'] = True
    atomic(path, tx)
    print(json.dumps({'job': jid, 'requeuehold_requested': True, 'checkpoint': detail['checkpoint_step']}), flush=True)


def finish(jid, wait_seconds):
    entry, path, tx = load(jid)
    assert tx and tx['hold_intent']
    if tx['released']:
        print(json.dumps({'job': jid, 'already_released': True}), flush=True)
        return
    deadline = time.monotonic() + wait_seconds
    while True:
        record, run = identity(jid)
        assert int(field(record, 'Restarts')) == int(field(tx['before'], 'Restarts')) + 1
        if tx['release_intent']:
            # A release may have succeeded before its receipt; reconcile by readback.
            if field(record, 'JobState') in ('PENDING', 'RUNNING') and int(field(record, 'Priority')) > 0:
                stable(tx['before'], record, 128, '1-12:00:00')
                tx.update(released=True, after=record, completed_at_utc=base.now())
                atomic(path, tx)
                return
        assert field(record, 'Priority') == '0'
        assert field(record, 'Reason') in base.OWN_HOLD_REASONS
        # Reconcile an ambiguous requeue call only against the expected held restart.
        if not tx['own_hold']:
            tx['own_hold'] = True
            atomic(path, tx)
        if field(record, 'JobState') == 'PENDING':
            break
        assert field(record, 'JobState') in ('RUNNING', 'COMPLETING')
        print(json.dumps({'job': jid, 'waiting_for_cleanup': field(record, 'JobState')}), flush=True)
        if time.monotonic() >= deadline:
            return
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    base.assert_own_hold(record)
    no_other_writer(jid, run)
    detail = checkpoint(jid, run)
    assert detail['checkpoint_step'] >= tx['checkpoint_before']['checkpoint_step']
    if detail['checkpoint_step'] == tx['checkpoint_before']['checkpoint_step']:
        assert detail['model_metadata_sha256'] == tx['checkpoint_before']['model_metadata_sha256']
    tx['checkpoint_after_writer_stopped'] = detail
    directory = path.parent / 'after_stop'
    directory.mkdir(exist_ok=True)
    if 'archives_after_stop' not in tx:
        tx['archives_after_stop'] = base.archive(jid, record, detail, directory)
        atomic(path, tx)
    # If an update succeeded before journaling, readback identifies that state.
    memory, limit = field(record, 'MinMemoryNode'), field(record, 'TimeLimit')
    assert memory in ('96G', '128G') and limit == '1-12:00:00'
    stable(tx['before'], record, int(memory[:-1]), limit)
    if memory != '128G':
        call('scontrol', 'update', f'JobId={jid}', 'MinMemoryNode=131072')
    record, _ = identity(jid)
    stable(tx['before'], record, 128, '1-12:00:00')
    base.assert_own_hold(record)
    tx.update(resized=True, held_after_update=record)
    atomic(path, tx)
    load(jid)
    tx['release_intent'] = True
    atomic(path, tx)
    call('scontrol', 'release', str(jid))
    after, _ = identity(jid)
    stable(tx['before'], after, 128, '1-12:00:00')
    assert field(after, 'JobState') in ('PENDING', 'RUNNING') and int(field(after, 'Priority')) > 0
    tx.update(released=True, after=after, completed_at_utc=base.now())
    atomic(path, tx)
    load(jid)
    print(json.dumps({'job': jid, 'released': True, 'state': field(after, 'JobState'),
                      'checkpoint': detail['checkpoint_step'], 'memory_gib': 128, 'hours': 36}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'stop', 'finish'])
    parser.add_argument('--job', type=int, choices=TARGETS)
    parser.add_argument('--wait-seconds', type=int, default=300)
    args = parser.parse_args()
    assert 0 <= args.wait_seconds <= 300
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare': prepare()
        else:
            assert args.job
            if args.action == 'stop': stop(args.job)
            else: finish(args.job, args.wait_seconds)
