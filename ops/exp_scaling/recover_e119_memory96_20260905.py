#!/usr/bin/env python3
"""Checkpoint-preserving 64→96 GiB recovery for exactly twelve reviewed E119 cells."""
from __future__ import annotations
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'ops/exp_scaling')]
import recover_e119_memory_pressure_20260905 as base
from recover_e119_health_20260905 import atomic, call, field, show

ART = ROOT / 'var/artifacts/e119_memory96_recovery_20260905'
PLAN = ART / 'plan.json'
IDS = (31045872, 31045873, 31045874, 31037849, 31048154, 31048159,
       31048160, 31048163, 31048156, 31048157, 31048158, 31048164)
PRESERVE = tuple(dict.fromkeys((*base.PRESERVE, 'QOS', 'Nice', 'ReqNodeList', 'Features', 'UserId')))
SOURCE = ROOT / 'var/artifacts/campaign_health_20260905_2310/e119/cgroups.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(jid: int, run: dict) -> str:
    assert jid in IDS and run['domain'] == 'countdown'
    record = base.live_identity(jid, run)
    assert field(record, 'UserId').startswith(getpass.getuser() + '(')
    assert field(record, 'Dependency') in ('(null)', '')
    return record


def stable(before: str, after: str, memory: str, same_attempt: bool = False) -> None:
    assert field(after, 'MinMemoryNode') == memory
    assert base.submitline(before) == base.submitline(after), 'Scientific exports/arguments changed'
    for key in PRESERVE:
        assert field(before, key) == field(after, key), key
    assert field(after, 'NumNodes') in ('1', '1-1')
    if same_attempt:
        for key in ('JobState', 'StartTime', 'Restarts', 'NodeList'):
            assert field(before, key) == field(after, key), key


def checkpoint(jid: int, run: dict) -> dict:
    detail = base.timing_and_checkpoint(jid, run)
    assert detail['checkpoint'] and not detail['fresh_restart'], 'Never restart from initialization'
    assert detail['saved_counter_validation']['saved_counters'] == dict.fromkeys(
        ('global_steps', 'global_step', 'prompt_batches_consumed_total'), detail['checkpoint_step'])
    assert not detail['rejected_checkpoints'], 'Inspect any active/partial writer before stopping'
    assert detail['unsaved_steps'] is not None and detail['unsaved_steps'] <= 96, 'Rollback exceeds reviewed bound'
    return detail


def prepare() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'transaction.json').exists(), 'Inspect prior execution before preparing'
    evidence = json.loads(SOURCE.read_text())
    assert set(evidence['confirmed_pressure']) == set(IDS)
    mapping = base.identities()
    entries = []
    for jid in IDS:
        run = mapping[jid]
        before = identity(jid, run)
        assert field(before, 'JobState') in ('RUNNING', 'PENDING')
        assert field(before, 'MinMemoryNode') == '64G'
        if field(before, 'JobState') == 'PENDING':
            assert base.ordinary_pending(before), 'Preserve existing holds'
        detail = checkpoint(jid, run)
        entries.append({'job_id': jid, 'run': run, 'before': before,
                        'original_submitline': base.submitline(before), 'checkpoint': detail,
                        'source_pressure': next(r for r in evidence['jobs'] if r['job_id'] == jid)})
    plan = {'created_at_utc': base.now(), 'authorization': 'User explicitly requested cancellation/restart with more memory for the twelve pressure-confirmed E119 runs.',
            'method': 'Same-ID requeuehold, validate stopped-writer checkpoints, change only MinMemoryNode to98304 MiB, audit and release.',
            'target_memory_gib': 96, 'ids': list(IDS), 'entries': entries,
            'script_sha256': sha(Path(__file__)), 'base_sha256': sha(Path(base.__file__)),
            'ledger_sha256': sha(base.campaign.E119_LEDGER),
            'continuations_sha256': sha(base.campaign.E119_CONTINUATIONS),
            'pressure_source': str(SOURCE), 'pressure_source_sha256': sha(SOURCE),
            'watchdog_note': 'Frozen train.sh checks restart_count only after a failed exit; job31045873 can start at13/12 but would not automatically requeue another watchdog failure. No retry-policy change is made.',
            'applied': False}
    atomic(PLAN, plan)
    return plan


def apply() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan['ids'] == list(IDS)
    assert [entry['job_id'] for entry in plan['entries']] == list(IDS)
    for path, key in ((Path(__file__), 'script_sha256'), (Path(base.__file__), 'base_sha256'),
                      (base.campaign.E119_LEDGER, 'ledger_sha256'),
                      (base.campaign.E119_CONTINUATIONS, 'continuations_sha256'),
                      (SOURCE, 'pressure_source_sha256')):
        assert sha(path) == plan[key], key
    receipt = ART / 'transaction.json'
    assert not receipt.exists(), 'Existing execution requires explicit reconciliation, never repeat blindly'
    transaction = {'created_at_utc': base.now(), 'plan_sha256': sha(PLAN), 'applied': False, 'jobs': []}
    # Validate every target before stopping any learner. Mutations below are sequential.
    for entry in plan['entries']:
        jid, run = entry['job_id'], entry['run']
        current = identity(jid, run)
        stable(entry['before'], current, '64G')
        state = field(current, 'JobState')
        assert state in ('RUNNING', 'PENDING')
        if state == 'RUNNING':
            stable(entry['before'], current, '64G', same_attempt=True)
            live_pressure = base.live_memory(jid, current)
        else:
            assert base.ordinary_pending(current)
            live_pressure = {'basis': 'Previously confirmed same cell naturally requeued before repair; resize its ordinary pending allocation.'}
        detail = checkpoint(jid, run)
        directory = ART / str(jid)
        directory.mkdir(exist_ok=True)
        archives = base.archive(jid, current, detail, directory)
        transaction['jobs'].append({'job_id': jid, 'run': run, 'before': current,
            'checkpoint_before': detail, 'live_pressure': live_pressure, 'archives_before': archives,
            'hold_intent': False, 'own_hold': False, 'resized': False, 'released': False})
    atomic(receipt, transaction)
    for action in transaction['jobs']:
        jid, run = action['job_id'], action['run']
        current = identity(jid, run)
        stable(action['before'], current, '64G', same_attempt=True)
        detail = checkpoint(jid, run)
        action['checkpoint_immediately_before_stop'] = detail
        state = field(current, 'JobState')
        assert state in ('RUNNING', 'PENDING')
        if state == 'PENDING':
            assert base.ordinary_pending(current)
        action['hold_intent'] = True
        action['hold_method'] = 'requeuehold' if state == 'RUNNING' else 'hold'
        atomic(receipt, transaction)
        call('scontrol', action['hold_method'], str(jid))
        action['own_hold'] = True
        atomic(receipt, transaction)
        deadline = time.monotonic() + 120
        while True:
            held = show(jid)
            if field(held, 'JobState') == 'PENDING':
                break
            assert state == 'RUNNING' and field(held, 'JobState') in ('RUNNING', 'COMPLETING')
            assert time.monotonic() < deadline, 'Cleanup delayed; leave owned hold for explicit recovery'
            time.sleep(2)
        base.assert_own_hold(held)
        stable(action['before'], held, '64G')
        detail = checkpoint(jid, run)
        assert detail['checkpoint_step'] >= action['checkpoint_before']['checkpoint_step']
        action['checkpoint_after_writer_stopped'] = detail
        stopped_archive = ART / str(jid) / 'after_stop'
        stopped_archive.mkdir(exist_ok=True)
        action['archives_after_stop'] = base.archive(jid, held, detail, stopped_archive)
        atomic(receipt, transaction)
        call('scontrol', 'update', f'JobId={jid}', 'MinMemoryNode=98304')
        resized = identity(jid, run)
        base.assert_own_hold(resized)
        stable(action['before'], resized, '96G')
        action.update(resized=True, held_after_update=resized)
        atomic(receipt, transaction)
        print(json.dumps({'job_id': jid, 'held_at_96g': True, 'resume_step': detail['checkpoint_step'], 'repeat_logged_updates': detail['unsaved_steps']}), flush=True)
    # Release only our own audited holds after all targets carry their new memory request.
    for action in transaction['jobs']:
        assert action['own_hold'] and action['resized'] and not action['released']
        jid, run = action['job_id'], action['run']
        held = identity(jid, run)
        base.assert_own_hold(held)
        stable(action['before'], held, '96G')
        call('scontrol', 'release', str(jid))
        after = identity(jid, run)
        assert field(after, 'JobState') in ('PENDING', 'RUNNING')
        stable(action['before'], after, '96G')
        assert field(after, 'Priority') != '0'
        action.update(released=True, after=after, released_at_utc=base.now())
        atomic(receipt, transaction)
        print(json.dumps({'job_id': jid, 'released': True, 'state': field(after, 'JobState'), 'memory': '96G'}), flush=True)
    assert sha(base.campaign.E119_LEDGER) == plan['ledger_sha256']
    assert sha(base.campaign.E119_CONTINUATIONS) == plan['continuations_sha256']
    transaction.update(applied=True, completed_at_utc=base.now())
    atomic(receipt, transaction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply:
        apply()
    else:
        plan = prepare()
        print(json.dumps({'dry_run': 'pass', 'plan': str(PLAN), 'ids': plan['ids'],
            'resume_steps': {e['job_id']: e['checkpoint']['checkpoint_step'] for e in plan['entries']}}))
