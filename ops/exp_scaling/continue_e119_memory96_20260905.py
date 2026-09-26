#!/usr/bin/env python3
"""Continue the exact owned-hold transaction after a Slurm cleanup timeout."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import recover_e119_memory96_20260905 as recovery


def main(apply=False):
    receipt = recovery.ART / 'transaction.json'
    transaction = json.loads(receipt.read_text())
    plan = json.loads(recovery.PLAN.read_text())
    assert recovery.sha(recovery.PLAN) == transaction['plan_sha256']
    assert recovery.sha(Path(recovery.__file__)) == plan['script_sha256']
    assert recovery.sha(Path(recovery.base.__file__)) == plan['base_sha256']
    assert recovery.sha(recovery.base.campaign.E119_LEDGER) == plan['ledger_sha256']
    assert recovery.sha(recovery.base.campaign.E119_CONTINUATIONS) == plan['continuations_sha256']
    actions = transaction['jobs']
    assert [a['job_id'] for a in actions] == list(recovery.IDS)
    assert not transaction['applied'] and not any(a['released'] for a in actions)
    for a in actions:
        record = recovery.identity(a['job_id'], a['run'])
        recovery.stable(a['before'], record, '96G' if a['resized'] else '64G')
        if a['own_hold']:
            assert a['hold_intent'] and a['hold_method'] == 'requeuehold'
            assert recovery.field(record, 'Priority') == '0'
            assert recovery.field(record, 'Reason') == 'job_requeued_in_held_state'
            assert recovery.field(record, 'JobState') in ('PENDING', 'COMPLETING')
            assert int(recovery.field(record, 'Restarts')) == int(recovery.field(a['before'], 'Restarts')) + 1
        else:
            assert not a['hold_intent'] and not a['resized']
            recovery.stable(a['before'], record, '64G', same_attempt=True)
            assert recovery.field(record, 'JobState') == 'RUNNING'
        recovery.checkpoint(a['job_id'], a['run'])
    if not apply:
        print(json.dumps({'dry_run': 'pass', 'owned_holds': [a['job_id'] for a in actions if a['own_hold']],
                          'remaining_running': [a['job_id'] for a in actions if not a['own_hold']]}))
        return
    assert not transaction.get('cleanup_continuation'), 'Inspect a prior continuation before repeating'
    recovery.atomic(recovery.ART / 'transaction.before_cleanup_continuation.json', transaction)
    transaction['cleanup_continuation'] = {'created_at_utc': recovery.base.now(), 'script_sha256': recovery.sha(Path(__file__)),
        'reason': 'Original controller timed out waiting for fourth allocation cleanup; four owned holds reconciled, three already96G. Stop remaining reviewed learners sequentially, allow independent node cleanups to overlap, wait up to600seconds and validate each stopped writer before resizing. Release only after all12 updates.'}
    recovery.atomic(receipt, transaction)
    for a in actions:
        if a['own_hold']:
            continue
        jid = a['job_id']
        record = recovery.identity(jid, a['run'])
        recovery.stable(a['before'], record, '64G', same_attempt=True)
        a['continuation_live_pressure'] = recovery.base.live_memory(jid, record)
        detail = recovery.checkpoint(jid, a['run'])
        directory = recovery.ART / str(jid) / 'continuation_before_stop'
        directory.mkdir(exist_ok=True)
        a['archives_continuation_before_stop'] = recovery.base.archive(jid, record, detail, directory)
        latest = recovery.identity(jid, a['run'])
        recovery.stable(a['before'], latest, '64G', same_attempt=True)
        a['checkpoint_immediately_before_stop'] = recovery.checkpoint(jid, a['run'])
        a.update(hold_intent=True, hold_method='requeuehold')
        recovery.atomic(receipt, transaction)
        recovery.call('scontrol', 'requeuehold', str(jid))
        a['own_hold'] = True
        recovery.atomic(receipt, transaction)
        print(json.dumps({'job_id': jid, 'stop_requested': True}), flush=True)
    deadline = time.monotonic() + 600
    while not all(a['resized'] for a in actions):
        for a in actions:
            if a['resized']:
                continue
            jid = a['job_id']
            held = recovery.identity(jid, a['run'])
            recovery.stable(a['before'], held, '64G')
            assert a['own_hold'] and recovery.field(held, 'Priority') == '0'
            if recovery.field(held, 'JobState') in ('RUNNING', 'COMPLETING'):
                continue
            recovery.base.assert_own_hold(held)
            detail = recovery.checkpoint(jid, a['run'])
            assert detail['checkpoint_step'] >= a['checkpoint_before']['checkpoint_step']
            a['checkpoint_after_writer_stopped'] = detail
            directory = recovery.ART / str(jid) / 'after_stop'
            directory.mkdir(exist_ok=True)
            a['archives_after_stop'] = recovery.base.archive(jid, held, detail, directory)
            recovery.atomic(receipt, transaction)
            recovery.call('scontrol', 'update', f'JobId={jid}', 'MinMemoryNode=98304')
            resized = recovery.identity(jid, a['run'])
            recovery.base.assert_own_hold(resized)
            recovery.stable(a['before'], resized, '96G')
            a.update(resized=True, held_after_update=resized)
            recovery.atomic(receipt, transaction)
            print(json.dumps({'job_id': jid, 'held_at_96g': True, 'resume_step': detail['checkpoint_step']}), flush=True)
        assert time.monotonic() < deadline, 'Cleanup still pending: preserve owned holds for inspection'
        if not all(a['resized'] for a in actions):
            time.sleep(2)
    for a in actions:
        assert a['own_hold'] and a['resized'] and not a['released']
        jid = a['job_id']
        held = recovery.identity(jid, a['run'])
        recovery.base.assert_own_hold(held)
        recovery.stable(a['before'], held, '96G')
        recovery.call('scontrol', 'release', str(jid))
        after = recovery.identity(jid, a['run'])
        recovery.stable(a['before'], after, '96G')
        assert recovery.field(after, 'JobState') in ('PENDING', 'RUNNING')
        assert recovery.field(after, 'Priority') != '0'
        a.update(released=True, after=after, released_at_utc=recovery.base.now())
        recovery.atomic(receipt, transaction)
        print(json.dumps({'job_id': jid, 'released_at_96g': True}), flush=True)
    assert recovery.sha(recovery.base.campaign.E119_LEDGER) == plan['ledger_sha256']
    assert recovery.sha(recovery.base.campaign.E119_CONTINUATIONS) == plan['continuations_sha256']
    transaction.update(applied=True, completed_at_utc=recovery.base.now())
    recovery.atomic(receipt, transaction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
