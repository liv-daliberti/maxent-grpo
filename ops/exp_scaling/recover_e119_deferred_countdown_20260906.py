#!/usr/bin/env python3
"""Bounded protected-save recovery with an explicit unrelated-ledger rebase."""
import argparse
import json
from pathlib import Path
import time

import recover_e119_throttling_20260906 as recovery

JID = 31045874
ART = recovery.ART
AMEND = ART / 'deferred_ledger_rebase_owner.json'
PREVIOUS_AMEND = ART / 'deferred_ledger_rebase.json'
PREVIOUS_CONTROLLER = ART / 'deferred_controller_before_owner_rebase.py'
BEFORE = ART / 'continuations_before_routing.json'
AFTER = ART / 'continuations_after_owner_routing.json'


def validate_change(before, after):
    assert set(before) == set(after)
    for key in before:
        if key != 'continuations':
            assert before[key] == after[key], key
    old = {r['original_job_id']: r for r in before['continuations']}
    new = {r['original_job_id']: r for r in after['continuations']}
    assert set(old) == set(new)
    assert [key for key in old if old[key] != new[key]] == [31014456]
    assert old[31014456]['continuation_job_id'] == 31048176
    assert new[31014456]['continuation_job_id'] == 31099622
    for key in ('domain', 'arm', 'seed', 'run_dir', 'run_stamp', 'original_job_id'):
        assert old[31014456][key] == new[31014456][key], key


def prepare():
    assert not AMEND.exists(), 'Never overwrite an existing ledger amendment'
    previous = json.loads(PREVIOUS_AMEND.read_text())
    assert recovery.sha(PREVIOUS_CONTROLLER) == previous['wrapper_sha256']
    assert recovery.sha(ART / 'continuations_after_routing.json') == previous['new_continuation_sha256']
    plan = json.loads((ART / 'plan.json').read_text())
    assert recovery.sha(BEFORE) == plan['continuation_sha256']
    assert recovery.sha(recovery.base.campaign.E119_LEDGER) == plan['ledger_sha256']
    after = recovery.base.campaign.E119_CONTINUATIONS.read_bytes()
    validate_change(json.loads(BEFORE.read_text()), json.loads(after))
    AFTER.write_bytes(after)
    entry = next(e for e in plan['entries'] if e['job_id'] == JID)
    record, run = recovery.identity(JID)
    recovery.stable(entry['before'], record, 96, same_attempt=True)
    assert run == entry['run']
    amendment = dict(created_at_utc=recovery.base.now(), job_id=JID,
                     basis='Independently verified the same MathIR cell moved from preempted lowprio31099437 to owner10531099622, including all102 scientific exports and executed receipt; remaining mutation target is still only Countdown31045874.',
                     old_continuation_sha256=recovery.sha(BEFORE),
                     new_continuation_sha256=recovery.sha(AFTER),
                     original_plan_sha256=recovery.sha(ART / 'plan.json'),
                     wrapper_sha256=recovery.sha(__file__),
                     previous_amendment_sha256=recovery.sha(PREVIOUS_AMEND),
                     previous_wrapper_sha256=recovery.sha(PREVIOUS_CONTROLLER),
                     independent_proof_sha256=recovery.sha(ART / 'second_rebase_independent_proof.json'),
                     only_changed_original_job_id=31014456,
                     old_effective_job_id=31048176, new_effective_job_id=31099622,
                     deferred_mapping_and_science_unchanged=True,
                     before_path=str(BEFORE), after_path=str(AFTER))
    recovery.atomic(AMEND, amendment)
    print(json.dumps(amendment), flush=True)


def load_rebased(jid):
    assert jid == JID, 'This ledger rebase permits only the deferred Countdown repair'
    plan = json.loads((ART / 'plan.json').read_text())
    amendment = json.loads(AMEND.read_text())
    assert amendment['original_plan_sha256'] == recovery.sha(ART / 'plan.json')
    assert amendment['wrapper_sha256'] == recovery.sha(__file__)
    assert amendment['previous_amendment_sha256'] == recovery.sha(PREVIOUS_AMEND)
    assert amendment['previous_wrapper_sha256'] == recovery.sha(PREVIOUS_CONTROLLER)
    assert amendment['independent_proof_sha256'] == recovery.sha(ART / 'second_rebase_independent_proof.json')
    assert recovery.sha(recovery.__file__) == plan['script_sha256']
    assert recovery.sha(recovery.base.__file__) == plan['base_sha256']
    assert recovery.sha(recovery.base.campaign.E119_LEDGER) == plan['ledger_sha256']
    assert recovery.sha(BEFORE) == plan['continuation_sha256'] == amendment['old_continuation_sha256']
    assert recovery.sha(AFTER) == recovery.sha(recovery.base.campaign.E119_CONTINUATIONS) == amendment['new_continuation_sha256']
    validate_change(json.loads(BEFORE.read_text()), json.loads(AFTER.read_text()))
    entry = next(e for e in plan['entries'] if e['job_id'] == jid)
    _, run = recovery.identity(jid)
    assert run == entry['run'], 'Deferred cell identity changed'
    path = ART / str(jid) / 'transaction.json'
    transaction = json.loads(path.read_text()) if path.exists() else None
    return plan, entry, path, transaction


def watch():
    recovery.load = load_rebased
    started = time.monotonic()
    observations = []
    while time.monotonic() - started < 3900:
        _, _, path, transaction = load_rebased(JID)
        if transaction and transaction['applied']:
            print(json.dumps({'job_id': JID, 'applied': True, 'receipt': str(path)}), flush=True)
            return
        if transaction:
            assert transaction['own_hold'], 'Uncertain hold ownership requires inspection'
            recovery.finish(JID)
            time.sleep(15)
            continue
        record, run = recovery.identity(JID)
        detail = recovery.checkpoint(JID, run, False)
        partial = []
        for name in detail['newer_partial_checkpoints']:
            partial.extend({'path': str(p), 'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
                           for p in Path(name).glob('*optim_states.pt'))
        observation = dict(at_utc=recovery.base.now(), step=detail['current_step'],
                           checkpoint=detail['checkpoint_step'], ready=detail['ready_to_stop'],
                           partial=partial)
        observations.append(observation)
        recovery.atomic(ART / 'deferred_save_observations.json', observations)
        print(json.dumps(observation), flush=True)
        if detail['ready_to_stop']:
            recovery.stop(JID, False)
        else:
            time.sleep(30)
    raise RuntimeError('65-minute bound reached; no blind extension or unsafe stop')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'watch'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        watch()
