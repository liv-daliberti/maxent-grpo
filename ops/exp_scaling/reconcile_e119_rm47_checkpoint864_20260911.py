#!/usr/bin/env python3
"""Preserve a newly completed864 checkpoint in the existing held RAM repair."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path

import recover_e119_rm47_memory128_20260911 as repair

ART = repair.ART
ADDENDUM = ART / 'checkpoint864_addendum.json'
PROTOCOL = repair.ROOT / 'paper/preregistration/e119_rm47_checkpoint864_20260911.md'
ORIGINAL_LOAD = repair.load
require, atomic = repair.require, repair.atomic
CHANGED = {'resume_step', 'allow_repeat_updates', 'checkpoint', 'checkpoint_files', 'partial_path', 'quarantine_path'}


def amend(plan, addendum):
    newer = copy.deepcopy(plan)
    newer.update(resume_step=864, allow_repeat_updates=0,
                 checkpoint=addendum['checkpoint'], checkpoint_files=addendum['checkpoint_files'],
                 partial_path=None, quarantine_path=None)
    require(all(newer[k] == plan[k] for k in plan if k not in CHANGED), 'Unrelated prepared invariant changed')
    return newer


def prefix_valid(tx, prefix):
    require(all(tx.get(k) == v for k, v in prefix.items() if k != 'events'), 'Original transaction nonce/state prefix changed')
    require(tx.get('events', [])[:len(prefix.get('events', []))] == prefix.get('events', []), 'Original transaction history changed')


def load():
    plan, tx = ORIGINAL_LOAD()
    addendum = json.loads(ADDENDUM.read_text())
    require(repair.sha(ADDENDUM) == tx.get('checkpoint864_addendum_sha256', repair.sha(ADDENDUM)), 'Recorded addendum digest differs')
    require(repair.sha(__file__) == addendum['controller_sha256'] and repair.sha(PROTOCOL) == addendum['protocol_sha256'], 'Addendum source/protocol changed')
    require(repair.sha(repair.PLAN) == addendum['original_plan_sha256'], 'Original plan changed')
    require(repair.sha(repair.OLD_TX) == addendum['old_final_tx_sha256'], 'Stopped old guard changed')
    prefix_valid(tx, addendum['transaction_prefix'])
    return amend(plan, addendum), tx


def held_before_memory(plan, tx):
    require(tx.get('hold_intent') and tx.get('old_cpu_stop_requested') and tx.get('old_final_tx_sha256'), 'Original stop transaction not acknowledged')
    require(tx.get('new_cpu_job_id') == 31246560 and tx.get('cpu_submission_receipt') == '31246560', 'Different CPU successor')
    require(not any(tx.get(k) for k in ('quarantine_intent', 'quarantined', 'memory_intent', 'memory_updated',
                                       'ledger_commit_intent', 'ledger_updated', 'guard_state_copied', 'target_release_requested')),
            'Transaction advanced beyond the reviewed positive-checkpoint boundary')
    repair.state(plan['item'], amended=False, held=True)
    require(repair.OLD_CPU not in repair.b.queue(), 'Old CPU still active')
    require(repair.sha(repair.OLD_TX) == tx['old_final_tx_sha256'], 'Old CPU transaction changed')
    repair.h.new_cpu(plan, tx['new_cpu_job_id'], held=True)


def prepare():
    require(not ADDENDUM.exists(), 'Immutable addendum already exists')
    plan, tx = ORIGINAL_LOAD(); held_before_memory(plan, tx)
    require(plan['resume_step'] == 768 and plan['allow_repeat_updates'] == 96, 'Unexpected original recovery')
    d = repair.memory.timing_and_checkpoint(repair.TARGET, plan['item']['identity'])
    require(d['checkpoint_step'] == 864 and d['current_step'] == 863 and not d['rejected_checkpoints']
            and d['unsaved_steps'] == 0 and not d['fresh_restart'], 'Durable864 zero-loss boundary not proven')
    require(d['saved_counter_validation']['saved_counters'] == dict.fromkeys(
        ('global_steps', 'global_step', 'prompt_batches_consumed_total'), 864), '864 counters disagree')
    require(Path(d['checkpoint']) == Path(plan['partial_path']), 'New checkpoint is not the protected864 path')
    addendum = {'schema':'e119-rm47-newly-durable864-addendum-v1', 'at_utc':repair.b.now(),
                'job_id':repair.TARGET, 'original_plan_sha256':repair.sha(repair.PLAN),
                'transaction_prefix_sha256':repair.sha(repair.TX), 'transaction_prefix':copy.deepcopy(tx),
                'old_final_tx_sha256':tx['old_final_tx_sha256'],
                'controller_sha256':repair.sha(__file__), 'protocol_sha256':repair.sha(PROTOCOL),
                'checkpoint':d, 'checkpoint_files':repair.checkpoint_files(d['checkpoint']),
                'resume_step':864, 'repeat_updates_bound':0, 'quarantine_required':False,
                'changed_in_memory_plan_keys':sorted(CHANGED), 'scheduler_mutations':False}
    newer = amend(plan, addendum)
    repair.checkpoint(newer)
    held_before_memory(plan, tx)
    atomic(ADDENDUM, addendum)
    print(json.dumps({'prepared':str(ADDENDUM), 'resume_step':864, 'repeat_updates_bound':0, 'scheduler_mutations':False}))


def check():
    plan, tx = load()
    held_before_memory(plan, tx); repair.checkpoint(plan)
    print(json.dumps({'valid':True, 'resume_step':864, 'repeat_updates_bound':0, 'scheduler_mutations':False}))


def record_addendum():
    plan, tx = load()
    if tx.get('checkpoint864_addendum_sha256'):
        require(tx.get('checkpoint864_addendum') == str(ADDENDUM), 'Recorded addendum path differs')
        return
    held_before_memory(plan, tx); repair.checkpoint(plan)
    tx.update(checkpoint864_addendum=str(ADDENDUM), checkpoint864_addendum_sha256=repair.sha(ADDENDUM),
              actual_resume_step=864, actual_repeat_updates_bound=0)
    repair.h.save(tx, 'Newly durable864 accepted after original stop; zero updates repeated and no quarantine; original768 plan retained')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare','check','route','activate','release'))
    a = p.parse_args(); repair.g.install_bounded_scheduler()
    with (ART/'amendment.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase in ('prepare','check'):
            with repair.g.LEDGER_LOCK.open('a+') as ledger:
                fcntl.flock(ledger, fcntl.LOCK_EX); globals()[a.phase]()
            return
        repair.load = load
        if a.phase == 'route':
            # Record provenance under the common ledger lock, then reuse the
            # original route. Its persisted hold/CPU-stop intents skip both calls.
            with repair.g.LEDGER_LOCK.open('a+') as ledger:
                fcntl.flock(ledger, fcntl.LOCK_EX); record_addendum()
            repair.route()
        elif a.phase == 'activate':
            load(); repair.h.activate()
        else:
            repair.release()


if __name__ == '__main__': main()
