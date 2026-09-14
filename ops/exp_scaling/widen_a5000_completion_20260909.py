#!/usr/bin/env python3
"""Widen five pending A5000 allocations without changing job IDs or runtimes."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path
import accelerate_a5000_completion_20260909 as parent

b = parent.base
ROOT = parent.ROOT
ART = ROOT / 'var/artifacts/campaign_a5000_widening_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/campaign_a5000_widening_20260909.md'
JOBS = {31158503, 31158504, 31158505, 31158506, 31158507}
OLD_NODES = {'node202', 'node203'}
NEW_NODES = {'node105', 'node202', 'node203', 'node204'}
PRESERVE = ('Account', 'Partition', 'QOS', 'NumCPUs', 'NumTasks', 'CPUs/Task',
            'MinMemoryNode', 'TresPerNode', 'ReqTRES', 'Nice', 'Requeue',
            'ExcNodeList', 'Command', 'WorkDir', 'StdOut', 'StdErr', 'TimeLimit',
            'Dependency', 'Comment')


def nodes(record):
    return set(b.command(['scontrol', 'show', 'hostnames', b.field(record, 'ReqNodeList')]).stdout.split())


def preserved(record, original):
    assert b.submit_tokens(record) == b.submit_tokens(original), 'submitted launcher or exports changed'
    for key in PRESERVE:
        assert b.field(record, key) == b.field(original, key), (key, b.field(record, key), b.field(original, key))
    assert b.field(record, 'NumNodes') in ('1', '1-1')


def identity(item):
    if item['cohort'] == 'e118':
        for path in (parent.SOURCE, parent.AGGREGATE):
            row = next(r for r in json.loads(path.read_text())['runs'] if r['run_dir'] == item['run_dir'])
            assert row['job_id'] == item['new_job_id'], ('mapping changed', item['new_job_id'], path)
            assert all(row[k] == item[k] for k in ('domain', 'arm', 'seed', 'run_dir', 'run_stamp'))
    else:
        mapping = parent.campaign.e120_continuation_jobs(parent.campaign.E120_LEDGER)
        assert mapping[item['row_before']['original_job_id']] == item['new_job_id']
        row = next(r for r in json.loads(parent.CONTINUATIONS.read_text())['continuations'] if r['run_dir'] == item['run_dir'])
        assert row['continuation_job_id'] == item['new_job_id']
        assert all(row[k] == item[k] for k in ('domain', 'seed', 'run_dir', 'run_stamp'))


def safe(item):
    identity(item)
    parent.safe_run(item, [item['new_job_id']])
    assert parent.digest(item['original_command'][-1]) == item['launcher_sha256']


def prepare():
    assert not PLAN.exists() and not TX.exists(), 'inspect existing transaction instead of replacing it'
    prior = json.loads(parent.TX.read_text())
    assert prior['status'] == 'complete'
    assert {i['new_job_id'] for i in prior['items']} == JOBS
    assert parent.runtime_fingerprints(prior['items']) == prior['runtime_fingerprints']
    node_records = {}
    for node in sorted(NEW_NODES):
        record = b.command(['scontrol', 'show', 'node', '-o', node]).stdout
        assert b.field(record, 'Gres') == 'gpu:a5000:10'
        assert 'lowprio' in b.field(record, 'Partitions').split(',')
        # Existing candidates may be drained; Slurm excludes those until healthy.
        if node not in OLD_NODES:
            assert not any(x in b.field(record, 'State') for x in ('DOWN', 'DRAIN', 'FAIL'))
        node_records[node] = record
    items = []
    for original in prior['items']:
        item = copy.deepcopy(original)
        item['job_id'] = item['new_job_id']
        item['before_widening'] = b.show(item['job_id'])
        item['status'] = 'planned'
        item.pop('owned_hold', None)
        if b.field(item['before_widening'], 'JobState') != 'PENDING':
            item['status'] = 'skipped_started'
        else:
            parent.new_guard(item, held=False)
            assert b.field(item['before_widening'], 'Priority') != '0'
            assert nodes(item['before_widening']) == OLD_NODES
            safe(item)
        items.append(item)
    plan = {'schema': 'campaign-a5000-pending-widening-v1', 'created_at': b.now(),
            'status': 'prepared', 'items': items, 'nodes': sorted(NEW_NODES),
            'node_records': node_records, 'runtime_fingerprints': prior['runtime_fingerprints'],
            'controller_sha256': parent.digest(__file__), 'protocol_sha256': parent.digest(PROTOCOL),
            'parent_controller_sha256': parent.digest(parent.__file__),
            'parent_transaction_sha256': parent.digest(parent.TX),
            'parent_protocol_sha256': parent.digest(parent.PROTOCOL),
            'scheduler_only': True, 'runtime_changed': False, 'job_ids_changed': False,
            'ledgers_written': False, 'events': []}
    b.atomic(PLAN, plan)
    print(json.dumps({'prepared': True, 'jobs': [i['job_id'] for i in items], 'nodes': plan['nodes']}))


def apply():
    plan = json.loads(PLAN.read_text())
    for path, key in ((__file__, 'controller_sha256'), (PROTOCOL, 'protocol_sha256'),
                      (parent.__file__, 'parent_controller_sha256'), (parent.TX, 'parent_transaction_sha256'),
                      (parent.PROTOCOL, 'parent_protocol_sha256')):
        assert parent.digest(path) == plan[key], ('provenance changed', path)
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    if tx['status'] == 'complete':
        print(json.dumps({'already_complete': True})); return
    def event(message):
        tx['events'].append({'at': b.now(), 'message': message}); b.atomic(TX, tx)
    assert parent.runtime_fingerprints(tx['items']) == tx['runtime_fingerprints']
    tx['status'] = 'applying'; event('Beginning pending-only node-pool expansion; no ledger writes')
    for item in tx['items']:
        if item['status'] in ('released', 'skipped_started', 'skipped_complete', 'skipped_preexisting_hold'):
            continue
        jid = item['job_id']; before = item['before_widening']
        current = b.show(jid)
        if b.field(current, 'JobState') != 'PENDING':
            item['status'] = 'skipped_started'; event(f'{jid}: started; left untouched'); continue
        if parent.recovery.complete(Path(item['run_dir'])):
            item['status'] = 'skipped_complete'; event(f'{jid}: terminal scientific receipt; left untouched'); continue
        preserved(current, before); safe(item)
        if not item.get('owned_hold'):
            if b.field(current, 'Priority') == '0' and not item.get('widen_hold_requested'):
                item['status'] = 'skipped_preexisting_hold'; event(f'{jid}: preexisting hold left untouched'); continue
            if not (item.get('widen_hold_requested') and b.field(current, 'Reason') == 'JobHeldUser'):
                assert nodes(current) == OLD_NODES
                item['widen_hold_requested'] = True; event(f'{jid}: requesting hold for node-only amendment')
                b.command(['scontrol', 'hold', str(jid)])
            held = b.show(jid)
            if b.field(held, 'JobState') != 'PENDING':
                b.command(['scontrol', 'release', str(jid)])
                item['status'] = 'skipped_started'; event(f'{jid}: allocation raced hold; released unchanged'); continue
            assert b.field(held, 'Reason') == 'JobHeldUser'
            item['owned_hold'] = True; event(f'{jid}: pending hold acknowledged')
        try:
            held = b.show(jid); preserved(held, before); safe(item)
            assert b.field(held, 'JobState') == 'PENDING' and b.field(held, 'Reason') == 'JobHeldUser'
            assert nodes(held) in (OLD_NODES, NEW_NODES)
            if nodes(held) != NEW_NODES:
                command = ['scontrol', 'update', f'JobId={jid}', f'NodeList={",".join(sorted(NEW_NODES))}']
                item['update_command'] = command; event(f'{jid}: adding node105 and node204 candidates')
                b.command(command)
            after = b.show(jid); preserved(after, before)
            assert nodes(after) == NEW_NODES and b.field(after, 'Reason') == 'JobHeldUser'
            item['held_after'] = after; event(f'{jid}: unchanged resources and full exports verified')
        except BaseException as exc:
            item['error'] = repr(exc)
            check = b.show(jid)
            try:
                preserved(check, before)
                assert nodes(check) in (OLD_NODES, NEW_NODES)
                assert b.field(check, 'JobState') == 'PENDING' and b.field(check, 'Reason') == 'JobHeldUser'
            except BaseException:
                tx['status'] = 'stopped_for_reconciliation'; event(f'{jid}: unexpected partial state; retained owned hold')
                raise
            b.command(['scontrol', 'release', str(jid)]); item['owned_hold'] = False
            tx['status'] = 'stopped_for_reconciliation'; event(f'{jid}: safe state verified; released owned hold after error')
            raise
        b.command(['scontrol', 'release', str(jid)]); item['owned_hold'] = False
        final = b.show(jid); preserved(final, before); assert nodes(final) == NEW_NODES
        assert b.field(final, 'Priority') != '0'
        item['after_release'] = final; item['status'] = 'released'; event(f'{jid}: widened job released')
    assert parent.runtime_fingerprints(tx['items']) == tx['runtime_fingerprints']
    assert parent.digest(parent.TX) == tx['parent_transaction_sha256']
    assert parent.digest(parent.PROTOCOL) == tx['parent_protocol_sha256']
    for item in tx['items']: identity(item)
    tx['status'] = 'complete'; tx['completed_at'] = b.now(); event('All pending candidates widened or skipped after starting; original transaction preserved')
    print(json.dumps({'status': tx['status'], 'jobs': [{'job': i['job_id'], 'status': i['status']} for i in tx['items']]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('prepare', 'apply'))
    args = parser.parse_args(); ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); globals()[args.phase]()
