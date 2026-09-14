#!/usr/bin/env python3
"""Roll back only two never-started held E119 node208 clones after node drain.

The original route controller and its receipts remain unchanged. Durable rollback
intents, exact held ownership, no allocation/data writes, and the shared ledger
lock protect cancellation, restoration and original-release reconciliation.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path

import accelerate_e119_node208_20260909 as launch

b = launch.b
ART = launch.ART / 'rollback_node_drain'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PAIRS = {31124279: 31163339, 31048181: 31163361}


def event(tx, message):
    tx['updated_at_utc'] = b.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'message': message})
    launch.save(TX, tx)


def no_gpu_writes(item):
    launch.require(not (Path(item['run_dir']) / f"debug_job{item['new_job_id']}").exists(),
                   'successor has a run directory; inspect possible writes')
    name = b.field(item['before'], 'JobName')
    for suffix in ('out', 'err'):
        path = launch.ROOT / f"var/artifacts/logs/{name}-{item['new_job_id']}.{suffix}"
        launch.require(not path.exists() or path.stat().st_size == 0,
                       'successor has nonempty logs; inspect possible allocation')


def never_started(item, held=True):
    record = launch.new_guard(item, held=held)
    for key, allowed in {'StartTime': {'Unknown', 'N/A'}, 'Restarts': {'0'},
                         'NodeList': {''}, 'RunTime': {'00:00:00'}}.items():
        launch.require(b.field(record, key) in allowed, f'successor was allocated: {key}')
    no_gpu_writes(item)
    return record


def prepare():
    launch.require(not PLAN.exists() and not TX.exists(), 'existing immutable rollback plan')
    route = launch.load_transaction()
    data = json.loads(launch.LEDGER.read_text())
    items = []
    for old, new in PAIRS.items():
        item = copy.deepcopy(launch.item_for(route, old))
        launch.require(item['new_job_id'] == new and item.get('old_held') and item.get('ledger_committed'),
                       'unexpected staged route identity')
        launch.require(not item.get('release_requested') and not item.get('released'),
                       'successor release was attempted; outside never-started rollback')
        launch.old_guard(item, True); launch.authoritative(item, new)
        item['held_successor_before'] = never_started(item)
        launch.safe_run(item, [old, new])
        item['promoted_row_before'] = copy.deepcopy(launch.row(data, item))
        item['rollback_status'] = 'prepared'
        items.append(item)
    plan = {'schema': 'e119-node208-never-started-rollback-v1', 'created_at_utc': b.now(),
            'reason': 'Node208 re-entered DRAIN after NHC reported six overheated GPUs; no E119 clone was released.',
            'items': items, 'events': [], 'controller_sha256': launch.digest(__file__),
            'source_plan_sha256': launch.digest(launch.PLAN),
            'source_transaction_sha256': launch.digest(launch.TX),
            'node_record': b.command(['scontrol', 'show', 'node', '-o', 'node208']).stdout}
    launch.save(PLAN, plan)
    print(json.dumps({'prepared': True, 'pairs': PAIRS, 'scheduler_mutations': False}))


def load():
    plan = json.loads(PLAN.read_text())
    launch.require(launch.digest(__file__) == plan['controller_sha256'], 'rollback controller drift')
    launch.require(launch.digest(launch.PLAN) == plan['source_plan_sha256'] and
                   launch.digest(launch.TX) == plan['source_transaction_sha256'],
                   'source route receipts changed; coordinate before rollback')
    launch.load_transaction()  # Original scientific/runtime/data/model fingerprints.
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    if TX.exists():
        launch.require(tx.get('plan_sha256') == launch.digest(PLAN), 'rollback plan binding changed')
    else:
        tx['plan_sha256'] = launch.digest(PLAN)
    return tx


def canceled_successor(item):
    launch.require(item.get('cancel_requested'), 'unowned successor cancellation')
    launch.require(item['new_job_id'] not in b.queue(), 'canceled successor still appears in queue')
    launch.require(launch.recovery.state(item['new_job_id']) == 'CANCELLED',
                   'successor cancellation not yet confirmed by accounting')
    record = never_started(item, held=False)
    launch.require(b.field(record, 'JobState') == 'CANCELLED', 'successor is not canceled')


def restore_row(tx, item):
    data = json.loads(launch.LEDGER.read_text())
    current = launch.row(data, item)
    marker = {'audit': str(TX), 'old': item['old_job_id'], 'canceled_new': item['new_job_id']}
    matching = [r for r in data.get('repair_history', [])
                if all(r.get(k) == v for k, v in marker.items())]
    if current == item['row_before']:
        launch.require(item.get('restore_requested') and len(matching) == 1,
                       'original mapping restored without this rollback intent')
        launch.authoritative(item, item['old_job_id'])
        item['mapping_restored'] = True
        event(tx, f"{item['old_job_id']}: reconciled restored original mapping")
        return
    launch.require(current == item['promoted_row_before'], 'target continuation changed before rollback')
    launch.authoritative(item, item['new_job_id'])
    launch.require(not matching, 'duplicate rollback history')
    launch.save(ART / f"ledger_before_{item['old_job_id']}.json", data)
    index = data['continuations'].index(current)
    data['continuations'][index] = copy.deepcopy(item['row_before'])
    data.setdefault('repair_history', []).append({**marker, 'at': b.now(),
        'reason': 'node208 drain; held successor never allocated', 'original_row_restored_exactly': True})
    launch.require(len(data['continuations']) == 75 and
                   len({r['continuation_job_id'] for r in data['continuations']}) == 75,
                   'continuation uniqueness or denominator changed')
    image = ART / f"ledger_after_{item['old_job_id']}.json"
    launch.save(image, data)
    item['restore_requested'] = True; item['restore_after_sha256'] = launch.digest(image)
    event(tx, f"{item['old_job_id']}: restoring exactly one original continuation row")
    launch.save(launch.LEDGER, data)
    launch.require(launch.digest(launch.LEDGER) == item['restore_after_sha256'], 'ledger restoration readback differs')
    launch.authoritative(item, item['old_job_id'])
    item['mapping_restored'] = True
    event(tx, f"{item['old_job_id']}: original mapping restored and verified")


def original_released(item):
    record = b.show(item['old_job_id'])
    launch.require(b.submit_tokens(record) == item['original_command'], 'original SubmitLine changed')
    for key in launch.OLD_FIELDS:
        launch.require(b.field(record, key) == b.field(item['before'], key), f'original field changed: {key}')
    launch.require(b.field(record, 'Priority') != '0' and
                   b.field(record, 'JobState') in {'PENDING', 'RUNNING', 'CONFIGURING'},
                   'original is not schedulable after release')
    return record


def apply(old_job):
    tx = load(); item = launch.item_for(tx, old_job)
    if item.get('rollback_complete'):
        launch.authoritative(item, old_job)
        print(json.dumps({'old': old_job, 'status': 'restored', 'canceled_new': item['new_job_id']})); return
    if not item.get('successor_canceled'):
        if item.get('cancel_requested') and item['new_job_id'] not in b.queue():
            canceled_successor(item)
        else:
            launch.old_guard(item, True); launch.authoritative(item, item['new_job_id'])
            never_started(item); launch.safe_run(item, [old_job, item['new_job_id']])
            item['cancel_requested'] = True
            event(tx, f"{old_job}: canceling exact never-allocated held successor {item['new_job_id']}")
            result = b.command(['scancel', str(item['new_job_id'])], check=False)
            item['cancel_result'] = {'returncode': result.returncode, 'stderr': result.stderr}
            canceled_successor(item)
        item['successor_canceled'] = True
        event(tx, f'{old_job}: successor cancellation and absence of writes confirmed')
    canceled_successor(item)
    if not item.get('mapping_restored'):
        launch.old_guard(item, True); launch.safe_run(item, [old_job])
        restore_row(tx, item)
    launch.authoritative(item, old_job)
    if item.get('old_release_requested') and b.field(b.show(old_job), 'Priority') != '0':
        item['original_release_record'] = original_released(item)
    else:
        launch.old_guard(item, True); launch.safe_run(item, [old_job])
        item['old_release_requested'] = True
        event(tx, f'{old_job}: releasing exact original hold after mapping restoration')
        result = b.command(['scontrol', 'release', str(old_job)], check=False)
        item['old_release_result'] = {'returncode': result.returncode, 'stderr': result.stderr}
        item['original_release_record'] = original_released(item)
    item.update(rollback_complete=True, rollback_status='restored')
    event(tx, f'{old_job}: original route restored; failed node208 attempt retained as audit history')
    print(json.dumps({'old': old_job, 'status': 'restored', 'canceled_new': item['new_job_id']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'apply'))
    parser.add_argument('--old-job-id', type=int)
    args = parser.parse_args()
    with launch.LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'prepare':
            prepare()
        else:
            launch.require(args.old_job_id in PAIRS, 'exact --old-job-id is required')
            apply(args.old_job_id)
