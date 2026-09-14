#!/usr/bin/env python3
"""Move two owned, never-released E119 clones from drained208 to healthy205/207.

Only ReqNodeList changes in Slurm. Immutable original submission and science
exports remain intact; a separate ledger audit records actual amended placement.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import re
from datetime import datetime
from pathlib import Path

import accelerate_e119_node208_20260909 as launch

b = launch.b
ART = launch.ART / 'healthy_route_amendment'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PAIRS = {31124279: 31163339, 31048181: 31163361}
POOL = {'node205', 'node207'}
NODELIST = 'node205,node207'
FIELDS = ('UserId', 'JobName', 'Account', 'Partition', 'ExcNodeList', 'MinMemoryNode',
          'TimeLimit', 'Nice', 'QOS', 'Dependency', 'Features', 'Requeue', 'NumCPUs',
          'NumTasks', 'CPUs/Task', 'TresPerNode', 'WorkDir', 'Command', 'Comment',
          'StdOut', 'StdErr', 'ReqTRES')


def event(tx, message):
    tx['updated_at_utc'] = b.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'message': message})
    launch.save(TX, tx)


def hosts(value):
    return set(b.command(['scontrol', 'show', 'hostnames', value]).stdout.split())


def healthy_nodes():
    records = {}
    for node in sorted(POOL):
        record = b.command(['scontrol', 'show', 'node', '-o', node]).stdout
        launch.require(not any(x in b.field(record, 'State') for x in ('DOWN', 'DRAIN', 'FAIL')),
                       f'{node} is no longer healthy/schedulable')
        launch.require('gpu:a6000:' in b.field(record, 'Gres') and
                       'lowprio' in b.field(record, 'Partitions').split(','), 'hardware/partition drift')
        records[node] = record
    return records


def route_command(item, original=False):
    source = item['before'] if original else item['held_before_amendment']
    updates = {'--account': b.field(source, 'Account'), '--partition': b.field(source, 'Partition'),
               '--nodelist': b.field(source, 'ReqNodeList') if original else NODELIST,
               '--mem': b.field(source, 'MinMemoryNode'), '--time': b.field(source, 'TimeLimit'),
               '--gres': b.field(source, 'TresPerNode').removeprefix('gres/'),
               '--nice': b.field(source, 'Nice')}
    result = [x for x in item['command'][:-1] if x.split('=', 1)[0] not in updates and x != '--hold']
    result += [f'{k}={v}' for k, v in updates.items()] + [item['command'][-1]]
    launch.require(b.exports(result) == b.exports(item['original_command']), 'forecast changed exports')
    return [result[0], '--test-only', *result[1:]]


def forecast(item):
    results = {}
    for name, original in [('healthy', False), ('original', True)]:
        command = route_command(item, original)
        result = launch.submit(command)
        output = result.stdout + result.stderr
        found = re.search(r'to start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)', output)
        launch.require(found is not None, f'{name} scheduler forecast unavailable')
        results[name] = {'start': found.group(1), 'stdout': result.stdout, 'stderr': result.stderr,
                         'command': command}
    launch.require(datetime.fromisoformat(results['healthy']['start']) <
                   datetime.fromisoformat(results['original']['start']),
                   'healthy route is not forecast earlier than the original exact resource request')
    return results


def profile(item, allowed_nodes, held=True):
    record = b.show(item['new_job_id'])
    launch.require(b.field(record, 'JobId') == str(item['new_job_id']), 'replacement job ID changed')
    for key in FIELDS:
        launch.require(b.field(record, key) == b.field(item['held_before_amendment'], key),
                       f'non-routing resource changed: {key}')
    launch.require(b.submit_tokens(record) == item['command'], 'original immutable SubmitLine changed')
    launch.require(b.exports(b.submit_tokens(record)) == b.exports(item['original_command']), 'science exports changed')
    launch.require(hosts(b.field(record, 'ReqNodeList')) == set(allowed_nodes), 'actual node pool changed')
    launch.require(b.field(record, 'NumNodes') in {'1', '1-1'}, 'node count changed')
    if held:
        for key, expected in {'JobState': 'PENDING', 'Reason': 'JobHeldUser', 'Priority': '0',
                              'StartTime': 'Unknown', 'Restarts': '0', 'NodeList': '', 'RunTime': '00:00:00'}.items():
            launch.require(b.field(record, key) == expected, f'job is not our never-allocated hold: {key}')
        launch.require(not (Path(item['run_dir']) / f"debug_job{item['new_job_id']}").exists(),
                       'held successor unexpectedly has a run directory')
    return record


def prepare():
    launch.require(not PLAN.exists() and not TX.exists(), 'existing immutable amendment plan')
    route = launch.load_transaction(); data = json.loads(launch.LEDGER.read_text()); items = []
    for old, new in PAIRS.items():
        item = copy.deepcopy(launch.item_for(route, old))
        launch.require(item['new_job_id'] == new and item.get('ledger_committed') and
                       not item.get('release_requested') and not item.get('released'), 'unexpected route state')
        launch.old_guard(item, True); launch.authoritative(item, new)
        item['held_before_amendment'] = launch.new_guard(item, held=True)
        profile(item, {'node208'}); launch.safe_run(item, [old, new])
        item['row_before_amendment'] = copy.deepcopy(launch.row(data, item))
        item['forecasts_before'] = forecast(item); items.append(item)
    plan = {'schema': 'e119-healthy-nodepool-amendment-v1', 'created_at_utc': b.now(), 'items': items,
            'controller_sha256': launch.digest(__file__), 'source_plan_sha256': launch.digest(launch.PLAN),
            'source_transaction_sha256': launch.digest(launch.TX), 'healthy_node_records': healthy_nodes(),
            'only_scheduler_change': {'ReqNodeList': {'before': ['node208'], 'after': sorted(POOL)}},
            'scientific_exports_unchanged': True, 'old_held_fallbacks_preserved': True, 'events': []}
    launch.save(PLAN, plan)
    print(json.dumps({'prepared': True, 'pairs': PAIRS, 'forecasts': {i['old_job_id']:
        {k: v['start'] for k, v in i['forecasts_before'].items()} for i in items}, 'scheduler_mutations': False}))


def load():
    plan = json.loads(PLAN.read_text())
    launch.require(launch.digest(__file__) == plan['controller_sha256'], 'amendment controller changed')
    launch.require(launch.digest(launch.PLAN) == plan['source_plan_sha256'] and
                   launch.digest(launch.TX) == plan['source_transaction_sha256'], 'original route receipts changed')
    launch.load_transaction()
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    if TX.exists():
        launch.require(tx.get('plan_sha256') == launch.digest(PLAN), 'transaction plan binding changed')
    else:
        tx['plan_sha256'] = launch.digest(PLAN)
    return tx


def update_ledger(tx, item):
    data = json.loads(launch.LEDGER.read_text()); current = launch.row(data, item)
    launch.authoritative(item, item['new_job_id'])
    if current.get('healthy_route_amendment') == str(TX):
        launch.require(item.get('ledger_update_requested') and current.get('actual_requested_nodes') == sorted(POOL),
                       'unowned ledger amendment')
        item['ledger_updated'] = True; event(tx, f"{item['old_job_id']}: reconciled ledger amendment"); return
    launch.require(current == item['row_before_amendment'], 'target continuation row changed')
    launch.save(ART / f"ledger_before_{item['old_job_id']}.json", data)
    current.update(held_scheduler_record=item['held_after_amendment'], healthy_route_amendment=str(TX),
                   actual_requested_nodes=sorted(POOL))
    data.setdefault('repair_history', []).append({'at': b.now(), 'audit': str(TX), 'old': item['old_job_id'],
        'new': item['new_job_id'], 'only_scheduler_change': 'ReqNodeList node208 -> node205,node207',
        'scientific_exports_unchanged': True})
    launch.require(len(data['continuations']) == 75 and
                   len({r['continuation_job_id'] for r in data['continuations']}) == 75, 'cohort denominator changed')
    image = ART / f"ledger_after_{item['old_job_id']}.json"; launch.save(image, data)
    item['ledger_update_requested'] = True; item['ledger_after_sha256'] = launch.digest(image)
    event(tx, f"{item['old_job_id']}: recording actual amended node pool without changing authoritative job ID")
    launch.save(launch.LEDGER, data)
    launch.require(launch.digest(launch.LEDGER) == item['ledger_after_sha256'], 'ledger readback mismatch')
    launch.authoritative(item, item['new_job_id']); item['ledger_updated'] = True
    event(tx, f"{item['old_job_id']}: node-pool provenance recorded")


def apply(old_job):
    tx = load(); item = launch.item_for(tx, old_job)
    launch.old_guard(item, True); launch.authoritative(item, item['new_job_id'])
    if item.get('release_requested') and b.field(b.show(item['new_job_id']), 'Priority') != '0':
        profile(item, POOL, held=False); item['released'] = True
        event(tx, f'{old_job}: reconciled healthy-route release')
        print(json.dumps({'old': old_job, 'new': item['new_job_id'], 'status': 'released'})); return
    actual = hosts(b.field(b.show(item['new_job_id']), 'ReqNodeList'))
    if actual == POOL:
        launch.require(item.get('update_requested'), 'node pool changed without our intent')
        profile(item, POOL); item['node_pool_updated'] = True
    else:
        profile(item, {'node208'}); healthy_nodes(); item['forecasts_at_apply'] = forecast(item)
        launch.safe_run(item, [old_job, item['new_job_id']])
        item['update_requested'] = True; event(tx, f'{old_job}: changing only held ReqNodeList to healthy205/207')
        result = b.command(['scontrol', 'update', f"JobId={item['new_job_id']}", f'NodeList={NODELIST}'], check=False)
        item['update_result'] = {'returncode': result.returncode, 'stderr': result.stderr}
        profile(item, POOL); item['node_pool_updated'] = True
        event(tx, f'{old_job}: unchanged science/resources and amended node pool verified')
    item['held_after_amendment'] = profile(item, POOL)
    update_ledger(tx, item)
    healthy_nodes(); item['forecasts_before_release'] = forecast(item)
    launch.safe_run(item, [old_job, item['new_job_id']]); launch.old_guard(item, True)
    profile(item, POOL)
    item['release_requested'] = True; event(tx, f'{old_job}: releasing held clone onto verified earlier healthy route')
    result = b.command(['scontrol', 'release', str(item['new_job_id'])], check=False)
    after = profile(item, POOL, held=False)
    launch.require(b.field(after, 'Priority') != '0' and b.field(after, 'JobState') in {'PENDING', 'RUNNING', 'CONFIGURING'},
                   'release did not become schedulable: ' + result.stderr)
    item.update(released=True, release_record=after)
    event(tx, f'{old_job}: healthy-route continuation released; original held fallback retained')
    print(json.dumps({'old': old_job, 'new': item['new_job_id'], 'status': 'released',
                      'state': b.field(after, 'JobState'), 'reason': b.field(after, 'Reason')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'apply')); parser.add_argument('--old-job-id', type=int)
    args = parser.parse_args()
    with launch.LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'prepare': prepare()
        else:
            launch.require(args.old_job_id in PAIRS, 'exact --old-job-id is required'); apply(args.old_job_id)
