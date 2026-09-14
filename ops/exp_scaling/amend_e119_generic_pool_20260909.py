#!/usr/bin/env python3
"""Broaden ten owned, pending E119 continuations to healthy A6000/A100 nodes.

Only actual GRES type and requested node pool change. Immutable SubmitLine,
science exports, job IDs, full predecessor history, memory and walltime remain.
No running allocation is cancelled or amended, and uncertain mutations are not
repeated. Every mutation requires the shared continuation-ledger lock.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path
import re

import accelerate_e119_healthy_a6000_20260909 as launch
import accelerate_e119_node208_20260909 as legacy

b = launch.b
ROOT = launch.ROOT
ART = ROOT / 'var/artifacts/e119_generic_pool_amendment_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROBE = ROOT / 'var/artifacts/e119_gres_type_probe_20260909/transaction.json'
SOURCES = (legacy.ART / 'healthy_route_amendment/transaction.json', launch.TX)
OLD_IDS = set(legacy.TARGETS)
BEFORE_POOL = {'node205', 'node207'}
POOL_TYPES = {'node205': 'a6000', 'node207': 'a6000', 'node302': 'a100'}
POOL = set(POOL_TYPES)
NODELIST = ','.join(sorted(POOL))
ACTIVE = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}
FIELDS = ('UserId', 'JobName', 'Account', 'Partition', 'ExcNodeList', 'MinMemoryNode',
          'TimeLimit', 'Nice', 'QOS', 'Dependency', 'Features', 'Requeue', 'NumCPUs',
          'NumTasks', 'CPUs/Task', 'WorkDir', 'Command', 'Comment', 'StdOut', 'StdErr',
          'Restarts', 'MinCPUsNode', 'TresPerTask', 'NtasksPerN:B:S:C', 'OverSubscribe',
          'Licenses', 'Contiguous')


def event(tx, message):
    tx['updated_at_utc'] = b.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'message': message})
    launch.save(TX, tx)
    print(json.dumps({'at': tx['updated_at_utc'], 'event': message}), flush=True)


def hosts(value):
    return set(b.command(['scontrol', 'show', 'hostnames', value]).stdout.split())


def tres(value):
    fields = value.split(',')
    result = dict(x.split('=', 1) for x in fields)
    launch.require(len(result) == len(fields), 'duplicate requested TRES')
    return result


def healthy_nodes():
    records = {}
    for node, gpu_type in sorted(POOL_TYPES.items()):
        record = b.command(['scontrol', 'show', 'node', '-o', node]).stdout
        launch.require(b.field(record, 'NodeName') == node, 'node identity changed')
        launch.require(not any(x in b.field(record, 'State') for x in ('DOWN', 'DRAIN', 'FAIL')),
                       f'{node} is no longer healthy/schedulable')
        launch.require(f'gpu:{gpu_type}:' in b.field(record, 'Gres') and
                       'lowprio' in b.field(record, 'Partitions').split(','),
                       f'{node} hardware/partition changed')
        records[node] = record
    return records


def profile(item, record, generic=False, held=False):
    before = item['pool_before_record']
    launch.require(b.field(record, 'JobId') == str(item['new_job_id']), 'job ID changed')
    for key in FIELDS:
        launch.require(b.field(record, key) == b.field(before, key), f'non-routing resource changed: {key}')
    launch.require(b.submit_tokens(record) == item['command'], 'immutable SubmitLine changed')
    launch.require(b.exports(b.submit_tokens(record)) == b.exports(item['original_command']), 'science exports changed')
    launch.require(hosts(b.field(record, 'ReqNodeList')) == (POOL if generic else BEFORE_POOL), 'actual node pool changed')
    launch.require(b.field(record, 'TresPerNode') == ('gres/gpu:1' if generic else 'gres/gpu:a6000:1'),
                   'actual GPU request changed')
    expected = tres(b.field(before, 'ReqTRES'))
    launch.require(expected.get('gres/gpu') == '1' and expected.get('gres/gpu:a6000') == '1',
                   'original request must be exactly one A6000')
    if generic:
        expected.pop('gres/gpu:a6000')
    launch.require(tres(b.field(record, 'ReqTRES')) == expected, 'requested TRES changed beyond GPU type')
    launch.require(b.field(record, 'NumNodes') in {'1', '1-1'}, 'node count changed')
    if held:
        launch.require(item.get('pool_hold_requested'), 'cannot claim an unowned hold')
        for key, expected_value in {'JobState': 'PENDING', 'Reason': 'JobHeldUser', 'Priority': '0',
                                    'Restarts': '0', 'NodeList': '', 'RunTime': '00:00:00', 'StartTime': 'Unknown'}.items():
            launch.require(b.field(record, key) == expected_value, f'not an owned never-allocated hold: {key}')
        launch.require(not (Path(item['run_dir']) / f"debug_job{item['new_job_id']}").exists(), 'job has already written runtime output')
    return record


def forecast_command(item, generic):
    before = item['pool_before_record']
    updates = {'--account': b.field(before, 'Account'), '--partition': b.field(before, 'Partition'),
               '--nodelist': NODELIST if generic else ','.join(sorted(BEFORE_POOL)),
               '--mem': b.field(before, 'MinMemoryNode'), '--time': b.field(before, 'TimeLimit'),
               '--gres': 'gpu:1' if generic else 'gpu:a6000:1', '--nice': b.field(before, 'Nice')}
    tokens = [x for x in item['command'][:-1] if x.split('=', 1)[0] not in updates and x != '--hold']
    tokens += [f'{key}={value}' for key, value in updates.items()] + [item['command'][-1]]
    launch.require(b.exports(tokens) == b.exports(item['original_command']), 'forecast changed scientific exports')
    return [tokens[0], '--test-only', *tokens[1:]]


def forecast(item, generic):
    command = forecast_command(item, generic)
    result = launch.submit(command)
    found = re.search(r'to start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)', result.stdout + result.stderr)
    launch.require(found is not None, 'scheduler forecast unavailable')
    return {'start': found.group(1), 'stdout': result.stdout, 'stderr': result.stderr, 'command': command}


def prepare():
    launch.require(not PLAN.exists() and not TX.exists(), 'existing immutable amendment plan')
    probe = json.loads(PROBE.read_text())
    launch.require(probe.get('type_change_supported') is True and
                   probe.get('status') == 'exact_held_probe_cancelled', 'owned held GRES probe did not pass')
    launch.load_transaction(); legacy.load_transaction()
    data = json.loads(launch.LEDGER.read_text()); items = []
    for source in SOURCES:
        route = json.loads(source.read_text())
        for source_item in route['items']:
            item = copy.deepcopy(source_item)
            launch.require(item.get('new_job_id') and item.get('released') and item.get('release_requested') and
                           item.get('ledger_committed'), 'all ten source routes must be complete and released')
            launch.old_guard(item, True); launch.authoritative(item, item['new_job_id'])
            record = b.show(item['new_job_id'])
            item['pool_before_record'] = record
            profile(item, record)
            launch.require(b.field(record, 'JobState') == 'PENDING' and b.field(record, 'Priority') != '0',
                           'only released pending successors may enter this amendment')
            launch.require(b.field(record, 'Restarts') == '0' and b.field(record, 'RunTime') == '00:00:00'
                           and not b.field(record, 'NodeList'), 'successor has already allocated')
            launch.safe_run(item, [item['old_job_id'], item['new_job_id']])
            item['row_before_pool'] = copy.deepcopy(launch.row(data, item))
            item['pool_source_transaction'] = str(source)
            items.append(item)
    launch.require(len(items) == 10 and {i['old_job_id'] for i in items} == OLD_IDS
                   and len({i['new_job_id'] for i in items}) == 10, 'exact ten-job scope changed')
    launch.require(len(data['continuations']) == 75 and len({r['continuation_job_id'] for r in data['continuations']}) == 75,
                   'continuation denominator changed')
    # Exact resource classes are forecast separately; broadening preserves all
    # prior eligible nodes and does not require every forecast to be earlier.
    forecasts = {}
    for item in items:
        key = '/'.join(b.field(item['pool_before_record'], x) for x in ('MinMemoryNode', 'TimeLimit', 'Nice', 'NumCPUs'))
        if key not in forecasts:
            forecasts[key] = {'before': forecast(item, False), 'generic': forecast(item, True)}
    plan = {'schema': 'e119-generic-gpu-pool-amendment-v1', 'created_at_utc': b.now(), 'items': items,
            'controller_sha256': launch.digest(__file__), 'probe_sha256': launch.digest(PROBE),
            'sources_sha256': {str(p): launch.digest(p) for p in SOURCES},
            'source_plans_sha256': {str(p): launch.digest(p) for p in (launch.PLAN, legacy.PLAN)},
            'helper_sha256': {str(Path(m.__file__)): launch.digest(m.__file__) for m in (launch, legacy)},
            'healthy_node_records': healthy_nodes(), 'resource_forecasts': forecasts,
            'only_scheduler_changes': {'ReqNodeList': {'before': sorted(BEFORE_POOL), 'after': sorted(POOL)},
                                       'GRES': {'before': 'gpu:a6000:1', 'after': 'gpu:1'}},
            'scientific_exports_unchanged': True, 'same_job_ids': True, 'old_held_fallbacks_preserved': True, 'events': []}
    launch.save(PLAN, plan)
    print(json.dumps({'prepared': True, 'job_ids': [i['new_job_id'] for i in items], 'resource_forecasts':
        {key: {route: details['start'] for route, details in rows.items()} for key, rows in forecasts.items()}, 'scheduler_mutations': False}))


def load():
    plan = json.loads(PLAN.read_text())
    launch.require(launch.digest(__file__) == plan['controller_sha256'] and
                   launch.digest(PROBE) == plan['probe_sha256'], 'controller/probe changed')
    for key in ('sources_sha256', 'source_plans_sha256', 'helper_sha256'):
        launch.require(all(launch.digest(p) == value for p, value in plan[key].items()), f'frozen {key} changed')
    launch.load_transaction(); legacy.load_transaction()
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    if TX.exists():
        launch.require(tx.get('plan_sha256') == launch.digest(PLAN), 'amendment plan binding changed')
    else:
        tx['plan_sha256'] = launch.digest(PLAN)
    return tx


def expected_row(item):
    row = copy.deepcopy(item['row_before_pool'])
    row.update(held_scheduler_record=item['pool_held_after'], generic_pool_amendment=str(TX),
               actual_requested_nodes=sorted(POOL), actual_requested_gres='gpu:1',
               actual_scheduler_profile={key: b.field(item['pool_held_after'], key)
                                         for key in (*FIELDS, 'ReqNodeList', 'TresPerNode', 'ReqTRES')})
    return row


def update_ledger(tx, item):
    data = json.loads(launch.LEDGER.read_text()); current = launch.row(data, item)
    launch.authoritative(item, item['new_job_id'])
    expected = expected_row(item)
    if current.get('generic_pool_amendment') == str(TX):
        launch.require(item.get('pool_ledger_requested') and current == expected, 'unowned/drifted pool ledger amendment')
        item['pool_ledger_updated'] = True
        event(tx, f"{item['old_job_id']}: reconciled exact generic-pool ledger amendment")
        return
    launch.require(current == item['row_before_pool'], 'target continuation row changed')
    launch.save(ART / f"ledger_before_{item['old_job_id']}.json", data)
    current.clear(); current.update(expected)
    data.setdefault('repair_history', []).append({'at': b.now(), 'audit': str(TX), 'old': item['old_job_id'],
        'new': item['new_job_id'], 'only_scheduler_changes': 'ReqNodeList 205/207 -> 205/207/302; GRES a6000:1 -> gpu:1',
        'same_job_id': True, 'scientific_exports_unchanged': True})
    launch.require(len(data['continuations']) == 75 and len({r['continuation_job_id'] for r in data['continuations']}) == 75,
                   'continuation denominator changed')
    image = ART / f"ledger_after_{item['old_job_id']}.json"
    launch.save(image, data)
    item['pool_ledger_requested'] = True; item['pool_ledger_sha256'] = launch.digest(image)
    event(tx, f"{item['old_job_id']}: recording actual GPU/node profile with IDs and historical submission intact")
    launch.save(launch.LEDGER, data)
    launch.require(launch.digest(launch.LEDGER) == item['pool_ledger_sha256'], 'ledger readback mismatch')
    launch.authoritative(item, item['new_job_id'])
    item['pool_ledger_updated'] = True
    event(tx, f"{item['old_job_id']}: generic-pool ledger provenance recorded")



def preserve_active(tx, item, record):
    """Undo only our confirmed hold if it raced with a still-live allocation."""
    profile(item, record)
    launch.require(b.field(record, 'JobState') in ACTIVE, 'allocation race is no longer active')
    if b.field(record, 'Priority') == '0':
        launch.require(item.get('pool_hold_requested') and
                       item.get('pool_hold_result', {}).get('returncode') == 0 and
                       b.field(item['pool_hold_before'], 'JobState') == 'PENDING' and
                       b.field(item['pool_hold_before'], 'Priority') != '0',
                       'active zero-priority hold is not our confirmed mutation; preserve for review')
        launch.require(b.field(record, 'Reason') not in {'JobHeldAdmin', 'job_requeued_in_held_state'},
                       'active hold ownership changed; do not release it')
        launch.require(not item.get('pool_active_hold_release_requested'),
                       'uncertain active-hold release remains held; manual reconciliation required')
        current = b.show(item['new_job_id']); profile(item, current)
        launch.require(b.field(current, 'JobState') in ACTIVE and b.field(current, 'Priority') == '0'
                       and b.field(current, 'Reason') not in {'JobHeldAdmin', 'job_requeued_in_held_state'},
                       'active hold changed before cleanup; preserve for review')
        item['pool_active_hold_release_requested'] = True
        event(tx, f"{item['old_job_id']}: releasing only our confirmed hold that raced with an active allocation; no resource update")
        result = b.command(['scontrol', 'release', str(item['new_job_id'])], check=False)
        item['pool_active_hold_release_result'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        event(tx, f"{item['old_job_id']}: active hold cleanup returned; auditing the unchanged live allocation")
        record = b.show(item['new_job_id']); profile(item, record)
        launch.require(b.field(record, 'JobState') in ACTIVE and b.field(record, 'Priority') != '0',
                       'active hold cleanup not confirmed; no retry/cancel/resource amendment')
    item['pool_allocation_preserved'] = True
    item['pool_allocation_race_record'] = record
    event(tx, f"{item['old_job_id']}: existing active allocation preserved; no GPU/node amendment")
    return False


def hold(tx, item):
    record = b.show(item['new_job_id'])
    profile(item, record)
    if b.field(record, 'JobState') in ACTIVE:
        return preserve_active(tx, item, record)
    launch.require(b.field(record, 'JobState') == 'PENDING', 'successor is not pending')
    if item.get('pool_hold_requested'):
        profile(item, record, held=True)
        item['pool_held'] = True
        event(tx, f"{item['old_job_id']}: reconciled owned pending hold")
        return True
    launch.require(b.field(record, 'Priority') != '0' and b.field(record, 'Reason') not in
                   {'JobHeldUser', 'JobHeldAdmin', 'job_requeued_in_held_state'}, 'preexisting hold is not owned by this amendment')
    launch.safe_run(item, [item['old_job_id'], item['new_job_id']]); launch.old_guard(item, True)
    # Recheck exact pending identity immediately before the hold.
    record = b.show(item['new_job_id']); profile(item, record)
    if b.field(record, 'JobState') in ACTIVE:
        return preserve_active(tx, item, record)
    launch.require(b.field(record, 'JobState') == 'PENDING' and b.field(record, 'Priority') != '0', 'pending state changed before hold')
    item['pool_hold_before'] = record
    item['pool_hold_requested'] = True
    event(tx, f"{item['old_job_id']}: intent to hold exact pending successor before changing only GPU type and node pool")
    result = b.command(['scontrol', 'hold', str(item['new_job_id'])], check=False)
    item['pool_hold_result'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    event(tx, f"{item['old_job_id']}: hold returned; auditing actual scheduler state")
    record = b.show(item['new_job_id']); profile(item, record)
    if b.field(record, 'JobState') in ACTIVE:
        return preserve_active(tx, item, record)
    profile(item, record, held=True)
    item['pool_held'] = True
    return True


def apply(old_job, tx=None):
    tx = load() if tx is None else tx
    item = launch.item_for(tx, old_job)
    launch.old_guard(item, True); launch.authoritative(item, item['new_job_id'])
    record = b.show(item['new_job_id'])
    if item.get('pool_release_requested') and b.field(record, 'Priority') == '0':
        profile(item, record, generic=True, held=True)
        launch.require(False, 'prior release intent remains held; manual reconciliation required, never repeat uncertain release')
    if item.get('pool_release_requested') and b.field(record, 'Priority') != '0':
        profile(item, record, generic=True)
        launch.require(b.field(record, 'JobState') in {'PENDING', *ACTIVE}, 'released job has unexpected state')
        launch.require(launch.row(json.loads(launch.LEDGER.read_text()), item) == expected_row(item), 'release ledger profile changed')
        item['pool_released'] = True
        event(tx, f'{old_job}: reconciled generic-pool release without repeating mutation')
        print(json.dumps({'old': old_job, 'new': item['new_job_id'], 'status': 'released'})); return
    if item.get('pool_update_requested'):
        profile(item, record, generic=True, held=True)
        item['pool_updated'] = True
        event(tx, f'{old_job}: reconciled exact GPU-type/node-pool amendment')
    else:
        if not hold(tx, item):
            print(json.dumps({'old': old_job, 'new': item['new_job_id'], 'status': 'active_allocation_preserved'})); return
        healthy_nodes()
        item['pool_forecasts_before_update'] = {'before': forecast(item, False), 'generic': forecast(item, True)}
        launch.safe_run(item, [old_job, item['new_job_id']]); launch.old_guard(item, True)
        profile(item, b.show(item['new_job_id']), held=True)
        item['pool_update_requested'] = True
        event(tx, f'{old_job}: intent to change only held Gres=gpu:1 and NodeList={NODELIST}')
        result = b.command(['scontrol', 'update', f"JobId={item['new_job_id']}", 'Gres=gpu:1', f'NodeList={NODELIST}'], check=False)
        item['pool_update_result'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        event(tx, f'{old_job}: resource amendment returned; auditing actual profile')
        profile(item, b.show(item['new_job_id']), generic=True, held=True)
        item['pool_updated'] = True
    held_after = profile(item, b.show(item['new_job_id']), generic=True, held=True)
    if 'pool_held_after' not in item:
        item['pool_held_after'] = held_after
    update_ledger(tx, item)
    healthy_nodes(); item['pool_forecast_before_release'] = forecast(item, True)
    launch.safe_run(item, [old_job, item['new_job_id']]); launch.old_guard(item, True)
    profile(item, b.show(item['new_job_id']), generic=True, held=True)
    launch.authoritative(item, item['new_job_id'])
    item['pool_release_requested'] = True
    event(tx, f'{old_job}: release intent after fresh health/export/checkpoint/sole-writer/profile checks')
    result = b.command(['scontrol', 'release', str(item['new_job_id'])], check=False)
    item['pool_release_result'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    record = profile(item, b.show(item['new_job_id']), generic=True)
    launch.require(b.field(record, 'Priority') != '0' and b.field(record, 'JobState') in {'PENDING', *ACTIVE},
                   'release did not become schedulable; reconcile existing intent before any retry')
    item.update(pool_released=True, pool_release_record=record)
    event(tx, f'{old_job}: generic route released; memory, walltime, science and all IDs preserved')
    print(json.dumps({'old': old_job, 'new': item['new_job_id'], 'status': 'released',
                      'state': b.field(record, 'JobState'), 'reason': b.field(record, 'Reason')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'apply', 'apply-all'))
    parser.add_argument('--old-job-id', type=int)
    args = parser.parse_args()
    if args.action == 'apply-all':
        launch.require(args.old_job_id is None, 'apply-all uses only the immutable ten-job scope')
        # Rehash immutable inputs once; every individual cell still gets a fresh
        # ledger lock, ownership/profile checks, checkpoint and writer audit.
        tx = load()
        launch.require(len(tx['items']) == 10 and {i['old_job_id'] for i in tx['items']} == OLD_IDS,
                       'immutable ten-job scope changed')
        for item in tx['items']:
            with launch.LOCK.open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                apply(item['old_job_id'], tx)
    else:
        with launch.LOCK.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.action == 'prepare':
                prepare()
            else:
                launch.require(args.old_job_id in OLD_IDS, 'exact --old-job-id required')
                apply(args.old_job_id)
