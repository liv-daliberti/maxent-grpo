#!/usr/bin/env python3
"""Amend only pending E118/E119/E120 scheduler requests, retaining job identities."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re

import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/campaign_pending_acceleration_20260908'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/campaign_pending_acceleration_20260908.md'
LEDGERS = (campaign.E118_LEDGER, base.LEDGER, campaign.E119_LEDGER,
           campaign.E119_CONTINUATIONS, campaign.E120_LEDGER, campaign.E120_CONTINUATIONS)
PRESERVE = ('Account', 'NumCPUs', 'NumTasks', 'CPUs/Task', 'MinMemoryNode',
            'TresPerNode', 'Nice', 'Requeue', 'ExcNodeList', 'Command', 'WorkDir',
            'StdOut', 'StdErr', 'TimeLimit', 'Partition')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities():
    result = {}
    for cohort, ledger in [('e118', campaign.E118_LEDGER), ('e119', campaign.E119_LEDGER),
                           ('e120', campaign.E120_LEDGER)]:
        mapping = (campaign.e119_continuation_jobs(ledger) if cohort == 'e119' else
                   campaign.e120_continuation_jobs(ledger) if cohort == 'e120' else {})
        for row in json.loads(ledger.read_text())['runs']:
            original = int(row['job_id'])
            jid = int(mapping.get(original, original))
            assert jid not in result
            result[jid] = {'cohort': cohort, 'job_id': jid, 'run_dir': row['run_dir'],
                           'run_stamp': row['run_stamp'], 'domain': row['domain'],
                           'seed': row['seed']}
    assert len(result) == 295
    return result


def hosts(record):
    requested = base.field(record, 'ReqNodeList')
    return set(base.command(['scontrol', 'show', 'hostnames', requested]).stdout.split())


def changes(record, cell, cells):
    changes = {}
    if base.field(record, 'JobState') != 'PENDING' or base.field(record, 'Priority') == '0':
        return changes
    assert base.field(record, 'ExcNodeList') == base.PVL
    env = base.exports(base.submit_tokens(record))
    assert env['SAVE_PATH'] == cell['run_dir'] and env['RUN_STAMP'] == cell['run_stamp']
    assert env['OAT_ZERO_AUTO_RESUME'] == '1'
    limit = base.field(record, 'TimeLimit')
    assert limit in ('12:00:00', '1-12:00:00', '3-00:00:00'), ('unexpected time limit', cell['job_id'], limit)
    dep = base.field(record, 'Dependency')
    if dep != '(null)':
        match = re.fullmatch(r'afterany:(\d+)\(unfulfilled\)', dep)
        assert match, ('unreviewed dependency', cell['job_id'], dep)
        parent = cells.get(int(match[1]))
        assert parent is not None and parent['run_dir'] != cell['run_dir'], ('unsafe dependency', dep)
        changes['Dependency'] = '0'
    if base.field(record, 'Account') == 'allcs':
        ratio = env['OAT_ZERO_VLLM_GPU_RATIO']
        cohort = cell['cohort']
        requested = hosts(record)
        partition = base.field(record, 'Partition')
        assert partition in {'cs', 'lowprio'}
        if cohort == 'e119':
            assert cell['domain'] == 'pantry_plan' and ratio == '0.25'
            assert requested <= {'node205', 'node206', 'node207'}
            target = {'node205', 'node206', 'node207'}
            assert partition == 'cs'
        elif cohort == 'e118' and ratio == '0.25':
            assert requested <= {'node205', 'node206', 'node207', 'node208'}
            target = {'node205', 'node206', 'node207'}
            if partition == 'lowprio':
                target.add('node208')
        elif ratio == '0.40':
            assert cohort in {'e118', 'e120'} and cell['domain'] != 'pantry_plan'
            assert requested <= {'node105', 'node202', 'node203', 'node204', 'node302'}
            target = {'node202', 'node203', 'node204'}
            if partition == 'lowprio':
                target.add('node105')
                if 'node302' in requested:
                    target.add('node302')
        else:
            raise RuntimeError(('unreviewed placement', cell['job_id'], cohort, ratio, requested))
        if requested != target:
            changes['NodeList'] = ','.join(sorted(target))
    else:
        assert base.field(record, 'Account') == 'mltheory'
        assert hosts(record) == {'node302'}, ('unreviewed owner pending placement', cell['job_id'])
    return changes


def matches(record, original):
    assert base.submit_tokens(record) == base.submit_tokens(original), 'submitted runtime/launcher changed'
    for key in PRESERVE:
        assert base.field(record, key) == base.field(original, key), (key, base.field(record, key))
    assert base.field(record, 'NumNodes') in ('1', '1-1')
    assert dict(t.split('=', 1) for t in base.field(record, 'ReqTRES').split(','))['gres/gpu'] == '1'


def audit(record, item):
    matches(record, item['before'])
    assert digest(item['launcher']) == item['launcher_sha256']
    for key in ('TimeLimit', 'Dependency', 'Partition'):
        expected = item['changes'].get(key, base.field(item['before'], key))
        if key == 'Dependency' and expected == '0':
            expected = '(null)'
        assert base.field(record, key) == expected, (item['job_id'], key, expected, base.field(record, key))
    expected_nodes = (set(item['changes']['NodeList'].split(',')) if 'NodeList' in item['changes']
                      else hosts(item['before']))
    assert hosts(record) == expected_nodes


def no_duplicate(cell, writers):
    assert not recovery.complete(Path(cell['run_dir'])), 'cell already complete'
    actual = writers.get(str(Path(cell['run_dir']).resolve()), set())
    assert actual <= {cell['job_id']}, ('multiple writers', cell['job_id'], actual)


def prepare():
    assert not PLAN.exists() and not TX.exists(), 'Inspect existing plan/transaction instead of overwriting'
    ART.mkdir(parents=True, exist_ok=True)
    cells = identities()
    active = base.queue()
    writers = recovery.active_writers()
    items = []
    for jid in sorted(set(cells) & set(active)):
        record = base.show(jid)
        delta = changes(record, cells[jid], cells)
        if not delta:
            continue
        no_duplicate(cells[jid], writers)
        launcher = base.submit_tokens(record)[-1]
        items.append(dict(cells[jid], before=record, changes=delta, launcher=launcher,
                          launcher_sha256=digest(launcher), status='planned'))
    plan = {'schema': 'campaign-pending-acceleration-v1', 'created_at': base.now(),
            'status': 'prepared', 'controller_sha256': digest(__file__),
            'protocol_sha256': digest(PROTOCOL), 'helper_sha256': digest(base.__file__),
            'scheduler_only': True, 'runtime_changed': False, 'ledgers_changed': False,
            'ledgers_before_sha256': {str(p): digest(p) for p in LEDGERS},
            'items': items, 'events': []}
    base.atomic(PLAN, plan)
    print(json.dumps({'prepared': True, 'count': len(items), 'changes': [
        {'job': x['job_id'], 'cohort': x['cohort'], 'changes': x['changes']} for x in items]}, indent=2))


def apply():
    plan = json.loads(PLAN.read_text())
    for p, key in ((__file__, 'controller_sha256'), (PROTOCOL, 'protocol_sha256'),
                   (base.__file__, 'helper_sha256')):
        assert digest(p) == plan[key], ('controller/protocol drift', p)
    tx = json.loads(TX.read_text()) if TX.exists() else dict(plan, status='applying')
    if tx['status'] == 'complete':
        print(json.dumps({'already_applied': True, 'count': len(tx['items'])})); return
    # Fail closed on an interrupted owned hold; the saved exact records enable reconciliation.
    assert not any(x.get('owned_hold') for x in tx['items']), 'Reconcile interrupted owned hold before retry'
    if not TX.exists():
        tx['ledgers_at_apply_sha256'] = {str(p): digest(p) for p in LEDGERS}
    def event(message):
        tx['events'].append({'at': base.now(), 'message': message})
        base.atomic(TX, tx)
    event('Validated controller; applying pending-only amendments without ledger writes')
    for item in tx['items']:
        if item['status'] in ('released', 'skipped_started', 'skipped_changed', 'skipped_terminal', 'skipped_noop'):
            continue
        jid = item['job_id']
        cells = identities()
        if jid not in cells or cells[jid]['run_dir'] != item['run_dir']:
            item['status'] = 'skipped_changed'; event(f'{jid}: authoritative mapping changed'); continue
        if jid not in base.queue():
            item['status'] = 'skipped_terminal'; event(f'{jid}: no longer queued'); continue
        current = base.show(jid)
        if base.field(current, 'JobState') != 'PENDING':
            item['status'] = 'skipped_started'; event(f'{jid}: running allocation left untouched'); continue
        assert base.field(current, 'Priority') != '0', ('preexisting hold', jid)
        matches(current, item['before'])
        assert changes(current, cells[jid], cells) == item['changes'], ('scheduler state drift', jid)
        no_duplicate(cells[jid], recovery.active_writers())
        item['hold_requested'] = True; event(f'{jid}: requesting pending hold')
        base.command(['scontrol', 'hold', str(jid)])
        held = base.show(jid)
        item['owned_hold'] = True; item['held_before'] = held; event(f'{jid}: hold acknowledged')
        if base.field(held, 'JobState') != 'PENDING':
            base.command(['scontrol', 'release', str(jid)])
            item['owned_hold'] = False; item['status'] = 'skipped_started'
            event(f'{jid}: allocation raced hold, released unchanged'); continue
        try:
            assert base.field(held, 'Reason') == 'JobHeldUser'
            matches(held, item['before'])
            # Only the explicitly enumerated scheduler fields enter scontrol.
            command = ['scontrol', 'update', f'JobId={jid}', *[f'{k}={v}' for k,v in item['changes'].items()]]
            item['update_command'] = command; item['update_requested'] = True
            event(f'{jid}: applying {item["changes"]}')
            result = base.command(command)
            item['update_stdout'] = result.stdout; item['update_stderr'] = result.stderr
            after = base.show(jid)
            assert base.field(after, 'JobState') == 'PENDING' and base.field(after, 'Reason') == 'JobHeldUser'
            audit(after, item)
            item['held_after'] = after; item['amendment_verified'] = True
            event(f'{jid}: unchanged runtime and requested scheduler fields verified')
        except BaseException as exc:
            item['error'] = repr(exc)
            after = base.show(jid); item['error_scheduler_record'] = after
            # A scheduler rejection that changed nothing can safely release our hold.
            unchanged = all(base.field(after, k) == base.field(held, k)
                            for k in ('Partition', 'TimeLimit', 'Dependency', 'ReqNodeList'))
            if unchanged and base.field(after, 'JobState') == 'PENDING':
                matches(after, held)
                base.command(['scontrol', 'release', str(jid)])
                item['owned_hold'] = False; item['status'] = 'rejected_released_unchanged'
                event(f'{jid}: scheduler rejected amendment; released original unchanged request')
            else:
                item['status'] = 'needs_reconciliation'
                event(f'{jid}: partial update requires reconciliation; saved exact scheduler record')
            raise
        item['release_requested'] = True; event(f'{jid}: releasing verified amendment')
        base.command(['scontrol', 'release', str(jid)])
        item['owned_hold'] = False; item['status'] = 'released'
        final = base.show(jid); audit(final, item)
        assert base.field(final, 'JobState') in ('PENDING', 'RUNNING', 'CONFIGURING')
        assert base.field(final, 'Priority') != '0'
        item['after_release'] = final; event(f'{jid}: released')
    tx['ledgers_after_sha256'] = {str(p): digest(p) for p in LEDGERS}
    tx['status'] = 'complete'; tx['completed_at'] = base.now()
    event('All planned cells amended or skipped because they started or changed; no ledger written')
    print(json.dumps({'status': tx['status'], 'counts': {s: sum(x['status'] == s for x in tx['items'])
          for s in sorted({x['status'] for x in tx['items']})}, 'audit': str(TX)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'apply'))
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
