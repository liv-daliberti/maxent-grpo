#!/usr/bin/env python3
"""Move the requeued Countdown seed74 cell to available A5000 borrowing capacity."""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery
sys.path.insert(0, str(base.ROOT / 'ops'))
from validate_deepspeed_checkpoint import select_latest_checkpoint, validate_checkpoint

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/countdown_s74_capacity_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/countdown_s74_capacity_20260909.md'
PHASE1 = ROOT / 'var/artifacts/campaign_a5000_completion_20260909/transaction.json'
SOURCE = base.LEDGER
AGGREGATE = campaign.E118_LEDGER
CONTINUATIONS = campaign.E120_CONTINUATIONS
MUTABLE = (SOURCE, AGGREGATE)
TARGETS = {31151411: ('e118', 'countdown', 'maxrl', 74, '0.40')}

PRESERVE = ('NumCPUs', 'NumTasks', 'CPUs/Task', 'MinMemoryNode',
            'TresPerNode', 'TimeLimit', 'Nice', 'Requeue', 'ExcNodeList', 'WorkDir')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    base.atomic(path, value)


def incoming(job):
    pattern = re.compile(r'(?<!\d)' + str(job) + r'(?!\d)')
    return [line for line in base.command(['squeue', '-h', '-o', '%i|%E']).stdout.splitlines()
            if pattern.search(line.split('|', 1)[1])]


def runtime_fingerprints(items):
    roots = {base.exports(item['original_command'])[key] for item in items
             for key in ('OAT_ZERO_SOURCE_ROOT', 'OAT_ZERO_OPS_SNAPSHOT_ROOT')}
    result = {}
    for root in sorted(roots):
        directory = Path(root)
        assert 'source_snapshots' in directory.parts
        files = sorted(p for p in directory.rglob('*') if p.is_file()
                       and '__pycache__' not in p.parts
                       and p.suffix in ('.py', '.sh', '.slurm', '.yaml', '.yml', '.toml'))
        assert files
        manifest = [(str(p.relative_to(directory)), digest(p)) for p in files]
        result[root] = {'files': len(files), 'sha256': hashlib.sha256(base.encoded(manifest)).hexdigest()}
    return result


def checkpoint(run):
    cp, rejected = select_latest_checkpoint(Path(run))
    if cp is None:
        assert not rejected, ('no valid checkpoint but rejected candidates exist', run, rejected)
        return {'path': None, 'step': 0, 'files': {}}
    assert not validate_checkpoint(cp)
    files = {}
    for p in sorted(cp.rglob('*.pt')):
        stat = p.stat()
        files[str(p)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}
    assert files
    return {'path': str(cp), 'step': int(cp.name[5:]), 'files': files}


def safe_run(item, allowed):
    assert not recovery.complete(Path(item['run_dir'])), 'cell already complete'
    assert checkpoint(item['run_dir']) == item['checkpoint'], 'durable checkpoint changed'
    writers = recovery.active_writers().get(str(Path(item['run_dir']).resolve()), set())
    assert writers <= set(allowed), ('unexpected writer', item['old_job_id'], writers, allowed)


def old_guard(item, held):
    record = base.show(item['old_job_id'])
    assert base.field(record, 'JobState') == 'PENDING', 'old allocation started; preserve it'
    if held:
        assert base.field(record, 'Reason') == 'JobHeldUser' and base.field(record, 'Priority') == '0'
    else:
        assert base.field(record, 'Priority') != '0', 'preexisting hold must not be changed'
    assert base.field(record, 'Partition') == 'cs'
    assert base.field(record, 'Account') == 'allcs'
    assert base.field(record, 'Dependency') == '(null)'
    assert base.submit_tokens(record) == item['original_command']
    for key in PRESERVE:
        assert base.field(record, key) == base.field(item['before'], key), (item['old_job_id'], key)
    assert base.field(record, 'ReqNodeList') == base.field(item['before'], 'ReqNodeList')
    assert digest(item['original_command'][-1]) == item['launcher_sha256']
    assert not incoming(item['old_job_id']), 'old allocation acquired a dependent'
    return record


def replacement_command(item):
    original = item['original_command']
    changes = {'--account': 'mltheory', '--partition': 'lowprio', '--nodelist': item['nodes'],
               '--gres': base.field(item['before'], 'TresPerNode').removeprefix('gres/'),
               '--mem': base.field(item['before'], 'MinMemoryNode'),
               '--cpus-per-task': base.field(item['before'], 'NumCPUs'),
               '--time': '3-00:00:00', '--nice': base.field(item['before'], 'Nice'),
               '--nodes': '1', '--ntasks': '1', '--ntasks-per-node': '1',
               '--exclude': base.PVL, '--comment': item['comment']}
    removed = set(changes) | {'--hold', '--dependency', '--begin'}
    command = [x for x in original[:-1] if x.split('=', 1)[0] not in removed]
    command += [f'{k}={v}' for k,v in changes.items()] + ['--hold', original[-1]]
    assert base.exports(command) == base.exports(original)
    return command


def new_guard(item, held):
    record = base.show(item['new_job_id'])
    expected = {'Partition': 'lowprio', 'Account': 'mltheory',
                'Dependency': '(null)', 'Comment': item['comment']}
    if held:
        expected.update(JobState='PENDING', Reason='JobHeldUser', Priority='0')
    for key, value in expected.items():
        assert base.field(record, key) == value, (item['new_job_id'], key, base.field(record, key), value)
    actual_nodes = set(base.command(['scontrol', 'show', 'hostnames', base.field(record, 'ReqNodeList')]).stdout.split())
    assert actual_nodes == set(item['nodes'].split(','))
    for key in PRESERVE:
        assert base.field(record, key) == base.field(item['before'], key), (item['new_job_id'], key)
    assert base.field(record, 'NumNodes') in ('1', '1-1')
    tres = dict(x.split('=', 1) for x in base.field(record, 'ReqTRES').split(','))
    assert tres['gres/gpu'] == '1'
    observed = base.submit_tokens(record)
    assert base.exports(observed) == base.exports(item['original_command'])
    assert observed[-1] == item['original_command'][-1]
    assert digest(observed[-1]) == item['launcher_sha256']
    return record


def prepare():
    assert not PLAN.exists() and not TX.exists(), 'Inspect existing transaction instead of overwriting'
    assert json.loads(PHASE1.read_text())['status'] == 'complete', 'Finish phase 1 before preparing'
    source = json.loads(SOURCE.read_text())
    aggregate = json.loads(AGGREGATE.read_text())
    items = []
    for old, (cohort, domain, arm, seed, ratio) in TARGETS.items():
        row = next(r for r in source['runs'] if int(r['job_id']) == old)
        assert row['domain'] == domain and row['seed'] == seed
        assert cohort != 'e118' or row['arm'] == arm
        assert cohort != 'e120' or row['model_key'] == 'qwen3b'
        before = base.show(old)
        original = base.submit_tokens(before)
        env = base.exports(original)
        assert env['SAVE_PATH'] == row['run_dir'] and env['RUN_STAMP'] == row['run_stamp']
        assert env['OAT_ZERO_VLLM_GPU_RATIO'] == ratio and env['OAT_ZERO_AUTO_RESUME'] == '1'
        assert base.field(before, 'TimeLimit') == '3-00:00:00'
        assert base.field(before, 'Account') == 'allcs' and base.field(before, 'ExcNodeList') == base.PVL
        assert base.field(before, 'NumCPUs') == '16' and base.field(before, 'Requeue') == '1'
        if cohort == 'e118':
            assert next(r for r in aggregate['runs'] if r['run_dir'] == row['run_dir'])['job_id'] == old
        item = {'cohort': cohort, 'old_job_id': old, 'new_job_id': None,
                'row_before': row, 'domain': domain, 'arm': arm, 'seed': seed,
                'run_dir': row['run_dir'], 'run_stamp': row['run_stamp'],
                'before': before, 'original_command': original, 'launcher_sha256': digest(original[-1]),
                'nodes': 'node202,node203',
                'comment': f'countdown-s74-capacity-20260909-old{old}', 'checkpoint': checkpoint(row['run_dir'])}
        assert item['checkpoint']['step'] == 1920
        assert base.field(before, 'MinMemoryNode') == '116G'
        assert base.field(before, 'TresPerNode') == 'gres/gpu:a5000:1'
        old_guard(item, held=False)
        safe_run(item, [old])
        item['command'] = replacement_command(item)
        test = base.command([item['command'][0], '--test-only', *[x for x in item['command'][1:] if x != '--hold']])
        item['test_only'] = {'stdout': test.stdout, 'stderr': test.stderr, 'returncode': test.returncode}
        items.append(item)
    plan = {'schema': 'countdown-s74-capacity-20260909-v1', 'status': 'prepared',
            'created_at': base.now(), 'items': items, 'controller_sha256': digest(__file__),
            'protocol_sha256': digest(PROTOCOL), 'helper_sha256': digest(base.__file__),
            'phase1_sha256': digest(PHASE1), 'runtime_fingerprints': runtime_fingerprints(items),
            'mutable_before_sha256': {str(p): digest(p) for p in MUTABLE},
            'e120_primary_sha256': digest(campaign.E120_LEDGER),
            'e120_continuation_sha256': digest(CONTINUATIONS),
            'helper_runtime_sha256': {str(Path(m.__file__).resolve()): digest(m.__file__) for m in (base, recovery, campaign)},
            'scheduler_only': True, 'scientific_runtime_exports_changed': False, 'events': []}
    for p in MUTABLE:
        (ART / (p.name + '.before')).write_bytes(p.read_bytes())
    save(PLAN, plan)
    print(json.dumps({'prepared': True, 'count': len(items), 'cells': [
        {'old': x['old_job_id'], 'checkpoint': x['checkpoint']['step'], 'nodes': x['nodes']} for x in items]}, indent=2))


def stage_ledgers(tx, event):
    if tx.get('staged'):
        return
    assert all(digest(p) == v for p,v in tx['mutable_before_sha256'].items())
    source = json.loads(SOURCE.read_text()); aggregate = json.loads(AGGREGATE.read_text())
    item, = tx['items']
    row = next(r for r in source['runs'] if r['job_id'] == item['old_job_id'])
    assert row == item['row_before']
    row['previous_job_ids'] = [*row.get('previous_job_ids', []), item['old_job_id']]
    row.update(job_id=item['new_job_id'], held_scheduler_record=item['new_held_record'],
               repair_audit=str(TX), scheduler_dependency='')
    target = next(r for r in aggregate['runs'] if r['run_dir'] == item['run_dir'])
    scale = target['scale']; target.clear(); target.update(row, scale=scale)
    source.setdefault('repair_history', []).append({'at': base.now(), 'audit': str(TX),
        'protocol': str(PROTOCOL), 'scheduler_only': True,
        'replacements': [{'old_job_id': item['old_job_id'], 'new_job_id': item['new_job_id']}]})
    assert len(source['runs']) == 50 and len(aggregate['runs']) == 150
    identity_keys = ('domain', 'arm', 'seed', 'run_dir', 'run_stamp')
    staged_map = {}
    for path, value in ((SOURCE, source), (AGGREGATE, aggregate)):
        before = json.loads(path.read_text())
        assert sorted(tuple(r[k] for k in identity_keys) for r in before['runs']) == sorted(tuple(r[k] for k in identity_keys) for r in value['runs'])
        assert len({r['job_id'] for r in value['runs']}) == len(value['runs'])
        for previous, updated in zip(before['runs'], value['runs']):
            if previous['run_dir'] != item['run_dir']:
                assert previous == updated, 'unrelated scientific row changed'
        staged = ART / (path.name + '.after'); save(staged, value)
        staged_map[str(path)] = {'path': str(staged), 'sha256': digest(staged)}
    tx['staged'] = staged_map
    event('Staged only the E118 source and aggregate images before promotion')


def promote(tx, event):
    for path, staged in tx['staged'].items():
        if digest(path) == staged['sha256']:
            continue
        assert digest(path) == tx['mutable_before_sha256'][path], ('concurrent ledger mutation', path)
        assert digest(staged['path']) == staged['sha256']
        event(f'Promoting staged ledger {path}')
        save(Path(path), json.loads(Path(staged['path']).read_text()))
        assert digest(path) == staged['sha256']
    assert digest(campaign.E120_LEDGER) == tx['e120_primary_sha256']
    assert digest(CONTINUATIONS) == tx['e120_continuation_sha256']
    item, = tx['items']
    for path in (SOURCE, AGGREGATE):
        assert next(r for r in json.loads(path.read_text())['runs'] if r['run_dir'] == item['run_dir'])['job_id'] == item['new_job_id']
    tx['ledgers_committed'] = True
    event('Both E118 mappings agree on the held replacement; all E120 ledgers unchanged')


def apply():
    plan = json.loads(PLAN.read_text())
    for path, key in ((__file__, 'controller_sha256'), (PROTOCOL, 'protocol_sha256'), (base.__file__, 'helper_sha256')):
        assert digest(path) == plan[key]
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    if tx['status'] == 'complete':
        print(json.dumps({'already_complete': True, 'replacements': [{x['old_job_id']: x['new_job_id']} for x in tx['items']]})); return
    def event(message):
        tx['events'].append({'at': base.now(), 'message': message}); save(TX, tx)
    assert all(digest(p) == h for p,h in tx['helper_runtime_sha256'].items())
    assert runtime_fingerprints(tx['items']) == tx['runtime_fingerprints']
    assert digest(campaign.E120_LEDGER) == tx['e120_primary_sha256']
    if not tx.get('staged'):
        assert all(digest(p) == v for p,v in tx['mutable_before_sha256'].items())
    tx['status'] = 'applying'; event('Beginning or reconciling the same one-cell capacity replacement transaction')
    try:
        if not tx.get('staged'):
            for item in tx['items']:
                if not item.get('old_held'):
                    if item.get('hold_requested') and base.field(base.show(item['old_job_id']), 'Reason') == 'JobHeldUser':
                        old_guard(item, held=True)
                    else:
                        old_guard(item, held=False); safe_run(item, [item['old_job_id']])
                        item['hold_requested'] = True; event(f"Holding predecessor {item['old_job_id']}")
                        base.command(['scontrol', 'hold', str(item['old_job_id'])])
                        held = base.show(item['old_job_id'])
                        if base.field(held, 'JobState') != 'PENDING':
                            base.command(['scontrol', 'release', str(item['old_job_id'])])
                            item['raced_hold_released'] = True
                            event(f"Predecessor {item['old_job_id']} started during hold; released unchanged")
                            raise RuntimeError('Allocation raced hold; preserve running predecessor and reconcile plan')
                        old_guard(item, held=True)
                    item['old_held'] = True; event(f"Predecessor {item['old_job_id']} is safely held")
                old_guard(item, held=True)
                safe_run(item, [item['old_job_id'], *([item['new_job_id']] if item['new_job_id'] else [])])
                if item['new_job_id'] is None:
                    if item.get('submission_uncertain'):
                        found = [j for j in base.queue() if base.field(base.show(j), 'Comment') == item['comment']]
                        assert len(found) == 1, ('uncertain submission: reconcile exact comment', item['comment'], found)
                        item['new_job_id'] = found[0]; item['submission_uncertain'] = False
                        event(f"Reconciled held replacement {found[0]}")
                    else:
                        item['submission_uncertain'] = True; event(f"Submitting held lowprio replacement for {item['old_job_id']}")
                        result = base.command(item['command'], check=False)
                        item['submission_result'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
                        if result.returncode:
                            event('Submission result requires reconciliation; no blind retry')
                            raise RuntimeError(result.stderr)
                        item['new_job_id'] = int(result.stdout.strip().split(';', 1)[0])
                        item['submission_uncertain'] = False; event(f"Recorded held replacement {item['new_job_id']}")
                item['new_held_record'] = new_guard(item, held=True)
                safe_run(item, [item['old_job_id'], item['new_job_id']])
                event(f"Audited preserved resources, full runtime exports and identity for {item['new_job_id']}")
            assert runtime_fingerprints(tx['items']) == tx['runtime_fingerprints']
            stage_ledgers(tx, event)
        promote(tx, event)
        for item in tx['items']:
            if item.get('old_cancelled'):
                continue
            if item['old_job_id'] in base.queue():
                old_guard(item, held=True); new_guard(item, held=True)
                safe_run(item, [item['old_job_id'], item['new_job_id']])
                item['cancel_requested'] = True; event(f"Retiring held predecessor {item['old_job_id']} after ledger commit")
                base.command(['scancel', str(item['old_job_id'])])
                assert item['old_job_id'] not in base.queue(), 'Wait for predecessor cancellation before release'
            else:
                assert item.get('cancel_requested'), 'Old job disappeared outside this transaction'
            item['old_cancelled'] = True; event(f"Predecessor {item['old_job_id']} no longer queued")
        assert runtime_fingerprints(tx['items']) == tx['runtime_fingerprints']
        assert all(item['old_job_id'] not in base.queue() for item in tx['items'])
        for item in tx['items']:
            if item.get('released'):
                continue
            current = base.show(item['new_job_id'])
            if item.get('release_requested') and base.field(current, 'Reason') != 'JobHeldUser':
                new_guard(item, held=False)
            else:
                new_guard(item, held=True); safe_run(item, [item['new_job_id']])
                item['release_requested'] = True; event(f"Releasing audited lowprio replacement {item['new_job_id']}")
                base.command(['scontrol', 'release', str(item['new_job_id'])])
            after = new_guard(item, held=False)
            assert base.field(after, 'JobState') in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'COMPLETED')
            assert base.field(after, 'Priority') != '0'
            item['released'] = True; item['after_release'] = after; event(f"Replacement {item['new_job_id']} released")
        assert digest(CONTINUATIONS) == tx['e120_continuation_sha256']
        assert digest(campaign.E120_LEDGER) == tx['e120_primary_sha256']
        tx['status'] = 'complete'; tx['completed_at'] = base.now()
        event('The one replacement was promoted and released; frozen recipes and scientific cells unchanged')
    except BaseException as exc:
        tx['status'] = 'stopped_for_reconciliation'; tx['error'] = repr(exc)
        event('Stopped with exact recorded IDs and holds; reconcile rather than create duplicate work')
        raise
    print(json.dumps({'status': tx['status'], 'replacements': [
        {'old': x['old_job_id'], 'new': x['new_job_id'], 'checkpoint': x['checkpoint']['step'],
         'state': base.field(x['after_release'], 'JobState'), 'reason': base.field(x['after_release'], 'Reason')}
        for x in tx['items']]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'apply'))
    args = parser.parse_args(); ART.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name in ('e118_ledger_promotion.lock',):
            lock = stack.enter_context((ROOT / 'var/artifacts' / name).open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
