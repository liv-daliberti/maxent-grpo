#!/usr/bin/env python3
"""Replace six pending lowprio allocations with audited, same-recipe cs requests."""
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
ART = ROOT / 'var/artifacts/campaign_cs_priority_replacements_20260908'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/campaign_cs_priority_replacements_20260908.md'
PHASE1 = ROOT / 'var/artifacts/campaign_pending_acceleration_20260908/transaction.json'
SOURCE = base.LEDGER
AGGREGATE = campaign.E118_LEDGER
CONTINUATIONS = campaign.E120_CONTINUATIONS
MUTABLE = (SOURCE, AGGREGATE, CONTINUATIONS)
TARGETS = {
    31146150: ('e118', 'countdown', 'maxrl', 73, '0.25'),
    31141497: ('e118', 'mathir', 'maxrl', 71, '0.40'),
    31141741: ('e118', 'mathir', 'replay_maxrl', 71, '0.40'),
    31141972: ('e118', 'countdown', 'maxrl', 74, '0.40'),
    31144883: ('e118', 'countdown', 'replay_maxrl', 74, '0.40'),
    31144919: ('e120', 'graph_coloring', None, 73, '0.40'),
}
PRESERVE = ('Account', 'NumCPUs', 'NumTasks', 'CPUs/Task', 'MinMemoryNode',
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
    assert base.field(record, 'Partition') == 'lowprio'
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
    changes = {'--account': 'allcs', '--partition': 'cs', '--nodelist': item['nodes'],
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
    expected = {'Partition': 'cs', 'Account': 'allcs',
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
    continuation = json.loads(CONTINUATIONS.read_text())
    aggregate = json.loads(AGGREGATE.read_text())
    items = []
    for old, (cohort, domain, arm, seed, ratio) in TARGETS.items():
        row = next(r for r in (source['runs'] if cohort == 'e118' else continuation['continuations'])
                   if int(r['job_id'] if cohort == 'e118' else r['continuation_job_id']) == old)
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
                'nodes': 'node205,node206,node207' if ratio == '0.25' else 'node202,node203,node204',
                'comment': f'campaign-cs-priority-20260908-old{old}', 'checkpoint': checkpoint(row['run_dir'])}
        old_guard(item, held=False)
        safe_run(item, [old])
        item['command'] = replacement_command(item)
        test = base.command([item['command'][0], '--test-only', *[x for x in item['command'][1:] if x != '--hold']])
        item['test_only'] = {'stdout': test.stdout, 'stderr': test.stderr, 'returncode': test.returncode}
        items.append(item)
    plan = {'schema': 'campaign-cs-priority-replacements-v1', 'status': 'prepared',
            'created_at': base.now(), 'items': items, 'controller_sha256': digest(__file__),
            'protocol_sha256': digest(PROTOCOL), 'helper_sha256': digest(base.__file__),
            'phase1_sha256': digest(PHASE1), 'runtime_fingerprints': runtime_fingerprints(items),
            'mutable_before_sha256': {str(p): digest(p) for p in MUTABLE},
            'e120_primary_sha256': digest(campaign.E120_LEDGER),
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
    source = json.loads(SOURCE.read_text())
    aggregate = json.loads(AGGREGATE.read_text())
    continuation = json.loads(CONTINUATIONS.read_text())
    for item in tx['items']:
        if item['cohort'] == 'e118':
            row = next(r for r in source['runs'] if r['job_id'] == item['old_job_id'])
            assert row == item['row_before']
            row['previous_job_ids'] = [*row.get('previous_job_ids', []), item['old_job_id']]
            row.update(job_id=item['new_job_id'], held_scheduler_record=item['new_held_record'],
                       repair_audit=str(TX), scheduler_dependency='')
            target = next(r for r in aggregate['runs'] if r['run_dir'] == item['run_dir'])
            scale = target['scale']; target.clear(); target.update(row, scale=scale)
        else:
            row = next(r for r in continuation['continuations'] if r['continuation_job_id'] == item['old_job_id'])
            assert row == item['row_before']
            row.setdefault('intermediate_job_ids', []).append(item['old_job_id'])
            row.update(continuation_job_id=item['new_job_id'], held_scheduler_record=item['new_held_record'],
                       released=False, repair_kind='same_recipe_cs_priority_20260908',
                       placement_amendment=str(PROTOCOL), priority_replacement_audit=str(TX),
                       resume_checkpoint=item['checkpoint']['step'], scientific_command_equal=True,
                       new_placement={'account': 'allcs', 'partition': 'cs', 'node': item['nodes'],
                                      'gpu': base.field(item['before'], 'TresPerNode'), 'cpus': 16,
                                      'memory': base.field(item['before'], 'MinMemoryNode'),
                                      'nice': int(base.field(item['before'], 'Nice'))},
                       fallback_resource_class='A5000 cs priority; no lowprio borrowing')
            row.setdefault('scheduler_placement_amendments', []).append(str(PROTOCOL))
    source.setdefault('repair_history', []).append({'at': base.now(), 'audit': str(TX), 'protocol': str(PROTOCOL),
        'scheduler_only': True, 'replacements': [{'old_job_id': x['old_job_id'], 'new_job_id': x['new_job_id']}
                                               for x in tx['items'] if x['cohort'] == 'e118']})
    continuation.setdefault('placement_amendments', []).append(str(PROTOCOL))
    assert len(source['runs']) == 50 and len(aggregate['runs']) == 150
    assert len({r['job_id'] for r in source['runs']}) == len({r['run_dir'] for r in source['runs']}) == 50
    assert len({r['job_id'] for r in aggregate['runs']}) == len({r['run_dir'] for r in aggregate['runs']}) == 150
    identity_keys = ('domain', 'arm', 'seed', 'run_dir', 'run_stamp')
    for path, value in ((SOURCE, source), (AGGREGATE, aggregate)):
        before = json.loads(path.read_text())
        assert sorted(tuple(r[k] for k in identity_keys) for r in before['runs']) == sorted(
            tuple(r[k] for k in identity_keys) for r in value['runs'])
    assert len(continuation['continuations']) == len(json.loads(CONTINUATIONS.read_text())['continuations'])
    staged_map = {}
    for path, value in [(SOURCE, source), (AGGREGATE, aggregate), (CONTINUATIONS, continuation)]:
        staged = ART / (path.name + '.after')
        save(staged, value)
        staged_map[str(path)] = {'path': str(staged), 'sha256': digest(staged)}
    for row in continuation['continuations']:
        if row.get('priority_replacement_audit') == str(TX):
            row['released'] = True
    final = ART / (CONTINUATIONS.name + '.released')
    save(final, continuation)
    released_image = {'path': str(final), 'sha256': digest(final)}
    # Publish the staging checkpoint only after every image is complete.
    tx['staged'] = staged_map
    tx['e120_released_image'] = released_image
    event('Staged source, aggregate and E120 continuation after-images before any promotion')


def promote(tx, event):
    for path, staged in tx['staged'].items():
        current = digest(path)
        accepted = {staged['sha256']}
        if path == str(CONTINUATIONS):
            accepted.add(tx['e120_released_image']['sha256'])
        if current in accepted:
            continue
        assert current == tx['mutable_before_sha256'][path], ('concurrent ledger mutation', path)
        assert digest(staged['path']) == staged['sha256']
        event(f'Promoting staged ledger {path}')
        save(Path(path), json.loads(Path(staged['path']).read_text()))
        assert digest(path) == staged['sha256']
    assert digest(campaign.E120_LEDGER) == tx['e120_primary_sha256']
    source = json.loads(SOURCE.read_text()); aggregate = json.loads(AGGREGATE.read_text())
    mapping = campaign.e120_continuation_jobs(campaign.E120_LEDGER)
    for item in tx['items']:
        if item['cohort'] == 'e118':
            assert next(r for r in source['runs'] if r['run_dir'] == item['run_dir'])['job_id'] == item['new_job_id']
            assert next(r for r in aggregate['runs'] if r['run_dir'] == item['run_dir'])['job_id'] == item['new_job_id']
        else:
            assert mapping[item['row_before']['original_job_id']] == item['new_job_id']
    tx['ledgers_committed'] = True; event('All effective mappings agree on held replacements; E120 primary unchanged')


def apply():
    plan = json.loads(PLAN.read_text())
    for path, key in ((__file__, 'controller_sha256'), (PROTOCOL, 'protocol_sha256'), (base.__file__, 'helper_sha256')):
        assert digest(path) == plan[key]
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    if tx['status'] == 'complete':
        print(json.dumps({'already_complete': True, 'replacements': [{x['old_job_id']: x['new_job_id']} for x in tx['items']]})); return
    def event(message):
        tx['events'].append({'at': base.now(), 'message': message}); save(TX, tx)
    assert runtime_fingerprints(tx['items']) == tx['runtime_fingerprints']
    assert digest(campaign.E120_LEDGER) == tx['e120_primary_sha256']
    if not tx.get('staged'):
        assert all(digest(p) == v for p,v in tx['mutable_before_sha256'].items())
    tx['status'] = 'applying'; event('Beginning or reconciling the same six-cell priority replacement transaction')
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
                        item['submission_uncertain'] = True; event(f"Submitting held cs replacement for {item['old_job_id']}")
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
                item['release_requested'] = True; event(f"Releasing audited cs replacement {item['new_job_id']}")
                base.command(['scontrol', 'release', str(item['new_job_id'])])
            after = new_guard(item, held=False)
            assert base.field(after, 'JobState') in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'COMPLETED')
            assert base.field(after, 'Priority') != '0'
            item['released'] = True; item['after_release'] = after; event(f"Replacement {item['new_job_id']} released")
        final = tx['e120_released_image']
        assert digest(final['path']) == final['sha256']
        assert digest(CONTINUATIONS) in {tx['staged'][str(CONTINUATIONS)]['sha256'], final['sha256']}
        save(CONTINUATIONS, json.loads(Path(final['path']).read_text()))
        assert digest(campaign.E120_LEDGER) == tx['e120_primary_sha256']
        tx['status'] = 'complete'; tx['completed_at'] = base.now()
        event('All six replacements promoted and released; frozen recipes and scientific cells unchanged')
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
        for name in ('e118_ledger_promotion.lock', 'e120_ledger_promotion.lock'):
            lock = stack.enter_context((ROOT / 'var/artifacts' / name).open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
