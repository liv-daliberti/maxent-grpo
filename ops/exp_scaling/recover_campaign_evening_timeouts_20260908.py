#!/usr/bin/env python3
"""Prepare/apply two existing E118/E119 Countdown timeout continuations."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path

import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/campaign_evening_timeouts_20260908'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/campaign_evening_timeout_recovery_20260908.md'
LOCK = ROOT / 'var/artifacts/e118_ledger_promotion.lock'
FILES = {'e118': base.LEDGER, 'e119': campaign.E119_CONTINUATIONS,
         'aggregate': campaign.E118_LEDGER}
ARCHIVE = ROOT / 'var/artifacts/e119_memory96_recovery_20260905/transaction.json'
TARGETS = {
    'e118': dict(old_job_id=31048124, seed=73, checkpoint_step=1728,
                 nodes='node205,node206,node207', gpu='a6000', mem='128G', cpus='16'),
    'e119': dict(old_job_id=31048160, seed=46, checkpoint_step=2496,
                 nodes='node202,node203,node204', gpu='a5000', mem='128G', cpus='8'),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(tx, event):
    tx['updated_at_utc'] = base.now()
    tx.setdefault('events', []).append(dict(at=tx['updated_at_utc'], event=event))
    base.atomic(TX, tx)


def command(original, item):
    env = base.exports(original)
    env.update(base.ROOT_EXPORTS)
    updates = {
        '--account': 'allcs', '--partition': 'cs', '--nodelist': item['nodes'],
        '--gres': f"gpu:{item['gpu']}:1", '--mem': item['mem'],
        '--time': '3-00:00:00', '--nodes': '1', '--ntasks': '1',
        '--ntasks-per-node': '1', '--cpus-per-task': item['cpus'], '--nice': '0',
        '--exclude': base.PVL, '--output': str(ROOT / 'var/artifacts/logs/%x-%j.out'),
        '--error': str(ROOT / 'var/artifacts/logs/%x-%j.err'), '--comment': item['marker'],
        '--export': 'ALL,' + ','.join(f'{k}={v}' for k, v in env.items()),
    }
    skip = set(updates) | {'--hold', '--dependency', '--begin'}
    result = [x for x in original[:-1] if x.split('=', 1)[0] not in skip]
    result += [f'{k}={v}' for k, v in updates.items()] + ['--hold', original[-1]]
    require(base.nonroot_exports(result) == base.nonroot_exports(original), 'Non-root exports changed')
    require('--requeue' in result, 'Requeue eligibility missing')
    return result


def hosts(value):
    return set(base.command(['scontrol', 'show', 'hostnames', value]).stdout.split())


def safe_cell(item, writers):
    old = item['old_job_id']
    require(recovery.state(old) == 'TIMEOUT' and old not in base.queue(), f'{old} is not an inactive TIMEOUT')
    run = Path(item['identity']['run_dir'])
    require(not recovery.complete(run), f'{old} already has a terminal receipt')
    recovery.no_other_writer(item, writers)
    latest, _ = recovery.select_latest_checkpoint(run)
    require(latest is not None and str(latest) == item['checkpoint'], f'{old} latest checkpoint changed')
    require(not recovery.validate_checkpoint(latest), f'{old} checkpoint validation failed')
    require(recovery.digest(item['original_command'][-1]) == item['launcher_sha256'], f'{old} frozen launcher changed')


def audit_record(item, *, held):
    record = base.show(item['new_job_id'])
    expected = dict(Account='allcs', Partition='cs', ExcNodeList=base.PVL,
                    MinMemoryNode=item['mem'], NumCPUs=item['cpus'], TimeLimit='3-00:00:00',
                    Requeue='1', Nice='0', Dependency='(null)',
                    TresPerNode=f"gres/gpu:{item['gpu']}:1", Comment=item['marker'])
    if held:
        expected.update(JobState='PENDING', Reason='JobHeldUser')
    for key, value in expected.items():
        require(base.field(record, key) == value, f"{item['old_job_id']} scheduler drift: {key}")
    require(base.field(record, 'NumNodes') in ('1', '1-1'), 'Single-node request changed')
    require(hosts(base.field(record, 'ReqNodeList')) == hosts(item['nodes']), 'Node pool changed')
    require('gres/gpu=1' in base.field(record, 'ReqTRES').split(','), 'GPU count changed')
    tokens = base.submit_tokens(record)
    require(base.nonroot_exports(tokens) == base.nonroot_exports(item['original_command']), 'Runtime/scientific exports changed')
    require(tokens[-1] == item['original_command'][-1], 'Frozen launcher changed')
    require(all(base.exports(tokens).get(k) == v for k, v in base.ROOT_EXPORTS.items()), 'Root exports changed')
    return record


def prepare():
    require(not TX.exists(), 'Existing transaction: inspect it; never overwrite')
    raw = {key: path.read_bytes() for key, path in FILES.items()}
    data = {key: json.loads(value) for key, value in raw.items()}
    require(data['e118'].get('released') is True and data['e118'].get('target_steps') == 3072, 'E118 contract drifted')
    require(len(data['aggregate']['runs']) == len({r['job_id'] for r in data['aggregate']['runs']}) == 150, 'E118 aggregate cardinality drifted')
    require(campaign.e119_continuation_jobs(campaign.E119_LEDGER).get(31014413) == 31048160, 'E119 continuation contract drifted')
    rows = []
    writers = recovery.active_writers()
    for cohort, target in TARGETS.items():
        key, id_key = ('runs', 'job_id') if cohort == 'e118' else ('continuations', 'continuation_job_id')
        matches = [r for r in data[cohort][key] if int(r[id_key]) == target['old_job_id']]
        require(len(matches) == 1, f'Expected one authoritative {cohort} cell')
        row = matches[0]
        require((row['domain'], row['arm'], row['seed']) == ('countdown', 'replay_maxrl', target['seed']), 'Scientific identity changed')
        if cohort == 'e118':
            aggregate_row = next(r for r in data['aggregate']['runs'] if r['job_id'] == target['old_job_id'])
            require(aggregate_row == dict(row, scale='qwen3b'), 'Source/aggregate mismatch')
            archived = row['held_scheduler_record']
        else:
            require(row['original_job_id'] == 31014413, 'E119 original cell changed')
            archived = next(r for r in json.loads(ARCHIVE.read_text())['jobs'] if r['job_id'] == target['old_job_id'])['after']
            require(base.field(archived, 'MinMemoryNode') == '96G', 'Audited E119 memory amendment changed')
        original = base.submit_tokens(archived)
        env = base.exports(original)
        require(env['SAVE_PATH'] == row['run_dir'] and env['RUN_STAMP'] == row['run_stamp'], 'Run identity exports changed')
        require(env['OAT_ZERO_AUTO_RESUME'] == '1' and env['OAT_ZERO_VLLM_GPU_RATIO'] == '0.25', 'Resume/runtime contract changed')
        require(env['OAT_ZERO_SOURCE_ROOT'] == str(Path(original[-1]).parents[2] / 'src'), 'Frozen source root changed')
        checkpoint, rejected = recovery.select_latest_checkpoint(Path(row['run_dir']))
        require(checkpoint is not None and int(checkpoint.name[5:]) == target['checkpoint_step'], 'Expected saved checkpoint changed')
        item = dict(target, cohort=cohort, identity={k: row[k] for k in base.IDENTITY},
                    original_command=original, launcher_sha256=recovery.digest(original[-1]),
                    checkpoint=str(checkpoint), rejected_checkpoints=rejected, new_job_id=None,
                    marker=f"campaign-evening-timeout-20260908-old{target['old_job_id']}",
                    released=False, submission_uncertain=False)
        item['command'] = command(original, item)
        safe_cell(item, writers)
        test = base.command([item['command'][0], '--test-only', *[x for x in item['command'][1:] if x != '--hold']])
        item['sbatch_test_only'] = dict(stdout=test.stdout, stderr=test.stderr)
        rows.append(item)
    dependencies = [Path(m.__file__).resolve() for m in (base, recovery, campaign)] + [ARCHIVE, campaign.E119_LEDGER]
    tx = dict(schema='campaign-evening-timeout-recovery-20260908-v1', created_at_utc=base.now(),
              authorization='User requested requeueing broken E118/E119/E120 cells and accelerating completion.',
              protocol=str(PROTOCOL), protocol_sha256=recovery.digest(PROTOCOL), controller_sha256=recovery.digest(__file__),
              dependency_sha256={str(p): recovery.digest(p) for p in dependencies},
              status='prepared', scheduler_only=True, scientific_exports_unchanged=True,
              existing_cells_only=True, outcomes_inspected=False, rows=rows,
              before_sha256={key: base.sha(value) for key, value in raw.items()},
              ledger_committed=False, events=[])
    ART.mkdir(parents=True, exist_ok=True)
    for key, value in raw.items():
        (ART / f'{key}.before.json').write_bytes(value)
    save(tx, 'Prepared two exact-cell checkpoint continuations; sbatch test-only passed; no submission')
    print(json.dumps(dict(transaction=str(TX), rows=[{k: r[k] for k in ('cohort', 'old_job_id', 'checkpoint_step', 'nodes', 'sbatch_test_only')} for r in rows]), indent=2))


def reconcile_submission(tx, item):
    if not item.get('submission_uncertain') or item.get('new_job_id'):
        return
    user = base.command(['id', '-un']).stdout.strip()
    listing = base.command(['squeue', '-h', '-u', user, '-o', '%i|%k']).stdout
    found = [int(line.split('|', 1)[0]) for line in listing.splitlines()
             if line.split('|', 1)[-1] == item['marker'] and line.split('|', 1)[0].isdigit()]
    require(len(found) == 1, 'Uncertain submission: reconcile exact scheduler comment/accounting; no blind retry')
    item['new_job_id'] = found[0]
    audit_record(item, held=True)
    item['submission_uncertain'] = False
    save(tx, f'Reconciled existing held job {found[0]}; no duplicate submission')


def stage_ledgers(tx):
    data = {}
    for key in FILES:
        path = ART / f'{key}.before.json'
        require(recovery.digest(path) == tx['before_sha256'][key], f'{key} backup changed')
        data[key] = json.loads(path.read_text())
    for item in tx['rows']:
        cohort, old = item['cohort'], item['old_job_id']
        key, id_key, history = ('runs', 'job_id', 'previous_job_ids') if cohort == 'e118' else ('continuations', 'continuation_job_id', 'previous_continuation_job_ids')
        row = next(r for r in data[cohort][key] if r[id_key] == old)
        require(all(row[k] == v for k, v in item['identity'].items()), 'Cell identity changed')
        row[history] = [*row.get(history, []), old]
        row[id_key] = item['new_job_id']
        row.update(held_scheduler_record=item['held_scheduler_record'], repair_audit=str(TX), scheduler_dependency=None)
        data[cohort].setdefault('repair_history', []).append(dict(at=base.now(), audit=str(TX), scheduler_only=True,
            old_job_id=old, new_job_id=item['new_job_id'], checkpoint=item['checkpoint']))
        if cohort == 'e118':
            index = next(i for i, r in enumerate(data['aggregate']['runs']) if r['job_id'] == old)
            data['aggregate']['runs'][index] = dict(copy.deepcopy(row), scale='qwen3b')
    require(len(data['aggregate']['runs']) == len({r['job_id'] for r in data['aggregate']['runs']}) == 150, 'Aggregate cardinality changed')
    after_hashes = {}
    for key, value in data.items():
        base.atomic(ART / f'{key}.after.json', value)
        after_hashes[key] = recovery.digest(ART / f'{key}.after.json')
    tx['after_sha256'] = after_hashes
    save(tx, 'Staged both authoritative ledgers and 150-cell aggregate; recorded all after hashes')


def apply():
    tx = json.loads(TX.read_text())
    require(tx['controller_sha256'] == recovery.digest(__file__), 'Controller changed since preparation')
    require(tx['protocol_sha256'] == recovery.digest(PROTOCOL), 'Protocol changed since preparation')
    require(all(recovery.digest(p) == h for p, h in tx['dependency_sha256'].items()), 'Frozen dependencies changed')
    if all(item['released'] for item in tx['rows']):
        print('Both continuations already released; no further mutation')
        return
    try:
        for item in tx['rows']:
            reconcile_submission(tx, item)
        if not tx['ledger_committed']:
            for key, path in FILES.items():
                require(recovery.digest(path) in {tx['before_sha256'][key], tx.get('after_sha256', {}).get(key)}, f'{key} changed outside transaction')
            writers = recovery.active_writers()
            for item in tx['rows']:
                safe_cell(item, writers)
                if item['new_job_id'] is None:
                    item['submission_uncertain'] = True
                    save(tx, f"Submitting held continuation for {item['old_job_id']}")
                    submitted = base.command(item['command']).stdout.strip()
                    item['new_job_id'] = int(submitted.split(';')[0])
                    item['submission_uncertain'] = False
                    save(tx, f"Submitted held continuation {item['new_job_id']}")
                item['held_scheduler_record'] = audit_record(item, held=True)
            writers = recovery.active_writers()
            for item in tx['rows']:
                safe_cell(item, writers)
            save(tx, 'Verified held resources, frozen exports, valid checkpoints and sole writers')
            if not tx.get('after_sha256'):
                stage_ledgers(tx)
            for key, path in FILES.items():
                staged = ART / f'{key}.after.json'
                require(recovery.digest(staged) == tx['after_sha256'][key], f'{key} staged file changed')
                current = recovery.digest(path)
                require(current in {tx['before_sha256'][key], tx['after_sha256'][key]}, f'{key} changed before promotion')
                if current != tx['after_sha256'][key]:
                    base.atomic(path, json.loads(staged.read_text()))
                save(tx, f'Promoted {key} atomically under shared lock')
            tx['ledger_committed'] = True
            save(tx, 'Both authoritative ledgers and E118 aggregate committed; replacements held')
        require(all(recovery.digest(path) == tx['after_sha256'][key] for key, path in FILES.items()), 'Ledger changed before release')
        require(campaign.e119_continuation_jobs(campaign.E119_LEDGER).get(31014413) == next(r['new_job_id'] for r in tx['rows'] if r['cohort'] == 'e119'), 'E119 continuation mapping invalid after promotion')
        for item in tx['rows']:
            if item['released']:
                continue
            record = audit_record(item, held=False)
            state, reason = base.field(record, 'JobState'), base.field(record, 'Reason')
            if state == 'PENDING' and reason == 'JobHeldUser':
                safe_cell(item, recovery.active_writers())
                item['release_requested'] = True
                save(tx, f"Releasing committed continuation {item['new_job_id']}")
                base.command(['scontrol', 'release', str(item['new_job_id'])])
            else:
                require(item.get('release_requested'), 'Replacement started without release request')
                require(state in {'PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING', 'COMPLETED'}, f'Unexpected release reconciliation state: {state}')
            item['release_record'] = base.show(item['new_job_id'])
            item['released'] = True
            save(tx, f"Released continuation {item['new_job_id']}; verify restoration/progress separately")
        tx['status'] = 'released'
        save(tx, 'Both checkpoint continuations released')
        print(json.dumps([{k: r[k] for k in ('cohort', 'old_job_id', 'new_job_id', 'checkpoint_step', 'released')} for r in tx['rows']], indent=2))
    except BaseException as error:
        tx['status'] = 'stopped'
        tx['last_error'] = repr(error)
        save(tx, 'Stopped with auditable state; apply reconciles this transaction only')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'apply'))
    args = parser.parse_args()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
