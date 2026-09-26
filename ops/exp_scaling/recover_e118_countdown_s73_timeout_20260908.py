#!/usr/bin/env python3
"""Prepare/apply the single registered E118 Countdown MaxRL seed-73 continuation."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path

import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery

ROOT = base.ROOT
OLD = 31048123
ART = ROOT / 'var/artifacts/e118_countdown_s73_timeout_20260908'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e118_countdown_s73_timeout_recovery_20260908.md'
SOURCE = base.LEDGER
AGGREGATE = ROOT / 'var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json'
LOCK = ROOT / 'var/artifacts/e118_ledger_promotion.lock'
MARKER = f'e118-countdown-s73-timeout-20260908-old{OLD}'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(tx, event):
    tx['updated_at_utc'] = base.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'event': event})
    base.atomic(TX, tx)


def command(original):
    env = base.exports(original)
    env.update(base.ROOT_EXPORTS)
    updates = {
        '--account': 'allcs', '--partition': 'lowprio', '--nodelist': 'node205',
        '--gres': 'gpu:a6000:1', '--mem': '128G', '--time': '3-00:00:00',
        '--nodes': '1', '--ntasks': '1', '--ntasks-per-node': '1',
        '--cpus-per-task': '16', '--nice': '200', '--exclude': base.PVL,
        '--output': str(ROOT / 'var/artifacts/logs/%x-%j.out'),
        '--error': str(ROOT / 'var/artifacts/logs/%x-%j.err'),
        '--comment': MARKER,
        '--export': 'ALL,' + ','.join(f'{k}={v}' for k, v in env.items()),
    }
    skip = set(updates) | {'--hold', '--dependency', '--begin'}
    result = [x for x in original[:-1] if x.split('=', 1)[0] not in skip]
    result += [f'{k}={v}' for k, v in updates.items()] + ['--hold', original[-1]]
    require(base.nonroot_exports(result) == base.nonroot_exports(original), 'Non-root exports changed')
    require('--requeue' in result, 'Requeue eligibility missing')
    return result


def capacity():
    record = base.command(['scontrol', 'show', 'node', '-o', 'node205']).stdout
    require(base.field(record, 'State').split('+')[0] in {'IDLE', 'MIXED', 'ALLOCATED'}, 'Node205 unavailable')
    require(int(base.field(record, 'RealMemory')) - int(base.field(record, 'AllocMem')) >= 128 * 1024, 'Node205 lacks 128 GiB unallocated host memory')
    require(int(base.field(record, 'CPUEfctv')) - int(base.field(record, 'CPUAlloc')) >= 16, 'Node205 lacks 16 unallocated CPUs')
    def tres(key):
        return dict(x.split('=', 1) for x in base.field(record, key).split(',') if '=' in x)
    total, used = tres('CfgTRES'), tres('AllocTRES')
    for key in ('gres/gpu', 'gres/gpu:a6000'):
        require(int(total[key]) - int(used.get(key, 0)) >= 1, f'Node205 lacks {key}')
    return record


def safe_cell(tx, *, check_checkpoint=True):
    require(recovery.state(OLD) == 'TIMEOUT' and OLD not in base.queue(), 'Predecessor is not an inactive TIMEOUT')
    run = Path(tx['identity']['run_dir'])
    require(not recovery.complete(run), 'Cell already has a terminal completion receipt')
    recovery.no_other_writer(tx, recovery.active_writers())
    if check_checkpoint:
        latest, _ = recovery.select_latest_checkpoint(run)
        require(latest is not None and str(latest) == tx['checkpoint'], 'Latest valid checkpoint changed')
        require(not recovery.validate_checkpoint(latest), 'Checkpoint validation failed')
    require(recovery.digest(tx['original_command'][-1]) == tx['launcher_sha256'], 'Frozen launcher changed')


def audit_record(tx, *, held):
    record = base.show(tx['new_job_id'])
    expected = dict(Account='allcs', Partition='lowprio', ReqNodeList='node205',
                    ExcNodeList=base.PVL, MinMemoryNode='128G', NumCPUs='16',
                    TimeLimit='3-00:00:00', Requeue='1', Nice='200',
                    Dependency='(null)', TresPerNode='gres/gpu:a6000:1', Comment=MARKER)
    if held:
        expected.update(JobState='PENDING', Reason='JobHeldUser')
    for key, value in expected.items():
        require(base.field(record, key) == value, f'Scheduler field drift: {key}')
    require('gres/gpu=1' in base.field(record, 'ReqTRES').split(','), 'GPU count changed')
    tokens = base.submit_tokens(record)
    require(base.nonroot_exports(tokens) == base.nonroot_exports(tx['original_command']), 'Runtime/scientific exports changed')
    require(tokens[-1] == tx['original_command'][-1], 'Frozen launcher path changed')
    require(all(base.exports(tokens).get(k) == v for k, v in base.ROOT_EXPORTS.items()), 'Repository-root exports changed')
    return record


def prepare():
    require(not TX.exists(), 'Existing transaction: review it and use apply; never overwrite')
    source_raw, aggregate_raw = SOURCE.read_bytes(), AGGREGATE.read_bytes()
    source, aggregate = json.loads(source_raw), json.loads(aggregate_raw)
    require(source.get('released') is True and source.get('target_steps') == 3072, 'Source scientific contract drifted')
    rows = [r for r in source['runs'] if int(r['job_id']) == OLD]
    require(len(rows) == 1, 'Expected one authoritative failed cell')
    row = rows[0]
    require((row['domain'], row['arm'], row['seed']) == ('countdown', 'maxrl', 73), 'Scientific identity drifted')
    aggregate_rows = [r for r in aggregate['runs'] if int(r['job_id']) == OLD]
    require(len(aggregate['runs']) == len({int(r['job_id']) for r in aggregate['runs']}) == 150, 'Aggregate cardinality drifted')
    require(len(aggregate_rows) == 1 and {k:v for k,v in aggregate_rows[0].items() if k != 'scale'} == row, 'Source/aggregate failed-cell mismatch')
    require(aggregate_rows[0]['scale'] == 'qwen3b', 'Aggregate scale drifted')
    original = base.submit_tokens(row['held_scheduler_record'])
    env = base.exports(original)
    require(env['SAVE_PATH'] == row['run_dir'] and env['RUN_STAMP'] == row['run_stamp'], 'Run identity exports changed')
    require(env['OAT_ZERO_AUTO_RESUME'] == '1' and env['OAT_ZERO_VLLM_GPU_RATIO'] == '0.25', 'Resume/runtime contract changed')
    require(env['OAT_ZERO_SOURCE_ROOT'] == str(Path(original[-1]).parents[2] / 'src'), 'Frozen source root changed')
    checkpoint, rejected = recovery.select_latest_checkpoint(Path(row['run_dir']))
    require(checkpoint is not None and checkpoint.name == 'step_01920', 'Expected latest valid step-1920 checkpoint')
    require(not recovery.validate_checkpoint(checkpoint), 'Invalid model/optimizer checkpoint')
    tx = dict(schema='e118-countdown-s73-timeout-20260908-v1', created_at_utc=base.now(),
              authorization='User explicitly requested fixing the one failed E118 cell.',
              protocol=str(PROTOCOL), protocol_sha256=recovery.digest(PROTOCOL),
              controller_sha256=recovery.digest(__file__), helper_sha256={str(Path(m.__file__).resolve()): recovery.digest(m.__file__) for m in (base, recovery)},
              status='prepared', scheduler_only=True, scientific_exports_unchanged=True,
              existing_cells_only=True, outcomes_inspected=False,
              old_job_id=OLD, new_job_id=None, identity={k:row[k] for k in base.IDENTITY},
              original_command=original, command=command(original),
              launcher_sha256=recovery.digest(original[-1]),
              checkpoint=str(checkpoint), checkpoint_step=1920, rejected_checkpoints=rejected,
              source_sha256_before=base.sha(source_raw), aggregate_sha256_before=base.sha(aggregate_raw),
              released=False, ledger_committed=False, submission_uncertain=False, events=[])
    safe_cell(tx)
    tx['node_before'] = capacity()
    test = base.command([tx['command'][0], '--test-only', *[x for x in tx['command'][1:] if x != '--hold']])
    tx['sbatch_test_only'] = dict(stdout=test.stdout, stderr=test.stderr)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / 'source.before.json').write_bytes(source_raw)
    (ART / 'aggregate.before.json').write_bytes(aggregate_raw)
    save(tx, 'Prepared one-cell checkpoint continuation; sbatch test-only passed; no submission')
    print(json.dumps(dict(transaction=str(TX), old_job_id=OLD, checkpoint=str(checkpoint), node='node205', test_only=test.stderr), indent=2))


def reconcile_submission(tx):
    if not tx.get('submission_uncertain') or tx.get('new_job_id'):
        return
    user = base.command(['id', '-un']).stdout.strip()
    listing = base.command(['squeue', '-h', '-u', user, '-o', '%i|%k']).stdout
    found = [int(line.split('|', 1)[0]) for line in listing.splitlines()
             if line.split('|', 1)[-1] == MARKER and line.split('|', 1)[0].isdigit()]
    require(len(found) == 1, 'Uncertain prior submission: reconcile exact scheduler comment/accounting before retry; no new job submitted')
    tx['new_job_id'] = found[0]
    audit_record(tx, held=True)
    tx['submission_uncertain'] = False
    save(tx, f'Reconciled existing held job {found[0]}; no duplicate submission')


def stage_ledgers(tx):
    source = json.loads((ART / 'source.before.json').read_text())
    aggregate = json.loads((ART / 'aggregate.before.json').read_text())
    require(recovery.digest(ART / 'source.before.json') == tx['source_sha256_before'], 'Source backup changed')
    require(recovery.digest(ART / 'aggregate.before.json') == tx['aggregate_sha256_before'], 'Aggregate backup changed')
    row = next(r for r in source['runs'] if int(r['job_id']) == OLD)
    require(all(row[k] == v for k, v in tx['identity'].items()), 'Source identity changed')
    row['previous_job_ids'] = [*row.get('previous_job_ids', []), OLD]
    row.update(job_id=tx['new_job_id'], held_scheduler_record=tx['held_scheduler_record'],
               repair_audit=str(TX), scheduler_dependency=None)
    source.setdefault('repair_history', []).append(dict(at=base.now(), audit=str(TX), scheduler_only=True,
        replacement=dict(old_job_id=OLD, new_job_id=tx['new_job_id']), checkpoint=tx['checkpoint']))
    for index, previous in enumerate(aggregate['runs']):
        if int(previous['job_id']) == OLD:
            aggregate['runs'][index] = dict(copy.deepcopy(row), scale='qwen3b')
    require(len(aggregate['runs']) == len({int(r['job_id']) for r in aggregate['runs']}) == 150, 'Replacement changed aggregate cardinality')
    base.atomic(ART / 'source.after.json', source)
    base.atomic(ART / 'aggregate.after.json', aggregate)
    tx['source_sha256_after'] = recovery.digest(ART / 'source.after.json')
    tx['aggregate_sha256_after'] = recovery.digest(ART / 'aggregate.after.json')
    save(tx, 'Staged source and 150-cell aggregate; recorded both after hashes before promotion')


def apply():
    tx = json.loads(TX.read_text())
    require(tx['controller_sha256'] == recovery.digest(__file__), 'Controller changed since preparation')
    require(tx['protocol_sha256'] == recovery.digest(PROTOCOL), 'Protocol changed since preparation')
    require(all(recovery.digest(p) == h for p, h in tx['helper_sha256'].items()), 'Recovery helper changed')
    if tx.get('released'):
        print(f"Already released continuation {tx['new_job_id']}; no further mutation")
        return
    reconcile_submission(tx)
    try:
        if not tx.get('ledger_committed'):
            safe_cell(tx)
            for path, prefix in ((SOURCE, 'source'), (AGGREGATE, 'aggregate')):
                allowed = {tx[f'{prefix}_sha256_before']}
                if tx.get(f'{prefix}_sha256_after'):
                    allowed.add(tx[f'{prefix}_sha256_after'])
                require(recovery.digest(path) in allowed, f'Authoritative {prefix} changed outside this transaction')
            if tx['new_job_id'] is None:
                capacity()
                tx['submission_uncertain'] = True
                save(tx, 'Submitting held one-cell continuation')
                submitted = base.command(tx['command']).stdout.strip()
                tx['new_job_id'] = int(submitted.split(';')[0])
                tx['submission_uncertain'] = False
                save(tx, f"Submitted held continuation {tx['new_job_id']}")
            tx['held_scheduler_record'] = audit_record(tx, held=True)
            safe_cell(tx)
            save(tx, 'Verified held resources, frozen launcher, exports, checkpoint and sole writer')
            if not tx.get('source_sha256_after'):
                stage_ledgers(tx)
            for path, prefix in ((SOURCE, 'source'), (AGGREGATE, 'aggregate')):
                staged = ART / f'{prefix}.after.json'
                require(recovery.digest(staged) == tx[f'{prefix}_sha256_after'], f'Staged {prefix} changed')
                current = recovery.digest(path)
                require(current in {tx[f'{prefix}_sha256_before'], tx[f'{prefix}_sha256_after']}, f'{prefix} changed before promotion')
                if current != tx[f'{prefix}_sha256_after']:
                    base.atomic(path, json.loads(staged.read_text()))
                save(tx, f'Promoted {prefix} atomically under the shared lock')
            tx['ledger_committed'] = True
            save(tx, 'Source and aggregate promotion complete; replacement remains held')
        require(recovery.digest(SOURCE) == tx['source_sha256_after'] and recovery.digest(AGGREGATE) == tx['aggregate_sha256_after'], 'Ledger changed before release')
        record = audit_record(tx, held=False)
        state, reason = base.field(record, 'JobState'), base.field(record, 'Reason')
        if state == 'PENDING' and reason == 'JobHeldUser':
            safe_cell(tx)
            tx['release_requested'] = True
            save(tx, f"Releasing committed continuation {tx['new_job_id']}")
            base.command(['scontrol', 'release', str(tx['new_job_id'])])
        else:
            require(tx.get('release_requested'), 'Held replacement unexpectedly started before release request')
            require(state in {'PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING', 'COMPLETED'}, f'Unexpected state while reconciling release: {state}')
        tx['release_record'] = base.show(tx['new_job_id'])
        tx['released'] = True
        tx['status'] = 'released'
        save(tx, 'Continuation released; verify checkpoint restore and fresh optimizer progress separately')
        print(json.dumps(dict(old_job_id=OLD, new_job_id=tx['new_job_id'], checkpoint_step=1920, node='node205', released=True)))
    except BaseException as error:
        tx['last_error'] = repr(error)
        save(tx, 'Stopped with auditable state; apply reconciles the same transaction only')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'apply'))
    args = parser.parse_args()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
