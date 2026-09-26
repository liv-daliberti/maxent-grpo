#!/usr/bin/env python3
"""Expand the three released E122 Countdown peers to healthy node208."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess

import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery
import launch_e122_level3_factorial as launcher
import e122_slurm_2511_compat as compat
import control_e122_level3_release as controller

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/e122_countdown_peers_node208_20260910'
PLAN, TX = ART / 'plan.json', ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e122_countdown_peers_node208_20260910.md'
JOBS = {31158679: 'replay_drgrpo', 31158680: 'maxrl', 31158681: 'replay_maxrl'}
JOB = None
OLD_NODES = {'node205', 'node206', 'node207', 'node302'}
NEW_NODES = OLD_NODES | {'node208'}
POOL = 'node205,node206,node207,node208,node302'
JOURNAL = controller.JOURNAL_ROOT
PRESERVE = ('UserId', 'JobName', 'Account', 'Partition', 'QOS', 'NumCPUs', 'NumTasks',
            'CPUs/Task', 'MinMemoryNode', 'TimeLimit', 'ExcNodeList', 'Requeue', 'Nice',
            'Features', 'Dependency', 'WorkDir', 'Command', 'TresPerNode', 'ReqTRES', 'Restarts')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(argv, *, check=True):
    return subprocess.run(argv, text=True, capture_output=True, timeout=45, check=check)


def nodes(record):
    return set(command(['scontrol', 'show', 'hostnames', base.field(record, 'ReqNodeList')]).stdout.split())


def save(tx, event):
    tx['updated_at_utc'] = base.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'event': event})
    base.atomic(TX, tx)


def authenticate():
    watch = json.loads((launcher.HERE / 'root_watch_arm_result.json').read_text())['command']
    src = watch[watch.index('--compat-source-sha256') + 1]
    test = watch[watch.index('--compat-test-sha256') + 1]
    compat.install_compat(launcher, src, test)
    args = controller.parse_args(watch[watch.index('--') + 1:])
    campaign = controller.load_campaign(args, launcher)
    job = next(r for r in campaign['jobs'] if int(r['job_id']) == JOB)
    require((job['cell']['domain'], job['cell']['arm'], job['cell']['seed']) == ('countdown', JOBS[JOB], 43),
            'Exact E122 identity changed')
    intent = JOURNAL / 'jobs' / f'{JOB}.intent.json'
    result = json.loads((JOURNAL / 'jobs' / f'{JOB}.result.json').read_text())
    require(result['job_id'] == str(JOB) and result['returncode'] == 0 and not result.get('error')
            and result['intent_sha256'] == recovery.digest(intent), 'Original successful release ownership is absent')
    status = controller.status(campaign, JOURNAL)
    require(not status['issues'] and not status['unknown'] and not status['needs_operator_review_job_ids'],
            'Existing E122 controller requires reconciliation')
    require(str(JOB) in status['released_nonterminal_job_ids'] and status['reserved_unfinished_slots'] == 4,
            'Four original released reservations changed')
    return campaign, job, status


def audit(record, before, cell):
    require(base.field(record, 'JobId') == str(JOB), 'Wrong scheduler ID')
    require(base.submit_tokens(record) == base.submit_tokens(before), 'Frozen SubmitLine changed')
    for key in PRESERVE:
        require(base.field(record, key) == base.field(before, key), f'Unapproved scheduler change: {key}')
    require(base.field(record, 'NumNodes') in {'1', '1-1'}, 'Single-node allocation changed')
    env = base.exports(base.submit_tokens(record))
    require(all(env.get(k) == v for k, v in cell['environment'].items()), 'Scientific or runtime exports changed')
    return record


def capacity():
    record = command(['scontrol', 'show', 'node', '-o', 'node208']).stdout
    state = base.field(record, 'State')
    require(not any(bad in state for bad in ('DOWN', 'DRAIN', 'FAIL', 'MAINT')), 'Node208 is unavailable or unhealthy')
    require('gpu:a6000:' in base.field(record, 'Gres'), 'Node208 GPU class changed')
    free_mib = int(base.field(record, 'RealMemory')) - int(base.field(record, 'AllocMem'))
    require('lowprio' in base.field(record, 'Partitions').split(','), 'Node208 lacks lowprio')
    return {'record': record, 'schedulable_memory_mib': free_mib}


def no_other_writer(cell):
    run = Path(cell['run_dir'])
    require(not recovery.complete(run), 'E122 cell already complete')
    writers = recovery.active_writers().get(str(run.resolve()), set())
    require(writers == {JOB}, f'Unexpected live writer candidates: {writers}')


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Existing preparation must be inspected, not overwritten')
    campaign, job, status = authenticate()
    cell = job['cell']; record = base.show(JOB)
    require(base.field(record, 'JobState') == 'PENDING' and base.field(record, 'Priority') != '0'
            and base.field(record, 'Reason') not in {'JobHeldUser', 'JobHeldAdmin'}, 'Exact peer job must be pending and released')
    require(base.field(record, 'RunTime') == '00:00:00' and base.field(record, 'Restarts') == '0', 'E122 cell has already started')
    require(nodes(record) == OLD_NODES, 'Original route changed')
    audit(record, record, cell); no_other_writer(cell)
    cap = capacity()
    trial = [x for x in cell['command'] if x != '--hold' and not x.startswith('--nodelist=')]
    trial.insert(1, '--test-only'); trial.insert(-1, '--nodelist=' + POOL)
    result = command(trial)
    files = {Path(__file__), PROTOCOL, Path(base.__file__), Path(recovery.__file__),
             Path(launcher.__file__), Path(controller.__file__), Path(compat.__file__), compat.TEST,
             launcher.PLAN, launcher.LEDGER, JOURNAL / 'context.json',
             JOURNAL / 'jobs' / f'{JOB}.intent.json', JOURNAL / 'jobs' / f'{JOB}.result.json'}
    plan = {'schema': 'e122-countdown-peer-released-node208-route-v1', 'created_at_utc': base.now(),
            'authorization': 'User requested starting E122 promptly while existing E118/E119/E120 finish.',
            'job_id': JOB, 'cell': cell, 'before': record, 'old_nodes': sorted(OLD_NODES),
            'new_nodes': sorted(NEW_NODES), 'controller_binding': campaign['binding'],
            'controller_status_before': {k:status[k] for k in ('blocked_reason','reserved_unfinished_slots','released_nonterminal_job_ids','issues')},
            'capacity_before': cap, 'sbatch_test_only': {'stdout':result.stdout,'stderr':result.stderr},
            'command': ['scontrol','update',f'JobId={JOB}',f'ReqNodeList={POOL}'],
            'files_sha256': {str(p.resolve()):recovery.digest(p) for p in files},
            'scientific_configuration_unchanged': True, 'hold_release_requeue_submission_forbidden': True,
            'original_controller_and_ledger_untouched': True, 'status': 'prepared', 'events': []}
    ART.mkdir(parents=True, exist_ok=True); base.atomic(PLAN, plan)
    print(json.dumps({k:plan[k] for k in ('status','job_id','old_nodes','new_nodes','command','sbatch_test_only')}, indent=2))


def apply():
    tx = json.loads((TX if TX.exists() else PLAN).read_text())
    require(all(recovery.digest(p) == sha for p,sha in tx['files_sha256'].items()), 'Frozen input or amendment changed')
    cell = tx['cell']; before = tx['before']
    record = audit(base.show(JOB), before, cell)
    route = nodes(record)
    if tx.get('update_intent'):
        # Never repeat a command whose acknowledgement may have been lost.
        require(route in (OLD_NODES, NEW_NODES), 'Unexpected route after submitted update')
        if route == NEW_NODES:
            tx.update(status='complete', after=record)
            save(tx, 'Reconciled exact accepted node208 expansion; no duplicate update')
            return
        require(tx.get('command_result', {}).get('returncode') is not None,
                'Uncertain update with original route: manual reconciliation; no blind retry')
        tx.update(status='not_applied', after=record)
        save(tx, 'Scheduler retained original route; no retry or running-job interruption')
        return
    require(route == OLD_NODES, 'Unowned placement change')
    if base.field(record, 'JobState') != 'PENDING':
        tx.update(status='skipped_started', after=record)
        save(tx, 'Job started before amendment; preserve active allocation unchanged')
        return
    require(base.field(record, 'Priority') != '0' and base.field(record, 'Reason') not in {'JobHeldUser','JobHeldAdmin'},
            'Preserve an existing hold')
    no_other_writer(cell); capacity()
    record = audit(base.show(JOB), before, cell)
    require(nodes(record) == OLD_NODES, 'Route changed during final validation')
    if base.field(record, 'JobState') != 'PENDING':
        tx.update(status='skipped_started', after=record)
        save(tx, 'Job raced final pending check; preserve its allocation')
        return
    tx['update_intent'] = True
    save(tx, 'Persisted one pending same-ID ReqNodeList expansion; no hold and no controller-lock interference')
    result = command(tx['command'], check=False)
    tx['command_result'] = {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    save(tx, 'Recorded scheduler update acknowledgement before readback')
    record = audit(base.show(JOB), before, cell); route = nodes(record)
    require(route in (OLD_NODES, NEW_NODES), 'Scheduler returned an unapproved node pool')
    tx.update(status='complete' if route == NEW_NODES else 'not_applied', after=record)
    require(result.returncode != 0 or route == NEW_NODES, 'Successful update did not expose the approved pool')
    require(all(recovery.digest(p) == sha for p,sha in tx['files_sha256'].items()), 'Frozen campaign changed during route update')
    save(tx, 'Verified node208 expansion' if route == NEW_NODES else 'Update rejected; original allocation request retained')
    print(json.dumps({'status':tx['status'],'job_id':JOB,'state':base.field(record,'JobState'),'nodes':sorted(route)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--job', type=int, choices=sorted(JOBS), required=True)
    args = parser.parse_args()
    JOB = args.job
    ART = ART / str(JOB)
    PLAN, TX = ART / 'plan.json', ART / 'transaction.json'
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'singleton.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (apply if args.apply else prepare)()
