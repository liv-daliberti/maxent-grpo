#!/usr/bin/env python3
"""One pending E119 node-pool amendment with an isolated existing-guard handoff."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

import guard_e119_healthy_completion_20260909 as guard

b, recovery = guard.base, guard.recovery
ROOT = b.ROOT
ART = ROOT / 'var/artifacts/e119_max44_node208_20260910'
PLAN, TX = ART / 'amendment_plan.json', ART / 'amendment_transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e119_max44_node208_20260910.md'
HEALTH = ROOT / 'var/artifacts/level3_start_push_20260910/node208_health_probe.json'
OLD_ART, OLD_PLAN, OLD_TX, OLD_LOCK = guard.ART, guard.PLAN, guard.TX, guard.LOCK
NEW = ART / 'guard'
TARGET, OLD_CPU = 31163361, 31164485
EXPECTED_TARGET_RESTARTS = '1'
NODES = 'node205,node207,node208,node302'
CPU_FIELDS = ('UserId', 'Account', 'Partition', 'ReqNodeList', 'MinMemoryNode',
              'NumCPUs', 'NumTasks', 'Requeue', 'TimeLimit', 'Command', 'Comment', 'JobName')
require, atomic = guard.require, guard.atomic


def configured_guard():
    guard.ART, guard.PLAN, guard.TX = NEW, NEW / 'plan.json', NEW / 'transaction.json'
    guard.REGISTRATION = NEW / 'supervisor.json'
    guard.LOCK = OLD_LOCK
    return guard


def health():
    proof = json.loads(HEALTH.read_text())
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(proof['at_utc'])).total_seconds()
    require(0 <= age <= 7200 and proof['node'] == 'node208', 'Physical node208 probe is missing or older than two hours')
    require(proof['gpu']['name'] == 'NVIDIA RTX A6000' and proof['gpu']['memory_total_MiB'] >= 48000
            and proof['gpu']['temperature_C'] < 80 and proof['host']['MemAvailable_kB'] >= 132 * 1024**2,
            'Physical node208 GPU/memory health does not qualify')
    node = b.command(['scontrol', 'show', 'node', '-o', 'node208']).stdout
    require(not any(x in b.field(node, 'State') for x in ('DRAIN', 'DOWN', 'FAIL')),
            'node208 became unhealthy; leave the target held')
    require('gpu:a6000:' in b.field(node, 'Gres') and 'lowprio' in b.field(node, 'Partitions').split(','),
            'node208 hardware or partition changed')
    return {'proof_sha256': recovery.digest(HEALTH), 'scheduler_node_record': node}


def target_record(item, *, amended, held):
    record = guard.show(TARGET)
    expected = copy.deepcopy(item)
    if amended:
        expected['resources']['ReqNodeList'] = NODES
    guard.stable(expected, record)
    require(b.field(record, 'JobId') == str(TARGET) and b.field(record, 'JobState') == 'PENDING',
            'Target is no longer pending; preserve any running allocation')
    require(b.field(record, 'Restarts') == EXPECTED_TARGET_RESTARTS and b.field(record, 'RunTime') == '00:00:00',
            'Target restart history changed or current allocation has run')
    debug = Path(item['identity']['run_dir']) / f'debug_job{TARGET}'
    require(not debug.exists() or not any(debug.iterdir()), 'Prior empty startup directory gained runtime output')
    if held:
        require(b.field(record, 'Reason') == 'JobHeldUser' and b.field(record, 'Priority') == '0',
                'Exact owned target hold is missing')
    else:
        require(b.field(record, 'Priority') != '0', 'Unexpected preexisting hold')
    guard.dormant(item)
    return record


def cpu_command():
    return ['sbatch', '--parsable', '--hold', '--job-name=e119-healthy-node208-guard',
            '--account=mltheory', '--partition=lowprio', '--nodelist=node915,node917',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=2G', '--gres=none',
            '--time=1-01:10:00', '--requeue', '--export=NONE', f'--chdir={ROOT}',
            f'--output={NEW}/supervisor-%j.out', f'--error={NEW}/supervisor-%j.err',
            '--comment=e119-max44-node208-guard-20260910', str(NEW / 'supervisor.slurm')]


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Existing immutable amendment must not be overwritten')
    original = json.loads(OLD_PLAN.read_text()); old_tx = json.loads(OLD_TX.read_text())
    require(old_tx['plan_sha256'] == recovery.digest(OLD_PLAN), 'Old guard transaction binding differs')
    item = copy.deepcopy(next(r for r in original['rows'] if r['job_id'] == TARGET))
    require(item['resources']['ReqNodeList'] == 'node205,node207,node302'
            and item['resources']['MinMemoryNode'] == '116G', 'Unexpected target route')
    target_record(item, amended=False, held=False)
    checkpoint = guard.checkpoint_and_writer(item)
    before_cpu = guard.show(OLD_CPU)
    require(b.field(before_cpu, 'JobState') == 'RUNNING', 'Original CPU guard is not running')
    registered = json.loads((OLD_ART / 'supervisor.json').read_text())
    require(registered['job_id'] == OLD_CPU and registered['plan_sha256'] == recovery.digest(OLD_PLAN)
            and b.submit_tokens(before_cpu) == registered['submit_tokens'], 'Old CPU ownership/registration differs')
    require(all(b.field(before_cpu, k) == v for k, v in registered['resources'].items()), 'Old CPU resources differ')
    ledger = json.loads(guard.CONTINUATIONS.read_text())
    row = next(r for r in ledger['continuations'] if r['continuation_job_id'] == TARGET)
    proof = health()
    NEW.mkdir(parents=True, exist_ok=True)
    script = ('#!/bin/bash\nset -euo pipefail\n'
              'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
              f'cd {ROOT}\nexec /usr/bin/python3 -u {Path(__file__).resolve()} watch\n')
    (NEW / 'supervisor.slurm').write_text(script)
    new_guard = copy.deepcopy(original)
    next(r for r in new_guard['rows'] if r['job_id'] == TARGET)['resources']['ReqNodeList'] = NODES
    for path in (Path(__file__), PROTOCOL):
        new_guard['helper_sha256'][str(path.resolve())] = recovery.digest(path)
    atomic(NEW / 'plan.json', new_guard)
    proposal = {'command': cpu_command(), 'script_sha256': recovery.digest(NEW / 'supervisor.slurm'),
                'scheduler_mutations': False}
    atomic(NEW / 'cpu_submission.json', proposal)
    plan = {'schema': 'e119-max44-node208-guard-handoff-v1', 'created_at_utc': b.now(),
            'job_id': TARGET, 'old_cpu_job_id': OLD_CPU, 'item': item, 'checkpoint': checkpoint,
            'ledger_row_before': row, 'old_cpu_submit_tokens': b.submit_tokens(before_cpu),
            'old_cpu_resources': {k: b.field(before_cpu, k) for k in CPU_FIELDS},
            'old_cpu_restarts': b.field(before_cpu, 'Restarts'),
            'old_plan_sha256': recovery.digest(OLD_PLAN), 'new_guard_plan_sha256': recovery.digest(NEW / 'plan.json'),
            'controller_sha256': recovery.digest(__file__), 'protocol_sha256': recovery.digest(PROTOCOL),
            'health_at_preparation': proof, 'cpu_submit_tokens': cpu_command(),
            'target_restarts': EXPECTED_TARGET_RESTARTS,
            'target_prior_startup': b.command(['sacct', '-n', '-P', '-D', '-X', '-j', str(TARGET),
                '-o', 'JobID,State,Start,End,Elapsed,NodeList,ExitCode']).stdout,
            'cpu_script_sha256': proposal['script_sha256'], 'scheduler_mutations': False,
            'rollback': {'only_before_target_release': True, 'restore_nodes': item['resources']['ReqNodeList'],
                         'restore_ledger_row': row, 'original_guard_files_untouched': True,
                         'original_guard_plan': str(OLD_PLAN), 'original_guard_transaction': str(OLD_TX),
                         'old_cpu_job_id': OLD_CPU, 'automatic_rollback': False}}
    atomic(PLAN, plan)
    print(json.dumps({'prepared': str(PLAN), 'target': TARGET, 'cpu_command': cpu_command(), 'scheduler_mutations': False}))


def load():
    plan = json.loads(PLAN.read_text())
    require(recovery.digest(__file__) == plan['controller_sha256'] and recovery.digest(PROTOCOL) == plan['protocol_sha256'],
            'Prepared controller/protocol changed')
    require(recovery.digest(OLD_PLAN) == plan['old_plan_sha256'] and
            recovery.digest(NEW / 'plan.json') == plan['new_guard_plan_sha256'], 'Guard plans changed')
    require(recovery.digest(NEW / 'supervisor.slurm') == plan['cpu_script_sha256'], 'CPU entrypoint changed')
    tx = json.loads(TX.read_text()) if TX.exists() else {'plan_sha256': recovery.digest(PLAN), 'events': []}
    require(tx['plan_sha256'] == recovery.digest(PLAN), 'Amendment transaction binding differs')
    return plan, tx


def save(tx, message):
    tx.setdefault('events', []).append({'at': b.now(), 'event': message})
    atomic(TX, tx); print(json.dumps({'event': message}), flush=True)


def new_cpu(plan, job, *, held):
    record = guard.show(job)
    require(b.submit_tokens(record) == plan['cpu_submit_tokens'], 'New CPU submission differs')
    require(b.field(record, 'UserId') == plan['old_cpu_resources']['UserId']
            and 'gres/gpu' not in b.field(record, 'ReqTRES'), 'New CPU owner or GPU request differs')
    require(b.field(record, 'MinMemoryNode') == '2G' and b.field(record, 'NumCPUs') == '2'
            and b.field(record, 'TimeLimit') == '1-01:10:00', 'New CPU actual resources differ')
    if held:
        require(b.field(record, 'JobState') == 'PENDING' and b.field(record, 'Reason') == 'JobHeldUser'
                and b.field(record, 'Priority') == '0' and b.field(record, 'Restarts') == '0',
                'New CPU must be a never-started held allocation')
    return record


def old_cpu(plan, *, running):
    record = guard.show(OLD_CPU)
    require(b.submit_tokens(record) == plan['old_cpu_submit_tokens'] and
            all(b.field(record, k) == v for k, v in plan['old_cpu_resources'].items()), 'Old CPU identity/resources changed')
    require(b.field(record, 'Restarts') == plan['old_cpu_restarts'], 'Old CPU restart changed')
    if running:
        require(b.field(record, 'JobState') == 'RUNNING', 'Exact old CPU is not running')
    return record


def final_old_state(plan):
    old = json.loads(OLD_TX.read_text())
    require(old['plan_sha256'] == plan['old_plan_sha256'], 'Old transaction plan binding differs')
    require(not any(not action.get('released') for state in old['jobs'].values()
                    for action in state.get('attempts', [])), 'Old guard has an unresolved retry; do not hand off')
    return old


def amended_row(plan):
    row = copy.deepcopy(plan['ledger_row_before'])
    row['actual_requested_nodes'] = NODES.split(',')
    row['actual_scheduler_profile']['ReqNodeList'] = NODES
    row['node208_guard_amendment'] = str(TX)
    return row


def route(job):
    plan, tx = load()
    require(not tx.get('target_release_requested'), 'Target release already began; do not repeat routing')
    require(tx.get('new_cpu_job_id', job) == job, 'Different successor CPU would duplicate the guard')
    new_cpu(plan, job, held=True)
    tx['new_cpu_job_id'] = job
    with guard.LEDGER_LOCK.open('a+') as ledger_lock:
        fcntl.flock(ledger_lock, fcntl.LOCK_EX)
        health()
        if not tx.get('target_hold_requested'):
            target_record(plan['item'], amended=False, held=False)
            guard.writer_check(plan['item'])
            tx['target_hold_requested'] = True; save(tx, 'Persisted exact pending target hold intent')
            b.command(['scontrol', 'hold', str(TARGET)])
        actual_amended = b.field(guard.show(TARGET), 'ReqNodeList') == NODES
        require(not actual_amended or tx.get('node_update_requested'), 'Node pool changed without this transaction')
        target_record(plan['item'], amended=actual_amended, held=True)
        if not tx.get('old_cpu_stop_requested'):
            old_cpu(plan, running=True); final_old_state(plan)
            tx['old_cpu_stop_requested'] = True; save(tx, 'Stopping only the verified owned CPU guard at the ledger boundary')
            b.command(['scancel', str(OLD_CPU)])
        # Never repeat an uncertain scancel; wait for the recorded stop to become observable.
        old_cpu(plan, running=False)
        with OLD_LOCK.open('a+') as old_lock:
            until = time.monotonic() + 45
            while True:
                if OLD_CPU not in b.queue():
                    try:
                        fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        pass
                require(time.monotonic() < until, 'Recorded old-CPU stop has not completed; target remains held')
                time.sleep(0.5)
            previous = final_old_state(plan)
            if not tx.get('old_final_tx_sha256'):
                tx['old_final_tx_sha256'] = recovery.digest(OLD_TX)
                atomic(ART / 'old_guard_transaction.final.json', previous)
                save(tx, 'Original CPU inactive and singleton acquired; froze final retry state')
            require(recovery.digest(OLD_TX) == tx['old_final_tx_sha256'], 'Original guard changed after stop')
            record = guard.show(TARGET)
            nodes = b.field(record, 'ReqNodeList')
            if nodes == NODES:
                require(tx.get('node_update_requested'), 'Node pool changed without recorded intent')
            else:
                target_record(plan['item'], amended=False, held=True)
                require(not tx.get('node_update_requested'), 'Uncertain node update requires inspection; never repeat it blindly')
                health(); guard.checkpoint_and_writer(plan['item'])
                tx['node_update_requested'] = True; save(tx, 'Adding only node208 to the held generic-GPU request')
                b.command(['scontrol', 'update', f'JobId={TARGET}', f'NodeList={NODES}'])
            tx['node_pool_updated'] = True
            target_record(plan['item'], amended=True, held=True)
            data = json.loads(guard.CONTINUATIONS.read_text())
            rows = [r for r in data['continuations'] if r['continuation_job_id'] == TARGET]
            require(len(rows) == 1 and len(data['continuations']) == 75, 'Continuation identity/cardinality changed')
            desired = amended_row(plan)
            if rows[0] != desired:
                require(rows[0] == plan['ledger_row_before'], 'Target continuation row changed independently')
                atomic(ART / 'continuation_ledger.before.json', data)
                rows[0].clear(); rows[0].update(desired)
                data.setdefault('repair_history', []).append({'at': b.now(), 'audit': str(TX), 'job_id': TARGET,
                    'only_scheduler_change': 'add node208 to ReqNodeList; same job and generic gpu:1'})
                tx['ledger_commit_requested'] = True; save(tx, 'Persisted one-row route provenance update intent')
                atomic(guard.CONTINUATIONS, data)
            tx['ledger_updated'] = True
            copied = copy.deepcopy(previous)
            copied['plan_sha256'] = plan['new_guard_plan_sha256']
            if (NEW / 'transaction.json').exists():
                require(json.loads((NEW / 'transaction.json').read_text()) == copied, 'Staged guard state differs')
            else:
                atomic(NEW / 'transaction.json', copied)
            tx['guard_state_copied'] = True; tx['status'] = 'routed_target_and_new_cpu_held'
            save(tx, 'Original guard files retained; copied all ten retry states and unchanged deadline; target stays held')
    print(json.dumps({'status': tx['status'], 'next': 'activate', 'new_cpu_job_id': job}))


def activate():
    plan, tx = load(); job = tx['new_cpu_job_id']
    require(tx.get('guard_state_copied') and tx.get('ledger_updated'), 'Route handoff is not staged')
    require(OLD_CPU not in b.queue(), 'Old CPU still active')
    if tx.get('cpu_release_requested'):
        record = new_cpu(plan, job, held=False)
        require(b.field(record, 'JobState') in {'RUNNING', 'PENDING', 'CONFIGURING'}
                and b.field(record, 'Priority') != '0', 'Uncertain CPU release; do not repeat')
        return
    new_cpu(plan, job, held=True)
    target_record(plan['item'], amended=True, held=True)
    configured_guard().register(job)
    tx['cpu_release_requested'] = True; save(tx, 'Registered held successor CPU; releasing only the CPU watcher')
    b.command(['scontrol', 'release', str(job)])
    tx['cpu_released'] = True; save(tx, 'CPU successor released; wait for fresh monitoring before target release')


def ready(plan, tx):
    require(OLD_CPU not in b.queue(), 'Old CPU became active')
    record = new_cpu(plan, tx['new_cpu_job_id'], held=False)
    require(b.field(record, 'JobState') == 'RUNNING', 'Successor CPU is not running')
    proof = json.loads((NEW / 'ready.json').read_text())
    require(proof['job_id'] == tx['new_cpu_job_id'] and proof['plan_sha256'] == plan['new_guard_plan_sha256'],
            'Successor ready receipt identity differs')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(proof['at'])).total_seconds()
    require(0 <= age < 180, 'Successor heartbeat is stale')
    status = json.loads((NEW / 'status.json').read_text())
    rows = {r['job_id']: r for r in status['jobs']}
    require(len(rows) == 10 and all(r['status'] == 'monitoring' or r['status'] == 'completed' for r in rows.values()),
            'Successor did not validate the entire inherited scope')
    require(rows[TARGET]['status'] == 'monitoring', 'Target guard is not monitoring')


def release_target():
    plan, tx = load()
    with guard.LEDGER_LOCK.open('a+') as ledger_lock:
        fcntl.flock(ledger_lock, fcntl.LOCK_EX)
        ready(plan, tx)
        if tx.get('target_release_requested'):
            record = guard.show(TARGET)
            item = copy.deepcopy(plan['item']); item['resources']['ReqNodeList'] = NODES
            guard.stable(item, record)
            require(b.field(record, 'JobState') in {'PENDING', 'RUNNING', 'CONFIGURING'} and b.field(record, 'Priority') != '0',
                    'Uncertain target release; do not repeat')
        else:
            target_record(plan['item'], amended=True, held=True); health()
            guard.writer_check(plan['item'])
            tx['target_release_requested'] = True; save(tx, 'Healthy successor monitoring verified; releasing owned target hold')
            b.command(['scontrol', 'release', str(TARGET)])
            record = guard.show(TARGET)
            item = copy.deepcopy(plan['item']); item['resources']['ReqNodeList'] = NODES
            guard.stable(item, record)
            require(b.field(record, 'JobState') in {'PENDING', 'RUNNING', 'CONFIGURING'} and b.field(record, 'Priority') != '0',
                    'Target did not become schedulable')
        tx.update(target_released=True, status='released', target_after=record)
        save(tx, 'One existing cell released onto expanded pool; original fallback and scientific identity preserved')
    print(json.dumps({'job_id': TARGET, 'state': b.field(record, 'JobState'), 'reason': b.field(record, 'Reason')}))


def watch():
    plan, tx = load()
    require(tx.get('guard_state_copied') and tx.get('cpu_release_requested'), 'Handoff is not authorized')
    require(os.environ.get('SLURM_JOB_ID') == str(tx.get('new_cpu_job_id')), 'Wrong CPU attempted handoff')
    require(OLD_CPU not in b.queue() and recovery.digest(OLD_TX) == tx['old_final_tx_sha256'], 'Original guard changed or is active')
    with OLD_LOCK.open('a+') as old_lock:
        fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configured_guard().run(watch=True, apply=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'once', 'route', 'activate', 'release', 'watch'))
    parser.add_argument('--job-id', type=int)
    args = parser.parse_args(); guard.install_bounded_scheduler()
    ART.mkdir(parents=True, exist_ok=True)
    if args.phase == 'watch':
        watch(); return
    with (ART / 'amendment.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare':
            with guard.LEDGER_LOCK.open('a+') as ledger:
                fcntl.flock(ledger, fcntl.LOCK_EX); prepare()
        elif args.phase == 'once':
            plan, tx = load()
            proof = health()
            report = {'at': b.now(), 'scheduler_mutations': False, 'target': TARGET,
                      'state': b.field(guard.show(TARGET), 'JobState'), 'physical_health': proof,
                      'new_guard_deadline_utc': json.loads((NEW / 'plan.json').read_text())['deadline_utc']}
            atomic(ART / 'dry_run.json', report); print(json.dumps(report))
        elif args.phase == 'route':
            require(args.job_id is not None, 'route requires the exact already-submitted held CPU ID'); route(args.job_id)
        elif args.phase == 'activate':
            activate()
        else:
            release_target()


if __name__ == '__main__':
    main()
