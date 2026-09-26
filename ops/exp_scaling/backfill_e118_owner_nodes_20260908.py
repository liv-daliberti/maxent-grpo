#!/usr/bin/env python3
"""Audited E118 owner-node continuations; preserve science and checkpoint identity."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
import sys
import time

import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery
sys.path.insert(0, str(base.ROOT / 'ops'))
from validate_deepspeed_checkpoint import select_latest_checkpoint, validate_checkpoint

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/e118_owner_backfill_20260908'
PROTOCOL = ROOT / 'paper/preregistration/e118_owner_backfill_20260908.md'
PROFILES = {
    31048110: dict(node='node302', memory_gib=116, gpu='a100', ratio='0.25',
                   domain='graph_coloring', arm='replay_maxrl', seed=71),
    31048115: dict(node='node105', memory_gib=128, gpu='a5000', ratio='0.40',
                   domain='graph_coloring', arm='maxrl', seed=74),
    31048133: dict(node='node105', memory_gib=128, gpu='a5000', ratio='0.40',
                   domain='mathir', arm='maxrl', seed=70, fresh=True),
    31048134: dict(node='node105', memory_gib=116, gpu='a5000', ratio='0.40',
                   domain='mathir', arm='replay_maxrl', seed=70, fresh=True),
}



def admission(profile):
    if not profile.get('fresh'):
        return {}
    receipts = {}
    names = ['node105_startup']
    if profile['memory_gib'] == 116:
        names.append('node302_startup')
    for name in names:
        path = ART / (name + '.json')
        assert time.time() - path.stat().st_mtime < 180, 'Admission receipt is stale'
        receipt = json.loads(path.read_text())
        current = receipt['current']
        expected_job = 31129015 if name == 'node105_startup' else 31128840
        assert receipt['job_id'] == expected_job
        observation_age = time.time() - datetime.fromisoformat(current['checked_at_utc']).timestamp()
        assert -5 <= observation_age < 180, 'Admission observation is stale'
        live = base.show(expected_job)
        assert base.field(live, 'JobState') == 'RUNNING'
        assert base.field(live, 'NodeList') == ('node105' if name == 'node105_startup' else 'node302')
        assert current['scheduler'].startswith('RUNNING|')
        assert current.get('restore_evidence')
        assert current.get('fresh_step', 0) > receipt['resume_step']
        assert current.get('metrics_age_seconds', 999999) + max(0, observation_age) < 180
        assert not current.get('fatal_errors')
        events = current['cgroup']['events']
        assert all(events[k] == 0 for k in ('high', 'oom', 'oom_kill'))
        assert current['cgroup']['noncache_gib'] < 100
        if name == 'node105_startup':
            timing = current['sleep_wake_metrics']
            assert any('sleep' in k and float(v) > 0 for k, v in timing.items())
            assert any('wake' in k and float(v) > 0 for k, v in timing.items())
        receipts[name] = receipt
    return receipts

def safe_cell(item):
    run = Path(item['run_dir'])
    assert not recovery.complete(run), 'Cell already has a completion receipt'
    allowed = {item['old_job_id']}
    if item.get('new_job_id'):
        allowed.add(item['new_job_id'])
    writers = recovery.active_writers().get(str(run.resolve()), set())
    assert not writers - allowed, f'Unexpected writer: {writers - allowed}'
    if item.get('checkpoint'):
        latest, _ = select_latest_checkpoint(run)
        assert str(latest) == item['checkpoint'], 'Selected checkpoint changed'
        assert not validate_checkpoint(latest), 'Checkpoint no longer valid'
    elif item.get('profile', {}).get('fresh') and not item.get('release_requested'):
        assert not any(run.glob('debug_job*/train_metrics.jsonl')), 'Fresh cell has training metrics'
        assert select_latest_checkpoint(run)[0] is None, 'Fresh cell has a checkpoint'


def capacity(profile):
    record = base.command(['scontrol', 'show', 'node', '-o', profile['node']]).stdout
    assert int(base.field(record, 'RealMemory')) - int(base.field(record, 'AllocMem')) >= profile['memory_gib'] * 1024
    assert int(base.field(record, 'CPUEfctv')) - int(base.field(record, 'CPUAlloc')) >= 16
    total = dict(x.split('=', 1) for x in base.field(record, 'CfgTRES').split(','))
    used = dict(x.split('=', 1) for x in base.field(record, 'AllocTRES').split(','))
    assert int(total['gres/gpu']) - int(used.get('gres/gpu', 0)) >= 1
    return record


def command(original, profile, old):
    env = base.exports(original)
    env.update(base.ROOT_EXPORTS)
    env['OAT_ZERO_VLLM_GPU_RATIO'] = profile['ratio']
    updates = {
        '--account': profile.get('account', 'mltheory'), '--partition': profile.get('partition', 'mltheory'),
        '--nodelist': profile['node'], '--gres': f'gpu:{profile["gpu"]}:1',
        '--mem': f'{profile["memory_gib"]}G', '--time': '3-00:00:00',
        '--nodes': '1', '--ntasks': '1', '--ntasks-per-node': '1',
        '--cpus-per-task': '16', '--nice': '200', '--exclude': base.PVL,
        '--output': str(ROOT / 'var/artifacts/logs/%x-%j.out'),
        '--error': str(ROOT / 'var/artifacts/logs/%x-%j.err'),
        '--comment': f'e118-owner-backfill-20260908-old{old}',
        '--export': 'ALL,' + ','.join(f'{k}={v}' for k, v in env.items()),
    }
    skip = set(updates) | {'--hold', '--dependency', '--begin'}
    result = [x for x in original[:-1] if x.split('=', 1)[0] not in skip]
    return result + [f'{k}={v}' for k, v in updates.items()] + ['--hold', original[-1]]


def audit_record(record, item, *, held):
    p = item['profile']
    expected = dict(Account=p.get('account', 'mltheory'), Partition=p.get('partition', 'mltheory'), ReqNodeList=p['node'],
                    ExcNodeList=base.PVL, MinMemoryNode=f'{p["memory_gib"]}G',
                    NumCPUs='16', TimeLimit='3-00:00:00', Requeue='1',
                    Dependency='(null)', TresPerNode=f'gres/gpu:{p["gpu"]}:1')
    if held:
        expected.update(JobState='PENDING', Reason='JobHeldUser')
        if p.get('fresh'):
            item['last_admission_receipts'] = admission(p)
    for key, value in expected.items():
        assert base.field(record, key) == value, (key, base.field(record, key), value)
    actual = base.submit_tokens(record)
    original = base.nonroot_exports(item['original_command'])
    original['OAT_ZERO_VLLM_GPU_RATIO'] = p['ratio']
    assert base.nonroot_exports(actual) == original, 'Unexpected runtime/scientific export change'
    assert all(base.exports(actual).get(k) == v for k, v in base.ROOT_EXPORTS.items())
    assert actual[-1] == item['original_command'][-1]
    assert recovery.digest(actual[-1]) == item['launcher_sha256']
    safe_cell(item)


def prepare(old):
    path = ART / f'{old}.json'
    assert not path.exists(), 'Existing transaction must be reconciled, not overwritten'
    ART.mkdir(parents=True, exist_ok=True)
    raw = base.LEDGER.read_bytes()
    source = json.loads(raw)
    row = next(r for r in source['runs'] if int(r['job_id']) == old)
    profile = PROFILES[old]
    assert all(row[k] == profile[k] for k in ('domain', 'arm', 'seed'))
    record = base.show(old)
    assert base.field(record, 'JobState') == 'PENDING'
    assert base.field(record, 'Dependency') == '(null)'
    for line in base.command(['squeue', '-h', '-o', '%i|%E']).stdout.splitlines():
        assert not re.search(r'(?<!\d)' + str(old) + r'(?!\d)', line.split('|', 1)[1]), 'Old job has dependants'
    original = base.submit_tokens(record)
    env = base.exports(original)
    assert env['OAT_ZERO_AUTO_RESUME'] == '1'
    assert env['SAVE_PATH'] == row['run_dir'] and env['RUN_STAMP'] == row['run_stamp']
    assert env['OAT_ZERO_VLLM_GPU_RATIO'] == '0.25'
    cp, rejected = select_latest_checkpoint(Path(row['run_dir']))
    if not profile.get('fresh'):
        assert cp is not None and not validate_checkpoint(cp)
    else:
        assert cp is None
    admitted = admission(profile)
    item = {k: row[k] for k in base.IDENTITY}
    item.update(old_job_id=old, new_job_id=None, before_record=record, profile=profile,
                original_command=original, command=command(original, profile, old),
                checkpoint=str(cp) if cp else None, checkpoint_step=int(cp.name[5:]) if cp else 0,
                rejected_checkpoints=rejected, launcher_sha256=recovery.digest(original[-1]))
    safe_cell(item)
    test = base.command([item['command'][0], '--test-only', *[x for x in item['command'][1:] if x != '--hold']])
    audit = dict(schema='e118-owner-backfill-20260908-v1', created_at=base.now(),
                 authorization='User requested more E118 jobs on node302/node105.',
                 protocol=str(PROTOCOL), source_ledger=str(base.LEDGER),
                 original_ledger_sha256=base.sha(raw), status='planned',
                 scheduler_only=profile['ratio'] == '0.25',
                 runtime_changes={} if profile['ratio'] == '0.25' else {
                     'OAT_ZERO_VLLM_GPU_RATIO': {'before': '0.25', 'after': profile['ratio']}},
                 same_scientific_cells=True, same_run_directories=True,
                 treatment_changed=False, outcomes_inspected=False,
                 replacements=[item], cs_placement_updates=[], events=[],
                 admission_receipts=admitted,
                 node_before=capacity(profile), controller_sha256=recovery.digest(__file__),
                 protocol_sha256=recovery.digest(PROTOCOL),
                 helper_sha256=recovery.digest(base.__file__),
                 sbatch_test_only=dict(stdout=test.stdout, stderr=test.stderr))
    (ART / f'{old}.source.before.json').write_bytes(raw)
    aggregate = ROOT / 'var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json'
    (ART / f'{old}.aggregate.before.json').write_bytes(aggregate.read_bytes())
    base.atomic(path, audit)
    print(json.dumps(dict(old_job=old, node=profile['node'], checkpoint_step=item['checkpoint_step'], planned=True)))


def apply(old):
    base.AUDIT = ART / f'{old}.json'
    base.PROTOCOL = PROTOCOL
    audit = json.loads(base.AUDIT.read_text())
    assert recovery.digest(__file__) == audit['controller_sha256']
    assert recovery.digest(base.__file__) == audit['helper_sha256']
    assert recovery.digest(PROTOCOL) == audit['protocol_sha256']
    if audit['status'] == 'complete':
        print('Already released:', audit['replacements'][0]['new_job_id'])
        return
    item = audit['replacements'][0]
    safe_cell(item)
    if not item.get('new_job_id'):
        capacity(item['profile'])
        admission(item['profile'])
    if item.get('submission_uncertain') and item.get('new_job_id') is None:
        raise RuntimeError('Uncertain prior submission: reconcile exact scheduler comment before continuing')
    def prepare_source(source):
        if not audit['runtime_changes']:
            return
        history = source['repair_history'][-1]
        assert history['audit'] == str(base.AUDIT)
        history.update(scheduler_only=False, runtime_changes=audit['runtime_changes'], treatment_changed=False)
        row = next(r for r in source['runs'] if int(r['job_id']) == item['new_job_id'])
        row['runtime_allocation_amendment'] = str(PROTOCOL)
    base.audit_record = audit_record
    base.apply(audit, prepare_source=prepare_source)
    print(json.dumps(dict(old_job=old, new_job=item['new_job_id'], node=item['profile']['node'], released=item['released'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'apply'))
    parser.add_argument('old_job', type=int, choices=PROFILES)
    args = parser.parse_args()
    with (ROOT / 'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase](args.old_job)
