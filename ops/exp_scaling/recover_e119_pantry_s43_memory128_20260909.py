#!/usr/bin/env python3
"""Prepare a bounded zero-update-loss Pantry seed43 recovery at128GiB."""
from __future__ import annotations
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import pickletools
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'ops/exp_scaling')]
import recover_e119_memory_pressure_20260905 as base
from recover_e119_health_20260905 import atomic, call, field, show
import recover_terminal_timeouts_20260908 as timeout_recovery

JID = 31048178
STAMP = 'e119_level2_pantry_maxrl_s43'
ART = ROOT / 'var/artifacts/e119_pantry_s43_memory128_20260909'
PLAN = ART / 'plan.json'
PROTOCOL = ROOT / 'paper/preregistration/e119_pantry_s43_memory128_20260909.md'
DIAGNOSIS = ROOT / 'var/artifacts/campaign_completion_push_20260909/pantry_s43_eval_process_probe.json'
PRESSURE_DELTA = ROOT / 'var/artifacts/campaign_completion_push_20260909/pantry_s43_eval_pressure_delta.json'
PRESERVE = tuple(dict.fromkeys((*base.PRESERVE, 'QOS', 'Nice', 'ReqNodeList', 'Features', 'UserId')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    run = base.identities()[JID]
    assert (run['run_stamp'], run['domain'], run['arm'], int(run['seed'])) == (STAMP, 'pantry_plan', 'maxrl', 43)
    record = show(JID)
    exp = base.exports(record)
    assert field(record, 'JobId') == str(JID)
    assert field(record, 'JobName') == 'e119-pantry-m-s43'
    assert field(record, 'UserId').startswith(getpass.getuser() + '(')
    assert exp['RUN_STAMP'] == STAMP and Path(exp['SAVE_PATH']).resolve() == Path(run['run_dir']).resolve()
    assert exp['OAT_ZERO_AUTO_RESUME'] == '1'
    assert field(record, 'Account') == 'allcs' and field(record, 'Partition') == 'cs'
    assert set(call('scontrol', 'show', 'hostnames', field(record, 'ReqNodeList')).splitlines()) == {'node205', 'node206', 'node207'}
    assert field(record, 'NumCPUs') == '8' and field(record, 'TresPerNode') == 'gres/gpu:1'
    assert field(record, 'Dependency') in ('(null)', '')
    assert field(record, 'TimeLimit') == '1-12:00:00' and field(record, 'Requeue') == '1'
    assert 'pvl' not in record.lower()
    return record, run


def stable(before, after, memory, same_attempt=False):
    assert field(after, 'MinMemoryNode') == memory
    assert base.submitline(before) == base.submitline(after)
    for k in PRESERVE:
        assert field(before, k) == field(after, k), k
    assert field(after, 'NumNodes') in ('1', '1-1')
    if same_attempt:
        for k in ('JobState', 'NodeList', 'StartTime', 'Restarts'):
            assert field(before, k) == field(after, k), k


def checkpoint(run):
    d = base.timing_and_checkpoint(JID, run)
    assert d['checkpoint'] and not d['fresh_restart']
    assert not d['rejected_checkpoints'], 'Inspect a newly partial save before proceeding'
    assert d['checkpoint_step'] == 960, 'Newer checkpoint requires refreshed review'
    assert d['saved_counter_validation']['saved_counters'] == dict.fromkeys(('global_steps', 'global_step', 'prompt_batches_consumed_total'), 960)
    assert d['unsaved_steps'] == 0 and d['current_step'] == 959
    model = Path(d['checkpoint']) / 'mp_rank_00_model_states.pt'
    with zipfile.ZipFile(model) as z:
        raw = z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))
    assert any(arg == 'online_canonical_bank_state' for _, arg, _ in pickletools.genops(raw))
    d['model_metadata_sha256'] = hashlib.sha256(raw).hexdigest()
    archives = {}
    for path in sorted(Path(d['checkpoint']).glob('*.pt')):
        stat = path.stat()
        with zipfile.ZipFile(path) as z:
            metadata = z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))
        archives[path.name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'pickle_metadata_sha256': hashlib.sha256(metadata).hexdigest()}
    d['archive_fingerprints'] = archives
    return d


def runtime_fingerprints(record):
    env = base.exports(record)
    paths = list(Path(env['OAT_ZERO_SOURCE_ROOT']).rglob('*.py'))
    paths += [Path(field(record, 'Command')), Path(env['OAT_ZERO_OPS_SNAPSHOT_ROOT']) / 'run_experiment.sh', Path(env['OAT_ZERO_OPS_SNAPSHOT_ROOT']) / 'slurm/train_node302.slurm']
    return {str(p): sha(p) for p in sorted(set(paths))}


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    assert not PLAN.exists() and not (ART / 'transaction.json').exists(), 'Inspect an existing plan or transaction before retry'
    record, run = identity()
    assert field(record, 'JobState') == 'RUNNING' and field(record, 'MinMemoryNode') == '96G'
    assert field(record, 'NodeList') == 'node205'
    detail = checkpoint(run)
    diagnosis = DIAGNOSIS
    review = PROTOCOL
    assert diagnosis.exists() and review.exists()
    evidence = json.loads(diagnosis.read_text())
    assert evidence['returncode'] == 0
    observed = json.loads(evidence['stdout'])
    assert 96 < observed['noncache_gib'] < 110
    events = dict(line.split() for line in observed['memory_events'].splitlines())
    assert int(events['high']) > 1000000 and int(events['oom']) == int(events['oom_kill']) == 0
    delta_receipt = json.loads(PRESSURE_DELTA.read_text())
    assert delta_receipt['returncode'] == 0
    delta = json.loads(delta_receipt['stdout'])
    assert delta['high_events_delta'] > 0
    for sample in [delta['before'], delta['after']]:
        assert int(sample['high']) == 96 * 2**30 and sample['noncache_gib'] > 96
        assert sample['inactive_file'] < 1024**2 and sample['events']['oom'] == sample['events']['oom_kill'] == 0
    node = call('scontrol', 'show', 'node', '-o', 'node205')
    assert int(field(node, 'RealMemory')) - int(field(node, 'AllocMem')) >=32*1024
    plan = {'job_id': JID, 'run': run, 'before': record, 'checkpoint': detail, 'created_at_utc': base.now(),
            'authorization': 'User requested broken-workload recovery and accelerated completion; parent approved this bounded same-ID memory repair.',
            'basis': 'Evaluation is stalled after durable checkpoint960. Measured noncache memory96.043GiB exceeds the96GiB threshold, with3573743memory.high events and noOOM. Increasing to128GiB provides about32GiB working-memory margin; this is a memory-throttling recovery, not a claim of OOM.',
            'target_memory_gib': 128, 'same_job_id': True, 'same_scientific_cell': True, 'same_effective_placement': True,
            'queue_tradeoff': 'node205 currently has39GiB unreserved, sufficient for the32GiB increment; recheck immediately before stopping. Preserve node205/node206/node207. Mandatory requeue delay and scheduler races may move the job back into the queue.',
            'script_sha256': sha(__file__), 'base_sha256': sha(base.__file__),
            'runtime_fingerprints': runtime_fingerprints(record),
            'helper_fingerprints': {str(Path(m.__file__).resolve()): sha(m.__file__) for m in [timeout_recovery, base.campaign]},
            'main_ledger_sha256': sha(base.campaign.E119_LEDGER), 'continuation_ledger_sha256': sha(base.campaign.E119_CONTINUATIONS),
            'diagnosis_sha256': sha(diagnosis), 'pressure_delta_sha256': sha(PRESSURE_DELTA), 'live_pressure_delta': delta, 'checkpoint_review_sha256': sha(review)}
    atomic(PLAN, plan)
    return plan


def apply():
    plan = json.loads(PLAN.read_text())
    assert plan['job_id'] == JID and plan['target_memory_gib'] == 128
    for path, key in [(Path(__file__), 'script_sha256'), (Path(base.__file__), 'base_sha256'),
                      (base.campaign.E119_LEDGER, 'main_ledger_sha256'), (base.campaign.E119_CONTINUATIONS, 'continuation_ledger_sha256'),
                      (DIAGNOSIS, 'diagnosis_sha256'), (PRESSURE_DELTA, 'pressure_delta_sha256'), (PROTOCOL, 'checkpoint_review_sha256')]:
        assert sha(path) == plan[key], key
    assert runtime_fingerprints(plan['before']) == plan['runtime_fingerprints']
    assert all(sha(p) == v for p, v in plan['helper_fingerprints'].items())
    receipt = ART / 'transaction.json'
    assert not receipt.exists(), 'Never repeat an existing transaction blindly'
    before, run = identity()
    stable(plan['before'], before, '96G', same_attempt=True)
    assert run['original_job_id'] == plan['run']['original_job_id']
    d = checkpoint(run)
    assert d['model_metadata_sha256'] == plan['checkpoint']['model_metadata_sha256']
    assert d['archive_fingerprints'] == plan['checkpoint']['archive_fingerprints']
    timeout_recovery.no_other_writer(dict(old_job_id=JID, new_job_id=JID, identity={'run_dir': run['run_dir'], 'run_stamp': run['run_stamp']}), timeout_recovery.active_writers())
    node = call('scontrol', 'show', 'node', '-o', 'node205')
    assert int(field(node, 'RealMemory')) - int(field(node, 'AllocMem')) >=32*1024, 'Required32GiB increment no longer fits'
    directory = ART / 'before_stop'
    directory.mkdir(exist_ok=True)
    archives = base.archive(JID, before, d, directory)
    latest, run = identity()
    stable(before, latest, '96G', same_attempt=True)
    d = checkpoint(run)
    assert d['model_metadata_sha256'] == plan['checkpoint']['model_metadata_sha256']
    assert d['archive_fingerprints'] == plan['checkpoint']['archive_fingerprints']
    t = {'job_id': JID, 'run': run, 'before': before, 'plan_sha256': sha(PLAN), 'created_at_utc': base.now(),
         'checkpoint_before': d, 'archives_before': archives, 'hold_intent': True,
         'own_hold': False, 'resized': False, 'released': False, 'applied': False}
    atomic(receipt, t)
    node = call('scontrol', 'show', 'node', '-o', 'node205')
    assert int(field(node, 'RealMemory')) - int(field(node, 'AllocMem')) >=32*1024, 'Capacity changed before stopping'
    latest, _ = identity()
    stable(before, latest, '96G', same_attempt=True)
    assert runtime_fingerprints(latest) == plan['runtime_fingerprints']
    assert checkpoint(run)['archive_fingerprints'] == plan['checkpoint']['archive_fingerprints']
    call('scontrol', 'requeuehold', str(JID))
    t['own_hold'] = True
    atomic(receipt, t)
    print(json.dumps({'job_id': JID, 'stop_requested': True}), flush=True)
    deadline = time.monotonic() + 600
    while True:
        held = show(JID)
        assert field(held, 'Priority') == '0'
        assert field(held, 'Reason') == 'job_requeued_in_held_state'
        if field(held, 'JobState') == 'PENDING':
            break
        assert field(held, 'JobState') in ('RUNNING', 'COMPLETING')
        assert time.monotonic() < deadline, 'Slow cleanup: preserve owned hold for reconciliation'
        time.sleep(2)
    base.assert_own_hold(held)
    stable(before, held, '96G')
    assert int(field(held, 'Restarts')) == int(field(before, 'Restarts')) + 1
    d = checkpoint(run)
    assert d['model_metadata_sha256'] == plan['checkpoint']['model_metadata_sha256']
    assert d['archive_fingerprints'] == plan['checkpoint']['archive_fingerprints']
    t['checkpoint_after_writer_stopped'] = d
    directory = ART / 'after_stop'
    directory.mkdir(exist_ok=True)
    t['archives_after_stop'] = base.archive(JID, held, d, directory)
    atomic(receipt, t)
    call('scontrol', 'update', f'JobId={JID}', 'MinMemoryNode=131072')
    resized, current_run = identity()
    assert current_run['original_job_id'] == run['original_job_id']
    base.assert_own_hold(resized)
    stable(before, resized, '128G')
    t.update(resized=True, held_after_update=resized)
    atomic(receipt, t)
    assert t['own_hold'] and t['resized'] and not t['released']
    assert runtime_fingerprints(resized) == plan['runtime_fingerprints']
    assert sha(base.campaign.E119_LEDGER) == plan['main_ledger_sha256'] and sha(base.campaign.E119_CONTINUATIONS) == plan['continuation_ledger_sha256']
    timeout_recovery.no_other_writer(dict(old_job_id=JID, new_job_id=JID, identity={'run_dir': run['run_dir'], 'run_stamp': run['run_stamp']}), timeout_recovery.active_writers())
    call('scontrol', 'release', str(JID))
    after, _ = identity()
    stable(before, after, '128G')
    assert field(after, 'JobState') in ('RUNNING', 'PENDING') and field(after, 'Priority') != '0'
    assert sha(base.campaign.E119_LEDGER) == plan['main_ledger_sha256']
    assert sha(base.campaign.E119_CONTINUATIONS) == plan['continuation_ledger_sha256']
    t.update(released=True, applied=True, after=after, completed_at_utc=base.now())
    atomic(receipt, t)
    print(json.dumps({'job_id': JID, 'applied': True, 'memory': '128G', 'state': field(after, 'JobState'), 'reason': field(after, 'Reason'), 'resume_step': d['checkpoint_step'], 'logged_updates_to_repeat': d['unsaved_steps']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply:
        apply()
    else:
        p = prepare()
        print(json.dumps({'dry_run': 'pass', 'job_id': JID, 'resume_step': p['checkpoint']['checkpoint_step'], 'repeat_logged_updates': p['checkpoint']['unsaved_steps']}))
