#!/usr/bin/env python3
"""Restart one severely degraded Pantry cell at96GiB from its validated save."""
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

JID = 31037832
STAMP = 'e119_level2_pantry_replay_drgrpo_s44'
ART = ROOT / 'var/artifacts/e119_pantry_s44_memory96_20260906'
PLAN = ART / 'plan.json'
PRESERVE = tuple(dict.fromkeys((*base.PRESERVE, 'QOS', 'Nice', 'ReqNodeList', 'Features', 'UserId')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    run = base.identities()[JID]
    assert (run['run_stamp'], run['domain'], run['arm'], int(run['seed'])) == (STAMP, 'pantry_plan', 'replay_drgrpo', 44)
    record = show(JID)
    exp = base.exports(record)
    assert field(record, 'JobId') == str(JID)
    assert field(record, 'JobName') == 'e119-pantry-rd-s44'
    assert field(record, 'UserId').startswith(getpass.getuser() + '(')
    assert exp['RUN_STAMP'] == STAMP and Path(exp['SAVE_PATH']).resolve() == Path(run['run_dir']).resolve()
    assert exp['OAT_ZERO_AUTO_RESUME'] == '1'
    assert field(record, 'Account') == 'allcs' and field(record, 'Partition') == 'cs'
    assert set(call('scontrol', 'show', 'hostnames', field(record, 'ReqNodeList')).splitlines()) == {'node205', 'node207'}
    assert field(record, 'NumCPUs') == '8' and field(record, 'TresPerNode') == 'gres/gpu:1'
    assert field(record, 'Dependency') in ('(null)', '')
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
    assert d['checkpoint_step'] == 192, 'Newer checkpoint requires refreshed review'
    assert d['saved_counter_validation']['saved_counters'] == dict.fromkeys(('global_steps', 'global_step', 'prompt_batches_consumed_total'), 192)
    assert d['unsaved_steps'] is not None and 0 <= d['unsaved_steps'] <= 16
    model = Path(d['checkpoint']) / 'mp_rank_00_model_states.pt'
    with zipfile.ZipFile(model) as z:
        raw = z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))
    assert any(arg == 'online_canonical_bank_state' for _, arg, _ in pickletools.genops(raw))
    d['model_metadata_sha256'] = hashlib.sha256(raw).hexdigest()
    return d


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'transaction.json').exists(), 'Inspect an existing transaction before retry'
    record, run = identity()
    assert field(record, 'JobState') == 'RUNNING' and field(record, 'MinMemoryNode') == '64G'
    assert field(record, 'NodeList') == 'node205'
    detail = checkpoint(run)
    diagnosis = ART / 'diagnosis/review.json'
    review = ART / 'checkpoint_review/review.json'
    assert diagnosis.exists() and review.exists()
    plan = {'job_id': JID, 'run': run, 'before': record, 'checkpoint': detail, 'created_at_utc': base.now(),
            'authorization': 'User explicitly requested repair of Pantry31037832 after the62-minute weight synchronization.',
            'basis': 'Severe observed throughput degradation. Live cgroup cause could not be measured;96GiB is a conservative intervention under uncertainty, not a claim of proven memory root cause.',
            'target_memory_gib': 96, 'same_job_id': True, 'same_scientific_cell': True, 'same_effective_placement': True,
            'queue_tradeoff': 'Current node205/207 routes lack96GiB headroom even after this64GiB job stops; job may queue normally.',
            'script_sha256': sha(__file__), 'base_sha256': sha(base.__file__),
            'main_ledger_sha256': sha(base.campaign.E119_LEDGER), 'continuation_ledger_sha256': sha(base.campaign.E119_CONTINUATIONS),
            'diagnosis_sha256': sha(diagnosis), 'checkpoint_review_sha256': sha(review)}
    atomic(PLAN, plan)
    return plan


def apply():
    plan = json.loads(PLAN.read_text())
    assert plan['job_id'] == JID and plan['target_memory_gib'] == 96
    for path, key in [(Path(__file__), 'script_sha256'), (Path(base.__file__), 'base_sha256'),
                      (base.campaign.E119_LEDGER, 'main_ledger_sha256'), (base.campaign.E119_CONTINUATIONS, 'continuation_ledger_sha256'),
                      (ART / 'diagnosis/review.json', 'diagnosis_sha256'), (ART / 'checkpoint_review/review.json', 'checkpoint_review_sha256')]:
        assert sha(path) == plan[key], key
    receipt = ART / 'transaction.json'
    assert not receipt.exists(), 'Never repeat an existing transaction blindly'
    before, run = identity()
    stable(plan['before'], before, '64G', same_attempt=True)
    assert run['original_job_id'] == plan['run']['original_job_id']
    d = checkpoint(run)
    assert d['model_metadata_sha256'] == plan['checkpoint']['model_metadata_sha256']
    directory = ART / 'before_stop'
    directory.mkdir(exist_ok=True)
    archives = base.archive(JID, before, d, directory)
    latest, run = identity()
    stable(before, latest, '64G', same_attempt=True)
    d = checkpoint(run)
    assert d['model_metadata_sha256'] == plan['checkpoint']['model_metadata_sha256']
    t = {'job_id': JID, 'run': run, 'before': before, 'plan_sha256': sha(PLAN), 'created_at_utc': base.now(),
         'checkpoint_before': d, 'archives_before': archives, 'hold_intent': True,
         'own_hold': False, 'resized': False, 'released': False, 'applied': False}
    atomic(receipt, t)
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
    stable(before, held, '64G')
    assert int(field(held, 'Restarts')) == int(field(before, 'Restarts')) + 1
    d = checkpoint(run)
    assert d['model_metadata_sha256'] == plan['checkpoint']['model_metadata_sha256']
    t['checkpoint_after_writer_stopped'] = d
    directory = ART / 'after_stop'
    directory.mkdir(exist_ok=True)
    t['archives_after_stop'] = base.archive(JID, held, d, directory)
    atomic(receipt, t)
    call('scontrol', 'update', f'JobId={JID}', 'MinMemoryNode=98304')
    resized, current_run = identity()
    assert current_run['original_job_id'] == run['original_job_id']
    base.assert_own_hold(resized)
    stable(before, resized, '96G')
    t.update(resized=True, held_after_update=resized)
    atomic(receipt, t)
    assert t['own_hold'] and t['resized'] and not t['released']
    call('scontrol', 'release', str(JID))
    after, _ = identity()
    stable(before, after, '96G')
    assert field(after, 'JobState') in ('RUNNING', 'PENDING') and field(after, 'Priority') != '0'
    assert sha(base.campaign.E119_LEDGER) == plan['main_ledger_sha256']
    assert sha(base.campaign.E119_CONTINUATIONS) == plan['continuation_ledger_sha256']
    t.update(released=True, applied=True, after=after, completed_at_utc=base.now())
    atomic(receipt, t)
    print(json.dumps({'job_id': JID, 'applied': True, 'memory': '96G', 'state': field(after, 'JobState'), 'reason': field(after, 'Reason'), 'resume_step': d['checkpoint_step'], 'logged_updates_to_repeat': d['unsaved_steps']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply:
        apply()
    else:
        p = prepare()
        print(json.dumps({'dry_run': 'pass', 'job_id': JID, 'resume_step': p['checkpoint']['checkpoint_step'], 'repeat_logged_updates': p['checkpoint']['unsaved_steps']}))
