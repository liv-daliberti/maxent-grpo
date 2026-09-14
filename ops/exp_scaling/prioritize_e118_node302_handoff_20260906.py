#!/usr/bin/env python3
"""Audited user-requested E119 requeue and one E118 node302 priority handoff."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re

import prioritize_e118_capacity_20260905 as e118
import recover_e119_throttling_20260906 as e119

ROOT = e118.ROOT
ART = ROOT / 'var/artifacts/e118_node302_handoff_20260906'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
AMENDMENT = ROOT / 'paper/preregistration/e118_node302_handoff_20260906.md'
E118_OLD = 31048107
E119_JOB = 31075341
LOCK = ROOT / 'var/artifacts/e118_ledger_promotion.lock'
e118.AUDIT = ART / 'e118_replacement.json'
e118.PROTOCOL = AMENDMENT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(tx, message):
    tx.setdefault('events', []).append({'at': e118.now(), 'event': message})
    e118.atomic(TX, tx)


def independent():
    rows = e118.command(['squeue', '-h', '-o', '%i|%E']).stdout.splitlines()
    assert not [r for r in rows if re.search(r'(?<!\d)' + str(E118_OLD) + r'(?!\d)', r.split('|', 1)[1])], 'E118 predecessor has dependent jobs'


def unchanged119(plan, record, dependency='(null)'):
    before = plan['e119_before']
    assert e119.base.submitline(record) == e119.base.submitline(before)
    for key in e119.PRESERVE:
        if key != 'Dependency':
            assert e119.field(record, key) == e119.field(before, key), key
    assert e119.field(record, 'MinMemoryNode') == '128G'
    actual = e119.field(record, 'Dependency')
    assert actual == dependency or (dependency != '(null)' and actual.startswith(dependency + '(')), actual
    for key, path in [('e119_ledger_sha256', e119.base.campaign.E119_LEDGER),
                      ('e119_continuation_sha256', e119.base.campaign.E119_CONTINUATIONS)]:
        assert sha(path) == plan[key], key
    assert sha(e119.field(record, 'Command')) == plan['e119_launcher_sha256']


def prepare():
    assert not PLAN.exists(), 'Inspect the existing handoff before preparing again'
    ART.mkdir(parents=True, exist_ok=True)
    source_bytes = e118.LEDGER.read_bytes()
    source = json.loads(source_bytes)
    row = next(r for r in source['runs'] if int(r['job_id']) == E118_OLD)
    assert (row['domain'], row['arm'], int(row['seed'])) == ('graph_coloring', 'maxrl', 70)
    record = e118.show(E118_OLD)
    assert e118.field(record, 'JobState') == 'PENDING' and int(e118.field(record, 'Priority')) > 0
    assert e118.field(record, 'Dependency') == '(null)'
    independent()
    original = e118.submit_tokens(record)
    checkpoint = e119.base.checked_checkpoint(row)
    assert checkpoint['step'] == 1728
    assert not (Path(row['run_dir']) / 'TRAINING_COMPLETE.json').exists()
    item = {key: row[key] for key in e118.IDENTITY}
    item.update(old_job_id=E118_OLD, before_record=record, lane='node302',
                original_command=original, command=e118.placed(original, 'node302', E118_OLD),
                resume_checkpoint=checkpoint['path'], checkpoint_validation=checkpoint, new_job_id=None)
    # Retain the audited parent controller's comment for idempotent submission reconciliation.
    test = e118.command([item['command'][0], '--test-only', *[x for x in item['command'][1:] if x != '--hold']])
    before119, run119 = e119.identity(E119_JOB)
    assert e119.field(before119, 'JobState') == 'RUNNING'
    assert e119.field(before119, 'MinMemoryNode') == '128G'
    detail119 = e119.checkpoint(E119_JOB, run119, True)
    plan = {'schema': 'e118-node302-priority-handoff-v1', 'created_at_utc': e118.now(),
            'authorization': 'User confirmed: Requeue it; prioritize E118. Requeue E11931075341 to resume later and start one existing E118 cell on node302.',
            'e118_item': item, 'e118_source_sha256': e118.sha(source_bytes),
            'e118_launcher_sha256': sha(original[-1]), 'e119_before': before119, 'e119_run': run119,
            'e119_checkpoint': detail119, 'e119_ledger_sha256': sha(e119.base.campaign.E119_LEDGER),
            'e119_continuation_sha256': sha(e119.base.campaign.E119_CONTINUATIONS),
            'e119_launcher_sha256': sha(e119.field(before119, 'Command')),
            'node_before': e118.command(['scontrol', 'show', 'node', '-o', 'node302']).stdout,
            'sbatch_test_only': {'stdout': test.stdout, 'stderr': test.stderr},
            'controller_sha256': sha(__file__), 'outcomes_inspected_for_selection': False}
    AMENDMENT.write_text('''# E118 node302 priority handoff — September 6, 2026

The user explicitly confirmed: “Requeue it; prioritize E118.” Requeue only
E119 Pantry Re:Dr.GRPO seed43, effective job31075341, into a user hold after
archiving logs and validating its latest complete model and optimizer checkpoint
and all three saved counters. Preserve its same job ID, registered run directory,
128GiB host memory,8CPUs,1GPU,36-hour walltime, frozen scientific/runtime exports,
checkpoint cadence and continuation-ledger identity. Validate again after its
writer stops. Never stop while a newer incomplete checkpoint is being written.
The original restart mapping from31014459 remains provenance; automatic resume
selects the newest valid checkpoint under the unchanged run directory.

Replace only pending E118 Graph MaxRL seed70 job31048107, selected by highest
valid pending checkpoint progress (1728; ties by existing job ID), without
examining efficacy outcomes. It has no live dependent jobs. Use the existing
held-and-audited E118 replacement workflow, preserve1GPU/16CPUs/128GiB/12hours,
all scientific/runtime exports and the frozen launcher, and change placement to
mltheory/node302 with the existing explicit excluded node list. Source and
aggregate ledgers are committed under their promotion lock before retiring the
old held pending allocation and releasing its replacement. No scientific cell
or treatment is added, and no other running allocation is interrupted.

After the E118 replacement starts, release E11931075341 with an afterany
dependency on that replacement. This queues safe later automatic resumption
without immediately reclaiming the handed-off slot. Only this scheduler
dependency changes for E119. Record actual repeated updates, full before/after
scheduler records, checkpoint provenance and startup verification in
var/artifacts/e118_node302_handoff_20260906/. If a stage fails, retain its recorded
transaction-owned holds for explicit reconciliation rather than spawning another
writer. Existing128GiB requests are never reduced to force admission.
''')
    plan['amendment_sha256'] = sha(AMENDMENT)
    (ART / 'e118_source.before.json').write_bytes(source_bytes)
    e118.atomic(PLAN, plan)
    print(json.dumps({'prepared': True, 'e118_old_job': E118_OLD, 'e118_checkpoint': 1728,
                      'e119_job': E119_JOB, 'e119_checkpoint': detail119['checkpoint_step'],
                      'e119_updates_to_repeat': detail119['unsaved_steps']}), flush=True)


def load():
    plan = json.loads(PLAN.read_text())
    assert sha(__file__) == plan['controller_sha256']
    assert sha(AMENDMENT) == plan['amendment_sha256']
    assert sha(plan['e118_item']['original_command'][-1]) == plan['e118_launcher_sha256']
    return plan


def stop119():
    plan = load()
    assert not TX.exists(), 'Never repeat a recorded requeue without reconciliation'
    record, run = e119.identity(E119_JOB)
    unchanged119(plan, record)
    assert e119.field(record, 'JobState') == 'RUNNING'
    assert e119.field(record, 'Restarts') == e119.field(plan['e119_before'], 'Restarts')
    detail = e119.checkpoint(E119_JOB, run, True)
    archive = ART / 'e119_before_requeue'
    archive.mkdir()
    archived = e119.base.archive(E119_JOB, record, detail, archive)
    # Recheck after archival; a new save may have begun in the meantime.
    detail = e119.checkpoint(E119_JOB, run, True)
    current, _ = e119.identity(E119_JOB)
    unchanged119(plan, current)
    assert e119.field(current, 'JobState') == 'RUNNING'
    tx = {'started_at_utc': e118.now(), 'plan_sha256': sha(PLAN), 'e119_before': current,
          'e119_checkpoint_before': detail, 'e119_archive': archived, 'requeue_requested': True}
    event(tx, 'validated complete E119 checkpoint; requesting authorized same-ID requeue into hold')
    e118.command(['scontrol', 'requeuehold', str(E119_JOB)])
    tx['e119_own_hold'] = True
    event(tx, 'E119 requeuehold accepted; wait for allocation cleanup before handoff')
    print(json.dumps({'requeue_requested': True, 'checkpoint': detail['checkpoint_step'],
                      'updates_to_repeat': detail['unsaved_steps']}), flush=True)


def launch118():
    plan = load()
    tx = json.loads(TX.read_text())
    assert tx.get('e119_own_hold') and not tx.get('e118_released')
    record, run = e119.identity(E119_JOB)
    unchanged119(plan, record)
    e119.base.assert_own_hold(record)
    assert int(e119.field(record, 'Restarts')) == int(e119.field(tx['e119_before'], 'Restarts')) + 1
    assert e119.field(record, 'AllocTRES') == '(null)', 'E119 allocation cleanup incomplete'
    detail = e119.checkpoint(E119_JOB, run, True)
    assert detail['checkpoint_step'] >= tx['e119_checkpoint_before']['checkpoint_step']
    if detail['checkpoint_step'] == tx['e119_checkpoint_before']['checkpoint_step']:
        assert detail['model_metadata_sha256'] == tx['e119_checkpoint_before']['model_metadata_sha256']
    tx['e119_checkpoint_after_stop'] = detail
    tx['e119_held_record'] = record
    node = e118.command(['scontrol', 'show', 'node', '-o', 'node302']).stdout
    assert int(e118.field(node, 'RealMemory')) - int(e118.field(node, 'AllocMem')) >= 128 * 1024
    assert int(e118.field(node, 'CPUEfctv')) - int(e118.field(node, 'CPUAlloc')) >= 16
    cfg = dict(x.split('=', 1) for x in e118.field(node, 'CfgTRES').split(','))
    used = dict(x.split('=', 1) for x in e118.field(node, 'AllocTRES').split(','))
    assert int(cfg['gres/gpu']) - int(used.get('gres/gpu', 0)) >= 1
    tx['node_after_e119_stop'] = node
    event(tx, 'E119 writer stopped; checkpoint intact; full128GiB E118 capacity available')
    independent()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if e118.AUDIT.exists():
            audit = json.loads(e118.AUDIT.read_text())
        else:
            assert e118.sha(e118.LEDGER.read_bytes()) == plan['e118_source_sha256']
            audit = {'schema': 'e118-node302-handoff-replacement-v1', 'created_at': e118.now(),
                     'protocol': str(AMENDMENT), 'source_ledger': str(e118.LEDGER),
                     'original_ledger_sha256': plan['e118_source_sha256'], 'status': 'planned',
                     'scheduler_only': True, 'outcomes_inspected': False,
                     'replacements': [plan['e118_item']], 'cs_placement_updates': [], 'events': []}
            e118.save(audit, 'prepared one authorized pending E118 replacement')
        e118.apply(audit)
    tx['e118_new_job'] = audit['replacements'][0]['new_job_id']
    tx['e118_released'] = True
    event(tx, 'released exact E118 continuation after authoritative ledger promotion')
    print(json.dumps({'e118_new_job': tx['e118_new_job'], 'e119_held': True}), flush=True)


def queue119():
    plan = load()
    tx = json.loads(TX.read_text())
    assert tx['e118_released'] and not tx.get('e119_released')
    new = tx['e118_new_job']
    record118 = e118.show(new)
    assert e118.field(record118, 'JobState') == 'RUNNING', 'Wait for E118 allocation before releasing E119'
    assert e118.field(record118, 'NodeList') == 'node302'
    run = plan['e119_run']
    record119 = e119.base.live_identity(E119_JOB, run)
    dependency = f'afterany:{new}'
    actual_dependency = e119.field(record119, 'Dependency')
    already_dependent = actual_dependency.startswith(dependency)
    unchanged119(plan, record119, dependency if already_dependent else '(null)')
    own_hold = e119.field(record119, 'Priority') == '0'
    if own_hold:
        e119.base.assert_own_hold(record119)
    else:
        assert already_dependent and tx.get('e119_dependency_intent') == dependency
        assert e119.field(record119, 'JobState') == 'PENDING'
    detail = e119.checkpoint(E119_JOB, run, True)
    assert detail['model_metadata_sha256'] == tx['e119_checkpoint_after_stop']['model_metadata_sha256']
    dependency = f'afterany:{new}'
    tx['e119_dependency_intent'] = dependency
    event(tx, 'queue E119 behind the new E118 allocation using afterany')
    if not already_dependent:
        e118.command(['scontrol', 'update', f'JobId={E119_JOB}', f'Dependency={dependency}'])
    if own_hold:
        held = e119.base.show(E119_JOB)
        unchanged119(plan, held, dependency)
        e119.base.assert_own_hold(held)
        e118.command(['scontrol', 'release', str(E119_JOB)])
    after = e119.base.show(E119_JOB)
    unchanged119(plan, after, dependency)
    assert e119.field(after, 'JobState') == 'PENDING' and int(e119.field(after, 'Priority')) > 0
    # Slurm may briefly report Reason=None before its next scheduler cycle;
    # pending state, positive priority and the exact dependency are authoritative.
    tx.update(e119_released=True, e119_after=after, e118_running_record=record118,
              completed_at_utc=e118.now(), status='handoff_complete')
    event(tx, 'E118 running onnode302; E119 queued with complete checkpoint and no user hold')
    print(json.dumps({'e118_job': new, 'e118_state': 'RUNNING', 'e119_job': E119_JOB,
                      'e119_state': 'PENDING', 'e119_dependency': dependency,
                      'e119_resume_step': detail['checkpoint_step'],
                      'e119_updates_to_repeat': detail['unsaved_steps']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'stop119', 'launch118', 'queue119'))
    args = parser.parse_args()
    globals()[args.phase]()
