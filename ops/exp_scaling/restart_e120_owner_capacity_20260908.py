#!/usr/bin/env python3
"""Resume the twelve unfinished E120-R1 cells in two audited owner-node lanes."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import backfill_e120_node105_20260905 as prior
import campaign_stats as campaign
from prioritize_e118_capacity_20260905 import command, exports, field, show, submit_tokens
from recover_e119_health_20260905 import checkpoint

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'var/artifacts/e120_owner_restart_20260908'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
AMENDMENT = ROOT / 'paper/preregistration/e120_owner_restart_20260908.md'
IDENTITY = ('domain', 'model_key', 'seed', 'run_dir', 'run_stamp')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    prior.save(path, value)


def event(tx, message):
    tx.setdefault('events', []).append({'at': prior.now(), 'message': message})
    save(TX, tx)


def capacity(node, memory, cpus, gpus):
    record = command(['scontrol', 'show', 'node', '-o', node]).stdout
    assert 'mltheory' in field(record, 'Partitions').split(',')
    assert not any(x in field(record, 'State') for x in ('DOWN', 'DRAIN', 'FAIL'))
    assert int(field(record, 'RealMemory')) - int(field(record, 'AllocMem')) >= memory * 1024
    assert int(field(record, 'CPUEfctv')) - int(field(record, 'CPUAlloc')) >= cpus
    allocated = field(record, 'AllocTRES')
    used_gpus = prior.gpu_count(allocated) if allocated not in ('', '(null)', None) else 0
    assert prior.gpu_count(field(record, 'CfgTRES')) - used_gpus >= gpus
    return record


def identity(item, *, held=True, dependency=None, placement=None):
    record = show(item['job_id'])
    assert field(record, 'JobState') == 'PENDING', item['job_id']
    if held:
        assert field(record, 'Priority') == '0', item['job_id']
    else:
        assert int(field(record, 'Priority')) > 0
    original = item['original_command']
    observed = submit_tokens(record)
    assert exports(observed) == exports(original)
    assert observed[-1] == original[-1]
    assert digest(observed[-1]) == item['wrapper_sha256']
    for key in ('NumCPUs', 'MinMemoryNode', 'TimeLimit', 'Nice', 'Requeue', 'WorkDir'):
        assert field(record, key) == field(item['before_record'], key), (item['job_id'], key)
    if dependency is not None:
        actual = field(record, 'Dependency')
        assert actual.startswith(dependency) if dependency else actual == '(null)', (actual, dependency)
    if placement:
        for key, value in placement.items():
            assert field(record, key) == value, (key, field(record, key), value)
    assert not (Path(item['run']['run_dir']) / 'TRAINING_COMPLETE.json').exists()
    return record


def falcon_command(original, old_id):
    changes = {'--partition': 'mltheory', '--account': 'mltheory', '--nodelist': 'node105',
               '--gres': 'gpu:a5000:1', '--exclude': prior.EXCLUSION,
               '--comment': f'e120-owner-restart-20260908-old{old_id}'}
    removed = set(changes) | {'--dependency', '--hold'}
    result = [v for v in original[:-1] if v.split('=', 1)[0] not in removed]
    result.extend(f'{k}={v}' for k, v in changes.items())
    result.extend(['--hold', original[-1]])
    assert exports(result) == exports(original)
    return result


def prepare():
    assert not PLAN.exists(), 'Inspect existing plan rather than overwrite it'
    ART.mkdir(parents=True, exist_ok=True)
    prior.patch_valid()
    main = json.loads(campaign.E120_LEDGER.read_text())
    mapping = campaign.e120_continuation_jobs(campaign.E120_LEDGER)
    assert len(mapping) == 7
    rows = []
    for run in main['runs']:
        if run['model_key'] != 'qwen3b' and run['job_id'] not in (31033700, 31033701):
            continue
        jid = mapping.get(run['job_id'], run['job_id'])
        record = show(jid)
        original = submit_tokens(record)
        env = exports(original)
        assert env['SAVE_PATH'] == run['run_dir'] and env['RUN_STAMP'] == run['run_stamp']
        assert env['OAT_ZERO_AUTO_RESUME'] == '1'
        assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING'] == 'fresh_frequency'
        assert field(record, 'Dependency') == '(null)'
        item = {'job_id': jid, 'original_job_id': run['job_id'], 'run': run,
                'before_record': record, 'original_command': original,
                'wrapper_sha256': digest(original[-1])}
        identity(item, held=run['model_key'] == 'qwen3b', dependency='')
        candidates = list(Path(run['run_dir']).glob('debug_job*/checkpoints/step_*'))
        item['checkpoint'] = checkpoint(run) if candidates else None
        item['resume_step'] = item['checkpoint']['step'] if item['checkpoint'] else 0
        if run['model_key'] == 'falcon1b':
            assert not candidates and not list(Path(run['run_dir']).glob('debug_job*/train_metrics.jsonl'))
            item['command'] = falcon_command(original, jid)
            test = command([item['command'][0], '--test-only', *[v for v in item['command'][1:] if v != '--hold']])
            item['sbatch_test_only'] = {'stdout': test.stdout, 'stderr': test.stderr}
        rows.append(item)
    qwen = sorted([r for r in rows if r['run']['model_key'] == 'qwen3b'],
                  key=lambda r: (-r['resume_step'], r['original_job_id']))
    assert len(qwen) == 10 and len(rows) == 12
    predecessors = [None, None]
    for index, item in enumerate(qwen):
        lane = index % 2
        item['lane'] = lane
        item['dependency'] = f'afterany:{predecessors[lane]}' if predecessors[lane] else ''
        predecessors[lane] = item['job_id']
    AMENDMENT.write_text('''# E120 owner capacity restart, September 8, 2026

The user explicitly requested repairing unfinished E118/E119/E120 cells, getting
them running, using node302 and node105, and finishing E120. This supersedes the
September 4 E118 400k priority holds and supplies the previously missing E120
replacement authorization recorded in the September 6 automatic approval rejection.
The ten Qwen-3B holds were resource priority decisions, not scientific gates.

Keep all ten Qwen-3B scheduler IDs and the immutable 45-cell primary ledger.
Validate six full model/optimizer checkpoints, and restart the remaining four
cells from initialization because no durable checkpoint exists. Pantry seeds
71–73 lost their pre-checkpoint work in the already recorded September 4 pause;
seed 74 never started. Preserve recipe, data, seed, eight-pass horizon, run path,
frozen repaired runtime, automatic resume, 128 GiB, 16 CPUs and one A100 per job.
Add the existing explicit excluded node list and release two afterany chains on
node302/mltheory, ordered by durable progress. At most two Qwen allocations run
simultaneously, leaving a third 128-GiB owner slot for E118/E119 recovery.

Move the two never-started Falcon Pantry seeds 55–56 from the priority-blocked
cs A6000 pool to node105/mltheory A5000. Same-treatment seed 57 completed on this
hardware. Preserve scientific/runtime exports byte for byte, wrapper, output
identity, 64 GiB, 8 CPUs, 72-hour walltime, Nice100 and automatic resume. Submit
held replacements, audit, register the two continuations, retire only the old
pending allocations, then release. Both may run concurrently. Preserve all
other jobs. Check the repaired runtime fingerprints before applying and verify
actual current-attempt optimizer progress after release.

All before/after scheduler state, checkpoint counters, ledger hashes and action
receipts are recorded in var/artifacts/e120_owner_restart_20260908/.
''')
    plan = {'schema': 'e120-owner-restart-plan-v1', 'created_at': prior.now(), 'status': 'prepared',
            'authorization': 'User explicitly requested E120 repair/restart and use of node302/node105 on 2026-09-08.',
            'previous_rejection': str(ROOT / 'var/artifacts/e120_falcon_node302_backfill_20260906/approval_review_rejection.json'),
            'main_ledger_sha256': digest(campaign.E120_LEDGER),
            'continuation_ledger_sha256': digest(campaign.E120_CONTINUATIONS),
            'controller_sha256': digest(__file__), 'amendment_sha256': digest(AMENDMENT),
            'scheduler_only': True, 'scientific_configuration_changed': False,
            'node302_before': capacity('node302', 256, 32, 2),
            'node105_before': capacity('node105', 128, 16, 2),
            'qwen': qwen, 'falcon': [r for r in rows if r['run']['model_key'] == 'falcon1b']}
    (ART / 'continuations.before.json').write_bytes(campaign.E120_CONTINUATIONS.read_bytes())
    save(PLAN, plan)
    print(json.dumps({'prepared': True, 'qwen': [{'job': r['job_id'], 'step': r['resume_step'], 'dependency': r['dependency']} for r in qwen],
                      'falcon': [r['job_id'] for r in plan['falcon']]}))


def apply():
    assert not TX.exists(), 'Inspect transaction before retrying'
    plan = json.loads(PLAN.read_text())
    for path, key in ((campaign.E120_LEDGER, 'main_ledger_sha256'),
                      (campaign.E120_CONTINUATIONS, 'continuation_ledger_sha256'),
                      (__file__, 'controller_sha256'), (AMENDMENT, 'amendment_sha256')):
        assert digest(path) == plan[key], (path, key)
    prior.patch_valid()
    capacity('node302', 256, 32, 2)
    capacity('node105', 128, 16, 2)
    tx = dict(plan, status='applying', started_at=prior.now())
    event(tx, 'Validated current explicit E120 authorization, runtime hashes and joint owner capacity')
    try:
        for item in tx['qwen']:
            identity(item, dependency='')
            if item['checkpoint']:
                assert checkpoint(item['run']) == item['checkpoint']
            command(['scontrol', 'update', f"JobId={item['job_id']}",
                     f"Dependency={item['dependency'] or '0'}", f'ExcNodeList={prior.EXCLUSION}'])
            item['held_after'] = identity(item, dependency=item['dependency'],
                placement={'ReqNodeList': 'node302', 'Partition': 'mltheory', 'Account': 'mltheory', 'ExcNodeList': prior.EXCLUSION})
            event(tx, f"Prepared Qwen job {item['job_id']} in lane {item['lane']}")
        for item in tx['falcon']:
            identity(item, held=False, dependency='')
            command(['scontrol', 'hold', str(item['job_id'])])
            item['old_held'] = identity(item, dependency='')
            item['submission_uncertain'] = True
            event(tx, f"Submitting held node105 replacement for {item['job_id']}")
            response = command(item['command']).stdout.strip()
            item['new_job_id'] = int(response.split(';')[0])
            item['submission_uncertain'] = False
            audit_item = dict(item, job_id=item['new_job_id'])
            record = identity(audit_item, dependency='', placement={'ReqNodeList': 'node105', 'Partition': 'mltheory',
                              'Account': 'mltheory', 'ExcNodeList': prior.EXCLUSION})
            assert field(record, 'TresPerNode') == 'gres/gpu:a5000:1'
            assert field(record, 'StdOut') == str(ROOT / f"slurm-{item['new_job_id']}.out")
            item['held_after'] = record
            event(tx, f"Audited held Falcon replacement {item['new_job_id']}")
        assert digest(campaign.E120_CONTINUATIONS) == plan['continuation_ledger_sha256']
        ledger = json.loads(campaign.E120_CONTINUATIONS.read_text())
        for item in tx['falcon']:
            assert not any(r['original_job_id'] == item['original_job_id'] for r in ledger['continuations'])
            row = {key: item['run'][key] for key in IDENTITY}
            row.update(original_job_id=item['original_job_id'], continuation_job_id=item['new_job_id'],
                       observed_step_before_continuation=0, resume_checkpoint=0, released=False,
                       pvl_excluded=True, repair_kind='owner_capacity_node105_20260908',
                       placement_amendment=str(AMENDMENT), held_scheduler_record=item['held_after'],
                       new_placement={'account': 'mltheory', 'partition': 'mltheory', 'node': 'node105',
                                      'gpu': 'a5000', 'cpus': 8, 'memory': '64G', 'nice': 100})
            ledger['continuations'].append(row)
        ledger.setdefault('placement_amendments', []).append(str(AMENDMENT))
        save(campaign.E120_CONTINUATIONS, ledger)
        tx['ledger_committed'] = True
        event(tx, 'Registered both held Falcon replacements; primary scientific ledger unchanged')
        for item in tx['falcon']:
            identity(item, dependency='')
            command(['scancel', str(item['job_id'])])
            item['old_cancelled'] = show(item['job_id'])
            assert field(item['old_cancelled'], 'JobState') == 'CANCELLED'
            event(tx, f"Retired superseded pending Falcon job {item['job_id']}")
        for item in [*tx['qwen'], *tx['falcon']]:
            jid = item.get('new_job_id', item['job_id'])
            command(['scontrol', 'release', str(jid)])
            record = show(jid)
            assert field(record, 'JobState') in ('PENDING', 'RUNNING', 'CONFIGURING')
            assert int(field(record, 'Priority')) > 0
            item['released'] = True
            item['release_record'] = record
            if 'new_job_id' in item:
                next(r for r in ledger['continuations'] if r['original_job_id'] == item['original_job_id'])['released'] = True
            event(tx, f"Released E120 job {jid}")
        save(campaign.E120_CONTINUATIONS, ledger)
        assert digest(campaign.E120_LEDGER) == plan['main_ledger_sha256']
        mapping = campaign.e120_continuation_jobs(campaign.E120_LEDGER)
        assert len(mapping) == 9
        assert all(mapping[r['original_job_id']] == r['new_job_id'] for r in tx['falcon'])
        tx['status'] = 'released'
        tx['continuation_ledger_after_sha256'] = digest(campaign.E120_CONTINUATIONS)
        event(tx, 'All twelve unfinished cells eligible: two Qwen chains and two Falcon owner allocations')
        print(json.dumps({'status': 'released', 'qwen': [r['job_id'] for r in tx['qwen']],
                          'falcon': [r['new_job_id'] for r in tx['falcon']]}))
    except BaseException as exc:
        tx['status'] = 'stopped_for_reconciliation'
        tx['error'] = repr(exc)
        event(tx, 'Stopped with transaction state recorded; inspect before retrying')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'apply'))
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'transaction.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.phase]()
