#!/usr/bin/env python3
"""Audited held E119 continuations on node208; every science export is preserved.

prepare is scheduler-read-only. stage/release require an explicit old job ID.
Interrupted submissions fail closed; use adopt only after independently locating
and verifying the exact held successor. The old job remains a dormant fallback.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import accelerate_a5000_completion_20260909 as prior
import campaign_stats as campaign
import recover_e119_health_20260905 as checkpoints
import recover_terminal_timeouts_20260908 as recovery

b = prior.base
ROOT = b.ROOT
ART = ROOT / 'var/artifacts/e119_node208_completion_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e119_node208_completion_20260909.md'
PRIMARY = campaign.E119_LEDGER
LEDGER = campaign.E119_CONTINUATIONS
LOCK = ROOT / 'var/artifacts/e118_ledger_promotion.lock'
IDENTITY = ('domain', 'arm', 'seed', 'run_dir', 'run_stamp')
# Only the ten independently audited, unguarded pending cells are in scope.
TARGETS = {
    31124279: ('maxrl', 45, 128, '3-00:00:00'),
    31048181: ('maxrl', 44, 116, '1-12:00:00'),
    31048187: ('maxrl', 46, 116, '1-12:00:00'),
    31048191: ('maxrl', 47, 116, '1-12:00:00'),
    31048180: ('drgrpo', 44, 116, '1-12:00:00'),
    31037836: ('replay_drgrpo', 45, 116, '1-12:00:00'),
    31048185: ('replay_maxrl', 45, 116, '1-12:00:00'),
    31048188: ('replay_maxrl', 46, 116, '1-12:00:00'),
    31037843: ('drgrpo', 47, 116, '1-12:00:00'),
    31037846: ('replay_maxrl', 47, 116, '1-12:00:00'),
}
OLD_FIELDS = ('UserId', 'JobName', 'Account', 'Partition', 'ReqNodeList',
              'ExcNodeList', 'MinMemoryNode', 'TimeLimit', 'Nice', 'QOS',
              'Dependency', 'Features', 'Requeue', 'NumCPUs', 'NumTasks',
              'CPUs/Task', 'TresPerNode', 'WorkDir', 'Command', 'Restarts')
PRESERVE = ('UserId', 'JobName', 'Nice', 'Requeue', 'NumCPUs', 'NumTasks',
            'CPUs/Task', 'ExcNodeList', 'WorkDir', 'Command', 'Features')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Durable receipt and parent-directory commit before any scheduler mutation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        handle.write(b.encoded(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def event(tx, message):
    tx['updated_at_utc'] = b.now()
    tx.setdefault('events', []).append({'at': tx['updated_at_utc'], 'message': message})
    save(TX, tx)


def clean_submit_environment():
    # --export=ALL retains routine cluster settings; remove ambient experiment
    # settings so the exact explicit original exports determine science.
    return {k: v for k, v in os.environ.items()
            if not k.startswith(('OAT_ZERO_', 'SBATCH_', 'SLURM_'))
            and k not in {'SAVE_PATH', 'RUN_STAMP', 'ROOT_DIR', 'PYTHONPATH',
                          'MAXENT_GRPO_ROOT', 'MAXENT_GRPO_VAR_ROOT',
                          'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'}}


def submit(command):
    return subprocess.run(command, text=True, capture_output=True, check=True,
                          env=clean_submit_environment())


def row(data, item):
    rows = [r for r in data['continuations'] if r['run_dir'] == item['run_dir']]
    require(len(rows) == 1, 'missing or duplicate continuation cell')
    require(all(rows[0][k] == item[k] for k in IDENTITY), 'scientific identity changed')
    return rows[0]


def authoritative(item, job):
    primary = json.loads(PRIMARY.read_text())
    original = [r for r in primary['runs'] if r['run_dir'] == item['run_dir']]
    require(len(original) == 1 and all(original[0][k] == item[k] for k in IDENTITY),
            'original scientific cell changed')
    original_id = item['row_before']['original_job_id']
    require(original[0]['job_id'] == original_id, 'original job identity changed')
    mapping = campaign.e119_continuation_jobs(PRIMARY)
    require(len(mapping) == 75 and mapping.get(original_id) == job,
            'authoritative continuation changed or ledger invalid')


def checkpoint(item):
    value = prior.checkpoint(item['run_dir'])
    if value['path'] is not None:
        detail = checkpoints.checkpoint(item)
        require(detail['step'] == value['step'], 'checkpoint/counter validation disagrees')
        require(0 < value['step'] < 3072, 'checkpoint is outside unfinished training')
        # Reject a newer partial save instead of silently resuming an older save.
        require(not any(int(Path(p).name[5:]) >= value['step'] for p in detail['rejected']),
                'newer incomplete checkpoint requires review')
    return value


def safe_run(item, allowed, initial=True):
    require(not recovery.complete(Path(item['run_dir'])), 'cell is already complete')
    writers = recovery.active_writers().get(str(Path(item['run_dir']).resolve()), set())
    require(writers <= set(allowed), f'unexpected same-cell writer: {writers - set(allowed)}')
    if initial:
        require(checkpoint(item) == item['checkpoint'], 'durable checkpoint changed')


def old_guard(item, held):
    record = b.show(item['old_job_id'])
    require(b.field(record, 'JobState') == 'PENDING', 'original allocated; preserve live writer')
    require(b.submit_tokens(record) == item['original_command'], 'original SubmitLine changed')
    for key in OLD_FIELDS:
        require(b.field(record, key) == b.field(item['before'], key), f'original field changed: {key}')
    require(b.field(record, 'NumNodes') in {'1', '1-1'}, 'original node count changed')
    require(not prior.incoming(item['old_job_id']), 'original has dependent jobs')
    if held:
        require(b.field(record, 'Reason') == 'JobHeldUser' and b.field(record, 'Priority') == '0',
                'original is not the exact owned user hold')
    else:
        require(b.field(record, 'Priority') != '0', 'do not claim a preexisting hold')
    return record


def build_command(item):
    original = item['original_command']
    changes = {'--account': 'mltheory', '--partition': 'lowprio', '--nodelist': 'node208',
               '--gres': 'gpu:a6000:1', '--mem': f"{item['memory_gib']}G",
               '--time': item['time_limit'], '--cpus-per-task': '8', '--nodes': '1',
               '--ntasks': '1', '--ntasks-per-node': '1', '--nice': b.field(item['before'], 'Nice'),
               '--exclude': b.PVL, '--comment': item['comment'],
               '--output': str(ROOT / 'var/artifacts/logs/%x-%j.out'),
               '--error': str(ROOT / 'var/artifacts/logs/%x-%j.err')}
    skip = set(changes) | {'--hold', '--dependency', '--begin', '--qos'}
    require(all('=' in t for t in original[1:-1] if t.split('=', 1)[0] in changes),
            'split scheduler options require explicit review')
    command = [x for x in original[:-1] if x.split('=', 1)[0] not in skip]
    command += [f'{k}={v}' for k, v in changes.items()] + ['--hold', original[-1]]
    require(b.exports(command) == b.exports(original), 'replacement changed scientific exports')
    require(command[-1] == original[-1] and '--requeue' in command,
            'frozen launcher or requeue setting changed')
    return command


def new_guard(item, held=False):
    record = b.show(item['new_job_id'])
    expected = {'JobId': str(item['new_job_id']), 'Account': 'mltheory', 'Partition': 'lowprio',
                'QOS': 'none', 'ReqNodeList': 'node208', 'TimeLimit': item['time_limit'],
                'MinMemoryNode': f"{item['memory_gib']}G", 'NumCPUs': '8',
                'TresPerNode': 'gres/gpu:a6000:1', 'Dependency': '(null)', 'Comment': item['comment']}
    if held:
        expected.update(JobState='PENDING', Reason='JobHeldUser', Priority='0')
    for key, value in expected.items():
        require(b.field(record, key) == value, f'replacement profile mismatch: {key}')
    for key in PRESERVE:
        require(b.field(record, key) == b.field(item['before'], key), f'preserved field changed: {key}')
    require(b.field(record, 'NumNodes') in {'1', '1-1'}, 'replacement node count changed')
    require(b.submit_tokens(record) == item['command'], 'replacement full SubmitLine changed')
    require(b.exports(b.submit_tokens(record)) == b.exports(item['original_command']),
            'replacement scientific exports changed')
    tres = dict(x.split('=', 1) for x in b.field(record, 'ReqTRES').split(','))
    require(tres.get('gres/gpu') == '1' and tres.get('gres/gpu:a6000') == '1',
            'replacement GPU count/type changed')
    job = item['new_job_id']; name = b.field(item['before'], 'JobName')
    for key, suffix in [('StdOut', 'out'), ('StdErr', 'err')]:
        require(b.field(record, key) == str(ROOT / f'var/artifacts/logs/{name}-{job}.{suffix}'),
                f'replacement log path mismatch: {key}')
    return record


def data_fingerprints(items):
    directories = {b.exports(i['original_command'])[key] for i in items
                   for key in ('OAT_ZERO_PROMPT_DATA', 'OAT_ZERO_EVAL_DATA')}
    result = {}
    for value in sorted(directories):
        directory = Path(value)
        require(directory.is_dir(), f'missing frozen dataset: {directory}')
        files = sorted(p for p in directory.rglob('*') if p.is_file())
        require(files, f'empty frozen dataset: {directory}')
        result[value] = {str(p.relative_to(directory)): digest(p) for p in files}
    return result


def model_fingerprints(items):
    result = {}
    for value in sorted({b.exports(i['original_command'])['OAT_ZERO_PRETRAIN'] for i in items}):
        directory = Path(value)
        files = sorted(directory.glob('*.safetensors'))
        require(files and (directory / 'config.json').is_file(), 'local model weights/config missing')
        result[value] = {'config_sha256': digest(directory / 'config.json'),
                         'weights': {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                     for p in files}}
    return result


def node_eligible():
    record = b.command(['scontrol', 'show', 'node', '-o', 'node208']).stdout
    require(not any(x in b.field(record, 'State') for x in ('DOWN', 'DRAIN', 'FAIL')),
            'node208 is not currently healthy/schedulable')
    require('gpu:a6000:' in b.field(record, 'Gres') and
            'lowprio' in b.field(record, 'Partitions').split(','), 'node208 hardware/partition changed')
    return record


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'immutable preparation already exists')
    ART.mkdir(parents=True, exist_ok=True)
    data = json.loads(LEDGER.read_text())
    require(len(data['continuations']) == 75, 'unexpected continuation denominator')
    items = []
    for job, (arm, seed, memory, limit) in TARGETS.items():
        matches = [r for r in data['continuations'] if r['continuation_job_id'] == job]
        require(len(matches) == 1, f'target mapping changed: {job}')
        old = matches[0]; record = b.show(job); command = b.submit_tokens(record); env = b.exports(command)
        require((old['domain'], old['arm'], old['seed']) == ('pantry_plan', arm, seed), 'wrong target cell')
        require(env['SAVE_PATH'] == old['run_dir'] and env['RUN_STAMP'] == old['run_stamp'], 'cell exports differ')
        require(env['OAT_ZERO_AUTO_RESUME'] == '1' and env['OAT_ZERO_VLLM_GPU_RATIO'] == '0.25',
                'original resume/Pantry GPU profile changed')
        require(env['OAT_ZERO_SOURCE_ROOT'].endswith('e76_tuned_scale_50d36295558a8958/src'),
                'unexpected scientific source snapshot')
        require(b.field(record, 'TimeLimit') == limit and b.field(record, 'Dependency') == '(null)',
                'current walltime/dependency differs from audited target')
        require(b.field(record, 'ExcNodeList') == b.PVL and b.field(record, 'NumCPUs') == '8'
                and b.field(record, 'Requeue') == '1', 'resource/exclusion profile changed')
        item = {k: old[k] for k in IDENTITY}
        item.update(old_job_id=job, new_job_id=None, row_before=copy.deepcopy(old), before=record,
                    original_command=command, memory_gib=memory, time_limit=limit,
                    comment=f'e119-node208-20260909-old{job}', status='prepared')
        authoritative(item, job); old_guard(item, False)
        item['checkpoint'] = checkpoint(item)
        safe_run(item, [job])
        item['command'] = build_command(item)
        dry = submit([item['command'][0], '--test-only', *[t for t in item['command'][1:] if t != '--hold']])
        item['test_only'] = {'stdout': dry.stdout, 'stderr': dry.stderr, 'returncode': dry.returncode}
        items.append(item)
    files = [Path(__file__), PROTOCOL, Path(prior.__file__), Path(b.__file__),
             Path(checkpoints.__file__), Path(recovery.__file__)]
    plan = {'schema': 'e119-node208-completion-v1', 'created_at_utc': b.now(), 'items': items,
            'authorization': 'User requested faster completion and all broken E119 cells queued.',
            'scientific_exports_unchanged': True, 'same_run_directories': True, 'existing_cells_only': True,
            'primary_sha256': digest(PRIMARY), 'frozen_files': {str(p): digest(p) for p in files},
            'runtime_fingerprints': prior.runtime_fingerprints(items),
            'data_fingerprints': data_fingerprints(items), 'model_fingerprints': model_fingerprints(items),
            'node_before': node_eligible(), 'events': []}
    save(ART / 'original_continuations.json', data)
    save(PLAN, plan)
    print(json.dumps({'prepared': True, 'cells': len(items), 'old_jobs': list(TARGETS), 'scheduler_mutations': False}))


def load_transaction():
    plan = json.loads(PLAN.read_text())
    require(all(digest(p) == h for p, h in plan['frozen_files'].items()), 'frozen controller/protocol drift')
    require(digest(PRIMARY) == plan['primary_sha256'], 'original scientific ledger changed')
    require(prior.runtime_fingerprints(plan['items']) == plan['runtime_fingerprints'], 'frozen runtime drift')
    require(data_fingerprints(plan['items']) == plan['data_fingerprints'], 'frozen dataset drift')
    require(model_fingerprints(plan['items']) == plan['model_fingerprints'], 'model files changed')
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    if TX.exists():
        require(tx.get('plan_sha256') == digest(PLAN), 'transaction/plan mismatch')
    else:
        tx['plan_sha256'] = digest(PLAN)
    return tx


def item_for(tx, old_job):
    matches = [i for i in tx['items'] if i['old_job_id'] == old_job]
    require(len(matches) == 1, 'old job is outside the immutable plan')
    return matches[0]


def promote(tx, item):
    # Caller holds e118_ledger_promotion.lock; reread the full ledger so unrelated
    # cells promoted since preparation are retained byte-for-byte as values.
    data = json.loads(LEDGER.read_text()); current = row(data, item)
    if current['continuation_job_id'] == item['new_job_id']:
        require(current.get('repair_audit') == str(TX) and
                current.get('dormant_fallback_job_id') == item['old_job_id'], 'unowned promotion')
        authoritative(item, item['new_job_id'])
        item['ledger_committed'] = True
        event(tx, f"{item['old_job_id']}: reconciled already committed promotion")
        return
    require(current == item['row_before'], 'target continuation changed before promotion')
    authoritative(item, item['old_job_id'])
    save(ART / f"continuations_before_{item['old_job_id']}.json", data)
    current['previous_continuation_job_ids'] = [*current.get('previous_continuation_job_ids', []), item['old_job_id']]
    current.update(continuation_job_id=item['new_job_id'], held_scheduler_record=item['new_held_record'],
                   repair_audit=str(TX), runtime_allocation_amendment=str(PROTOCOL),
                   dormant_fallback_job_id=item['old_job_id'])
    data.setdefault('repair_history', []).append({'at': b.now(), 'audit': str(TX),
        'old': item['old_job_id'], 'new': item['new_job_id'], 'scientific_exports_unchanged': True})
    require(len(data['continuations']) == 75 and
            len({r['continuation_job_id'] for r in data['continuations']}) == 75,
            'continuation denominator or uniqueness changed')
    image = ART / f"continuations_after_{item['old_job_id']}.json"
    save(image, data)
    item['promotion_requested'] = True; item['promotion_after_sha256'] = digest(image)
    event(tx, f"{item['old_job_id']}: committing exactly one authoritative continuation")
    save(LEDGER, data)
    require(digest(LEDGER) == item['promotion_after_sha256'], 'promotion readback differs')
    authoritative(item, item['new_job_id'])
    item['ledger_committed'] = True
    event(tx, f"{item['old_job_id']}: promotion verified; original held fallback retained")


def stage(old_job):
    tx = load_transaction(); item = item_for(tx, old_job)
    if item.get('released'):
        authoritative(item, item['new_job_id']); old_guard(item, True); new_guard(item)
        print(json.dumps({'status': 'released', 'old_job_id': old_job, 'new_job_id': item['new_job_id']})); return
    if item.get('status') == 'skipped_started':
        print(json.dumps({'status': 'skipped_started', 'old_job_id': old_job})); return
    if not item.get('old_held'):
        if item.get('hold_requested') and b.field(b.show(old_job), 'Reason') == 'JobHeldUser':
            old_guard(item, True)
        else:
            authoritative(item, old_job); old_guard(item, False); safe_run(item, [old_job])
            item['hold_requested'] = True; event(tx, f'{old_job}: holding exact pending predecessor')
            result = b.command(['scontrol', 'hold', str(old_job)], check=False)
            after = b.show(old_job)
            if b.field(after, 'JobState') != 'PENDING':
                # Release only the hold requested by this transaction. Never cancel
                # or requeue the allocation that won the scheduling race.
                b.command(['scontrol', 'release', str(old_job)], check=False)
                item['status'] = 'skipped_started'; event(tx, f'{old_job}: allocation raced hold; live writer preserved')
                return
            require(result.returncode == 0 or b.field(after, 'Reason') == 'JobHeldUser',
                    'original hold was rejected or uncertain')
            old_guard(item, True)
        item['old_held'] = True; event(tx, f'{old_job}: owned dormant fallback hold verified')
    old_guard(item, True)
    require(not item.get('submission_uncertain'), 'uncertain held submission: independently locate and adopt exact successor')
    if item['new_job_id'] is None:
        safe_run(item, [old_job])
        item['submission_uncertain'] = True; event(tx, f'{old_job}: submitting one held continuation')
        result = submit(item['command']).stdout.strip()
        require(re.fullmatch(r'\d+(;[^\s]+)?', result) is not None, 'ambiguous sbatch acknowledgement')
        item['new_job_id'] = int(result.split(';')[0]); item['submission_uncertain'] = False
        event(tx, f"{old_job}: recorded held continuation {item['new_job_id']}")
    item['new_held_record'] = new_guard(item, True)
    safe_run(item, [old_job, item['new_job_id']])
    promote(tx, item)
    item['status'] = 'staged'; event(tx, f'{old_job}: audited staged continuation ready for release')
    print(json.dumps({'status': 'staged', 'old_job_id': old_job, 'new_job_id': item['new_job_id']}))


def adopt(old_job, new_job):
    tx = load_transaction(); item = item_for(tx, old_job)
    require(item.get('submission_uncertain') and item['new_job_id'] is None,
            'adoption only reconciles an ambiguous submission')
    old_guard(item, True); authoritative(item, old_job)
    proposed = copy.deepcopy(item); proposed['new_job_id'] = new_job
    proposed['new_held_record'] = new_guard(proposed, True)
    safe_run(proposed, [old_job, new_job])
    item.update(new_job_id=new_job, new_held_record=proposed['new_held_record'], submission_uncertain=False)
    event(tx, f'{old_job}: independently identified exact held continuation {new_job}; stage remains required')


def release(old_job):
    tx = load_transaction(); item = item_for(tx, old_job)
    require(item.get('ledger_committed'), 'authoritative held promotion must precede release')
    authoritative(item, item['new_job_id']); old_guard(item, True)
    record = new_guard(item)
    if item.get('release_requested') and b.field(record, 'Priority') != '0':
        safe_run(item, [old_job, item['new_job_id']], initial=False)
        item['released'] = True; item['status'] = 'released'
        event(tx, f'{old_job}: reconciled schedulable continuation release')
        print(json.dumps({'status': 'released', 'old_job_id': old_job, 'new_job_id': item['new_job_id']})); return
    new_guard(item, True); safe_run(item, [old_job, item['new_job_id']]); node_eligible()
    item['release_requested'] = True; event(tx, f"{old_job}: releasing audited continuation {item['new_job_id']}")
    result = b.command(['scontrol', 'release', str(item['new_job_id'])], check=False)
    after = new_guard(item)
    require(b.field(after, 'Priority') != '0' and b.field(after, 'JobState') in {'PENDING', 'RUNNING', 'CONFIGURING'},
            'release did not become schedulable: ' + result.stderr)
    item.update(released=True, status='released', release_record=after)
    event(tx, f'{old_job}: continuation released; scientific recipe and fallback unchanged')
    print(json.dumps({'status': 'released', 'old_job_id': old_job, 'new_job_id': item['new_job_id'],
                      'state': b.field(after, 'JobState'), 'reason': b.field(after, 'Reason')}))


def status():
    tx = load_transaction()
    print(json.dumps({'items': [{k: i.get(k) for k in ('old_job_id', 'new_job_id', 'arm', 'seed',
        'status', 'ledger_committed', 'submission_uncertain')} for i in tx['items']]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'stage', 'release', 'adopt', 'status'))
    parser.add_argument('--old-job-id', type=int); parser.add_argument('--new-job-id', type=int)
    args = parser.parse_args()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action in {'stage', 'release', 'adopt'}:
            require(args.old_job_id is not None, '--old-job-id is required')
            if args.action == 'adopt':
                require(args.new_job_id is not None, '--new-job-id is required')
                adopt(args.old_job_id, args.new_job_id)
            else:
                globals()[args.action](args.old_job_id)
        else:
            globals()[args.action]()
