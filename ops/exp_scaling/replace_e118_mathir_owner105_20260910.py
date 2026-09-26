#!/usr/bin/env python3
"""Protect the still-pending E118 MathIR Re:Max continuation on node105's owning partition."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path
import re
import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery
import replace_campaign_cs_priority_20260908 as prior

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/e118_mathir_owner105_20260910'
PLAN, TX = ART / 'plan.json', ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e118_mathir_owner105_20260910.md'
SOURCE, AGGREGATE = base.LEDGER, campaign.E118_LEDGER
PARENTS = (31158503,)
DEPENDENCY = 'afterany:' + ':'.join(map(str, PARENTS))
TARGETS = {31158504: ('replay_maxrl', 72)}
PRESERVE = ('UserId', 'JobName', 'Account', 'QOS', 'NumCPUs', 'NumTasks', 'CPUs/Task',
            'MinMemoryNode', 'TresPerNode', 'TimeLimit', 'Nice', 'Requeue', 'ExcNodeList',
            'WorkDir', 'Command', 'Features')
sha, save = prior.digest, base.atomic


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def safe(item, allowed):
    prior.safe_run(item, allowed)


def old_guard(item, held):
    r = base.show(item['old_job_id'])
    require(base.field(r, 'JobState') == 'PENDING', 'Predecessor is no longer pending; preserve it')
    require(base.field(r, 'Partition') == 'lowprio' and base.field(r, 'Dependency') == '(null)', 'Original route differs')
    require(base.submit_tokens(r) == item['original_command'], 'Original frozen SubmitLine changed')
    for k in PRESERVE + ('ReqNodeList', 'Restarts'):
        require(base.field(r, k) == base.field(item['before'], k), 'Original resource changed: ' + k)
    if held:
        require(base.field(r, 'Reason') == 'JobHeldUser' and base.field(r, 'Priority') == '0', 'Owned hold is absent')
    else:
        require(int(base.field(r, 'Priority')) > 0 and 'held' not in base.field(r, 'Reason').lower()
                and 'hold' not in base.field(r, 'Reason').lower(), 'Preserve pre-existing hold')
    require(not prior.incoming(item['old_job_id']), 'Predecessor has incoming dependencies')
    return r


def check_dependency(r):
    dep = base.field(r, 'Dependency')
    require(dep == '(null)' or all(x.startswith('afterany:') for x in dep.split(',')), 'Unexpected dependency type')
    remaining = set(map(int, re.findall(r'(?:afterany:|:)(\d+)', dep)))
    require(remaining <= set(PARENTS), 'Unexpected dependency parent')
    active = set(base.queue())
    require(not ((set(PARENTS) - remaining) & active), 'Live parent disappeared from dependency')


def new_guard(item, held):
    r = base.show(item['new_job_id'])
    expected = {'Account': 'mltheory', 'Partition': 'mltheory', 'QOS': 'none',
                'ReqNodeList': 'node105', 'Comment': item['comment'], 'Restarts': '0'}
    if held:
        expected.update(JobState='PENDING', Reason='JobHeldUser', Priority='0')
    require(all(base.field(r, k) == v for k, v in expected.items()), 'Protected replacement resources/hold differ')
    for k in PRESERVE:
        require(base.field(r, k) == base.field(item['before'], k), 'Replacement changed: ' + k)
    require(base.field(r, 'NumNodes') in ('1', '1-1'), 'Replacement is not one node')
    tres = dict(x.split('=', 1) for x in base.field(r, 'ReqTRES').split(','))
    require(tres['gres/gpu'] == '1' and tres['cpu'] == '16' and tres['mem'] == '116G', 'Replacement TRES differ')
    tokens = base.submit_tokens(r)
    require(base.exports(tokens) == base.exports(item['original_command']), 'Frozen runtime exports changed')
    require(tokens[-1] == item['original_command'][-1] and sha(tokens[-1]) == item['launcher_sha256'], 'Frozen launcher changed')
    check_dependency(r)
    return r


def replacement_command(item):
    changes = {'--account': 'mltheory', '--partition': 'mltheory', '--nodelist': 'node105',
               '--gres': 'gpu:a5000:1', '--mem': '116G', '--cpus-per-task': '16',
               '--time': '3-00:00:00', '--nice': '200', '--nodes': '1', '--ntasks': '1',
               '--ntasks-per-node': '1', '--exclude': base.PVL,
               '--dependency': DEPENDENCY, '--comment': item['comment']}
    removed = set(changes) | {'--hold', '--begin'}
    cmd = [x for x in item['original_command'][:-1] if x.split('=', 1)[0] not in removed]
    cmd += [f'{k}={v}' for k, v in changes.items()] + ['--hold', item['original_command'][-1]]
    require(base.exports(cmd) == base.exports(item['original_command']), 'Export change in prepared replacement')
    return cmd


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Inspect existing transaction instead of overwriting')
    source, aggregate = (json.loads(p.read_text()) for p in (SOURCE, AGGREGATE))
    require(len(source['runs']) == 50 and len(aggregate['runs']) == 150, 'Campaign cardinality differs')
    items = []
    for old, identity in TARGETS.items():
        row = next(r for r in source['runs'] if r['job_id'] == old)
        require((row['arm'], row['seed']) == identity and row['domain'] == 'mathir', 'Wrong science cell')
        require(next(r for r in aggregate['runs'] if r['job_id'] == old) == dict(row, scale='qwen3b'), 'Source/aggregate mismatch')
        before = base.show(old); original = base.submit_tokens(before); env = base.exports(original)
        require(env['SAVE_PATH'] == row['run_dir'] and env['RUN_STAMP'] == row['run_stamp'], 'Wrong run identity')
        require(env['OAT_ZERO_AUTO_RESUME'] == '1' and env['OAT_ZERO_VLLM_GPU_RATIO'] == '0.40'
                and env['OAT_ZERO_ADAM_OFFLOAD'] == env['OAT_ZERO_ACTIVATION_OFFLOADING'] == '1', 'Runtime differs')
        expected = {'Account': 'mltheory', 'QOS': 'none', 'MinMemoryNode': '116G', 'NumCPUs': '16',
                    'Nice': '200', 'TimeLimit': '3-00:00:00', 'TresPerNode': 'gres/gpu:a5000:1',
                    'ExcNodeList': base.PVL, 'Requeue': '1'}
        require(all(base.field(before, k) == v for k, v in expected.items()), 'Original resources differ')
        cp = prior.checkpoint(row['run_dir'])
        require(cp['step'] == {31158503: 2688, 31158504: 2304}[old], 'Latest durable checkpoint changed')
        item = dict(old_job_id=old, new_job_id=None, row_before=row, run_dir=row['run_dir'], run_stamp=row['run_stamp'],
                    before=before, original_command=original, launcher_sha256=sha(original[-1]), checkpoint=cp,
                    comment=f'e118-mathir-owner105-20260910-old{old}')
        old_guard(item, False); safe(item, [old]); item['command'] = replacement_command(item)
        result = base.command([item['command'][0], '--test-only', *[x for x in item['command'][1:] if x != '--hold']])
        item['test_only'] = {'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode}
        items.append(item)
    plan = dict(schema='e118-mathir-protected-owner105-v1', status='prepared', created_at=base.now(), items=items,
                dependency=DEPENDENCY, parents_before={str(j): base.show(j) for j in PARENTS},
                pins={str(p): sha(p) for p in (Path(__file__), PROTOCOL, Path(base.__file__), Path(prior.__file__), Path(recovery.__file__), Path(campaign.__file__))},
                runtime_fingerprints=prior.runtime_fingerprints(items),
                mutable_before_sha256={str(p): sha(p) for p in (SOURCE, AGGREGATE)}, events=[])
    for p in (SOURCE, AGGREGATE):
        (ART / (p.name + '.before')).write_bytes(p.read_bytes())
    save(PLAN, plan)
    print(json.dumps({'prepared': str(PLAN), 'items': [{'old': i['old_job_id'], 'checkpoint': i['checkpoint']['step'], 'test_only': i['test_only']} for i in items]}, indent=2))


def stage(tx, event):
    require(all(sha(p) == v for p, v in tx['mutable_before_sha256'].items()), 'Ledger changed before staging')
    source, aggregate = (json.loads(p.read_text()) for p in (SOURCE, AGGREGATE))
    for i in tx['items']:
        row = next(r for r in source['runs'] if r['job_id'] == i['old_job_id'])
        require(row == i['row_before'], 'Predecessor ledger row changed')
        row.setdefault('previous_job_ids', []).append(i['old_job_id'])
        row.update(job_id=i['new_job_id'], held_scheduler_record=i['new_held_record'],
                   repair_audit=str(TX), runtime_allocation_amendment=str(PROTOCOL),
                   scheduler_dependency=DEPENDENCY, owner105_resume_checkpoint=i['checkpoint'],
                   protected_owner105=True)
        target = next(r for r in aggregate['runs'] if r['run_dir'] == i['run_dir'])
        target.clear(); target.update(row, scale='qwen3b')
    source.setdefault('repair_history', []).append(dict(at=base.now(), audit=str(TX), protocol=str(PROTOCOL),
        scheduler_only=True, replacements=[{'old': i['old_job_id'], 'new': i['new_job_id']} for i in tx['items']]))
    tx['staged'] = {}
    for path, value, count in ((SOURCE, source, 50), (AGGREGATE, aggregate, 150)):
        require(len(value['runs']) == len({r['job_id'] for r in value['runs']}) == len({r['run_dir'] for r in value['runs']}) == count, 'Ledger identity count differs')
        before = json.loads(path.read_text())
        keys = ('domain', 'arm', 'seed', 'run_dir', 'run_stamp')
        require(sorted(tuple(r[k] for k in keys) for r in before['runs']) == sorted(tuple(r[k] for k in keys) for r in value['runs']), 'Science identity changed')
        staged = ART / (path.name + '.after'); save(staged, value)
        tx['staged'][str(path)] = {'path': str(staged), 'sha256': sha(staged)}
    event('Both authoritative after-images staged before promotion')


def apply():
    tx = json.loads((TX if TX.exists() else PLAN).read_text())
    require(all(sha(p) == v for p, v in tx['pins'].items()), 'Prepared helper/protocol changed')
    if tx['status'] == 'complete':
        print(json.dumps({'already_complete': True})); return
    require(prior.runtime_fingerprints(tx['items']) == tx['runtime_fingerprints'], 'Frozen runtime changed')
    def event(message):
        tx['events'].append({'at': base.now(), 'message': message}); save(TX, tx)
    tx['status'] = 'applying'; event('Beginning or reconciling exact pending-cell protected owner transaction')
    try:
        if not tx.get('staged'):
            require(all(sha(p) == v for p, v in tx['mutable_before_sha256'].items()), 'Unrelated ledger change; reprepare safely')
            for i in tx['items']:
                if not i.get('old_held'):
                    if i.get('hold_requested') and base.field(base.show(i['old_job_id']), 'Reason') == 'JobHeldUser':
                        old_guard(i, True)
                    else:
                        old_guard(i, False); safe(i, [i['old_job_id']]); i['hold_requested'] = True
                        event(f"Holding pending predecessor {i['old_job_id']}")
                        base.command(['scontrol', 'hold', str(i['old_job_id'])])
                        if base.field(base.show(i['old_job_id']), 'JobState') != 'PENDING':
                            base.command(['scontrol', 'release', str(i['old_job_id'])])
                            event('Start race: restored own hold without changing running allocation')
                            raise RuntimeError('Predecessor started during hold; preserve active job')
                        old_guard(i, True)
                    i['old_held'] = True; event('Verified owned pending hold')
                old_guard(i, True); safe(i, [i['old_job_id']] + ([i['new_job_id']] if i['new_job_id'] else []))
                if i['new_job_id'] is None:
                    if i.get('submission_uncertain'):
                        found = [j for j in base.queue() if base.field(base.show(j), 'Comment') == i['comment']]
                        require(len(found) == 1, 'Ambiguous submission requires exact-comment reconciliation')
                        i['new_job_id'] = found[0]; i['submission_uncertain'] = False
                    else:
                        i['submission_uncertain'] = True; event('Persisted held replacement submission intent')
                        result = base.command(i['command'], check=False)
                        i['submission_result'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
                        event('Recorded raw submission response before parsing')
                        require(result.returncode == 0, 'Submission failed; do not retry blindly')
                        i['new_job_id'] = int(result.stdout.strip().split(';', 1)[0]); i['submission_uncertain'] = False
                    event(f"Recorded replacement {i['new_job_id']}")
                i['new_held_record'] = new_guard(i, True); safe(i, [i['old_job_id'], i['new_job_id']]); event('Verified full frozen runtime and protected resources')
            require(prior.runtime_fingerprints(tx['items']) == tx['runtime_fingerprints'], 'Frozen runtime changed before promotion')
            stage(tx, event)
        for p, staged in tx['staged'].items():
            require(sha(staged['path']) == staged['sha256'], 'Staged after-image changed')
            if sha(p) != staged['sha256']:
                require(sha(p) == tx['mutable_before_sha256'][p], 'Concurrent ledger mutation during promotion')
                event('Promoting staged ledger ' + p); save(Path(p), json.loads(Path(staged['path']).read_text()))
        for p in (SOURCE, AGGREGATE):
            rows = json.loads(p.read_text())['runs']
            require(all(next(r for r in rows if r['run_dir'] == i['run_dir'])['job_id'] == i['new_job_id'] for i in tx['items']), 'Effective mappings disagree')
        tx['ledgers_committed'] = True; event('Both effective mappings point to held successors')
        for i in tx['items']:
            if i.get('old_cancelled'): continue
            if i['old_job_id'] in base.queue():
                old_guard(i, True); new_guard(i, True); safe(i, [i['old_job_id'], i['new_job_id']])
                i['cancel_requested'] = True; event(f"Retiring superseded held predecessor {i['old_job_id']}")
                base.command(['scancel', str(i['old_job_id'])])
                require(i['old_job_id'] not in base.queue(), 'Old writer still queued')
            else:
                require(i.get('cancel_requested'), 'Unexpected predecessor disappearance')
            i['old_cancelled'] = True; event('Verified predecessor absent')
        require(all(i['old_job_id'] not in base.queue() for i in tx['items']), 'Predecessor remains active')
        for i in tx['items']:
            if i.get('released'): continue
            if not (i.get('release_requested') and base.field(base.show(i['new_job_id']), 'Reason') != 'JobHeldUser'):
                new_guard(i, True); safe(i, [i['new_job_id']]); i['release_requested'] = True
                event(f"Releasing protected successor {i['new_job_id']} with active-job afterany dependency")
                base.command(['scontrol', 'release', str(i['new_job_id'])])
            r = new_guard(i, False)
            require(base.field(r, 'JobState') in ('PENDING', 'RUNNING', 'CONFIGURING') and int(base.field(r, 'Priority')) > 0, 'Replacement release failed')
            i['released'] = True; i['after_release'] = r; event('Verified replacement release and unchanged checkpoint source')
        tx['status'] = 'complete'; tx['completed_at'] = base.now(); event('Protected owner continuation migration complete')
    except BaseException as exc:
        tx['status'] = 'stopped_for_reconciliation'; tx['error'] = repr(exc); event('Stopped without blind retry; exact IDs and holds retained'); raise
    print(json.dumps({'status': tx['status'], 'replacements': [{'old': i['old_job_id'], 'new': i['new_job_id'], 'checkpoint': i['checkpoint']['step'], 'dependency': base.field(i['after_release'], 'Dependency')} for i in tx['items']]}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('prepare', 'apply')); args = p.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); globals()[args.phase]()
