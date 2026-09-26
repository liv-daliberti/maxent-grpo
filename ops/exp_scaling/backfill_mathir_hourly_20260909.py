#!/usr/bin/env python3
"""Prepare/stage/release two guarded one-hour MathIR checkpoint continuations."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path
import accelerate_a5000_completion_20260909 as prior

b = prior.base
ROOT = prior.ROOT
ART = ROOT / 'var/artifacts/campaign_mathir_hourly_backfill_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
GUARD_READY = ART / 'guard_ready.json'
PROTOCOL = ROOT / 'paper/preregistration/campaign_mathir_hourly_backfill_20260909.md'
SOURCE = prior.SOURCE
AGGREGATE = prior.AGGREGATE
MUTABLE = (SOURCE, AGGREGATE)
TARGETS = {31158503: ('maxrl', 72), 31158504: ('replay_maxrl', 72)}
SHORT_NODES = {'node105', 'node202', 'node204'}
LONG_NODES = {'node105', 'node202', 'node203', 'node204'}
PRESERVE = ('Account', 'NumCPUs', 'NumTasks', 'CPUs/Task', 'MinMemoryNode',
            'Nice', 'Requeue', 'ExcNodeList', 'WorkDir', 'TresPerNode')
RUNTIME_CHANGE = {'OAT_ZERO_RESUME_STEPS': {'before': '192', 'after': '48'}}


def save(path, value):
    b.atomic(Path(path), value)


def nodes(record):
    return set(b.command(['scontrol', 'show', 'hostnames', b.field(record, 'ReqNodeList')]).stdout.split())


def desired_exports(original):
    env = b.exports(original)
    assert env['OAT_ZERO_RESUME_STEPS'] == '192'
    assert env['OAT_ZERO_VLLM_GPU_RATIO'] == '0.40'
    assert env['OAT_ZERO_AUTO_RESUME'] == '1'
    env['OAT_ZERO_RESUME_STEPS'] = '48'
    return env


def build_command(item, *, short):
    original = item['original_command']
    changes = {'--account': 'mltheory', '--partition': 'all' if short else 'lowprio',
               '--nodelist': ','.join(sorted(SHORT_NODES if short else LONG_NODES)),
               '--time': '1:00:00' if short else '3-00:00:00', '--gres': 'gpu:a5000:1',
               '--cpus-per-task': '16', '--mem': '116G', '--nodes': '1', '--ntasks': '1',
               '--ntasks-per-node': '1', '--nice': b.field(item['before'], 'Nice'),
               '--exclude': b.PVL,
               '--comment': item['comment'] if short else f"mathir-long-fallback-from-{item['old_job_id']}"}
    env = desired_exports(original) if short else b.exports(original)
    changes['--export'] = 'ALL,' + ','.join(f'{k}={v}' for k,v in env.items())
    removed = set(changes) | {'--hold', '--dependency', '--begin'}
    command = [x for x in original[:-1] if x.split('=', 1)[0] not in removed]
    command += [f'{k}={v}' for k,v in changes.items()] + ['--hold', original[-1]]
    assert b.exports(command) == env and command[-1] == original[-1]
    return command


def current_identity(item, expected):
    for path in MUTABLE:
        row = next(r for r in json.loads(path.read_text())['runs'] if r['run_dir'] == item['run_dir'])
        assert row['job_id'] == expected, ('effective mapping changed', path, item['run_dir'])
        assert all(row[k] == item[k] for k in ('domain', 'arm', 'seed', 'run_dir', 'run_stamp'))


def safe_run(item, allowed):
    prior.safe_run(item, allowed)


def old_guard(item, *, held):
    record = b.show(item['old_job_id'])
    assert b.field(record, 'JobState') == 'PENDING', 'predecessor started; preserve it'
    assert b.submit_tokens(record) == item['original_command']
    assert b.field(record, 'Partition') == 'lowprio'
    assert b.field(record, 'TimeLimit') == '3-00:00:00'
    assert b.field(record, 'QOS') == 'none'
    assert nodes(record) == LONG_NODES
    assert b.field(record, 'Dependency') == '(null)'
    for key in PRESERVE:
        assert b.field(record, key) == b.field(item['before'], key), (item['old_job_id'], key)
    if held:
        assert b.field(record, 'Reason') == 'JobHeldUser' and b.field(record, 'Priority') == '0'
    else:
        assert b.field(record, 'Priority') != '0', 'preexisting hold must remain untouched'
    assert prior.digest(item['original_command'][-1]) == item['launcher_sha256']
    assert not prior.incoming(item['old_job_id']), 'predecessor acquired a scheduler dependent'
    return record


def new_guard(item, *, held):
    record = b.show(item['new_job_id'])
    expected = {'Account': 'mltheory', 'Partition': 'all', 'QOS': 'none',
                'TimeLimit': '01:00:00', 'TresPerNode': 'gres/gpu:a5000:1',
                'Dependency': '(null)', 'Comment': item['comment']}
    if held:
        expected.update(JobState='PENDING', Reason='JobHeldUser', Priority='0')
    for key,value in expected.items():
        assert b.field(record,key) == value, (item['new_job_id'], key, b.field(record,key),value)
    assert nodes(record) == SHORT_NODES
    for key in PRESERVE:
        assert b.field(record,key) == b.field(item['before'],key), (item['new_job_id'], key)
    assert b.field(record,'NumNodes') in ('1','1-1')
    tres = dict(x.split('=',1) for x in b.field(record,'ReqTRES').split(','))
    assert tres['gres/gpu'] == '1' and tres['gres/gpu:a5000'] == '1'
    observed = b.submit_tokens(record)
    assert b.exports(observed) == desired_exports(item['original_command'])
    assert observed[-1] == item['original_command'][-1]
    assert prior.digest(observed[-1]) == item['launcher_sha256']
    return record


def prepare():
    assert not PLAN.exists() and not TX.exists(), 'inspect existing plan instead of replacing it'
    baseline = json.loads(prior.TX.read_text())
    assert baseline['status'] == 'complete'
    items = []
    for old,(arm,seed) in TARGETS.items():
        row = next(r for r in json.loads(SOURCE.read_text())['runs'] if r['job_id'] == old)
        assert row['domain'] == 'mathir' and row['arm'] == arm and row['seed'] == seed
        record = b.show(old); command = b.submit_tokens(record); env = b.exports(command)
        assert env['SAVE_PATH'] == row['run_dir'] and env['RUN_STAMP'] == row['run_stamp']
        for key,value in {'Account':'mltheory','MinMemoryNode':'116G','NumCPUs':'16','Requeue':'1','ExcNodeList':b.PVL}.items():
            assert b.field(record,key) == value
        desired_exports(command)
        item = {k: row[k] for k in ('domain','arm','seed','run_dir','run_stamp')}
        item.update(old_job_id=old, new_job_id=None, row_before=copy.deepcopy(row), before=record,
                    original_command=command, launcher_sha256=prior.digest(command[-1]),
                    checkpoint=prior.checkpoint(row['run_dir']),
                    comment=f'mathir-hourly-backfill-20260909-old{old}')
        current_identity(item,old); old_guard(item,held=False); safe_run(item,[old])
        item['command'] = build_command(item,short=True)
        item['long_route_command'] = build_command(item,short=False)
        dry = b.command([item['command'][0],'--test-only',*[x for x in item['command'][1:] if x!='--hold']])
        item['test_only'] = {'returncode':dry.returncode,'stdout':dry.stdout,'stderr':dry.stderr}
        items.append(item)
    plan = {'schema':'campaign-mathir-hourly-backfill-v1','status':'prepared','created_at':b.now(),
            'items':items,'runtime_changes':RUNTIME_CHANGE,'scientific_settings_changed':False,
            'fallback_resume_interval':192,'fallback_automatic':True,'fallback_uses_dormant_originals':True,
            'controller_sha256':prior.digest(__file__),'protocol_sha256':prior.digest(PROTOCOL),
            'helper_sha256':prior.digest(prior.__file__),'base_helper_sha256':prior.digest(b.__file__),
            'original_five_transaction_sha256':prior.digest(prior.TX),
            'original_five_protocol_sha256':prior.digest(prior.PROTOCOL),
            'runtime_fingerprints':prior.runtime_fingerprints(items),
            'mutable_before_sha256':{str(p):prior.digest(p) for p in MUTABLE},
            'guard_required_before_release':True,'events':[]}
    for p in MUTABLE: (ART/(p.name+'.before')).write_bytes(p.read_bytes())
    save(PLAN,plan)
    print(json.dumps({'prepared':True,'cells':[{'old':i['old_job_id'],'checkpoint':i['checkpoint']['step'],'test':i['test_only']} for i in items]}))


def load_transaction():
    plan = json.loads(PLAN.read_text())
    for path,key in ((__file__,'controller_sha256'),(PROTOCOL,'protocol_sha256'),
                     (prior.__file__,'helper_sha256'),(b.__file__,'base_helper_sha256'),
                     (prior.TX,'original_five_transaction_sha256'),(prior.PROTOCOL,'original_five_protocol_sha256')):
        assert prior.digest(path) == plan[key], ('controller or provenance drift',path)
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    assert prior.runtime_fingerprints(tx['items']) == tx['runtime_fingerprints']
    return tx


def event(tx,message):
    tx['events'].append({'at':b.now(),'message':message}); save(TX,tx)


def stage_ledgers(tx):
    if not tx.get('staged'):
        assert all(prior.digest(p)==h for p,h in tx['mutable_before_sha256'].items()), 'ledger changed before staging'
        source=json.loads(SOURCE.read_text());aggregate=json.loads(AGGREGATE.read_text())
        for item in tx['items']:
            row=next(r for r in source['runs'] if r['job_id']==item['old_job_id'])
            assert row==item['row_before']
            row['previous_job_ids']=[*row.get('previous_job_ids',[]),item['old_job_id']]
            row.update(job_id=item['new_job_id'], held_scheduler_record=item['new_held_record'],
                       repair_audit=str(TX), runtime_allocation_amendment=str(PROTOCOL),
                       hourly_backfill_runtime_changes=RUNTIME_CHANGE,
                       hourly_backfill_long_route_command=item['long_route_command'])
            target=next(r for r in aggregate['runs'] if r['run_dir']==item['run_dir'])
            scale=target['scale'];target.clear();target.update(row,scale=scale)
        source.setdefault('repair_history',[]).append({'at':b.now(),'protocol':str(PROTOCOL),'audit':str(TX),
            'runtime_changes':RUNTIME_CHANGE,'replacements':[{'old':i['old_job_id'],'new':i['new_job_id']} for i in tx['items']]})
        assert len(source['runs'])==50 and len(aggregate['runs'])==150
        assert len({r['job_id'] for r in source['runs']})==50 and len({r['job_id'] for r in aggregate['runs']})==150
        staged={}
        for path,value in ((SOURCE,source),(AGGREGATE,aggregate)):
            before=json.loads(path.read_text())
            keys=('domain','arm','seed','run_dir','run_stamp')
            assert sorted(tuple(r[k] for k in keys) for r in before['runs'])==sorted(tuple(r[k] for k in keys) for r in value['runs'])
            p=ART/(path.name+'.after');save(p,value);staged[str(path)]={'path':str(p),'sha256':prior.digest(p)}
        tx['staged']=staged;event(tx,'Staged both ledger after-images before promotion')
    for path,image in tx['staged'].items():
        if prior.digest(path)==image['sha256']:continue
        assert prior.digest(path)==tx['mutable_before_sha256'][path], ('concurrent ledger mutation',path)
        assert prior.digest(image['path'])==image['sha256']
        event(tx,f'Promoting staged ledger {path}');save(path,json.loads(Path(image['path']).read_text()))
        assert prior.digest(path)==image['sha256']
    for item in tx['items']:current_identity(item,item['new_job_id'])
    tx['ledgers_committed']=True;event(tx,'Both effective mappings point to held hourly replacements')


def stage():
    tx=load_transaction()
    if tx['status'] in ('staged_awaiting_guard','complete'):
        print(json.dumps({'status':tx['status'],'new_jobs':[i['new_job_id'] for i in tx['items']]}));return
    tx['status']='staging';event(tx,'Staging held hourly replacements; no release before guard readiness')
    if not tx.get('staged'):
        assert all(prior.digest(p)==h for p,h in tx['mutable_before_sha256'].items())
        for item in tx['items']:
            current_identity(item,item['old_job_id'])
            if not item.get('old_held'):
                record=b.show(item['old_job_id'])
                if item.get('hold_requested') and b.field(record,'Reason')=='JobHeldUser':
                    old_guard(item,held=True)
                else:
                    old_guard(item,held=False);safe_run(item,[item['old_job_id']])
                    item['hold_requested']=True;event(tx,f"Holding predecessor {item['old_job_id']}")
                    b.command(['scontrol','hold',str(item['old_job_id'])])
                    record=b.show(item['old_job_id'])
                    if b.field(record,'JobState')!='PENDING':
                        b.command(['scontrol','release',str(item['old_job_id'])])
                        event(tx,'Predecessor started during hold; released unchanged')
                        raise RuntimeError('allocation raced hold; reconcile while preserving running job')
                    old_guard(item,held=True)
                item['old_held']=True;event(tx,f"Predecessor {item['old_job_id']} held")
            old_guard(item,held=True)
            if item['new_job_id'] is None:
                if item.get('submission_uncertain'):
                    found=[j for j in b.queue() if b.field(b.show(j),'Comment')==item['comment']]
                    assert len(found)==1, ('reconcile uncertain held submission',found)
                    item['new_job_id']=found[0];item['submission_uncertain']=False
                else:
                    item['submission_uncertain']=True;event(tx,f"Submitting held hourly job for {item['old_job_id']}")
                    result=b.command(item['command'],check=False)
                    item['submission_result']={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
                    if result.returncode: event(tx,'Submission requires reconciliation; no blind retry');raise RuntimeError(result.stderr)
                    item['new_job_id']=int(result.stdout.strip().split(';',1)[0]);item['submission_uncertain']=False
                event(tx,f"Recorded held replacement {item['new_job_id']}")
            item['new_held_record']=new_guard(item,held=True)
            safe_run(item,[item['old_job_id'],item['new_job_id']])
            event(tx,f"Verified full exports, resources and checkpoint for {item['new_job_id']}")
    stage_ledgers(tx)
    tx['status']='staged_awaiting_guard';event(tx,'Held replacements staged; install the bounded timeout guard before release')
    print(json.dumps({'status':tx['status'],'cells':[{'old':i['old_job_id'],'new':i['new_job_id'],'checkpoint':i['checkpoint']['step']} for i in tx['items']]}))


def guard_ready(tx):
    ready=json.loads(GUARD_READY.read_text())
    ids=sorted(i['new_job_id'] for i in tx['items'])
    assert sorted(ready['active_job_ids'])==ids
    assert {str(k):int(v) for k,v in ready['initial_checkpoints'].items()}=={str(i['new_job_id']):i['checkpoint']['step'] for i in tx['items']}
    assert isinstance(ready['max_requeues'],int) and ready['max_requeues']>0
    assert isinstance(ready['deadline'],str) and ready['deadline']
    assert prior.digest(ready['guard_script'])==ready['guard_script_sha256']
    record=b.show(int(ready['guard_job_id']))
    assert b.field(record,'JobState')=='RUNNING', 'bounded timeout guard must be running before release'
    return dict(ready,scheduler_record=record)


def release():
    tx=load_transaction()
    if tx['status']=='complete':print(json.dumps({'already_complete':True}));return
    assert tx.get('ledgers_committed')
    tx['guard_at_release']=guard_ready(tx);tx['status']='releasing';event(tx,'Bounded advancing-checkpoint timeout guard verified running')
    for item in tx['items']:
        current_identity(item,item['new_job_id'])
        old_guard(item,held=True)
        safe_run(item,[item['old_job_id'],item['new_job_id']])
        item['old_fallback_held']=True
        event(tx,f"Preserving exact held predecessor {item['old_job_id']} as dormant fallback")
    for item in tx['items']:
        if item.get('released'):continue
        record=b.show(item['new_job_id'])
        if item.get('release_requested') and b.field(record,'Reason')!='JobHeldUser':
            new_guard(item,held=False)
        else:
            new_guard(item,held=True);old_guard(item,held=True);safe_run(item,[item['old_job_id'],item['new_job_id']])
            item['release_requested']=True;event(tx,f"Releasing guarded hourly job {item['new_job_id']}")
            b.command(['scontrol','release',str(item['new_job_id'])])
        after=new_guard(item,held=False)
        assert b.field(after,'Priority')!='0' and b.field(after,'JobState') in ('PENDING','RUNNING','CONFIGURING','COMPLETING','COMPLETED')
        item['released']=True;item['after_release']=after;event(tx,f"Hourly job {item['new_job_id']} released")
    assert prior.runtime_fingerprints(tx['items'])==tx['runtime_fingerprints']
    tx['status']='complete';tx['completed_at']=b.now();event(tx,'Both hourly allocations released; original long jobs remain owned held fallbacks; original five-cell transaction unchanged')
    print(json.dumps({'status':tx['status'],'new_jobs':[i['new_job_id'] for i in tx['items']]}))


def fallback(new_job_id, reason):
    """Caller must own e118_ledger_promotion.lock; CLI acquires it externally."""
    assert reason in {'no_progress','requeue_cap','deadline'}
    tx=load_transaction()
    item=next(i for i in tx['items'] if i['new_job_id']==int(new_job_id))
    f=item.setdefault('fallback',{'reason':reason,'status':'preparing'})
    if f['status']=='complete':
        current_identity(item,item['old_job_id'])
        print(json.dumps({'already_restored':True,'job':item['old_job_id']}));return
    assert item.get('old_fallback_held') and item.get('released')
    assert f['reason']==reason
    assert not prior.recovery.complete(Path(item['run_dir'])), 'scientific cell already complete; retire its fallback instead'
    queue=b.queue()
    if item['new_job_id'] in queue:
        assert reason=='deadline', 'only expired pending hourly jobs may be retired without TIMEOUT'
        active=new_guard(item,held=False)
        assert b.field(active,'JobState')=='PENDING', 'do not interrupt a running hourly allocation'
        old_guard(item,held=True)
        writers=prior.recovery.active_writers().get(str(Path(item['run_dir']).resolve()),set())
        assert writers<={item['old_job_id'],item['new_job_id']}
        if not (f.get('deadline_hold_requested') and b.field(active,'Reason')=='JobHeldUser'):
            assert b.field(active,'Priority')!='0', 'preserve an unrelated preexisting hourly hold'
            f['deadline_hold_requested']=True;event(tx,'Holding expired pending hourly request for fallback')
            b.command(['scontrol','hold',str(item['new_job_id'])])
        active=b.show(item['new_job_id'])
        if b.field(active,'JobState')!='PENDING':
            b.command(['scontrol','release',str(item['new_job_id'])])
            f['deadline_hold_requested']=False;event(tx,'Hourly allocation raced deadline hold; released unchanged')
            raise RuntimeError('hourly allocation started; defer fallback until inactive')
        new_guard(item,held=True)
        f['hourly_cancel_requested']=True;event(tx,'Retiring only the expired owned-held hourly request')
        b.command(['scancel',str(item['new_job_id'])])
        assert item['new_job_id'] not in b.queue()
        f['hourly_cancelled_for_deadline']=True;event(tx,'Expired hourly request inactive; long fallback may be restored')
    assert item['new_job_id'] not in b.queue(), 'hourly allocation must be inactive before fallback'
    state=prior.recovery.state(item['new_job_id'])
    assert state=='TIMEOUT' or (reason=='deadline' and f.get('hourly_cancel_requested') and state=='CANCELLED'), ('unexpected terminal hourly state',state)
    # A release acknowledgement may precede persistence; reconcile that exact state.
    if f.get('release_requested') and f.get('ledgers_committed'):
        current_identity(item,item['old_job_id'])
        old=b.show(item['old_job_id'])
        if b.field(old,'Reason')!='JobHeldUser':
            assert b.field(old,'JobState') in {'PENDING','RUNNING','CONFIGURING','COMPLETING','COMPLETED'}
            assert b.submit_tokens(old)==item['original_command'] and nodes(old)==LONG_NODES
            for key in PRESERVE:assert b.field(old,key)==b.field(item['before'],key)
            assert b.field(old,'TimeLimit')=='3-00:00:00' and b.field(old,'Partition')=='lowprio'
            item['old_fallback_held']=False;f.update(status='complete',completed_at=b.now(),after_release=old)
            event(tx,'Reconciled previously acknowledged long-route release');return
    old=old_guard(item,held=True)
    writers=prior.recovery.active_writers().get(str(Path(item['run_dir']).resolve()),set())
    assert writers <= {item['old_job_id']}, ('unexpected writer during fallback',writers)
    cp=prior.checkpoint(item['run_dir']);assert cp['step']>=item['checkpoint']['step']
    f['checkpoint_at_fallback']=cp;event(tx,f"Fallback requested for inactive hourly job {new_job_id}: {reason}")
    if not f.get('staged'):
        current_identity(item,item['new_job_id'])
        source=json.loads(SOURCE.read_text());aggregate=json.loads(AGGREGATE.read_text())
        before_hashes={str(p):prior.digest(p) for p in MUTABLE}
        row=next(r for r in source['runs'] if r['run_dir']==item['run_dir'])
        row['previous_job_ids']=[j for j in row.get('previous_job_ids',[]) if j!=item['old_job_id']]
        if item['new_job_id'] not in row['previous_job_ids']:row['previous_job_ids'].append(item['new_job_id'])
        row.update(job_id=item['old_job_id'],held_scheduler_record=old,repair_audit=str(TX),
                   hourly_backfill_status='returned_to_original_long_route',
                   hourly_backfill_returned_from=item['new_job_id'],
                   hourly_backfill_runtime_changes={'OAT_ZERO_RESUME_STEPS':{'before':'48','after':'192'}})
        target=next(r for r in aggregate['runs'] if r['run_dir']==item['run_dir'])
        scale=target['scale'];target.clear();target.update(row,scale=scale)
        source.setdefault('repair_history',[]).append({'at':b.now(),'audit':str(TX),'hourly_fallback_reason':reason,
            'hourly_job_id':item['new_job_id'],'restored_long_job_id':item['old_job_id']})
        staged={}
        for path,value in ((SOURCE,source),(AGGREGATE,aggregate)):
            out=ART/f"fallback_{new_job_id}_{path.name}.after";save(out,value)
            staged[str(path)]={'path':str(out),'sha256':prior.digest(out)}
        f['before_sha256']=before_hashes;f['staged']=staged
        event(tx,'Staged fallback mappings before releasing the preserved long allocation')
    for path,image in f['staged'].items():
        if prior.digest(path)==image['sha256']:continue
        assert prior.digest(path)==f['before_sha256'][path], ('concurrent fallback ledger mutation',path)
        assert prior.digest(image['path'])==image['sha256']
        save(path,json.loads(Path(image['path']).read_text()));assert prior.digest(path)==image['sha256']
    current_identity(item,item['old_job_id']);f['ledgers_committed']=True
    event(tx,'Both canonical mappings restored to original long job')
    assert item['new_job_id'] not in b.queue()
    old=old_guard(item,held=True)
    assert prior.recovery.active_writers().get(str(Path(item['run_dir']).resolve()),set())<={item['old_job_id']}
    f['release_requested']=True;event(tx,f"Releasing restored long job {item['old_job_id']}")
    b.command(['scontrol','release',str(item['old_job_id'])])
    after=b.show(item['old_job_id'])
    assert b.field(after,'JobState') in {'PENDING','RUNNING','CONFIGURING'} and b.field(after,'Priority')!='0'
    assert b.submit_tokens(after)==item['original_command'] and nodes(after)==LONG_NODES
    for key in PRESERVE:assert b.field(after,key)==b.field(item['before'],key)
    assert b.field(after,'TimeLimit')=='3-00:00:00' and b.field(after,'Partition')=='lowprio'
    item['old_fallback_held']=False;f.update(status='complete',completed_at=b.now(),after_release=after)
    event(tx,'Hourly requeue route ended; original 72-hour allocation restored and released')
    print(json.dumps({'restored_long_job':item['old_job_id'],'inactive_hourly_job':item['new_job_id'],'reason':reason}))


def retire_completed(new_job_id):
    """Retire only the owned-held dormant original after scientific completion."""
    tx=load_transaction();item=next(i for i in tx['items'] if i['new_job_id']==int(new_job_id))
    if item.get('fallback_retired_complete'):return
    assert prior.recovery.complete(Path(item['run_dir'])), 'terminal science receipt required'
    assert item.get('old_fallback_held') and item.get('released')
    current_identity(item,item['new_job_id'])
    if item['old_job_id'] in b.queue():
        old_guard(item,held=True)
        item['complete_retire_requested']=True;event(tx,'Scientific completion verified; retiring exact held long fallback')
        b.command(['scancel',str(item['old_job_id'])]);assert item['old_job_id'] not in b.queue()
    else:assert item.get('complete_retire_requested'), 'dormant original disappeared outside controller'
    item['old_fallback_held']=False;item['fallback_retired_complete']=True
    event(tx,'Completed scientific cell has no dormant retraining request')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','stage','release','fallback','retire-completed'))
    parser.add_argument('new_job_id',nargs='?',type=int)
    parser.add_argument('--reason',choices=('no_progress','requeue_cap','deadline'))
    args=parser.parse_args();ART.mkdir(parents=True,exist_ok=True)
    with (ROOT/'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.phase=='fallback':
            assert args.new_job_id is not None and args.reason is not None
            fallback(args.new_job_id,args.reason)
        elif args.phase=='retire-completed':
            assert args.new_job_id is not None
            retire_completed(args.new_job_id)
        else:globals()[args.phase]()
