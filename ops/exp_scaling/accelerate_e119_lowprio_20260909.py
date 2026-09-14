#!/usr/bin/env python3
"""One guarded E119 Pantry hourly continuation with a held long-route fallback."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path
import accelerate_a5000_completion_20260909 as prior
import campaign_stats as campaign
import recover_terminal_timeouts_20260908 as recovery
import recover_e119_health_20260905 as e119_health

b = prior.base
ROOT = prior.ROOT
ART = ROOT / 'var/artifacts/e119_lowprio_completion_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
GUARD_READY = ART / 'guard_ready.json'
PROTOCOL = ROOT / 'paper/preregistration/e119_lowprio_completion_20260909.md'
PRIMARY = campaign.E119_LEDGER
LEDGER = campaign.E119_CONTINUATIONS
OLD = 31037832
POOL = {'node205', 'node206', 'node207', 'node302'}
IDENTITY = ('domain', 'arm', 'seed', 'run_dir', 'run_stamp')
PRESERVE = ('NumCPUs', 'NumTasks', 'CPUs/Task', 'Nice', 'Requeue', 'ExcNodeList', 'WorkDir', 'Command', 'StdOut', 'StdErr')
OLD_RESOURCES = PRESERVE + ('Account', 'Partition', 'TimeLimit', 'ReqNodeList', 'MinMemoryNode', 'TresPerNode')


def save(path, value):
    b.atomic(Path(path), value)


def event(tx, message):
    tx['events'].append({'at': b.now(), 'message': message})
    save(TX, tx)


def nodes(record):
    return set(b.command(['scontrol', 'show', 'hostnames', b.field(record, 'ReqNodeList')]).stdout.split())


def row(data, item):
    rows = [r for r in data['continuations'] if r['run_dir'] == item['run_dir']]
    assert len(rows) == 1
    assert all(rows[0][k] == item[k] for k in IDENTITY)
    return rows[0]


def current_identity(item, job):
    found = row(json.loads(LEDGER.read_text()), item)
    assert int(found['continuation_job_id']) == job, 'authoritative continuation changed'
    originals = [r for r in json.loads(PRIMARY.read_text())['runs'] if r['run_dir'] == item['run_dir']]
    assert len(originals) == 1 and all(originals[0][k] == item[k] for k in IDENTITY)
    assert int(originals[0]['job_id']) == item['row_before']['original_job_id']


def safe_run(item, allowed, *, initial=False):
    assert not recovery.complete(Path(item['run_dir'])), 'training already complete'
    actual = recovery.active_writers().get(str(Path(item['run_dir']).resolve()), set())
    assert actual <= set(allowed), ('unexpected active writer', actual)
    cp = prior.checkpoint(item['run_dir'])
    checked = e119_health.checkpoint(item)
    assert checked['step'] == cp['step']
    if initial:
        assert cp == item['checkpoint'], 'checkpoint changed before admission'
    return cp


def old_guard(item, held=True):
    record = b.show(item['old_job_id'])
    assert b.field(record, 'JobState') == 'PENDING', 'original started; preserve its allocation'
    assert b.submit_tokens(record) == item['original_command']
    for key in OLD_RESOURCES:
        assert b.field(record,key) == b.field(item['before'],key), ('original profile changed',key)
    assert b.field(record,'Dependency') == '(null)'
    if held:
        assert b.field(record,'Reason') == 'JobHeldUser' and b.field(record,'Priority') == '0'
    else:
        assert b.field(record,'Priority') != '0', 'do not claim a preexisting hold'
    assert not prior.incoming(item['old_job_id']), 'original acquired dependent'
    return record


def new_guard(item, held=False):
    record = b.show(item['new_job_id'])
    expected = {'Account':'mltheory','Partition':'all','QOS':'none','TimeLimit':'01:00:00',
                'MinMemoryNode':'116G','NumCPUs':'8','TresPerNode':'gres/gpu:1',
                'Dependency':'(null)','Comment':item['comment'],'Requeue':'1','ExcNodeList':b.PVL}
    if held:
        expected.update(JobState='PENDING',Reason='JobHeldUser',Priority='0')
    for key,value in expected.items():
        assert b.field(record,key) == value, ('replacement profile mismatch', key, b.field(record,key), value)
    assert nodes(record) == POOL
    for key in PRESERVE:
        if key in ('StdOut','StdErr'): continue  # same %x-%j template, new scheduler identity
        assert b.field(record,key) == b.field(item['before'],key), key
    assert b.field(record,'NumNodes') in ('1','1-1')
    assert dict(x.split('=',1) for x in b.field(record,'ReqTRES').split(','))['gres/gpu'] == '1'
    command = b.submit_tokens(record)
    assert b.exports(command) == b.exports(item['original_command'])
    assert command[-1] == item['original_command'][-1]
    return record


def build_command(item):
    original = item['original_command']
    changes = {'--account':'mltheory','--partition':'all','--nodelist':','.join(sorted(POOL)),
               '--gres':'gpu:1','--mem':'116G','--cpus-per-task':'8','--time':'01:00:00',
               '--nodes':'1','--ntasks':'1','--ntasks-per-node':'1',
               '--nice':b.field(item['before'],'Nice'),'--exclude':b.PVL,
               '--output':str(ROOT/'var/artifacts/logs/%x-%j.out'),
               '--error':str(ROOT/'var/artifacts/logs/%x-%j.err'),'--comment':item['comment']}
    skip = set(changes) | {'--hold','--dependency','--begin'}
    command = [x for x in original[:-1] if x.split('=',1)[0] not in skip]
    command += [f'{k}={v}' for k,v in changes.items()] + ['--hold',original[-1]]
    assert b.exports(command) == b.exports(original)
    return command


def prepare():
    assert not PLAN.exists() and not TX.exists(), 'existing immutable preparation'
    ART.mkdir(parents=True,exist_ok=True)
    data=json.loads(LEDGER.read_text())
    before=next(r for r in data['continuations'] if r['continuation_job_id']==OLD)
    record=b.show(OLD);command=b.submit_tokens(record);env=b.exports(command)
    assert before['domain']=='pantry_plan' and before['arm']=='replay_drgrpo' and before['seed']==44
    assert env['SAVE_PATH']==before['run_dir'] and env['RUN_STAMP']==before['run_stamp']
    assert env['OAT_ZERO_AUTO_RESUME']=='1' and env['OAT_ZERO_VLLM_GPU_RATIO']=='0.25'
    assert env['OAT_ZERO_SOURCE_ROOT'].endswith('e76_tuned_scale_50d36295558a8958/src')
    for key,value in {'Account':'allcs','Partition':'cs','MinMemoryNode':'96G','NumCPUs':'8','Requeue':'1','ExcNodeList':b.PVL}.items():
        assert b.field(record,key)==value
    assert nodes(record)=={'node205','node206','node207'}
    item={k:before[k] for k in IDENTITY}
    item.update(old_job_id=OLD,new_job_id=None,row_before=copy.deepcopy(before),before=record,
                original_command=command,checkpoint=prior.checkpoint(before['run_dir']),
                comment=f'e119-lowprio-completion-20260909-old{OLD}')
    current_identity(item,OLD);old_guard(item,False);safe_run(item,[OLD],initial=True)
    item['command']=build_command(item)
    dry=b.command([item['command'][0],'--test-only',*[x for x in item['command'][1:] if x!='--hold']])
    item['test_only']={'stdout':dry.stdout,'stderr':dry.stderr,'returncode':dry.returncode}
    hashes={str(p):prior.digest(p) for p in [Path(__file__),PROTOCOL,Path(prior.__file__),Path(b.__file__),Path(recovery.__file__),Path(e119_health.__file__)]}
    plan={'schema':'e119-lowprio-completion-20260909-v1','status':'prepared','created_at':b.now(),
          'item':item,'events':[],'frozen_files':hashes,'runtime_fingerprints':prior.runtime_fingerprints([item]),
          'scientific_exports_unchanged':True,'effective_resume_steps':96,'primary_sha256':prior.digest(PRIMARY),
          'controller_sha256':prior.digest(__file__),'protocol_sha256':prior.digest(PROTOCOL),
          'guard_required_before_release':True,'memory_change':{'old':'96G','new':'116G'}}
    for p in (PRIMARY,LEDGER):(ART/(p.name+'.before')).write_bytes(p.read_bytes())
    save(PLAN,plan)
    print(json.dumps({'prepared':True,'old':OLD,'checkpoint':item['checkpoint']['step'],'test_only':item['test_only']}))


def load_transaction():
    plan=json.loads(PLAN.read_text())
    assert all(prior.digest(p)==h for p,h in plan['frozen_files'].items()), 'frozen controller/protocol drift'
    assert prior.digest(PRIMARY)==plan['primary_sha256'], 'primary scientific ledger changed'
    tx=json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    assert prior.runtime_fingerprints([tx['item']])==tx['runtime_fingerprints'], 'frozen runtime drift'
    return tx


def stage():
    tx=load_transaction();i=tx['item']
    if tx['status'] in ('staged_awaiting_guard','released'):
        print(json.dumps({'status':tx['status'],'new_job_id':i['new_job_id']}));return
    if not tx.get('ledger_committed'):
        existing=row(json.loads(LEDGER.read_text()),i)
        if i.get('new_job_id') and int(existing['continuation_job_id'])==i['new_job_id']:
            assert existing.get('repair_audit')==str(TX) and existing.get('dormant_fallback_job_id')==i['old_job_id']
            tx['ledger_committed']=True;event(tx,'Reconciled completed ledger promotion after interrupted acknowledgement')
        else:
            current_identity(i,i['old_job_id'])
    if not i.get('old_held'):
        if i.get('hold_requested') and b.field(b.show(i['old_job_id']),'Reason')=='JobHeldUser':
            old_guard(i)
        else:
            old_guard(i,False);safe_run(i,[i['old_job_id']],initial=True)
            i['hold_requested']=True;event(tx,'Holding exact still-pending predecessor')
            b.command(['scontrol','hold',str(i['old_job_id'])])
            after=b.show(i['old_job_id'])
            if b.field(after,'JobState')!='PENDING':
                b.command(['scontrol','release',str(i['old_job_id'])])
                raise RuntimeError('Original allocated during hold; preserved current writer')
            old_guard(i)
        i['old_held']=True;event(tx,'Original retained as exact dormant held fallback')
    old_guard(i)
    if i.get('submission_uncertain'):
        raise RuntimeError('Uncertain held submission: reconcile exact comment before retrying')
    if i['new_job_id'] is None:
        safe_run(i,[i['old_job_id']],initial=True)
        i['submission_uncertain']=True;event(tx,'Submitting one held replacement')
        result=b.command(i['command']).stdout.strip()
        i['new_job_id']=int(result.split(';')[0]);i['submission_uncertain']=False
        event(tx,'Recorded held replacement identity')
    i['new_held_record']=new_guard(i,True)
    safe_run(i,[i['old_job_id'],i['new_job_id']],initial=True)
    if not tx.get('ledger_committed'):
        data=json.loads(LEDGER.read_text());r=row(data,i)
        if int(r['continuation_job_id'])==i['old_job_id']:
            assert r==i['row_before'], 'target row changed before promotion'
            r['previous_continuation_job_ids']=[*r.get('previous_continuation_job_ids',[]),i['old_job_id']]
            r.update(continuation_job_id=i['new_job_id'],held_scheduler_record=i['new_held_record'],
                     repair_audit=str(TX),runtime_allocation_amendment=str(PROTOCOL),dormant_fallback_job_id=i['old_job_id'])
            data.setdefault('repair_history',[]).append({'at':b.now(),'audit':str(TX),'old':i['old_job_id'],'new':i['new_job_id'],'scientific_exports_unchanged':True})
            assert len(data['continuations'])==75
            save(ART/'continuation_ledger.after.json',data);event(tx,'Staged single authoritative continuation after-image')
            save(LEDGER,data)
        else:
            assert int(r['continuation_job_id'])==i['new_job_id'] and r.get('repair_audit')==str(TX)
        current_identity(i,i['new_job_id']);tx['ledger_committed']=True
    tx['status']='staged_awaiting_guard';event(tx,'Authoritative continuation points to held replacement; original remains dormant')
    print(json.dumps({'status':tx['status'],'old':i['old_job_id'],'new_job_id':i['new_job_id']}))


def release():
    tx=load_transaction();i=tx['item']
    assert tx['status'] in ('staged_awaiting_guard','released') and tx.get('ledger_committed')
    current_identity(i,i['new_job_id']);old_guard(i)
    ready=json.loads(GUARD_READY.read_text())
    assert ready['new_job_id']==i['new_job_id'] and ready['controller_sha256']==tx['controller_sha256']
    assert ready['plan_sha256']==prior.digest(PLAN)
    assert b.field(b.show(ready['guard_job_id']),'JobState')=='RUNNING', 'durable guard must be running'
    if i.get('release_requested') and b.field(b.show(i['new_job_id']),'Priority')!='0':
        new_guard(i);i['released']=True;tx['status']='released';event(tx,'Reconciled acknowledged or uncertain release');return
    new_guard(i,True);safe_run(i,[i['old_job_id'],i['new_job_id']],initial=True)
    i['release_requested']=True;event(tx,'Releasing guarded one-hour continuation')
    result=b.command(['scontrol','release',str(i['new_job_id'])],check=False)
    after=new_guard(i)
    assert b.field(after,'Priority')!='0', ('release did not become schedulable',result.stderr)
    i['released']=True;tx['status']='released';i['release_record']=after;event(tx,'Replacement schedulable with unchanged science and exact dormant fallback')


def fallback(new_id,reason):
    """Caller owns ledger lock. Never use lower-memory fallback for memory failure."""
    assert reason in {'no_progress','requeue_cap','deadline','node_failure'}, 'memory/process failure requires separate conservative repair'
    tx=load_transaction();i=tx['item'];assert i['new_job_id']==int(new_id)
    assert tx.get('ledger_committed') and not recovery.complete(Path(i['run_dir']))
    if i.get('fallback_complete'):
        current_identity(i,i['old_job_id']);return {'status':'fallback_released','old_job_id':i['old_job_id']}
    if not i.get('fallback_mapping_restored'):
        existing=row(json.loads(LEDGER.read_text()),i)
        if int(existing['continuation_job_id'])==i['old_job_id']:
            assert existing.get('fallback_from_job_id')==i['new_job_id'] and existing.get('fallback_reason')==reason
            i['fallback_mapping_restored']=True;event(tx,'Reconciled completed fallback mapping after interrupted acknowledgement')
    if not i.get('fallback_mapping_restored'):
        current_identity(i,i['new_job_id']);old_guard(i)
        if int(new_id) in b.queue():
            record=new_guard(i)
            assert b.field(record,'JobState')=='PENDING' and reason=='deadline', 'current allocation must finish before fallback'
            if not i.get('deadline_hold_requested'):
                assert b.field(record,'Priority')!='0', 'unowned pending hold'
                i['deadline_hold_requested']=True;event(tx,'Holding exact pending successor at scheduling deadline')
                b.command(['scontrol','hold',str(new_id)])
            held=new_guard(i)
            if b.field(held,'JobState')!='PENDING':
                b.command(['scontrol','release',str(new_id)])
                raise RuntimeError('continuation allocated; defer fallback until inactive')
            assert b.field(held,'Reason')=='JobHeldUser' and b.field(held,'Priority')=='0'
            b.command(['scancel',str(new_id)])
            assert int(new_id) not in b.queue(), 'successor cancellation still settling'
        assert recovery.state(new_id) in {'TIMEOUT','CANCELLED','NODE_FAIL','BOOT_FAIL'}, 'unsupported inactive state'
        safe_run(i,[i['old_job_id']]);old_guard(i)
        data=json.loads(LEDGER.read_text());r=row(data,i)
        assert int(r['continuation_job_id'])==i['new_job_id']
        r['previous_continuation_job_ids']=[*r.get('previous_continuation_job_ids',[]),i['new_job_id']]
        r.update(continuation_job_id=i['old_job_id'],fallback_from_job_id=i['new_job_id'],fallback_reason=reason)
        r.pop('dormant_fallback_job_id',None)
        save(LEDGER,data);i['fallback_mapping_restored']=True;event(tx,'Restored exact original authoritative mapping after successor inactive')
    current_identity(i,i['old_job_id'])
    record=b.show(i['old_job_id'])
    if i.get('fallback_release_requested') and b.field(record,'Priority')!='0':
        i['fallback_complete']=True;tx['status']='fallback_released';event(tx,'Reconciled original release');return {'status':'fallback_released','old_job_id':i['old_job_id']}
    old_guard(i);safe_run(i,[i['old_job_id']])
    i['fallback_release_requested']=True;event(tx,'Releasing exact original route after coherent mapping')
    result=b.command(['scontrol','release',str(i['old_job_id'])],check=False)
    after=b.show(i['old_job_id']);assert b.field(after,'Priority')!='0',result.stderr
    i['fallback_complete']=True;tx['status']='fallback_released';event(tx,'Original long route fallback released')
    return {'status':'fallback_released','old_job_id':i['old_job_id']}


def retire_completed(new_id):
    """Caller owns ledger lock; retire only exact dormant predecessor on receipt."""
    tx=load_transaction();i=tx['item'];assert i['new_job_id']==int(new_id)
    assert recovery.complete(Path(i['run_dir']))
    if i.get('retired_complete'):return {'status':'complete'}
    assert not i.get('fallback_mapping_restored'), 'fallback became authoritative; coordinate retirement'
    current_identity(i,i['new_job_id'])
    if i['old_job_id'] in b.queue():
        old_guard(i);i['retire_requested']=True;event(tx,'Retiring exact dormant original after complete receipt')
        b.command(['scancel',str(i['old_job_id'])])
        assert i['old_job_id'] not in b.queue(), 'retirement still settling'
    else:
        assert i.get('retire_requested') and recovery.state(i['old_job_id'])=='CANCELLED'
    i['retired_complete']=True;tx['status']='complete';event(tx,'Terminal scientific receipt; dormant fallback retired')
    return {'status':'complete'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','stage','release','fallback','retire-completed'])
    p.add_argument('--new-job-id',type=int);p.add_argument('--reason');args=p.parse_args()
    with (ROOT/'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.action=='fallback':print(json.dumps(fallback(args.new_job_id,args.reason)))
        elif args.action=='retire-completed':print(json.dumps(retire_completed(args.new_job_id)))
        else:globals()[args.action]()
