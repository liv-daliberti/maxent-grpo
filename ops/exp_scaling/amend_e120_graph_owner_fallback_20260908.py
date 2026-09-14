#!/usr/bin/env python3
"""Apply reviewed same-ID E120 owner fallback without changing runtime exports."""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path

import campaign_stats as campaign
import backfill_e120_graph_node204_20260908 as parent
import prioritize_e118_capacity_20260905 as base

ART = parent.ART / 'owner_fallback'
PROTOCOL = ART / 'protocol.md'
RECEIPT = ART / 'transaction.json'
JOB = 31144919
NODES = 'node202,node203,node204,node302'
PARTITIONS = 'lowprio,mltheory'


def row_digest(row):
    return hashlib.sha256(base.encoded(row)).hexdigest()


def inspect_child(initial):
    rec = base.show(parent.CHILD)
    assert base.field(rec,'JobState') == 'PENDING'
    assert base.field(rec,'Dependency') == f'afterany:{JOB}(unfulfilled)'
    assert base.field(rec,'Priority') != '0'
    assert base.submit_tokens(rec) == base.submit_tokens(initial['child_before'])
    assert base.field(rec,'ReqNodeList') == 'node302'
    assert base.field(rec,'ExcNodeList') == parent.prior.EXCLUSION
    return rec


def same_runtime(rec, initial):
    tokens = base.submit_tokens(rec)
    assert tokens[-1] == initial['command'][-1]
    assert base.exports(tokens) == base.exports(initial['command'])
    assert base.exports(tokens)[parent.RATIO] == '0.40'
    assert base.field(rec,'ExcNodeList') == parent.prior.EXCLUSION
    assert base.field(rec,'MinMemoryNode') == '128G'
    assert base.field(rec,'NumCPUs') == '16'
    assert base.field(rec,'NumTasks') == '1'
    assert base.field(rec,'NumNodes') in ('1','1-1')
    assert base.field(rec,'TimeLimit') == '3-00:00:00'
    assert base.field(rec,'Requeue') == '1'
    assert dict(v.split('=',1) for v in base.field(rec,'ReqTRES').split(','))['gres/gpu'] == '1'
    assert all(parent.digest(p)==v for p,v in initial['runtime_sha256'].items())
    assert 'pvl' not in rec.lower()


def apply():
    assert not RECEIPT.exists(), 'Existing amendment: reconcile rather than repeat'
    initial=json.loads(parent.TX.read_text())
    assert initial['new_job_id']==JOB and initial['status']=='released'
    initial_hash=parent.digest(parent.TX)
    before=base.show(JOB)
    assert base.field(before,'JobState')=='PENDING' and base.field(before,'Priority')!='0'
    assert base.field(before,'Account')=='allcs' and base.field(before,'Partition')=='lowprio'
    assert base.field(before,'ReqNodeList')=='node204'
    same_runtime(before,initial)
    child_before=inspect_child(initial)
    parent.no_completion(initial['run']['run_dir'])
    assert parent.checkpoint(initial['run'])==initial['checkpoint']
    assert parent.checkpoint_file_identity(initial['checkpoint'])==initial['checkpoint_file_identity']
    assert parent.writers(initial['run']['run_dir'],parent.live_records())==[JOB]
    assert parent.incoming(JOB)=={parent.CHILD:f'afterany:{JOB}(unfulfilled)'}
    ledger=json.loads(campaign.E120_CONTINUATIONS.read_text())
    old_ledger_hash=parent.digest(campaign.E120_CONTINUATIONS)
    assert parent.digest(campaign.E120_LEDGER)==initial['primary_sha256']
    mapping=campaign.e120_continuation_jobs(campaign.E120_LEDGER)
    assert len(mapping)==9 and mapping[parent.ORIGINAL]==JOB
    row=next(r for r in ledger['continuations'] if r['original_job_id']==parent.ORIGINAL)
    assert row['continuation_job_id']==JOB
    old_row=json.loads(json.dumps(row))
    replacements={'--account':'mltheory','--partition':PARTITIONS,'--nodelist':NODES,'--gres':'gpu:1'}
    cmd=[v.split('=',1)[0]+'='+replacements[v.split('=',1)[0]] if v.split('=',1)[0] in replacements else v for v in initial['command']]
    assert base.exports(cmd)==base.exports(initial['command']) and cmd[-1]==initial['command'][-1]
    test=base.command([cmd[0],'--test-only',*cmd[1:]])
    tx={'schema':'e120-graph-owner-fallback-same-id-v1','created_at_utc':base.now(),'job_id':JOB,
        'status':'validated','controller_sha256':parent.digest(__file__),'protocol_sha256':parent.digest(PROTOCOL),
        'initial_transaction_sha256':initial_hash,'primary_sha256':initial['primary_sha256'],
        'continuations_before_sha256':old_ledger_hash,'row_before_sha256':row_digest(row),
        'row_before':old_row,'scheduler_before':before,'child_before':child_before,
        'runtime_exports_changed':False,'scientific_settings_changed':False,'checkpoint':initial['checkpoint'],
        'test_only':{'command':cmd,'stdout':test.stdout,'stderr':test.stderr,'returncode':test.returncode},'events':[]}
    def event(message):
        tx['events'].append({'at':base.now(),'message':message});base.atomic(RECEIPT,tx)
    event('Validated same job, checkpoint, writer, child, runtime and accepted lowprio-first test-only request')
    try:
        tx['holding_job']=True;event('Holding pending job before scheduler resource amendment')
        base.command(['scontrol','hold',str(JOB)])
        held=base.show(JOB)
        assert base.field(held,'JobState')=='PENDING' and base.field(held,'Reason')=='JobHeldUser'
        same_runtime(held,initial);inspect_child(initial)
        tx['held_before']=held
        tx['updating_scheduler']=True;event('Adding owner fallback and allowed A5000 node pool without creating a job')
        base.command(['scontrol','update',f'JobId={JOB}','Account=mltheory',f'Partition={PARTITIONS}',f'NodeList={NODES}','Gres=gpu:1'])
        after=base.show(JOB)
        same_runtime(after,initial)
        assert base.field(after,'JobState')=='PENDING' and base.field(after,'Reason')=='JobHeldUser'
        assert base.field(after,'Priority')=='0' and base.field(after,'Account')=='mltheory'
        assert base.field(after,'Partition')==PARTITIONS
        assert set(base.command(['scontrol','show','hostnames',base.field(after,'ReqNodeList')]).stdout.splitlines())==set(NODES.split(','))
        assert base.field(after,'TresPerNode')=='gres/gpu:1'
        assert base.field(after,'Dependency')=='(null)'
        tx['scheduler_after_held']=after
        tx['child_after_held']=inspect_child(initial)
        assert parent.writers(initial['run']['run_dir'],parent.live_records())==[JOB]
        parent.no_completion(initial['run']['run_dir'])
        assert parent.checkpoint_file_identity(initial['checkpoint'])==initial['checkpoint_file_identity']
        assert parent.digest(campaign.E120_CONTINUATIONS)==old_ledger_hash
        row['new_placement']={'account':'mltheory','partition':PARTITIONS,'node':NODES,'gpu':'generic:1','cpus':16,'memory':'128G','nice':200}
        row['placement_amendment']=str(PROTOCOL)
        row.setdefault('scheduler_placement_amendments',[]).append(str(PROTOCOL))
        row['held_scheduler_record']=after
        ledger.setdefault('placement_amendments',[]).append(str(PROTOCOL))
        tx['row_after_sha256']=row_digest(row)
        tx['row_after']=json.loads(json.dumps(row))
        tx['committing_ledger']=True;event('Promoting audited same-ID resource placement before release')
        base.atomic(campaign.E120_CONTINUATIONS,ledger)
        mapping=campaign.e120_continuation_jobs(campaign.E120_LEDGER)
        assert len(mapping)==9 and mapping[parent.ORIGINAL]==JOB
        assert parent.digest(campaign.E120_LEDGER)==initial['primary_sha256']
        assert parent.digest(parent.TX)==initial_hash
        tx['ledger_committed']=True
        tx['continuations_after_sha256']=parent.digest(campaign.E120_CONTINUATIONS)
        event('Same continuation ID and exact child dependency retained; primary and initial transaction unchanged')
        same_runtime(base.show(JOB),initial);inspect_child(initial)
        tx['releasing_job']=True;event('Releasing same job with owner fallback and A5000 borrowing enabled')
        base.command(['scontrol','release',str(JOB)])
        final=base.show(JOB);same_runtime(final,initial)
        assert base.field(final,'JobState') in ('PENDING','RUNNING','CONFIGURING') and base.field(final,'Priority')!='0'
        tx['scheduler_after_release']=final;tx['child_after_release']=inspect_child(initial)
        tx['released']=True;tx['status']='released';tx['observed_at_utc']=base.now()
        tx['scheduler_timezone']='America/New_York'
        tx['optimizer_verified']=False
        event('Released same E120 job; allocation and optimizer startup are not claimed')
        print(json.dumps({'job_id':JOB,'state':base.field(final,'JobState'),'node':base.field(final,'NodeList'),
                          'reason':base.field(final,'Reason'),'account':base.field(final,'Account'),
                          'partition':base.field(final,'Partition'),'eligible_nodes':base.field(final,'ReqNodeList'),
                          'forecast_scheduler_local':base.field(final,'StartTime')},indent=2))
    except BaseException as exc:
        tx['status']='stopped_for_reconciliation';tx['error']=repr(exc)
        event('Stopped fail closed; reconcile held job and receipt rather than repeat amendment')
        raise


if __name__=='__main__':
    ART.mkdir(parents=True,exist_ok=True)
    with (parent.ROOT/'var/artifacts/e120_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);apply()
