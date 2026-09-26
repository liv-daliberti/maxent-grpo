#!/usr/bin/env python3
"""Apply the reviewed E120 Falcon Pantry serial64GiB owner-partition backfill."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import backfill_e120_falcon_pantry_node302_20260906 as prep
from prioritize_e118_capacity_20260905 import command,exports,field,show,submit_tokens,atomic,queue

ART=prep.ART
TX=ART/'transaction.json'
IDENTITY=('domain','model_key','seed','run_dir','run_stamp')


def event(tx,message):
    tx.setdefault('events',[]).append({'at':prep.prior.now(),'message':message})
    atomic(TX,tx)


def pending_identity(jid,item,held=False):
    s=show(jid)
    assert field(s,'JobState')=='PENDING',jid
    assert (field(s,'Reason')=='JobHeldUser' and field(s,'Priority')=='0') if held else int(field(s,'Priority'))>0
    assert exports(submit_tokens(s))==exports(item['original_command'])
    assert submit_tokens(s)[-1]==item['original_command'][-1]
    for key in ('MinMemoryNode','NumCPUs','TimeLimit','Nice','Requeue','Dependency','WorkDir'):
        assert field(s,key)==field(item['before_record'],key),(jid,key)
    assert not list(Path(item['run']['run_dir']).glob('debug_job*/train_metrics.jsonl'))
    assert not (Path(item['run']['run_dir'])/'TRAINING_COMPLETE.json').exists()
    return s


def apply():
    assert not TX.exists(),'transaction exists; reconcile evidence before any retry'
    plan=json.loads(prep.PLAN.read_text())
    assert plan['status']=='prepared_only'
    for path,key in [(prep.campaign.E120_LEDGER,'main_ledger_sha256'),(prep.campaign.E120_CONTINUATIONS,'continuation_ledger_sha256'),
                     (prep.AMEND,'amendment_sha256'),(prep.__file__,'controller_sha256'),(prep.prior.PATCH,'runtime_amendment_sha256')]:
        assert prep.sha(path)==plan[key],(path,key)
    prep.prior.patch_valid()
    main=json.loads(prep.campaign.E120_LEDGER.read_text())
    mapping=prep.campaign.e120_continuation_jobs(prep.campaign.E120_LEDGER)
    assert len(mapping)==7
    holds=prep.protected_holds(main,mapping)
    assert {r['job_id'] for r in holds}=={r['job_id'] for r in plan['protected_qwen3b_holds']}
    cap=prep.capacity()
    for item in plan['rows']:
        assert prep.sha(item['original_command'][-1])==item['wrapper_sha256']
        pending_identity(item['old_job_id'],item)
    tx={'schema':'e120-falcon-pantry-node302-backfill-transaction-v1','started_at_utc':prep.prior.now(),
        'authorization':'Root confirmed final E119128GiB repairs released; user authorized bounded backfill after recovery.',
        'status':'started','plan_sha256':prep.sha(prep.PLAN),'apply_controller_sha256':prep.sha(__file__),
        'capacity_before_holds':cap,'protected_holds_before':holds,'rows':plan['rows'],'events':[]}
    event(tx,'validated live capacity, scientific identity, repaired runtime and ten protected holds')
    try:
        for item in tx['rows']:
            command(['scontrol','hold',str(item['old_job_id'])])
            item['old_held_record']=pending_identity(item['old_job_id'],item,held=True)
            event(tx,f"held superseded zero-step pending job {item['old_job_id']}")
        previous=None
        for item in tx['rows']:
            dependency=f'afterany:{previous}' if previous else None
            item['dependency']=dependency
            item['command']=prep.placed(item['original_command'],item['old_job_id'],dependency)
            item['submission_uncertain']=True
            event(tx,f"submitting held replacement for {item['old_job_id']}")
            result=command(item['command']).stdout
            item['new_job_id']=int(result.strip().split(';')[0]);item['submission_uncertain']=False
            event(tx,f"submitted held replacement {item['new_job_id']}")
            record=show(item['new_job_id']);prep.held_audit(record,item,dependency)
            item['held_scheduler_record']=record;previous=item['new_job_id']
            event(tx,f"audited held replacement {item['new_job_id']}")
        tx['capacity_before_commit']=prep.capacity()
        assert prep.sha(prep.campaign.E120_LEDGER)==plan['main_ledger_sha256']
        assert prep.sha(prep.campaign.E120_CONTINUATIONS)==plan['continuation_ledger_sha256']
        ledger=json.loads(prep.campaign.E120_CONTINUATIONS.read_text())
        existing={r['original_job_id'] for r in ledger['continuations']}
        for item in tx['rows']:
            pending_identity(item['old_job_id'],item,held=True)
            assert item['original_job_id'] not in existing
            row={key:item['run'][key] for key in IDENTITY}
            row.update(original_job_id=item['original_job_id'],continuation_job_id=item['new_job_id'],
                observed_step_before_continuation=0,resume_checkpoint=0,pvl_excluded=True,released=False,
                repair_kind='node302_owner_serial_64g_backfill',placement_amendment=str(prep.AMEND),
                held_scheduler_record=item['held_scheduler_record'],dependency=item['dependency'],
                new_placement={'account':'mltheory','partition':'mltheory','node':'node302','gpu':'a100',
                    'cpus':8,'memory':'64G','nice':100})
            ledger['continuations'].append(row)
        atomic(prep.campaign.E120_CONTINUATIONS,ledger)
        tx['ledger_committed']=True;tx['continuation_ledger_sha256']=prep.sha(prep.campaign.E120_CONTINUATIONS)
        event(tx,'registered both held replacements in continuation ledger; primary scientific ledger unchanged')
        mapping=prep.campaign.e120_continuation_jobs(prep.campaign.E120_LEDGER)
        assert len(mapping)==9
        assert all(mapping[i['original_job_id']]==i['new_job_id'] for i in tx['rows'])
        for item in tx['rows']:
            pending_identity(item['old_job_id'],item,held=True)
            command(['scancel',str(item['old_job_id'])])
            s=show(item['old_job_id'])
            assert field(s,'JobState')=='CANCELLED',(item['old_job_id'],field(s,'JobState'))
            item['old_cancelled_record']=s
            event(tx,f"cancelled superseded pending job {item['old_job_id']}")
        assert not any(i['old_job_id'] in queue() for i in tx['rows'])
        for item in tx['rows']:
            prep.held_audit(show(item['new_job_id']),item,item['dependency'])
            command(['scontrol','release',str(item['new_job_id'])])
            s=show(item['new_job_id'])
            assert field(s,'JobState') in ('RUNNING','PENDING','CONFIGURING') and field(s,'Priority')!='0'
            item['release_record']=s;item['released']=True
            next(r for r in ledger['continuations'] if r['original_job_id']==item['original_job_id'])['released']=True
            event(tx,f"released replacement {item['new_job_id']}")
        atomic(prep.campaign.E120_CONTINUATIONS,ledger)
        assert prep.sha(prep.campaign.E120_LEDGER)==plan['main_ledger_sha256']
        tx['protected_holds_after']=prep.protected_holds(main,mapping)
        tx['continuation_ledger_sha256']=prep.sha(prep.campaign.E120_CONTINUATIONS)
        tx['status']='released';event(tx,'complete: one serial64GiB slot; all ten Qwen3B holds remain protected')
        print(json.dumps({'status':tx['status'],'replacements':[{'old':i['old_job_id'],'new':i['new_job_id'],'dependency':i['dependency'],'state':field(i['release_record'],'JobState')} for i in tx['rows']]}))
    except BaseException as exc:
        tx['status']='stopped_for_review';tx['error']=repr(exc);event(tx,'stopped safely; inspect recorded holds and mappings before retry');raise


if __name__=='__main__':apply()
