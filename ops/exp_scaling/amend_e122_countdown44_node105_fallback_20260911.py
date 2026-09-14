#!/usr/bin/env python3
"""Add qualified node105 after Slurm definitively rejected an account change."""
from pathlib import Path
import fcntl
import amend_e122_countdown44_node105_20260911 as reviewed

recovery = reviewed.recovery
c = reviewed.c
require = reviewed.require
ROOT = reviewed.ROOT
ART = ROOT/'var/artifacts/e122_countdown44_node105_fallback_20260911'
PLAN = ART/'plan.json'
SOURCE = Path(__file__).resolve()
TEST = ROOT/'tests/test_amend_e122_countdown44_node105_fallback_20260911.py'
PROTOCOL = ROOT/'paper/preregistration/e122_countdown44_node105_fallback_20260911.md'
OLD_PLAN_SHA = 'f5cd4fed38b8931804a7652051b31ae81bf683da046d7b2d9ff782775c03303e'


def rejected_account_attempt():
    result_path = reviewed.ART/'31158682.result.json'
    value = recovery.read(result_path)
    require(value['returncode'] == 1 and value['error'] is None
            and 'Account may not be modified after submission' in value['stderr'],
            'Prior owner update lacks definite account rejection')
    require({p.name for p in reviewed.ART.glob('*.result.json')} == {'31158682.result.json'},
            'Unexpected additional owner transaction mutation')
    return result_path


def verify(expected):
    plan = c.pinned_json(PLAN,expected)
    require(plan['schema'] == 'e122_countdown44_node105_fallback_v1'
            and plan['job_ids'] == list(reviewed.JOBS)
            and set(plan['new_nodes']) == reviewed.NEW_NODES, 'Fallback plan differs')
    for path,pin in plan['source_pins'].items():
        require(c.digest(Path(path)) == pin, 'Fallback source/evidence changed: '+path)
    reviewed.verify_plan(OLD_PLAN_SHA)
    rejected_account_attempt()
    return plan


def prepare(campaign):
    require(not PLAN.exists(), 'Fallback plan already exists')
    old = reviewed.verify_plan(OLD_PLAN_SHA)
    require(campaign['binding'] == old['campaign_binding'], 'Original authenticated campaign differs')
    rejection = rejected_account_attempt()
    rows = {}
    for jid,row in old['rows'].items():
        record = reviewed.prior.base.show(int(jid))
        reviewed.audit(record,row['original_record'],jid,row['cell'])
        require(reviewed.pending(record) and reviewed.prior.nodes(record) == reviewed.OLD_NODES,
                'Prior rejected update did not preserve exact original pending request')
        reviewed.require_released(record)
        rows[jid] = {**row,'before':record,'sole_writer_at_prepare':reviewed.sole_writer(jid,row['cell']),
            'command':['scontrol','update','JobId='+jid,'ReqNodeList='+reviewed.POOL]}
    paths = [SOURCE,TEST,PROTOCOL,reviewed.PLAN,reviewed.SOURCE,
             reviewed.ART/'apply.intent.json',reviewed.ART/'31158682.intent.json',rejection]
    value = {'schema':'e122_countdown44_node105_fallback_v1','at_utc':c.now(),
        'job_ids':list(reviewed.JOBS),'new_nodes':sorted(reviewed.NEW_NODES),'rows':rows,
        'prior_owner_update_definitively_rejected':True,'prior_owner_plan_sha256':OLD_PLAN_SHA,
        'source_pins':{str(p):c.digest(p) for p in paths},'node_health':reviewed.node_health(),
        'account_partition_qos_science_resources_unchanged':True,'new_submissions_or_releases':False}
    c.immutable_json(PLAN,value)
    return value


def apply(expected):
    plan = verify(expected)
    with (ART/'singleton.lock').open('a') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with recovery.admission_locks(), c.locked(c.JOURNAL_ROOT):
            require(not (ART/'apply.intent.json').exists(), 'Fallback intent consumed; never repeat')
            c.immutable_json(ART/'apply.intent.json',{'at_utc':c.now(),'plan_sha256':expected})
            outcomes = {}
            for jid,row in plan['rows'].items():
                verify(expected)
                record = reviewed.prior.base.show(int(jid))
                reviewed.audit(record,row['original_record'],jid,row['cell'])
                require(reviewed.prior.nodes(record) == reviewed.OLD_NODES, 'Unowned route change')
                if not reviewed.pending(record):
                    outcomes[jid] = {'status':'preserved_started','record':record}; continue
                reviewed.require_released(record); reviewed.node_health()
                writer = reviewed.sole_writer(jid,row['cell'])
                record = reviewed.prior.base.show(int(jid))
                reviewed.audit(record,row['original_record'],jid,row['cell'])
                require(reviewed.prior.nodes(record) == reviewed.OLD_NODES, 'Route changed before update')
                if not reviewed.pending(record):
                    outcomes[jid] = {'status':'preserved_started','record':record}; continue
                reviewed.require_released(record)
                c.immutable_json(ART/f'{jid}.intent.json',{'at_utc':c.now(),'before':record,
                    'command':row['command'],'sole_writer':writer,'plan_sha256':expected})
                try:
                    result = c.command(row['command'])
                    ack = {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'error':None}
                except Exception as error:
                    ack = {'returncode':None,'error':type(error).__name__+': '+str(error)}
                c.immutable_json(ART/f'{jid}.result.json',{'at_utc':c.now(),**ack})
                require(ack['returncode'] == 0, 'Fallback update uncertain; never repeat')
                after = reviewed.prior.base.show(int(jid))
                reviewed.audit(after,row['original_record'],jid,row['cell'])
                require(reviewed.prior.nodes(after) == reviewed.NEW_NODES, 'Acknowledged fallback route differs')
                outcomes[jid] = {'status':'expanded','record':after}
                c.immutable_json(ART/f'{jid}.readback.json',outcomes[jid])
            c.immutable_json(ART/'complete.json',{'at_utc':c.now(),'plan_sha256':expected,'outcomes':outcomes})
            return outcomes
