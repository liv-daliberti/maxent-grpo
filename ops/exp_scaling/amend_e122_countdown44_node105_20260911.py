#!/usr/bin/env python3
"""Route three released E122 jobs to qualified capacity, retaining resources."""
from __future__ import annotations
from pathlib import Path
import fcntl
import json
import resume_e122_shared_release_20260911 as recovery
import amend_e122_countdown_peers_node208_20260910 as prior

ROOT = recovery.ROOT
ART = ROOT/'var/artifacts/e122_countdown44_node105_20260911'
PLAN = ART/'plan.json'
SOURCE = Path(__file__).resolve()
TEST = ROOT/'tests/test_amend_e122_countdown44_node105_20260911.py'
PROTOCOL = ROOT/'paper/preregistration/e122_countdown44_node105_20260911.md'
AUDIT = ROOT/'var/artifacts/capacity_audit_20260911/e122_countdown44/audit.json'
QUAL = AUDIT.with_name('a5000_qualification.json')
OWNER_PROOF = AUDIT.with_name('owner_route_test_only.json')
EVIDENCE = {AUDIT:'3f6955d27f34c899496d1bbdbca135117dcb68f2aede2f6d92ef399bf418f807',
            QUAL:'1728b05a434da833e4b45f9e3be7509034a8ab55c1014b0997c17d875fee7965',
            OWNER_PROOF:'2d39d6560ce342237ffa904c0fd8ca144101abdefc138174d755ee4ac4879e69'}
JOBS = {'31158682':'drgrpo','31158683':'replay_drgrpo','31158684':'maxrl'}
OLD_NODES = {'node205','node206','node207','node302'}
NEW_NODES = OLD_NODES | {'node105'}
POOL = ','.join(sorted(NEW_NODES))
OWNER_JOB = '31158682'
ROUTES = {jid:{'nodes':sorted({'node105'} if jid == OWNER_JOB else NEW_NODES),
    'Account':'mltheory' if jid == OWNER_JOB else 'allcs',
    'Partition':'mltheory' if jid == OWNER_JOB else 'lowprio','QOS':'medium'} for jid in JOBS}
c = recovery.c
require = recovery.require


def node_health():
    record = c.command(['scontrol','show','node','-o','node105']).stdout
    require('gpu:a5000:' in c.field(record,'Gres'), 'Qualified node105 GPU changed')
    require(not any(x in c.field(record,'State') for x in ('DOWN','DRAIN','FAIL','MAINT','UNKNOWN')),
            'Qualified node105 no longer healthy')
    require({'lowprio','mltheory'} <= set(c.field(record,'Partitions').split(',')), 'Qualified partitions changed')
    return record


def audit(record, original, job_id, cell, *, after=False):
    require(c.field(record,'JobId') == job_id, 'Wrong exact job ID')
    require(prior.base.submit_tokens(record) == prior.base.submit_tokens(original), 'Original SubmitLine changed')
    for key in prior.PRESERVE:
        expected = ROUTES[job_id][key] if after and key in ('Account','Partition','QOS') else c.field(original,key)
        observed = c.field(record,key)
        if after and job_id == OWNER_JOB and key == 'ReqTRES':
            # Billing is derived from the owner partition; requested resources are exact.
            strip_billing = lambda text: sorted(x for x in text.split(',') if not x.startswith('billing='))
            require(strip_billing(observed) == strip_billing(expected), 'Requested TRES changed')
        else:
            require(observed == expected, 'Resource or identity changed: '+key)
    require(c.field(record,'NumNodes') in ('1','1-1'), 'One-node allocation changed')
    env = prior.base.exports(prior.base.submit_tokens(record))
    require(all(env.get(k) == v for k,v in cell['environment'].items()), 'Frozen scientific exports changed')
    require(c.field(record,'MinMemoryNode') == '128G' and c.field(record,'NumCPUs') == '8'
            and c.field(record,'TresPerNode') == 'gres/gpu:1'
            and c.field(record,'TimeLimit') == '1-12:00:00', 'Original resource profile changed')


def pending(record):
    return c.field(record,'JobState') == 'PENDING'


def require_released(record):
    require(c.field(record,'Priority') != '0'
            and c.field(record,'Reason') not in ('JobHeldUser','JobHeldAdmin'), 'Existing hold must be preserved')
    require(c.field(record,'RunTime') == '00:00:00' and c.field(record,'Restarts') == '0', 'Previously started job is outside amendment')


def verify_plan(expected):
    value = c.pinned_json(PLAN,expected)
    require(value['schema'] == 'e122_countdown44_node105_placement_v1'
            and value['job_ids'] == list(JOBS) and value['routes'] == ROUTES,
            'Exact placement plan differs')
    for name,digest in value['source_pins'].items():
        require(c.digest(Path(name)) == digest, 'Placement source/evidence changed: '+name)
    recovery.verify_plan()
    return value


def sole_writer(job_id, cell):
    supervisor = recovery.verify_plan()
    storage = recovery.importlib.import_module(supervisor['storage_module'])
    report = storage.storage_report()
    require(report['allowed'] and not report['errors'], 'Shared writer snapshot unresolved')
    matches = [x for x in report['writer_profiles'] if x['run_dir'] == cell['run_dir']]
    require(len(matches) == 1 and matches[0]['job_id'] == job_id,
            'Expected sole writer for exact run directory changed')
    return {'observed_at':report['observed_at'],'writer_identity_sha256':report['writer_identity_sha256'],
            'writer':matches[0],'required_bytes':report['required_bytes'],'margin_bytes':report['margin_bytes']}


def prepare(campaign):
    require(not PLAN.exists(), 'Placement plan already exists')
    supervisor = recovery.verify_plan()
    require(campaign['binding'] == supervisor['binding'], 'Original authenticated campaign differs')
    for path,expected in EVIDENCE.items():
        require(c.digest(path) == expected, 'Independent placement proof changed')
    status = recovery.ORIGINAL_STATUS(campaign,c.JOURNAL_ROOT)
    require(not status['issues'] and not status['unknown_job_ids'] and not status['needs_operator_review_job_ids'],
            'Original controller requires reconciliation')
    require(status['reserved_unfinished_slots'] == 4, 'Original cap4 reservations changed')
    rows = {}
    pins = {str(p):c.digest(p) for p in [SOURCE,TEST,PROTOCOL,Path(prior.__file__),Path(prior.base.__file__),
            recovery.PLAN,recovery.SOURCE,Path(c.__file__),Path(recovery.launcher.__file__),*EVIDENCE]}
    for jid,arm in JOBS.items():
        job = next(x for x in campaign['jobs'] if x['job_id'] == jid)
        require((job['cell']['domain'],job['cell']['arm'],job['cell']['seed']) == ('countdown',arm,44),
                'Wrong fixed cell identity')
        intent = c.JOURNAL_ROOT/'jobs'/f'{jid}.intent.json'
        result = intent.with_name(f'{jid}.result.json')
        old = recovery.read(intent); ack = recovery.read(result)
        require(ack['returncode'] == 0 and not ack['error'] and ack['intent_sha256'] == c.digest(intent),
                'Exact successful release ownership missing')
        record = prior.base.show(int(jid))
        audit(record,old['held_scheduler_record'],jid,job['cell'])
        require(pending(record) and prior.nodes(record) == OLD_NODES, 'Exact original pending route required')
        require_released(record)
        writer = sole_writer(jid,job['cell'])
        argv = ['scontrol','update','JobId='+jid,'ReqNodeList='+','.join(ROUTES[jid]['nodes'])]
        if jid == OWNER_JOB:
            argv += ['Account=mltheory','Partition=mltheory','QOS=medium']
        rows[jid] = {'cell':job['cell'],'original_record':old['held_scheduler_record'],'before':record,
                     'command':argv,'sole_writer_at_prepare':writer}
        pins.update({str(p):c.digest(p) for p in (intent,result)})
    value = {'schema':'e122_countdown44_node105_placement_v1','at_utc':c.now(),'job_ids':list(JOBS),
        'old_nodes':sorted(OLD_NODES),'routes':ROUTES,'rows':rows,'source_pins':pins,
        'campaign_binding':campaign['binding'],'node_health':node_health(),
        'preserved_resource_fields':list(prior.PRESERVE),'same_ids_same_writers':True,
        'science_resources_and_submit_lines_unchanged':True,'new_submissions_or_releases':False,
        'only_owner_job_changes_account_partition':OWNER_JOB,'qos_remains_medium':True,
        'derived_partition_billing_may_change_only_for_owner_job':True}
    c.immutable_json(PLAN,value)
    return value


def apply(expected):
    plan = verify_plan(expected)
    with (ART/'singleton.lock').open('a') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with recovery.admission_locks(), c.locked(c.JOURNAL_ROOT):
            require(not (ART/'apply.intent.json').exists(), 'Placement intent already consumed; never repeat')
            c.immutable_json(ART/'apply.intent.json',{'at_utc':c.now(),'plan_sha256':expected})
            outcomes = {}
            for jid,row in plan['rows'].items():
                verify_plan(expected)
                record = prior.base.show(int(jid))
                audit(record,row['original_record'],jid,row['cell'])
                require(prior.nodes(record) == OLD_NODES, 'Unowned placement change')
                if not pending(record):
                    outcomes[jid] = {'status':'preserved_started','record':record}; continue
                require_released(record); node_health()
                writer = sole_writer(jid,row['cell'])
                record = prior.base.show(int(jid))
                audit(record,row['original_record'],jid,row['cell'])
                require(prior.nodes(record) == OLD_NODES, 'Placement changed during final validation')
                if not pending(record):
                    outcomes[jid] = {'status':'preserved_started','record':record}; continue
                require_released(record)
                c.immutable_json(ART/f'{jid}.intent.json',{'at_utc':c.now(),'before':record,
                    'command':row['command'],'plan_sha256':expected,'sole_writer':writer})
                try:
                    result = c.command(row['command'])
                    ack = {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'error':None}
                except Exception as error:
                    ack = {'returncode':None,'error':type(error).__name__+': '+str(error)}
                c.immutable_json(ART/f'{jid}.result.json',{'at_utc':c.now(),**ack})
                require(ack['returncode'] == 0, 'Placement acknowledgement uncertain; never repeat')
                after = prior.base.show(int(jid))
                audit(after,row['original_record'],jid,row['cell'],after=True)
                require(prior.nodes(after) == set(ROUTES[jid]['nodes']), 'Acknowledged placement differs')
                outcomes[jid] = {'status':'expanded','record':after}
                c.immutable_json(ART/f'{jid}.readback.json',outcomes[jid])
            c.immutable_json(ART/'complete.json',{'at_utc':c.now(),'plan_sha256':expected,'outcomes':outcomes})
            return outcomes
