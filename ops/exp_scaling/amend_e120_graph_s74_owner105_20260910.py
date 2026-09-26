#!/usr/bin/env python3
"""Audited pending-only E120 Graph seed74 route to the existing node105 pool member."""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

import accelerate_a5000_completion_20260909 as parent
import widen_a5000_completion_20260909 as widening

b = parent.base
ROOT = parent.ROOT
ART = ROOT / 'var/artifacts/e120_graph_s74_owner105_20260910'
PLAN, TX = ART / 'plan.json', ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e120_graph_s74_owner105_20260910.md'
JOB = 31158507
OLD_NODES = {'node105', 'node202', 'node203', 'node204'}
PRESERVE = ('UserId', *widening.PRESERVE, 'Restarts', 'Features')
LEDGERS = (parent.campaign.E120_LEDGER, parent.CONTINUATIONS)


def bounded(parts, *, check=True):
    return subprocess.run(parts, capture_output=True, text=True, check=check, timeout=45)


b.command = bounded


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def node_health():
    record = b.command(['scontrol', 'show', 'node', '-o', 'node105']).stdout
    require(not any(s in b.field(record, 'State') for s in ('DOWN', 'DRAIN', 'FAIL')), 'node105 unhealthy')
    require(b.field(record, 'Gres') == 'gpu:a5000:10', 'node105 GPU resource changed')
    require('lowprio' in b.field(record, 'Partitions').split(','), 'node105 pool unavailable')
    return record


def node_set(record):
    return set(b.command(['scontrol', 'show', 'hostnames', b.field(record, 'ReqNodeList')]).stdout.split())


def stable(plan, record):
    require(b.field(record, 'JobId') == str(JOB), 'job identity changed')
    require(b.submit_tokens(record) == plan['submit_tokens'], 'submission or scientific exports changed')
    for key, value in plan['resources'].items():
        require(b.field(record, key) == value, f'resource changed: {key}')
    require(b.field(record, 'NumNodes') in ('1', '1-1'), 'node count changed')


def science(plan):
    require(all(parent.digest(p) == plan['ledger_sha256'][str(p)] for p in LEDGERS), 'E120 ledger changed')
    require(parent.campaign.e120_continuation_jobs(LEDGERS[0]).get(31033709) == JOB, 'authoritative writer changed')
    require(parent.runtime_fingerprints([{'original_command':plan['submit_tokens']}]) == plan['runtime_fingerprints'], 'runtime snapshot changed')
    parent.safe_run(plan['item'], [JOB])
    require(not parent.incoming(JOB), 'job acquired a dependent')


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'preparation already exists')
    record = b.show(JOB)
    require(b.field(record, 'JobState') == 'PENDING' and b.field(record, 'Priority') != '0', 'job must be pending and unheld')
    require(b.field(record, 'RunTime') == '00:00:00' and node_set(record) == OLD_NODES, 'pending route changed')
    tokens = b.submit_tokens(record); env = b.exports(tokens)
    require(env['OAT_ZERO_AUTO_RESUME'] == '1', 'automatic resume unavailable')
    for key,value in {'Account':'mltheory','Partition':'lowprio','MinMemoryNode':'128G','NumCPUs':'16',
                      'TimeLimit':'3-00:00:00','TresPerNode':'gres/gpu:a5000:1','Nice':'100','Requeue':'1'}.items():
        require(b.field(record,key)==value, f'unexpected {key}')
    hourly = json.loads((ROOT/'var/artifacts/preemption_hourly_timeout_guard_20260909/transaction.json').read_text())
    require(hourly['jobs']['31159786']['status'] == 'fallback_complete', 'hourly fallback unresolved')
    require(datetime.now(timezone.utc) > datetime.fromisoformat(hourly['cleanup_deadline_utc']), 'hourly guard deadline still active')
    queue = b.queue(); require(31159786 not in queue and 31048518 not in queue, 'predecessor remains queued')
    item={'old_job_id':JOB,'run_dir':env['SAVE_PATH'],'checkpoint':parent.checkpoint(env['SAVE_PATH'])}
    require(0 < item['checkpoint']['step'] < 3072, 'missing unfinished resume checkpoint')
    plan={'schema':'e120-graph-s74-owner105-20260910-v1','created_at_utc':b.now(), 'before':record,
          'item':item,'submit_tokens':tokens,'resources':{k:b.field(record,k) for k in PRESERVE},
          'ledger_sha256':{str(p):parent.digest(p) for p in LEDGERS},
          'runtime_fingerprints':parent.runtime_fingerprints([{'original_command':tokens}]),
          'controller_sha256':parent.digest(__file__),'protocol_sha256':parent.digest(PROTOCOL),
          'healthy_node':node_health(),'hourly_fallback_status':'fallback_complete',
          'accounting':b.command(['sacct','-n','-P','-D','-X','-j',str(JOB),'-S','2026-09-09',
                                 '-o','JobID,State,ExitCode,Start,End,NodeList,Elapsed']).stdout,
          'scheduler_only':True,'scientific_exports_changed':False,'ledgers_written':False}
    science(plan); b.atomic(PLAN,plan)
    print(json.dumps({'prepared':str(PLAN),'job':JOB,'checkpoint':item['checkpoint']['step'],'scheduler_mutations':False}))


def apply():
    plan=json.loads(PLAN.read_text())
    require(parent.digest(__file__)==plan['controller_sha256'] and parent.digest(PROTOCOL)==plan['protocol_sha256'], 'prepared implementation changed')
    if TX.exists():
        tx=json.loads(TX.read_text())
        require(tx.get('status')=='complete', 'existing transaction requires reconciliation; do not repeat mutation')
        print(json.dumps({'already_complete':True}));return
    tx={'plan_sha256':parent.digest(PLAN),'events':[],'status':'applying'}
    def event(message):
        tx['events'].append({'at_utc':b.now(),'event':message});b.atomic(TX,tx)
    record=b.show(JOB);stable(plan,record);science(plan);node_health()
    if b.field(record,'JobState')!='PENDING':
        tx['status']='skipped_started';event('Allocation started; left unchanged');return
    require(b.field(record,'Priority')!='0' and node_set(record)==OLD_NODES, 'preexisting hold or route change')
    tx['hold_intent']=True;event('Persisted pending-only owned hold intent')
    b.command(['scontrol','hold',str(JOB)])
    record=b.show(JOB);stable(plan,record)
    if b.field(record,'JobState')!='PENDING':
        tx['release_intent']=True;event('Start raced owned hold; releasing unchanged allocation')
        b.command(['scontrol','release',str(JOB)])
        tx['status']='skipped_started';tx['final']=b.show(JOB);event('Running allocation preserved');return
    require(b.field(record,'Reason')=='JobHeldUser' and b.field(record,'Priority')=='0','owned hold not observed')
    require(node_set(record)==OLD_NODES,'route changed before owned update');science(plan);node_health()
    tx['held_before']=record;tx['node_update_intent']=True;event('Changing only ReqNodeList to node105')
    b.command(['scontrol','update',f'JobId={JOB}','NodeList=node105'])
    record=b.show(JOB);stable(plan,record)
    require(b.field(record,'JobState')=='PENDING' and b.field(record,'Reason')=='JobHeldUser' and node_set(record)=={'node105'},'held route amendment differs')
    science(plan);node_health();tx['held_after']=record
    tx['release_intent']=True;event('Validated unchanged science/checkpoint/resources/ledgers; releasing owned hold')
    b.command(['scontrol','release',str(JOB)])
    record=b.show(JOB);stable(plan,record)
    require(b.field(record,'JobState') in ('PENDING','RUNNING','CONFIGURING') and b.field(record,'Priority')!='0' and node_set(record)=={'node105'},'route release differs')
    require(all(parent.digest(p)==plan['ledger_sha256'][str(p)] for p in LEDGERS),'E120 ledger changed')
    tx['final']=record;tx['status']='complete';event('Same E120 job eligible only on node105; no job or ledger created')
    print(json.dumps({'status':'complete','job':JOB,'nodes':['node105'],'state':b.field(record,'JobState'),'reason':b.field(record,'Reason')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','apply']);args=parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for name in ('e118_ledger_promotion.lock','e120_ledger_promotion.lock'):
            handle=stack.enter_context((ROOT/'var/artifacts'/name).open('a+'));fcntl.flock(handle,fcntl.LOCK_EX)
        globals()[args.phase]()
