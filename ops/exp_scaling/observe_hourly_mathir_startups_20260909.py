#!/usr/bin/env python3
"""Read-only successor preserving six-cell startup coverage and approved fallbacks."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import observe_a5000_startups_20260909 as monitor

ROOT=monitor.ROOT
ART=ROOT/'var/artifacts/campaign_hourly_startup_observer_20260909'
PLAN=ART/'plan.json'
STATE=ART/'latest.json'
PROTOCOL=ROOT/'paper/preregistration/campaign_hourly_startup_observer_20260909.md'
DEPLOYMENT=ROOT/'var/artifacts/campaign_mathir_hourly_backfill_20260909/transaction.json'
OLD_ART=ROOT/'var/artifacts/campaign_startup_observer_20260909'
base=monitor.base


def prepare():
    assert not PLAN.exists() and not STATE.exists()
    prior=json.loads((OLD_ART/'plan.json').read_text()); prior_state=json.loads((OLD_ART/'latest.json').read_text())
    deployment=json.loads(DEPLOYMENT.read_text());replacements={r['old_job_id']:r for r in deployment['items']}
    assert set(replacements)=={31158503,31158504}
    rows=[]
    for old in prior['rows']:
        row=dict(old)
        if old['job_id'] in replacements:
            item=replacements[old['job_id']]
            assert old['run_dir']==item['run_dir'] and isinstance(item['new_job_id'],int)
            row.update(job_id=item['new_job_id'],checkpoint_step=item['checkpoint']['step'],expected_exports=base.exports(item['command']),
                       launcher=item['command'][-1],launcher_sha256=monitor.digest(item['command'][-1]),
                       dormant_fallback_job_id=item['old_job_id'],fallback_exports=base.exports(item['long_route_command']))
        rows.append(row)
    assert len(rows)==6 and len({r['job_id'] for r in rows})==6
    plan={**prior,'rows':rows,'prepared_at_utc':base.now(),'schema':'campaign-hourly-startup-observer-20260909-v1',
          'controller_sha256':monitor.digest(monitor.__file__),'wrapper_sha256':monitor.digest(__file__),
          'helper_sha256':monitor.digest(base.__file__),'protocol_sha256':monitor.digest(PROTOCOL),
          'prior_plan_sha256':monitor.digest(OLD_ART/'plan.json'),'approved_deployment':str(DEPLOYMENT),
          'approved_deployment_controller_sha256':monitor.digest(ROOT/'ops/exp_scaling/backfill_mathir_hourly_20260909.py')}
    base.atomic(PLAN,plan)
    state={'schema':plan['schema'],'plan_sha256':monitor.digest(PLAN),
           'started_at_utc':prior_state['started_at_utc'],'deadline_utc':prior_state['deadline_utc'],
           'jobs':{key:value for key,value in prior_state['jobs'].items() if int(key) not in replacements}}
    base.atomic(STATE,state)
    print(json.dumps({'prepared':True,'job_ids':[r['job_id'] for r in rows],'deadline_utc':state['deadline_utc'],
                      'other_four_coverage_preserved':True}),flush=True)


def run(watch):
    plan=json.loads(PLAN.read_text())
    assert monitor.digest(__file__)==plan['wrapper_sha256']
    assert monitor.digest(ROOT/'ops/exp_scaling/backfill_mathir_hourly_20260909.py')==plan['approved_deployment_controller_sha256']
    monitor.ART=ART;monitor.PLAN=PLAN;monitor.STATE=STATE;monitor.PROTOCOL=PROTOCOL
    original=monitor.observe
    def observe(item,previous):
        effective=dict(item)
        if 'dormant_fallback_job_id' in item:
            tx=json.loads(DEPLOYMENT.read_text())
            deployed=next(r for r in tx['items'] if r['new_job_id']==item['job_id'])
            assert deployed['old_job_id']==item['dormant_fallback_job_id'] and deployed['run_dir']==item['run_dir']
            if deployed.get('fallback',{}).get('status')=='complete':
                effective.update(job_id=item['dormant_fallback_job_id'],expected_exports=item['fallback_exports'])
        result=original(effective,previous)
        result['registered_hourly_job_id']=item['job_id']
        result['following_approved_long_fallback']=effective['job_id']!=item['job_id']
        return result
    monitor.observe=observe
    monitor.run(watch)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','once','watch'));args=parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/'observer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare() if args.phase=='prepare' else run(args.phase=='watch')
