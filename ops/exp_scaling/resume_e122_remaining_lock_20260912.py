#!/usr/bin/env python3
"""Resume only the proven pre-admission-lock failure; never repeat a release."""
from pathlib import Path
from contextlib import ExitStack
import importlib.util
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
p=ROOT/'ops/exp_scaling/expand_e122_remaining_20260912.py'
s=importlib.util.spec_from_file_location('remaining_e122',p);w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
m=w.m
SHA='9eafeb77a45f46df2d801222b34575f50cdff1afc15c67ee8b9779fe0a1a9b04'
plan=m.validate(SHA)
claim=m.read(m.ART/'claim.json');m.require(claim['plan_sha256']==SHA,'claim differs')
log=ROOT/'paper/audits/paper_refresh_20260912/e122_remaining_release.log'
m.require('BlockingIOError: [Errno 11]' in log.read_text() and 'with old.admission_locks()' in log.read_text(),'failure not the documented pre-lock failure')
m.require(not (m.ART/'result.json').exists() and not list(m.ART.glob('release_*.json')) and not list((m.ART/'routes').glob('*/intent.json')),'some action already occurred')
m.setup(plan['max_unfinished_slots'],plan['candidate_job_ids'])
with ExitStack() as stack:
    for attempt in range(30):
        try:
            stack.enter_context(m.old.admission_locks());break
        except BlockingIOError:
            if attempt==29:raise
            time.sleep(1)
    stack.enter_context(m.c.locked(m.c.JOURNAL_ROOT))
    campaign=m.migration.merged_campaign(m.MIGRATION_SHA)
    jobs={j['job_id']:j for j in campaign['jobs']}
    proof={}
    for jid in plan['candidate_job_ids']:
        for root in [m.c.JOURNAL_ROOT,m.migration.NEW_JOURNAL]:
            m.require(not (root/'jobs'/f'{jid}.intent.json').exists(),'candidate has existing durable release intent')
        raw=m.old.launcher.audit_held(int(jid),jobs[jid]['cell'])
        m.require(m.c.field(raw,'RunTime')=='00:00:00' and m.c.field(raw,'Restarts')=='0','candidate has execution history')
        proof[jid]=raw
    m.new(m.ART/'lock_failure_reconciliation.json',{'at':m.c.now(),'plan_sha256':SHA,'preserved_claim_sha256':m.digest(m.ART/'claim.json'),'failure_log_sha256':m.digest(log),'source_sha256':m.digest(Path(__file__)),'exact_held_no_intent_proof':proof,'jobs_previously_released':0})
    released=[]
    for i in range(plan['max_additional_releases']):
        m.validate(SHA)
        result=m.c.advance_once(m.old.fresh_args(),m.old.launcher,root=m.migration.NEW_JOURNAL)
        m.new(m.ART/f'release_{i}.json',result)
        jid=result.get('last_release_job_id')
        if not jid:break
        released.append(jid)
        print(json.dumps({'released':jid,'count':len(released)}),flush=True)
        m.require(not result['issues'] and not result['needs_operator_review_job_ids'],'journal needs review')
        m.route_pending(jid,jobs[jid]['cell'])
    final=m.c.status(m.migration.merged_campaign(m.MIGRATION_SHA),m.migration.NEW_JOURNAL)
    m.new(m.ART/'result.json',{'status':'finite_expansion_complete','released_job_ids':released,'count':len(released),'final_status':final,'persistent_cap':4,'plan_sha256':SHA,'reconciled_only_prelock_failure':True})
print(json.dumps({'released_job_ids':released,'persistent_cap':4}),flush=True)
