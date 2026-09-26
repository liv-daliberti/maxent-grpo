#!/usr/bin/env python3
"""Complete only the ledger/commit tail after delayed cancellation accounting.

Never submit jobs, cancel jobs, retry inference, or relax admission/resource gates.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import migrate_level3_neutral_v5 as m
import watch_level3_neutral_calibration_v5_r2_20260911 as w
SOURCE=Path(__file__).resolve()


def main():
 expected=m.digest(m.PLAN);plan=m.validate(expected)
 m.require(not m.COMMIT.exists(),'migration already committed')
 intent=m.read(m.ART/'cancellation_intent.json');m.require(intent['plan_sha256']==expected,'cancellation ownership mismatch')
 m.new(m.ART/'accounting_reconciliation_intent.json',{'plan_sha256':expected,'code_sha256':m.digest(SOURCE),'reason':'Immediate post-cancellation accounting was incomplete; complete the original commit tail using existing replacement receipts and fresh scheduler evidence only.','jobs_submitted':0,'jobs_cancelled':0,'created_at':m.calibration.now()})
 m.old.compat.install_compat(m.old.launcher,m.old.COMPAT_SHA,m.old.COMPAT_TEST_SHA)
 with m.old.admission_locks(),m.c.locked(m.c.JOURNAL_ROOT):
  replacements=[]
  for i,row in enumerate(plan['rows']):
   result=m.read(m.ART/f'submission_{i:02d}_result.json');audit=m.read(m.ART/f'submission_{i:02d}_audit.json')
   jid=result['stdout'].strip().split(';')[0]
   m.require(result['returncode']==0 and result['command']==row['cell']['command'] and jid==str(audit['job_id']) and audit['cell']==row['cell'],'replacement receipt binding mismatch')
   raw=m.held_without_training(jid,row['cell'],row['campaign'])
   replacements.append({**row,'job_id':jid,'held_record':raw})
  m.require(len(replacements)==22 and [r['job_id'] for r in replacements]==intent['new_held_job_ids']
    and [r['old_job_id'] for r in replacements]==intent['old_job_ids'],'replacement union differs from cancellation intent')
  accounts=m.run(['sacct','-X','-n','-P','-j',','.join(intent['old_job_ids']),'-o','JobIDRaw,State,ElapsedRaw'])
  records={p[0]:p[1:] for line in accounts.splitlines() if (p:=line.split('|')) and p[0] in set(intent['old_job_ids'])}
  m.require(len(records)==22 and all(v[0].startswith('CANCELLED') and v[1]=='0' for v in records.values()),'old cancellations are still unconfirmed')
  m.require(not any((m.ART/name).exists() for name in ['cancellation_result.json','e122_jobs.json','e124_transaction.json']),'partial commit tail requires separate inspection')
  m.new(m.ART/'cancellation_result.json',{'sacct':accounts,'old_jobs_cancelled_without_runtime':22,'reconciled_after_accounting_delay':True})
  ledger=deepcopy(m.read(ROOT/'var/artifacts/e122_level3_factorial_jobs.json'))
  mapping={r['old_job_id']:r for r in replacements if r['campaign']=='e122'}
  for i,row in enumerate(ledger['runs']):
   replacement=mapping.get(str(row['job_id']))
   if replacement:
    ledger['runs'][i]={**row,**{k:replacement['cell'][k] for k in m.c.CELL_FIELDS},'job_id':int(replacement['job_id']),'held_scheduler_record':replacement['held_record']}
  ledger.update(schema='e122_neutral_migration_current_jobs_v1',migration_plan_sha256=expected)
  m.new(m.ART/'e122_jobs.json',ledger)
  tx=m.read(m.ART/'e124_original_transaction.json');tx['prompt_dataset_migration_sha256']=expected
  for r in replacements:
   if r['campaign']=='e124':
    cell_id=r['cell']['cell_id'];tx['rows'][cell_id]={**tx['rows'][cell_id],'job_id':int(r['job_id']),'command':r['cell']['command'],'status':'held','superseded_job_id':int(r['old_job_id'])}
  m.new(m.ART/'e124_transaction.json',tx)
  m.new(m.COMMIT,{'plan_sha256':expected,'replacements':replacements,'created_at':m.calibration.now(),'old_jobs_cancelled':22,'new_jobs_held':22,'original_registrations_preserved':True,'accounting_reconciliation_sha256':m.digest(m.ART/'accounting_reconciliation_intent.json')})
 status=w.finish_migration(m.calibration,w.advance(m.calibration,w.EXPECTED),m)
 status.update(updated_at=m.calibration.now(),registration_sha256=w.EXPECTED,accounting_reconciliation=True)
 m.new(m.calibration.ART/'parallel_continuation_result.json',status)
 p=m.calibration.ART/'automatic_continuation_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(p)
 m.new(m.ART/'accounting_reconciliation_result.json',{'status':'committed','old_jobs_cancelled_without_runtime':22,'new_jobs_verified_held':22,'new_submissions':0,'new_cancellations':0,'commit_sha256':m.digest(m.COMMIT)})
 print(json.dumps(status,sort_keys=True),flush=True)
if __name__=='__main__':main()
