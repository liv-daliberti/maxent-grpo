#!/usr/bin/env python3
"""Continue the single registered neutral calibration through fresh admission.

Own one observer lock and the campaign's mutation lock. Never retry a failed
scientific gate, an ambiguous submission, or a partially materialized dataset.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='b2efff5f9cab38de3d4110626b87ca30a3fab651c83ebacf669bb0a52db50fdf'


def advance(c,expected):
    """Called while holding the campaign lock; tolerate another owner finishing a step."""
    c.validate_plan(expected)
    if (c.ART/'retirement_result.json').exists():
        return {'status':'retired_incomplete_development_diagnostic','terminal':True}
    completed=[Path(c.task(t)['output']).is_file() for t in range(4)]
    if not all(completed):
        return {'status':'waiting_development','complete_tiers':sum(completed)}
    recipe_path=c.ART/'recipe.json'
    recipe=c.read(recipe_path) if recipe_path.exists() else c.fit(expected)
    c.require(recipe['registration_sha256']==expected,'recipe registration mismatch')
    c.common.verify_pins(recipe['receipts_sha256'])
    if recipe['development_fit_pass'] is not True:
        return {'status':'development_outside_tolerance','terminal':True,'gates':recipe['gates']}
    if not (c.DATA/'identity.json').exists():
        c.require(not c.DATA.exists(),'partial dataset materialization; preserve for inspection')
        c.finalize(expected)
    c.common.verify_pins(c.read(c.DATA/'identity.json')['files_sha256'])
    confirmation=Path(c.task()['output'])
    if not confirmation.exists():
        receipt=c.ART/'confirmation_submission_result.json'
        if not receipt.exists():
            c.require(not (c.ART/'confirmation_submission_intent.json').exists(),
                      'ambiguous confirmation submission; preserve for inspection')
            job=c.submit_once('confirmation',c.gpu_command('confirmation',expected))
        else:
            submitted=c.read(receipt)
            c.require(submitted['returncode']==0,'failed confirmation submission; preserve for inspection')
            job=submitted['stdout'].strip().split(';')[0]
        return {'status':'waiting_confirmation','job_id':job,'dataset':str(c.DATA)}
    report=c.ART/'confirmation_report.json'
    result=c.read(report) if report.exists() else c.audit(expected)
    c.require(result['registration_sha256']==expected,'confirmation registration mismatch')
    if result['admitted']:
        c.require((c.ART/'admission.json').is_file(),'passing report is missing its admission record')
    return {'status':result['status'],'admitted':result['admitted'],'terminal':True,
            'report':str(report),'dataset':str(c.DATA),'neutral_level3':result['neutral_level3'],
            'deltas':result['deltas'],'gates':result['gates']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-sha256',default=EXPECTED)
    parser.add_argument('--deadline-hours',type=float,default=2.8)
    args=parser.parse_args()
    sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
    import calibrate_modebench_level3_neutral_v2 as c
    c.require(args.registration_sha256==EXPECTED,'unexpected calibration registration')
    registration=c.read(c.ART/'automatic_continuation_registration.json')
    c.require(registration['watcher_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'continuation implementation changed after registration')
    status_path=c.ART/'automatic_continuation_status.json'
    started=time.monotonic()
    with (c.ART/'automatic_continuation.lock').open('a') as observer:
        fcntl.flock(observer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                with (c.ART/'controller.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    status=advance(c,args.registration_sha256)
            except BlockingIOError:
                status={'status':'waiting_for_active_controller'}
            except Exception as error:
                status={'status':'requires_inspection','terminal':True,'error':str(error),'error_type':type(error).__name__}
            if time.monotonic()-started > args.deadline_hours*3600 and not status.get('terminal'):
                status={**status,'status':'continuation_deadline','terminal':True}
            status.update(updated_at=datetime.now(timezone.utc).isoformat(),registration_sha256=EXPECTED)
            tmp=status_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(status,indent=2,sort_keys=True)+'\n');tmp.replace(status_path)
            print(json.dumps(status,sort_keys=True),flush=True)
            if status.get('terminal'):return
            time.sleep(30)

if __name__=='__main__':main()
