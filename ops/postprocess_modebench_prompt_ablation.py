#!/usr/bin/env python3
"""Resume isolated offline grading as frozen local/hosted cohorts complete.

No model or network calls. The scientific analyzer is a separate immutable
snapshot; this controller only schedules complete cohort postprocessing.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)


def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--scope',choices=('local','hosted','full'),default='local')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.workers not in (1,2) or not 1<=args.poll_seconds<=60:parser.error('Use 1–2 workers and 1–60 second polls')
    base=args.base.resolve()
    frozen=json.loads((base/'analysis_source_manifest.json').read_text())
    for source in frozen['files'].values():
        if file_sha(source['path'])!=source['sha256']:raise ValueError('Frozen analysis source changed')
    analyzer=Path(frozen['files']['analyzer']['path'])
    local_plan=base/'local/plan_v2.json'
    plan=json.loads(local_plan.read_text())
    tasks=[]
    if args.scope in ('local','full'):
        for checkpoint in plan['checkpoints']:
            directory=Path(plan['output_root'])/checkpoint['label']
            tasks.append({'label':checkpoint['label'],'family':'local','directory':directory,
                          'ready':directory/'result.json','graded':directory/'prompt_ablation_normalization_audit.json',
                          'command':[sys.executable,str(analyzer),'--base',str(base),'--local-plan',str(local_plan),
                                     '--grade-local',checkpoint['label']]})
    if args.scope in ('hosted','full'):
        registry=json.loads((base/'hosted_analysis_runs.json').read_text())
        for entry in registry['runs']:
            directory=Path(entry['run_dir'])
            tasks.append({'label':entry['model_id']+'_'+entry['arm'],'family':'frontier','directory':directory,
                          'ready':directory/'status.json','graded':directory/'prompt_ablation_grading_audit.json',
                          'command':[sys.executable,str(analyzer),'--base',str(base),'--grade-hosted',str(directory)]})
    output=(args.output or base/('analysis_complete' if args.scope=='full' else 'analysis_'+args.scope+'_complete')).resolve()
    logs=base/'analysis_logs';logs.mkdir(exist_ok=True)
    active,done,failed={},set(),{}
    report_process=None
    def ready(task):
        if not task['ready'].exists():return False
        record=json.loads(task['ready'].read_text())
        return record.get('status')=='complete' if task['family']=='local' else record.get('complete') is True
    with (base/('.postprocess_'+args.scope+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            for label,(process,handle,task) in list(active.items()):
                code=process.poll()
                if code is None:continue
                handle.close();del active[label]
                if code==0 and task['graded'].exists():done.add(label)
                else:failed[label]={'exit_code':code,'log':str(logs/(label+'.log'))}
            for task in tasks:
                label=task['label']
                if label in done or label in failed or label in active:continue
                if not ready(task) or len(active)>=args.workers:continue
                # The analyzer authenticates an existing sidecar before reusing it.
                handle=(logs/(label+'.log')).open('a')
                process=subprocess.Popen(task['command'],stdout=handle,stderr=subprocess.STDOUT,cwd=base)
                active[label]=(process,handle,task)
                print(json.dumps({'at_utc':now(),'event':'grading_started','label':label,'pid':process.pid}),flush=True)
            state={'at_utc':now(),'scope':args.scope,'controller_pid':os.getpid(),'api_calls':0,
                   'registered_cohorts':len(tasks),'complete_cohorts':sum(ready(t) for t in tasks),
                   'graded_cohorts':len(done),'active':list(active),'failed':failed,
                   'analyzer':frozen['files']['analyzer'],'output':str(output)}
            if len(done)==len(tasks) and report_process is None:
                handle=(logs/('final_'+args.scope+'.log')).open('a')
                command=[sys.executable,str(analyzer),'--base',str(base),'--scope',args.scope,'--output',str(output)]
                report_process=(subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,cwd=base),handle)
                state['report_pid']=report_process[0].pid
                print(json.dumps({'at_utc':now(),'event':'final_report_started','pid':report_process[0].pid}),flush=True)
            if report_process:
                code=report_process[0].poll()
                state['report_exit_code']=code
                if code is not None:
                    report_process[1].close()
                    state['status']='complete' if code==0 else 'failed'
                    atomic(base/('postprocessing_'+args.scope+'_status.json'),state)
                    print(json.dumps(state),flush=True)
                    raise SystemExit(code)
            state['status']='failed' if failed else 'running' if active or report_process else 'waiting'
            atomic(base/('postprocessing_'+args.scope+'_status.json'),state)
            if failed and not active:raise SystemExit(1)
            if not args.watch and not active and not report_process:
                print(json.dumps(state),flush=True)
                return
            time.sleep(args.poll_seconds)


if __name__=='__main__':main()
