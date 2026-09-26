#!/usr/bin/env python3
"""Finish registered inference waves using the tested bounded launcher."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'src')]
from launch_modebench_fresh_concentration import (BASE,assert_ready,submit,load_execution_amendment,
                                                effective_indices,assert_runtime_hardware)
from restore_modebench_fresh_concentration import atomic_new,digest
from analyze_modebench_fresh_panel import build_report
from analyze_modebench_fresh_concentration import write_artifacts

def cycle():
    pp=BASE/'plan.json';plan=json.loads(pp.read_text())
    receipts=[json.loads(p.read_text()) for p in (BASE/'slurm'/'submissions').glob('*.json')]
    if any(r['plan_sha256']!=digest(pp) for r in receipts):raise ValueError('submission plan changed')
    amendment=load_execution_amendment(BASE,plan,receipts)
    submitted={i for r in receipts for i in effective_indices(r,amendment)}
    complete=set()
    for i,t in enumerate(plan['tasks']):
        folder=Path(plan['output_root'])/t['task_id'];p=folder/'result.json'
        if p.exists():
            r=json.loads(p.read_text());run=json.loads((folder/'run.json').read_text())
            if r.get('status')!='complete' or r.get('task_id')!=t['task_id'] or run['identity']['plan_sha256']!=digest(pp):
                raise ValueError('completed task is not bound to this plan')
            complete.add(i)
    ids=','.join(r['job_id'] for r in receipts)
    active=subprocess.run(['squeue','-h','-r','-j',ids,'-o','%i %T'],capture_output=True,text=True,check=True).stdout.strip().splitlines() if ids else []
    state={'at_utc':datetime.now(timezone.utc).isoformat(),'registered':len(plan['tasks']),
           'submitted':len(submitted),'complete_receipts':len(complete),'active_scheduler_tasks':len(active),
           'running':sum(line.split()[-1]=='RUNNING' for line in active),'phase':'collecting'}
    if sum(line.split()[-1] in ('RUNNING','COMPLETING','CONFIGURING') for line in active)>8:
        raise ValueError('campaign concurrency limit exceeded')
    if not active:
        missing=submitted-complete
        if missing:raise ValueError(f'submitted tasks ended without completion; inspect exact job logs: {sorted(missing)}')
        if len(complete)==len(plan['tasks']):
            hardware={'plan_sha256':digest(pp),'tasks':[assert_runtime_hardware(t,amendment,plan['output_root'])
                                                       for t in plan['tasks']],
                      'execution_amendment':amendment['reference'] if amendment else None}
            hardware_path=BASE/'audits/final_runtime_hardware.json'
            hardware_path.parent.mkdir(exist_ok=True)
            if hardware_path.exists():
                if json.loads(hardware_path.read_text())!=hardware:raise ValueError('final hardware audit changed')
            else:atomic_new(hardware_path,hardware)
            output=BASE/'fresh_panel'
            if output.exists():
                report=json.loads((output/'report.json').read_text())
                audit=report['completeness_audit']
                if (report.get('status')!='complete' or audit['plan_source']['sha256']!=digest(pp)
                        or audit.get('status')!='complete' or audit.get('authenticated_tasks')!=150
                        or audit.get('authenticated_response_slots')!=1228800):
                    raise ValueError('existing final analysis differs')
                manifest=json.loads((output/'manifest.json').read_text())
                if manifest.get('status')!='complete' or manifest['source_report_sha256']!=digest(output/'report.json'):
                    raise ValueError('published report manifest differs')
                for name,expected in manifest['files'].items():
                    if Path(name).is_absolute() or '..' in Path(name).parts or digest(output/name)!=expected:
                        raise ValueError('published analysis artifact differs')
            else:
                report=build_report(pp)
                write_artifacts(report,output)
            state.update(phase='analysis_complete',report=str(output/'report.json'))
            atomic_new(BASE/'collection_and_analysis_complete.json',{
                'plan_sha256':digest(pp),'tasks':150,'response_slots':1228800,
                'report':str(output/'report.json'),'report_sha256':digest(output/'report.json'),
                'runtime_hardware_audit':{'path':str(hardware_path),'sha256':digest(hardware_path)},
                'scientific_interpretation_pending':True})
        else:
            ready=[]
            for i,t in enumerate(plan['tasks']):
                if i in submitted:continue
                try:assert_ready(t)
                except ValueError as error:
                    if not str(error).startswith('checkpoint is not restored:'):
                        raise
                    state.setdefault('waiting_checkpoint_ids',[]).append(t['task_id'])
                    continue
                ready.append(i)
            if ready:
                falcon=[i for i in ready if plan['tasks'][i]['model_scale']=='falcon1b']
                qwen=[i for i in ready if plan['tasks'][i]['model_scale']!='falcon1b']
                if amendment and not {50,51}<=complete:
                    if {50,51}<=set(falcon):selected,phase=[50,51],'falcon_interface_validation'
                    elif qwen:selected,phase=qwen,'full_panel'
                    else:selected,phase=[],None
                elif falcon:selected,phase=falcon,'full_panel'
                else:selected,phase=qwen,'full_panel'
                if selected:
                    receipt=submit(selected,phase)
                    state.update(new_job_id=receipt['job_id'],newly_submitted=len(selected))
                else:state['phase']='waiting_for_falcon_interface_inputs'
            else:state['phase']='waiting_for_checkpoint_restoration'
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        state=cycle();print(json.dumps(state,sort_keys=True),flush=True)
        if a.once or state['phase']=='analysis_complete':break
        time.sleep(45)
