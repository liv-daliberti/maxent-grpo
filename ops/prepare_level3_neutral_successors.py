"""Prepare fresh runtime/cell configurations for the unstarted Python L3 jobs.

Old snapshots, plans, ledgers and scheduler jobs remain immutable. Commands are
held submissions for an explicit version migration, never automatic releases.
"""
from pathlib import Path
from copy import deepcopy
import importlib.util,json,shutil,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'ops/exp_scaling'),str(ROOT/'src')]
from followup_metrics import atomic_new,file_sha,sha
from modebench_current_contract import CURRENT,training_environment,make_messages
BASE=ROOT/'artifacts/modebench_level3_neutral_default_20260911'


def new_cell(cell,campaign,runtime):
    result={k:deepcopy(cell[k]) for k in ('domain','arm','seed','target_steps','resources')};result['level']=3
    result['run_stamp']=cell['run_stamp']+'_neutral_v1'
    result['run_dir']=str(ROOT/'var/data'/('modebench_'+CURRENT)/campaign/cell['arm']/('s'+str(cell['seed'])))
    env=training_environment(3,'python_factors',cell['environment'])
    env.update(SAVE_PATH=result['run_dir'],RUN_STAMP=result['run_stamp'],OAT_ZERO_SOURCE_ROOT=str(runtime/'src'),OAT_ZERO_OPS_SNAPSHOT_ROOT=str(runtime/'ops'))
    result['environment']=dict(sorted(env.items()));result['environment_sha256']=sha(result['environment']);result['prompt_condition']=CURRENT;result['source_cell_sha256']=sha(cell)
    result['source_difficulty_calibration_applies']=False;result['previous_run_may_be_resumed']=False
    return result


def main():
    registration=json.loads((BASE/'registration.json').read_text())
    for p,h in registration['source_sha256'].items():assert file_sha(p)==h
    records=[]
    for campaign,path in [('e122',ROOT/'var/artifacts/e122_level3_factorial_plan.json'),('e124',ROOT/'var/artifacts/e124_qwen7b_three_level/plan.json')]:
        plan=json.loads(path.read_text());oldroot=Path(plan['snapshot']['root']);snap=BASE/'runtime'/campaign
        for relative,info in plan['snapshot']['inventory'].items():assert file_sha(oldroot/relative)==info['sha256']
        snap.parent.mkdir(exist_ok=True)
        if not snap.exists():shutil.copytree(oldroot,snap)
        snap.chmod(snap.stat().st_mode | 0o200)
        system=registration['system_prompt']
        add='\n\n# Prospective prompt amendment '+CURRENT+'; historical renderers remain unchanged.\n'
        add+='def _apply_python_level3_neutral_v1(question):\n    return render_chat_prompt("qwen", '+repr(system)+', question)\n'
        for alias in ('qwen_level3_python_factors','qwen_level3_python_factors_neutral_v1'):
            add+='TEMPLATE_FACTORY['+repr(alias)+'] = _apply_python_level3_neutral_v1\nPROMPT_TEMPLATE_ROLES['+repr(alias)+'] = "boxed"\n'
        target=snap/'src/oat_drgrpo/templates.py'
        amendment=snap/'PROMPT_AMENDMENT_IDENTITY.json'
        if amendment.exists():
            prior=json.loads(amendment.read_text());assert prior['parent_plan_sha256']==file_sha(path)
            for relative,h in prior['inventory_sha256'].items():assert file_sha(snap/relative)==h
        else:
            for relative,info in plan['snapshot']['inventory'].items():assert file_sha(snap/relative)==info['sha256']
            target.chmod(target.stat().st_mode | 0o200)
            target.write_bytes(target.read_bytes()+add.encode())
        spec=importlib.util.spec_from_file_location('_successor_templates_'+campaign,target);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for r in registration['exact_existing_neutral_prompts']:
            m=r['messages'];sm,um,am=mod.CHAT_SURFACES['qwen'];assert mod.TEMPLATE_FACTORY['qwen_level3_python_factors_neutral_v1'](m[1]['content'])==sm+m[0]['content']+um+m[1]['content']+am
        inventory={str(p.relative_to(snap)):file_sha(p) for p in snap.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        if not amendment.exists():atomic_new(amendment,{'schema':'modebench-runtime-prompt-amendment-v1','condition':CURRENT,'parent_snapshot':str(oldroot),'parent_plan_sha256':file_sha(path),'only_changed_parent_file':'src/oat_drgrpo/templates.py','inventory_sha256':inventory,'note':'Any copied historical snapshot manifest identifies the parent snapshot; this amendment manifest identifies the successor.'})
        cells=[]
        for c in plan['cells']:
            if c['domain']!='python_factors' or c.get('level',3)!=3:continue
            new=new_cell(c,campaign,snap)
            if campaign=='e122':command=list(c['command'])
            else:
                from launch_e124_qwen7b_three_level import job_command
                command=job_command(plan,c)
            commands=[]
            for part in command:
                if part.startswith('--export='):part='--export=ALL,'+','.join(k+'='+v for k,v in new['environment'].items())
                elif part.startswith('--job-name='):part='--job-name='+new['run_stamp']
                elif part.startswith('--comment='):part='--comment='+CURRENT+':'+new['environment_sha256'][:16]
                part=part.replace(str(oldroot),str(snap));commands.append(part)
            assert '--hold' in commands and commands[-1]==str(snap/'ops/slurm/train_node302.slurm')
            new['held_submission_command']=commands;cells.append(new)
        records.append({'campaign':campaign,'source_plan':str(path),'source_plan_sha256':file_sha(path),'runtime':str(snap),'runtime_amendment_sha256':file_sha(snap/'PROMPT_AMENDMENT_IDENTITY.json'),'cells':cells})
    atomic_new(BASE/'successor_plans.json',{'schema':'modebench-python-l3-neutral-successors-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'condition':CURRENT,'status':'prepared_not_submitted','campaigns':records,'source_sha256':file_sha(__file__),'old_held_job_ids':[r['job_id'] for r in registration['existing_jobs_pending_version_migration']],'new_training_jobs':0,'original_registrations_unchanged':True,'required_queue_transition':'Replace the held legacy jobs and update their source-bound release controllers as one explicit migration; do not release these as an additional parallel training cohort.'})
    print(json.dumps({'status':'prepared','successor_cells':sum(len(r['cells']) for r in records),'new_training_jobs':0}))

if __name__=='__main__':main()
