"""Scratch-only operational recovery tests; no real scheduler, model or grader."""
import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('scratch_node_failure_recovery',
        ROOT/'artifacts/recover_modebench_scale_revised_node_failure_20260912.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


@pytest.fixture
def c(tmp_path, monkeypatch):
    r = load(); base = tmp_path/'scratch'; base.mkdir()
    def file(name, value='scratch only'):
        p = base/name; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value) if isinstance(value,(dict,list)) else value)
        return p
    for key,relative in {'ROOT':'.','SOURCE':'artifacts/helper.py','PYTHON':'venv/bin/python',
        'VIEW_RUNNER':'artifacts/runner.py','VIEW_HELPER':'artifacts/old_recovery.py',
        'TRANSPORT':'artifacts/transport.py','COMPOSITE':'composite','ORIGINAL_PLAN':'composite/dev/plan.json',
        'AUTHORITY':'authority','AUTHORITY_TERMINAL':'authority/terminal.json','INCIDENT':'incident/snapshot.json',
        'CONTROLLER_LOCK':'composite/controller.lock','RECOVERY_ROOT':'recovery','CLAIM':'composite/claim.json',
        'ACTION_LOCK':'composite/action.lock'}.items():
        monkeypatch.setattr(r,key,base/relative)
    for name in ('artifacts/helper.py','artifacts/runner.py','artifacts/old_recovery.py','artifacts/transport.py',
                 'ops/repo_env.sh','composite/controller.lock','python-target'):
        file(name)
    r.PYTHON.parent.mkdir(parents=True); r.PYTHON.symlink_to(base/'python-target')
    monkeypatch.setattr(r,'CONTROLLER_INODE',r.CONTROLLER_LOCK.stat().st_ino)
    manifest = file('view/manifest.json', {'runner':{'path':str(r.VIEW_RUNNER)},
        'proot':{'path':str(file('proot'))},'preserved_source':{'path':str(file('preserved.py'))},
        'mappings':[{'source':str(file('mapped.py'))}]})
    file('view/manifest.sha256')
    cells = []
    for i in range(7):
        tasks = [{'output':str(base/f'outputs/{i}/difficulty_{t}.json')} for t in range(4)]
        tasks_path = file(f'composite/dev/tasks/{i}.json',tasks)
        command = [str(r.PYTHON),str(base/'ops/evaluator.py'),'--tasks-json',str(tasks_path),'--resume']
        cells.append({'id':r.CELL_IDS[i-5] if i>=5 else f'completed_{i}', 'level':'level5',
            'model_label':'14b','phase':'dev','source_kind':'domain_revision_v1',
            'tasks':str(tasks_path),'command':command})
    plan = {'cells':cells,'immutable_inputs_sha256':{str(r.PYTHON):r.sha(r.PYTHON)}}
    r.ORIGINAL_PLAN.write_text(json.dumps(plan)); monkeypatch.setattr(r,'ORIGINAL_PLAN_SHA',r.sha(r.ORIGINAL_PLAN))
    for name in ('plan.sha256.json','submission_intent.json','submission_result.json',
                 'frozen_transport/plan.json','frozen_transport/plan.sha256.json',
                 'frozen_transport/submission_intent.json','frozen_transport/submission_result.json'):
        file('composite/dev/'+name,{'scratch':True})
    file('composite/dev/submission_result.json',{'array_job_id':r.ORIGINAL_ARRAY,'status':'submitted'})
    for i in range(6):
        file(f'composite/dev/runtime/{i}.json', {'array_job_id':r.ORIGINAL_ARRAY,'cell':cells[i]['id'],
            'plan_sha256':r.ORIGINAL_PLAN_SHA,'command':cells[i]['command']})
        file(f'composite/dev/frozen_transport/runtime/{i}.json', {'array_job_id':r.ORIGINAL_ARRAY,'array_index':i,
            'stage_plan_sha256':r.ORIGINAL_PLAN_SHA,'evaluator_command':cells[i]['command']})
        for suffix in ('.out','.err'): file(f'composite/dev/logs/{r.ORIGINAL_ARRAY}_{i}'+suffix)
    for tier,count in enumerate([116,116,116,61]):
        file(f'outputs/5/difficulty_{tier}.json.batches/run.json',{'identity':'scratch'})
        for k in range(count): file(f'outputs/5/difficulty_{tier}.json.batches/seed-1__rows-{k:06d}.json',{'draw':k})
        if tier<3: file(f'outputs/5/difficulty_{tier}.json',{'complete':True})
    terminal = {'at_utc':'2026-09-12T13:29:23.067411+00:00','last_sweep':266,
        'result':{'jobs':[{'job':'31254520_5','state':'NODE_FAIL'}],'status':'execution_failed'}}
    file('authority/terminal.json',terminal)
    old_rows = []
    for i in range(7):
        old_rows.append([str(800+i) if i<6 else str(r.ORIGINAL_ARRAY),f'{r.ORIGINAL_ARRAY}_{i}',
            'NODE_FAIL' if i==5 else 'FAILED' if i<5 else 'PENDING', '143:0' if i==2 else '1:0' if i<6 else '0:0',
            '2026-09-12T10:00:00' if i<6 else 'Unknown','2026-09-12T13:28:44' if i<6 else 'Unknown',
            'node105' if i<6 else 'None assigned','6' if i<6 else '0','60G',
            'cpu=6,mem=60G,node=1,gres/gpu:a5000=2' if i<6 else '', '2026-09-12T02:52:21'])
    data = SimpleNamespace(r=r,base=base,root=r.RECOVERY_ROOT,manifest=manifest,file=file,plan=plan,
        old_rows=old_rows,phase='pending',priority=10,calls=[],snapshot_error=False,
        mutation_error=None,mutation_nonzero=None,stdout='900\n',hold_race=False)
    def observation(command,stdout):
        return {'command':command,'environment':{'TZ':'UTC'},'returncode':0,'stdout':stdout,'stderr':'',
            'at_utc':'2026-09-12T16:00:00+00:00','observer_host':'spin.cs.princeton.edu'}
    data.observation = observation
    incident = {'schema':'modebench_scale_node105_incident_snapshot_v1','observer_host':'spin.cs.princeton.edu',
        'wash_pid_os_state':'unverified_from_spin','original_stage_plan':{'path':str(r.ORIGINAL_PLAN),'sha256':r.ORIGINAL_PLAN_SHA},
        'prior_authority_terminal':{'path':str(r.AUTHORITY_TERMINAL),'sha256':r.sha(r.AUTHORITY_TERMINAL),'value':terminal},
        'saved_outputs_sha256':r.inventory(plan,initial=True)['files_sha256'],
        'observations':[observation(['sacct','-j',str(r.ORIGINAL_ARRAY),'-n','-P','--format='+r.FIELDS],
            '\n'.join('|'.join(row) for row in old_rows))]}
    file('incident/snapshot.json',incident); monkeypatch.setattr(r,'INCIDENT_SHA',r.sha(r.INCIDENT))
    data.incident = incident
    def scheduler(command):
        if data.snapshot_error: raise ValueError('scratch scheduler unavailable')
        if command[0]=='squeue':
            out = '' if data.phase=='terminal' else f'{r.ORIGINAL_ARRAY}_6|'+('RUNNING' if data.phase=='running' else 'PENDING')+'\n'
        elif command[0]=='sacct' and '--name='+getattr(data,'job_name','missing') in command:
            out = data.recovery_accounting
        elif command[0]=='sacct':
            assert '--array' in command
            rows=copy.deepcopy(data.old_rows)
            if data.phase=='terminal': rows[6][2]='CANCELLED by 1000'; rows[6][5]='2026-09-12T16:00:00'
            if data.phase=='running': rows[6][2]='RUNNING'; rows[6][4]='2026-09-12T16:00:00'; rows[6][6]='node105'; rows[6][7]='6'
            out='\n'.join('|'.join(row) for row in rows)+'\n'
        elif command==['scontrol','show','job','31254520_6','-o']:
            out=f'JobId=31254520 ArrayJobId=31254520 ArrayTaskId=6 JobState=PENDING Priority={data.priority}\n'
        else: pytest.fail('unexpected scheduler read '+str(command))
        return observation(command,out)
    monkeypatch.setattr(r,'run_read',scheduler)
    data.real_original=r.original
    monkeypatch.setattr(r,'original',lambda:(plan,{'inputs_sha256':{str(r.TRANSPORT):r.sha(r.TRANSPORT)}}))
    monkeypatch.setattr(r,'frozen_view',lambda path:r.read(path))
    def guarded_mutation(command,**kwargs):
        data.calls.append(command)
        assert kwargs.get('close_fds') is True
        assert (r.RECOVERY_ROOT/(('hold' if command[0]=='scontrol' else 'cancel' if command[0]=='scancel' else 'submission')+'_intent.json')).exists()
        if data.mutation_error==command[0]: raise subprocess.TimeoutExpired(command,60)
        if data.mutation_nonzero==command[0]: return SimpleNamespace(returncode=1,stdout='',stderr='scratch failure')
        if command[0]=='scontrol':
            data.priority=0
            if data.hold_race: data.phase='running'
        elif command[0]=='scancel': data.phase='terminal'
        elif command[0]!='sbatch': pytest.fail('unexpected process '+str(command))
        return SimpleNamespace(returncode=0,stdout=data.stdout if command[0]=='sbatch' else '',stderr='')
    monkeypatch.setattr(r.subprocess,'run',guarded_mutation)
    return data


def prepare(c):
    c.r.prepare(c.root,c.manifest,c.r.INCIDENT,c.r.INCIDENT_SHA)
    value=c.r.read(c.root/'plan.json'); c.job_name=value['job_name']
    return value


def cancel(c):
    prepare(c); c.r.cancel_pending(c.root)


def submit(c):
    cancel(c); return c.r.submit(c.root)


def rewrite(path,value):
    path.chmod(0o644); path.write_text(json.dumps(value))


def test_prepare_preserves_partial_outputs_and_actual_interpreter_symlink(c):
    before={str(p):p.read_bytes() for p in (c.base/'outputs').rglob('*') if p.is_file()}
    plan=prepare(c)
    assert c.r.verify(c.root)['original_indices']==[5,6]
    assert len(plan['preserved_outputs_sha256'])==416
    assert all(Path(p).read_bytes()==value for p,value in before.items())
    assert plan['input_symlink_targets']=={str(c.r.PYTHON):str(c.base/'python-target')}
    assert plan['cells'][0]['command'][0]==str(c.r.PYTHON)
    assert c.calls==[]


def test_exact_operational_amendment_and_separate_original_mapping(c):
    p=prepare(c); command=p['submit_command']
    assert '--partition=all' in command and '--account=mltheory' in command
    assert '--array=0-1%1' in command and '--dependency=afterany:31254520' in command
    assert '--gres=gpu:a5000:2' in command and '--mem=60G' in command and '--cpus-per-task=6' in command
    assert '--no-requeue' in command and not any(v.startswith('--nodelist') for v in command)
    assert [x['original_index'] for x in p['cells']]==[5,6]
    assert p['hardware']['allowed_worker_nodes']==['node105','node202','node203','node204']
    assert str(c.manifest) in (c.root/'worker.slurm').read_text()


def test_no_second_prepare_or_noncanonical_root(c):
    prepare(c)
    with pytest.raises(ValueError,match='already prepared'): c.r.prepare(c.root,c.manifest,c.r.INCIDENT,c.r.INCIDENT_SHA)
    with pytest.raises(ValueError,match='fixed'): c.r.verify(c.root/'alternate')


@pytest.mark.parametrize('path_kind',['batch','runtime','incident','terminal','helper','interpreter'])
def test_changed_input_refuses_before_scheduler_mutation(c,path_kind):
    prepare(c)
    p={'batch':next((c.base/'outputs').rglob('seed-*')),'runtime':c.r.ORIGINAL_PLAN.parent/'runtime/5.json',
       'incident':c.r.INCIDENT,'terminal':c.r.AUTHORITY_TERMINAL,'helper':c.r.SOURCE,'interpreter':c.base/'python-target'}[path_kind]
    p.chmod(0o644); p.write_text('changed')
    with pytest.raises(ValueError): c.r.cancel_pending(c.root)
    assert c.calls==[]


def test_same_content_interpreter_link_retarget_is_rejected(c):
    prepare(c); c.file('another-python','scratch only'); c.r.PYTHON.unlink(); c.r.PYTHON.symlink_to(c.base/'another-python')
    with pytest.raises(ValueError,match='symlink target'): c.r.verify(c.root)


def test_incident_saved_output_change_refuses_preparation(c):
    next((c.base/'outputs').rglob('seed-*')).write_text('changed')
    with pytest.raises(ValueError,match='pinned input changed'): prepare(c)
    assert not c.r.CLAIM.exists() and not c.root.exists()


def test_initial_inventory_rejects_new_batch_or_python_manifest(c):
    c.file('outputs/6/difficulty_0.json.batches/run.json',{})
    with pytest.raises(ValueError,match='output evidence'): prepare(c)
    assert c.calls==[]


def test_actual_expanded_pending_raw_id_is_retained(c):
    observed=c.r.scheduler_snapshot(pending=True)
    assert observed['rows_by_original_index']['6'][:2]==['31254520','31254520_6']
    assert '--array' in observed['sacct']['command']


@pytest.mark.parametrize('mutation', ['running','other_failure_changed','missing_row','wrong_raw_id'])
def test_original_state_gate_rejects_changed_accounting(c,mutation):
    if mutation=='running': c.phase='running'
    elif mutation=='other_failure_changed': c.old_rows[2][3]='1:0'
    elif mutation=='missing_row': c.old_rows.pop(0)
    else: c.old_rows[5][0]='999'
    with pytest.raises(ValueError): c.r.scheduler_snapshot(pending=True)


def test_only_proven_unstarted_cell_6_is_held_then_cancelled(c):
    cancel(c)
    assert c.calls==[['scontrol','hold','31254520_6'],['scancel','31254520_6']]
    proof=c.r.cancellation_gate(c.root)
    assert proof['status']=='cancelled_only_unstarted_original_cell_6'
    assert proof['terminal']['rows_by_original_index']['6'][4]=='Unknown'
    assert not (c.r.ORIGINAL_PLAN.parent/'runtime/6.json').exists()


def test_start_race_after_hold_never_cancels_started_worker(c):
    prepare(c); c.hold_race=True
    with pytest.raises(ValueError): c.r.cancel_pending(c.root)
    assert c.calls==[['scontrol','hold','31254520_6']]
    assert not (c.root/'cancel_intent.json').exists()


def test_ambiguous_hold_can_be_reconciled_without_repeating_hold(c):
    prepare(c); c.mutation_error='scontrol'
    with pytest.raises(ValueError,match='ambiguous'): c.r.cancel_pending(c.root)
    c.mutation_error=None; c.priority=0
    c.r.reconcile_hold(c.root)
    c.r.cancel_pending(c.root)
    assert c.calls==[['scontrol','hold','31254520_6'],['scancel','31254520_6']]
    c.r.cancellation_gate(c.root)


def test_hold_reconciliation_refuses_unheld_or_started_job(c):
    prepare(c); c.mutation_error='scontrol'
    with pytest.raises(ValueError): c.r.cancel_pending(c.root)
    with pytest.raises(ValueError,match='not authenticated'): c.r.reconcile_hold(c.root)
    c.phase='running'; c.priority=0
    with pytest.raises(ValueError): c.r.reconcile_hold(c.root)
    assert len(c.calls)==1


def test_ambiguous_cancel_reconciles_only_actual_terminal_and_never_recancels(c):
    prepare(c); c.mutation_error='scancel'
    with pytest.raises(ValueError,match='ambiguous'): c.r.cancel_pending(c.root)
    with pytest.raises(ValueError): c.r.reconcile_cancellation(c.root)
    c.phase='terminal'; c.r.reconcile_cancellation(c.root)
    c.r.cancellation_gate(c.root)
    with pytest.raises(ValueError): c.r.cancel_pending(c.root)
    assert c.calls==[['scontrol','hold','31254520_6'],['scancel','31254520_6']]


def test_submit_waits_for_original_terminal_and_is_exactly_once(c):
    cancel(c); c.phase='pending'
    with pytest.raises(ValueError,match='fully terminal'): c.r.submit(c.root)
    assert not (c.root/'submission_intent.json').exists()
    c.phase='terminal'; value=c.r.submit(c.root)
    assert value['array_job_id']==900
    assert c.r.submission_identity(c.root)==value
    with pytest.raises(ValueError,match='already attempted'): c.r.submit(c.root)
    assert sum(cmd[0]=='sbatch' for cmd in c.calls)==1


@pytest.mark.parametrize('outcome',['timeout','nonzero','malformed'])
def test_ambiguous_or_failed_submission_never_retries(c,outcome):
    cancel(c)
    if outcome=='timeout': c.mutation_error='sbatch'
    elif outcome=='nonzero': c.mutation_nonzero='sbatch'
    else: c.stdout='maybe submitted'
    with pytest.raises(ValueError): c.r.submit(c.root)
    with pytest.raises(ValueError,match='already attempted'): c.r.submit(c.root)
    assert sum(cmd[0]=='sbatch' for cmd in c.calls)==1
    assert not (c.root/'submission_identity.json').exists()


def accounting(c,job=900):
    value=c.r.read(c.root/'plan.json')
    return '\n'.join('|'.join([str(job+i),f'{job}_{i}',value['job_name'],value['recovery_token'],
        value['user'],'mltheory','all','cpu=6,mem=60G,node=1,gres/gpu:a5000=2','480']) for i in range(2))+'\n'


def test_reconciled_submission_identity_rechecks_actual_accounting(c):
    cancel(c); c.mutation_error='sbatch'
    with pytest.raises(ValueError): c.r.submit(c.root)
    c.recovery_accounting=accounting(c)
    c.r.reconcile_submission(c.root)
    assert c.r.submission_identity(c.root)['array_job_id']==900
    value=c.r.read(c.root/'submission_identity.json'); value['array_job_id']=901
    rewrite(c.root/'submission_identity.json',value)
    with pytest.raises(ValueError,match='differs from actual accounting'): c.r.submission_identity(c.root)
    assert sum(cmd[0]=='sbatch' for cmd in c.calls)==1


@pytest.mark.parametrize('change',['missing_cell','wrong_account','wrong_partition','wrong_gpu','wrong_cpu','wrong_mem',
                                     'wrong_token','wrong_time','duplicate_cell','different_array'])
def test_recovery_accounting_requires_exact_two_cell_identity_and_resources(c,change):
    cancel(c); c.mutation_error='sbatch'
    with pytest.raises(ValueError): c.r.submit(c.root)
    out=accounting(c)
    replacements={'wrong_account':('mltheory','other'),'wrong_partition':('|all|','|cs|'),
        'wrong_gpu':('a5000','a100'),'wrong_cpu':('cpu=6','cpu=4'),'wrong_mem':('60G','59G'),
        'wrong_token':(c.r.read(c.root/'plan.json')['recovery_token'],'different-token'),
        'wrong_time':('|480','|60'),'duplicate_cell':('900_1','900_0'),'different_array':('900_1','901_1')}
    if change=='missing_cell': out=out.splitlines()[0]+'\n'
    else: out=out.replace(*replacements[change])
    c.recovery_accounting=out
    with pytest.raises(ValueError): c.r.reconcile_submission(c.root)
    assert not (c.root/'submission_identity.json').exists()


def setup_worker_env(c,monkeypatch,index=0,host='node202.ionic.cs.princeton.edu',gpu='NVIDIA RTX A5000'):
    for k,v in c.r.ENVIRONMENT.items(): monkeypatch.setenv(k,v)
    for k,v in {'SLURM_ARRAY_JOB_ID':'900','SLURM_ARRAY_TASK_ID':str(index),'SLURM_JOB_ID':str(900+index),
        'SLURMD_NODENAME':host.split('.')[0],'SLURM_CPUS_PER_TASK':'6','SLURM_MEM_PER_NODE':'61440',
        'SLURM_JOB_ACCOUNT':'mltheory','SLURM_JOB_PARTITION':'all'}.items(): monkeypatch.setenv(k,v)
    monkeypatch.setattr(c.r.socket,'gethostname',lambda:host)
    monkeypatch.setattr(importlib.metadata,'version',lambda package:'0.8.4')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,
        get_device_name=lambda i:gpu)))


@pytest.mark.parametrize('host',['node105.ionic.cs.princeton.edu','node202.ionic.cs.princeton.edu',
                                  'node203.ionic.cs.princeton.edu','node204.ionic.cs.princeton.edu'])
def test_worker_uses_real_approved_host_and_same_a5000(c,monkeypatch,host):
    setup_worker_env(c,monkeypatch,host=host)
    value=c.r.worker_environment({'array_job_id':900},0)
    assert value['hostname']==host and value['environment']['SLURM_JOB_ID']=='900'


@pytest.mark.parametrize('key,value',[('SLURM_ARRAY_JOB_ID','31254520'),('SLURM_ARRAY_TASK_ID','6'),
    ('SLURM_JOB_ACCOUNT','other'),('SLURM_JOB_PARTITION','mltheory'),('SLURM_MEM_PER_NODE','60416'),
    ('SLURM_CPUS_PER_TASK','4'),('VLLM_USE_V1','1'),('VLLM_ATTENTION_BACKEND','FLASH_ATTN'),
    ('SLURMD_NODENAME','node200')])
def test_worker_rejects_wrong_allocation_backend_and_identity(c,monkeypatch,key,value):
    setup_worker_env(c,monkeypatch); monkeypatch.setenv(key,value)
    with pytest.raises(ValueError): c.r.worker_environment({'array_job_id':900},0)


@pytest.mark.parametrize('gpu,host',[('NVIDIA A100','node202.ionic.cs.princeton.edu'),
                                    ('NVIDIA RTX A5000','node200.ionic.cs.princeton.edu')])
def test_worker_rejects_different_gpu_type_or_unregistered_node(c,monkeypatch,gpu,host):
    setup_worker_env(c,monkeypatch,gpu=gpu,host=host)
    with pytest.raises(ValueError): c.r.worker_environment({'array_job_id':900},0)


def test_worker_records_separate_truthful_runtime_and_exact_argv_once(c,monkeypatch):
    submit(c); setup_worker_env(c,monkeypatch)
    seen=[]
    monkeypatch.setattr(c.r.os,'execv',lambda executable,argv:seen.append((executable,argv)))
    c.r.worker(c.root,0)
    runtime=c.r.read(c.root/'runtime/0.json')
    assert runtime['array_job_id']==900 and runtime['original_index']==5 and runtime['recovery_index']==0
    assert runtime['hostname']=='node202.ionic.cs.princeton.edu'
    assert runtime['outputs_before_exec']['batches_by_original_index']['5']==[116,116,116,61]
    assert seen==[(c.plan['cells'][5]['command'][0],c.plan['cells'][5]['command'])]
    assert c.r.read(c.r.ORIGINAL_PLAN.parent/'runtime/5.json')['array_job_id']==31254520
    with pytest.raises(ValueError,match='already attempted'): c.r.worker(c.root,0)
    assert len(seen)==1


@pytest.mark.parametrize('index',[0,1])
def test_worker_refuses_unregistered_new_output_before_first_exec(c,monkeypatch,index):
    submit(c); setup_worker_env(c,monkeypatch,index=index)
    c.file(f'outputs/{5+index}/difficulty_3.json.batches/new.json',{'unexpected':True})
    monkeypatch.setattr(c.r.os,'execv',lambda *args:pytest.fail('unexpected evaluator exec'))
    with pytest.raises(ValueError,match='unexpected output'): c.r.worker(c.root,index)
    assert not (c.root/f'runtime/{index}.json').exists()


def test_other_recovery_cell_growth_does_not_impose_a_scientific_dependency(c,monkeypatch):
    submit(c); setup_worker_env(c,monkeypatch,index=1)
    c.file('outputs/5/difficulty_3.json.batches/additional_registered_cell_output.json',{})
    seen=[]; monkeypatch.setattr(c.r.os,'execv',lambda *args:seen.append(args))
    c.r.worker(c.root,1)
    assert len(seen)==1 and c.r.read(c.root/'runtime/1.json')['original_index']==6
    assert not (c.r.ORIGINAL_PLAN.parent/'runtime/6.json').exists()


def test_original_authentication_uses_historical_transport_without_host_spoofing(c,monkeypatch):
    transport=SimpleNamespace(LAUNCHER=Path('scratch-launcher'),LAUNCHER_SHA='scratch',
        module=lambda *args:SimpleNamespace(), verify_transport=lambda launcher,path:(c.plan,{}))
    monkeypatch.setattr(c.r,'module',lambda *args:transport)
    value,_=c.real_original()
    assert value==c.plan
    c.file('composite/dev/runtime/6.json',{})
    with pytest.raises(ValueError,match='cell 6 has started'): c.real_original()


@pytest.mark.parametrize('field,value',[('observer_host','wash.cs.princeton.edu'),
                                       ('wash_pid_os_state','unverified'),
                                       ('wash_pid_os_state','dead')])
def test_incident_retains_exact_truthful_observer_and_unverified_remote_pid(c,monkeypatch,field,value):
    changed=copy.deepcopy(c.incident); changed[field]=value
    rewrite(c.r.INCIDENT,changed); monkeypatch.setattr(c.r,'INCIDENT_SHA',c.r.sha(c.r.INCIDENT))
    with pytest.raises(ValueError,match='incident identity'): c.r.incident_capture()


def test_controller_fence_refuses_a_live_other_owner(c):
    import fcntl
    with c.r.CONTROLLER_LOCK.open('r+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): prepare(c)
    assert not c.r.CLAIM.exists() and not c.root.exists() and c.calls==[]


def test_changed_controller_lock_inode_refuses_preparation(c):
    c.r.CONTROLLER_LOCK.rename(c.r.CONTROLLER_LOCK.with_suffix('.old'))
    c.r.CONTROLLER_LOCK.write_text('new inode')
    with pytest.raises(ValueError,match='fence identity'): prepare(c)
    assert not c.r.CLAIM.exists() and c.calls==[]


def test_nonzero_hold_does_not_cancel_or_implicitly_retry(c):
    prepare(c); c.mutation_nonzero='scontrol'
    with pytest.raises(ValueError,match='hold failed'): c.r.cancel_pending(c.root)
    with pytest.raises((ValueError,FileNotFoundError)): c.r.cancel_pending(c.root)
    assert c.calls==[['scontrol','hold','31254520_6']]


def test_worker_verification_preserves_every_original_incident_output(c,monkeypatch):
    extra=c.file('outputs/0/finished.json',{'original':'must remain'})
    snapshot=c.r.read(c.r.INCIDENT); snapshot['saved_outputs_sha256'][str(extra)]=c.r.sha(extra)
    rewrite(c.r.INCIDENT,snapshot); monkeypatch.setattr(c.r,'INCIDENT_SHA',c.r.sha(c.r.INCIDENT))
    submit(c); extra.write_text('changed original completed output')
    with pytest.raises(ValueError,match='pinned input changed'): c.r.verify(c.root,require_initial=False)


def test_cancel_intent_without_recorded_outcome_reconciles_actual_terminal_without_retry(c):
    prepare(c); c.mutation_error='scancel'
    with pytest.raises(ValueError): c.r.cancel_pending(c.root)
    (c.root/'cancel_ambiguous.json').unlink()  # Simulate process loss before the outcome write.
    c.phase='terminal'; proof=c.r.reconcile_cancellation(c.root)
    assert proof['cancellation_outcome_evidence']=='missing_after_durable_intent'
    c.r.cancellation_gate(c.root)
    assert c.calls==[['scontrol','hold','31254520_6'],['scancel','31254520_6']]


def test_cancel_reconciliation_rejects_conflicting_result_and_ambiguity(c):
    prepare(c); c.mutation_error='scancel'
    with pytest.raises(ValueError): c.r.cancel_pending(c.root)
    c.file('recovery/cancel_result.json',{'contradictory':True})
    c.phase='terminal'
    with pytest.raises(ValueError,match='contradictory'): c.r.reconcile_cancellation(c.root)
    assert not (c.root/'cancellation.json').exists()
