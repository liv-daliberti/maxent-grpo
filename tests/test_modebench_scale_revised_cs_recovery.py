"""Synthetic scratch retirement/submit/worker tests; no real scheduler or grading."""
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

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_cs_scratch_base_tests',ROOT/'tests/test_modebench_scale_revised_node_failure_recovery.py')
previous=importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
c=previous.c


def load():
    spec=importlib.util.spec_from_file_location('_scratch_cs_recovery',ROOT/'artifacts/recover_modebench_scale_revised_cs_20260912.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def rewrite(path,value):
    path.chmod(0o644); path.write_text(json.dumps(value))


@pytest.fixture
def d(c,monkeypatch):
    r=load(); b=c.r
    for key,value in {'ROOT':c.base,'SOURCE':c.file('artifacts/cs.py'),'BASE':b.SOURCE,
        'OLD_ADAPTER':c.file('artifacts/adapter.py'),'OLD':c.root,'RECOVERY':c.base/'cs',
        'RETIREMENT':c.base/'retirement/retirement.json','CLAIM':c.base/'composite/cs_claim.json'}.items():
        monkeypatch.setattr(r,key,value)
    baseline=b.inventory(c.plan,initial=True)
    old_inputs={str(b.PYTHON):b.sha(b.PYTHON),str(b.SOURCE):b.sha(b.SOURCE)}
    old={'view_manifest':str(c.manifest),'original_indices':[5,6],'inputs_sha256':old_inputs,
         'user':r.pwd.getpwuid(os.getuid()).pw_name}
    c.file('recovery/plan.json',old)
    for relative in ('plan.sha256.json','worker.slurm','submission_intent.json','submission_result.json','submission_identity.json',
        'none_start_transport/plan.json','none_start_transport/plan.sha256.json','none_start_transport/worker.slurm',
        'none_start_transport/submission_intent.json','none_start_transport/submission_result.json'):
        c.file('recovery/'+relative,{'synthetic':True})
    c.file('recovery/preserved_inventory.json',baseline)
    monkeypatch.setattr(r,'OLD_PLAN_SHA',r.sha(r.OLD/'plan.json'))
    context=SimpleNamespace(base=b,adapter=SimpleNamespace(),proof={'transport':{'inputs_sha256':{str(r.OLD_ADAPTER):r.sha(r.OLD_ADAPTER)}},
        'submission':{'array_job_id':r.OLD_JOB}},old_plan=old,original=c.plan)
    monkeypatch.setattr(r,'historical_context',lambda:context)
    data=SimpleNamespace(r=r,b=b,c=c,context=context,root=r.RECOVERY,calls=[],error=None,returncode=0,stdout='900;synthetic\n',
                         phase='CANCELLED',accounting='',queue_extra='',execs=[])
    def obs(command,stdout,at='2026-09-12T16:49:10+00:00'):
        return {'command':command,'environment':{'TZ':'UTC'},'returncode':0,'stdout':stdout,'stderr':'',
                'at_utc':at,'observer_host':'synthetic-spin'}
    data.obs=obs
    def rows(state):
        return [[str(r.OLD_JOB),f'{r.OLD_JOB}_{i}',state,'0:0','None' if state=='CANCELLED' else 'Unknown',
            '2026-09-12T16:49:00' if state=='CANCELLED' else 'Unknown','None assigned','0','60G','',
            '2026-09-12T16:06:25','mltheory','mltheory'] for i in (0,1)]
    data.rows=rows
    def snapshot(at,held=False):
        account=obs(r.accounting_command(),'\n'.join('|'.join(row) for row in rows('PENDING'))+'\n',at)
        queue=obs(['squeue','-u',old['user'],'-r','-h','-o','%i|%T'],f'{r.OLD_JOB}_0|PENDING\n{r.OLD_JOB}_1|PENDING\n',at)
        controls=[obs(['scontrol','show','job','-o',f'{r.OLD_JOB}_{i}'],
            f'ArrayJobId={r.OLD_JOB} ArrayTaskId={i} JobState=PENDING Account=mltheory Partition=mltheory NodeList= '
            f'RunTime=00:00:00 StartTime=Unknown AllocTRES=(null) Restarts=0 Requeue=0 Priority={0 if held else 99} '
            f'Command={r.OLD}/none_start_transport/worker.slurm\n',at) for i in (0,1)]
        return {'captured_at_utc':at,'observer_host':'synthetic-spin','terminal_accounting':account,'queue_observation':queue,
            'control_observations':controls,'retired_cells':rows('PENDING'),'absent_runtime_paths':r.absent_runtime_paths(),
            'output_inventory':baseline}
    historical={str(p):r.sha(p) for p in r.OLD.rglob('*') if p.is_file()}
    historical.update({str(p):r.sha(p) for p in (r.BASE,b.INCIDENT,r.OLD_ADAPTER)})
    historical.update(b.read(b.INCIDENT)['saved_outputs_sha256'])
    pending=snapshot('2026-09-12T16:47:00+00:00'); pending['historical_execution_files_sha256']=historical
    held=snapshot('2026-09-12T16:48:00+00:00',held=True)
    c.file('retirement/pending.json',pending);c.file('retirement/held.json',held);c.file('retirement/routing_policy.json',{'synthetic':True})
    for label,command,before,times in (
        ('hold',['scontrol','hold',str(r.OLD_JOB)],{k:v for k,v in pending.items() if k!='historical_execution_files_sha256'},
         ('2026-09-12T16:47:01+00:00','2026-09-12T16:47:02+00:00')),
        ('cancel',['scancel',str(r.OLD_JOB)],held,('2026-09-12T16:48:59+00:00','2026-09-12T16:49:00+00:00'))):
        intent={'command':command,'environment':{'TZ':'UTC'},'retired_array_job_id':r.OLD_JOB,'before':before,'at_utc':times[0]}
        path=c.file('retirement/'+label+'_intent.json',intent)
        c.file('retirement/'+label+'_result.json',{'command':command,'environment':{'TZ':'UTC'},'returncode':0,
            'intent_sha256':r.sha(path),'at_utc':times[1]})
    links={'pending_observation_sha256':'pending.json','held_observation_sha256':'held.json','routing_policy_sha256':'routing_policy.json',
        'hold_intent_sha256':'hold_intent.json','hold_result_sha256':'hold_result.json',
        'cancel_intent_sha256':'cancel_intent.json','cancel_result_sha256':'cancel_result.json'}
    capture={'schema':r.RETIREMENT_SCHEMA,'captured_at_utc':'2026-09-12T16:49:10+00:00','observer_host':'synthetic-spin',
        'retired_array_job_id':r.OLD_JOB,'reason':r.RETIREMENT_REASON,'no_model_execution':True,
        'absent_runtime_paths':r.absent_runtime_paths(),'historical_execution_files_sha256':historical,
        'final_output_inventory':baseline,'preserved_outputs_sha256':baseline['files_sha256'],
        'terminal_accounting':obs(r.accounting_command(),'\n'.join('|'.join(row) for row in rows('CANCELLED'))+'\n'),
        'queue_observation':obs(['squeue','-u',old['user'],'-r','-h','-o','%i|%T'],''),'retired_cells':rows('CANCELLED'),
        **{key:r.sha(r.RETIREMENT.parent/name) for key,name in links.items()}}
    c.file('retirement/retirement.json',capture);data.digest=r.sha(r.RETIREMENT);data.links=links
    context.adapter.snapshot=lambda base,terminal: {'status':'original_array_terminal','synthetic':True}
    def run_read(command):
        if command==r.accounting_command():
            out='\n'.join('|'.join(row) for row in rows(data.phase))+'\n'
        elif command[0]=='squeue': out=data.queue_extra
        elif command[0]=='sacct' and any(s.startswith('--name=') for s in command): out=data.accounting
        else: pytest.fail('unexpected real-like read '+repr(command))
        return obs(command,out)
    monkeypatch.setattr(b,'run_read',run_read)
    def run(command,**kwargs):
        assert command==r.read(data.root/'plan.json')['submit_command']
        assert r.read(data.root/'submission_intent.json')['command']==command
        assert kwargs['close_fds'] is True and kwargs['env']['TZ']=='UTC'
        data.calls.append(command)
        if data.error: raise subprocess.TimeoutExpired(command,60)
        return SimpleNamespace(returncode=data.returncode,stdout=data.stdout,stderr='')
    monkeypatch.setattr(r,'subprocess',SimpleNamespace(run=run,TimeoutExpired=subprocess.TimeoutExpired))
    return data


def prepare(d):
    d.r.prepare(d.root,d.c.manifest,d.digest);return d.r.read(d.root/'plan.json')


def submit(d):
    prepare(d);return d.r.submit(d.root)


def repin_capture(d):
    value=d.r.read(d.r.RETIREMENT)
    value.update({key:d.r.sha(d.r.RETIREMENT.parent/name) for key,name in d.links.items()})
    rewrite(d.r.RETIREMENT,value);d.digest=d.r.sha(d.r.RETIREMENT)


def account(d,job=900):
    p=d.r.read(d.root/'plan.json')
    return '\n'.join('|'.join([str(job),f'{job}_{i}',p['job_name'],p['recovery_token'],p['user'],'allcs','cs',
        'cpu=6,mem=60G,node=1,gres/gpu=2,gres/gpu:a5000=2','480']) for i in (0,1))+'\n'


def worker_env(d,monkeypatch,index=0):
    values={**d.r.ENVIRONMENT,'SLURMD_NODENAME':'node202','SLURM_ARRAY_JOB_ID':'900','SLURM_ARRAY_TASK_ID':str(index),
        'SLURM_JOB_ID':str(900+index),'SLURM_JOB_ACCOUNT':'allcs','SLURM_JOB_PARTITION':'cs','SLURM_CPUS_PER_TASK':'6',
        'SLURM_MEM_PER_NODE':'61440','PYTHONPYCACHEPREFIX':'/tmp/NONPRODUCTION-scale-frozen-pycache-synthetic'}
    for key,value in values.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(d.r.socket,'gethostname',lambda:'node202.synthetic')
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'0.8.4')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,get_device_name=lambda i:'NVIDIA RTX A5000')))
    def execv(path,args):d.execs.append((path,args))
    monkeypatch.setattr(d.r.os,'execv',execv)


def test_prepare_verifies_full_composition_and_preserves_old_files(d):
    before={str(p):p.read_bytes() for base in (d.r.OLD,d.r.RETIREMENT.parent,d.c.base/'outputs') for p in base.rglob('*') if p.is_file()}
    p=prepare(d)
    assert d.r.verify(d.root,require_initial=True)==p
    assert len(p['preserved_outputs_sha256'])==416
    assert all(Path(path).read_bytes()==raw for path,raw in before.items())
    assert d.calls==[]
    command=p['submit_command']
    assert '--account=allcs' in command and '--partition=cs' in command
    assert '--dependency=afterany:31254520:31258973' in command and '--array=0-1%1' in command
    assert '--gres=gpu:a5000:2' in command and '--cpus-per-task=6' in command and '--mem=60G' in command
    assert '--no-requeue' in command and not any(x.startswith('--nodelist') for x in command)
    assert command[-1]==str(d.root/'worker.slurm')
    assert str(d.r.SOURCE) in (d.root/'worker.slurm').read_text()
    assert d.r.OLD_ADAPTER.name not in (d.root/'worker.slurm').read_text()
    assert [x['command'] for x in p['cells']]==[d.c.plan['cells'][i]['command'] for i in (5,6)]
    assert p['input_symlink_targets'][str(d.b.PYTHON)]==os.readlink(d.b.PYTHON)


def test_no_second_prepare_or_noncanonical_root(d):
    prepare(d)
    with pytest.raises(ValueError,match='already prepared'):prepare(d)
    with pytest.raises(ValueError,match='one fresh canonical'):d.r.verify(d.root/'other')


@pytest.mark.parametrize('field,new',[(4,'2026-09-12T16:20:00'),(6,'node105'),(7,'6'),(9,'cpu=6'),(11,'allcs'),(12,'cs'),(3,'1:0')])
def test_retirement_refuses_started_or_changed_terminal_cell(d,field,new):
    value=d.r.read(d.r.RETIREMENT);row=value['retired_cells'][0];row[field]=new
    value['terminal_accounting']['stdout']='\n'.join('|'.join(r) for r in value['retired_cells'])
    rewrite(d.r.RETIREMENT,value)
    with pytest.raises(ValueError):d.r.retirement(d.context,d.r.sha(d.r.RETIREMENT))
    assert d.calls==[]


@pytest.mark.parametrize('old,new',[
    ('StartTime=Unknown','StartTime=2026-09-12T16:00:00'),('AllocTRES=(null)','AllocTRES=cpu=6'),
    ('Restarts=0','Restarts=1'),('Requeue=0','Requeue=1'),('worker.slurm','other.slurm'),('Priority=0','Priority=5'),
    ('RunTime=00:00:00','RunTime=00:00:01'),('ArrayTaskId=0','ArrayTaskId=1')])
def test_held_snapshot_rejects_each_no_start_control_violation(d,old,new):
    value=d.r.read(d.r.RETIREMENT.parent/'held.json')
    value['control_observations'][0]['stdout']=value['control_observations'][0]['stdout'].replace(old,new)
    with pytest.raises(ValueError):d.r.pending_snapshot(value,d.context,held=True)


def test_retirement_requires_explicit_digest_and_successful_mutations(d):
    with pytest.raises(ValueError,match='explicit actual'):d.r.retirement(d.context,'0'*64)
    p=d.r.RETIREMENT.parent/'cancel_result.json';value=d.r.read(p);value['returncode']=1;rewrite(p,value);repin_capture(d)
    with pytest.raises(ValueError,match='successful retirement'):d.r.retirement(d.context,d.digest)


def test_retirement_rejects_forged_inventory_and_omitted_incident_file(d):
    value=d.r.read(d.r.RETIREMENT);value['preserved_outputs_sha256'].pop(next(iter(value['preserved_outputs_sha256'])))
    rewrite(d.r.RETIREMENT,value)
    with pytest.raises(ValueError,match='exact partial'):d.r.retirement(d.context,d.r.sha(d.r.RETIREMENT))


@pytest.mark.parametrize('path_kind',['original_output','old_runtime','new_python'])
def test_prepare_refuses_any_unregistered_work(d,path_kind):
    if path_kind=='original_output':Path(next(iter(d.c.incident['saved_outputs_sha256']))).write_text('changed')
    elif path_kind=='old_runtime':d.c.file('recovery/runtime/0.json',{})
    else:d.c.file('outputs/6/difficulty_0.json',{})
    with pytest.raises(ValueError):prepare(d)
    assert d.calls==[] and not d.r.CLAIM.exists()


@pytest.mark.parametrize('field,value',[('hardware',{}),('dependency_ids',[31254520]),('cells',[]),('environment',{})])
def test_resealed_but_changed_contract_rejected(d,field,value):
    p=prepare(d);p[field]=value;rewrite(d.root/'plan.json',p)
    rewrite(d.root/'plan.sha256.json',{'sha256':d.r.sha(d.root/'plan.json')})
    with pytest.raises(ValueError,match='canonical cs'):d.r.verify(d.root)


def test_unrelated_future_retirement_note_does_not_change_plan_closure(d):
    prepare(d);d.c.file('retirement/unrelated_review.json',{'future_note':True});d.r.verify(d.root)


def test_fresh_parent_must_remain_exactly_retired_before_submission(d):
    prepare(d);d.phase='PENDING'
    with pytest.raises(ValueError):d.r.submit(d.root)
    assert d.calls==[] and not (d.root/'submission_intent.json').exists()


def test_retired_array_hidden_live_sibling_refuses(d):
    prepare(d);d.queue_extra=f'{d.r.OLD_JOB}_[0-1]|PENDING\n'
    with pytest.raises(ValueError,match='live queue'):d.r.submit(d.root)
    assert d.calls==[]


def test_submission_single_canonical_command_and_identity(d):
    value=submit(d)
    assert value['array_job_id']==900 and d.r.submission_identity(d.root)==value
    assert len(d.calls)==1
    with pytest.raises(ValueError,match='already attempted'):d.r.submit(d.root)
    assert len(d.calls)==1


@pytest.mark.parametrize('mode',['timeout','nonzero','empty','parent_id'])
def test_ambiguous_or_failed_submit_never_resubmits(d,mode):
    prepare(d)
    if mode=='timeout':d.error=True
    elif mode=='nonzero':d.returncode=1
    elif mode=='empty':d.stdout=''
    else:d.stdout='31258973\n'
    with pytest.raises(ValueError):d.r.submit(d.root)
    with pytest.raises(ValueError,match='already attempted'):d.r.submit(d.root)
    assert len(d.calls)==1


def test_readonly_reconciliation_of_timeout_uses_actual_canonical_accounting(d):
    prepare(d);d.error=True
    with pytest.raises(ValueError):d.r.submit(d.root)
    d.accounting=account(d);identity=d.r.reconcile_submission(d.root)
    assert identity['array_job_id']==900 and d.r.submission_identity(d.root)==identity and len(d.calls)==1


@pytest.mark.parametrize('change',['partition','account','gpu','duplicate','identity','outcome'])
def test_accounting_reconciliation_drift_rejected(d,change):
    prepare(d);d.error=True
    with pytest.raises(ValueError):d.r.submit(d.root)
    d.accounting=account(d)
    if change=='partition':d.accounting=d.accounting.replace('|cs|','|all|')
    elif change=='account':d.accounting=d.accounting.replace('|allcs|','|mltheory|')
    elif change=='gpu':d.accounting=d.accounting.replace('a5000=2','a100=2')
    elif change=='duplicate':d.accounting+=d.accounting.splitlines()[0]+'\n'
    elif change=='identity':d.accounting=d.accounting.replace('900_1','901_1')
    else:
        d.c.file('cs/submission_result.json',{'intent_sha256':d.r.sha(d.root/'submission_intent.json')})
    with pytest.raises(ValueError):d.r.reconcile_submission(d.root)
    assert len(d.calls)==1


def test_successful_result_cannot_reconcile_to_different_job(d):
    submit(d);(d.root/'submission_identity.json').unlink();d.accounting=account(d,901)
    with pytest.raises(ValueError,match='contradicts'):d.r.reconcile_submission(d.root)
    assert len(d.calls)==1


def test_reconciled_identity_revalidates_embedded_actual_resources(d):
    prepare(d);d.error=True
    with pytest.raises(ValueError):d.r.submit(d.root)
    d.accounting=account(d);value=d.r.reconcile_submission(d.root)
    value['accounting']['stdout']=value['accounting']['stdout'].replace('|cs|','|all|')
    rewrite(d.root/'submission_identity.json',value)
    with pytest.raises(ValueError):d.r.submission_identity(d.root)


@pytest.mark.parametrize('index',[0,1])
def test_direct_worker_has_truthful_mapping_and_exact_original_exec(d,monkeypatch,index):
    submit(d);worker_env(d,monkeypatch,index);d.r.worker(d.root,index)
    runtime=d.r.read(d.root/'runtime'/f'{index}.json')
    assert runtime['original_index']==index+5 and runtime['recovery_index']==index
    assert runtime['environment']['SLURM_JOB_ACCOUNT']=='allcs' and runtime['environment']['SLURM_JOB_PARTITION']=='cs'
    assert runtime['retirement_sha256']==d.digest and runtime['parents_terminal']['original_terminal']['synthetic']
    assert d.execs==[(d.c.plan['cells'][index+5]['command'][0],d.c.plan['cells'][index+5]['command'])]
    assert not any(Path(p).exists() for p in d.r.absent_runtime_paths())
    with pytest.raises(ValueError,match='already attempted'):d.r.worker(d.root,index)
    assert len(d.execs)==1


@pytest.mark.parametrize('key,new',[('SLURM_JOB_ACCOUNT','cs'),('SLURM_JOB_PARTITION','allcs'),('SLURM_JOB_PARTITION','mltheory'),
    ('SLURM_ARRAY_JOB_ID','901'),('SLURM_CPUS_PER_TASK','12'),('SLURM_MEM_PER_NODE','60000'),('VLLM_USE_V1','1')])
def test_worker_refuses_route_identity_or_science_environment_drift(d,monkeypatch,key,new):
    submit(d);worker_env(d,monkeypatch);monkeypatch.setenv(key,new)
    with pytest.raises(ValueError):d.r.worker(d.root,0)
    assert d.execs==[] and not (d.root/'runtime/0.json').exists()


@pytest.mark.parametrize('index',[0,1])
def test_worker_refuses_new_target_outputs_before_claim(d,monkeypatch,index):
    submit(d);worker_env(d,monkeypatch,index)
    d.c.file(f'outputs/{index+5}/difficulty_3.json.batches/seed-1__rows-new.json',{'unregistered':True})
    with pytest.raises(ValueError,match='before its single worker claim'):d.r.worker(d.root,index)
    assert d.execs==[]


def test_python_worker_allows_registered_pantrys_independent_growth(d,monkeypatch):
    submit(d);worker_env(d,monkeypatch,0);d.r.worker(d.root,0)
    d.c.file('outputs/5/difficulty_3.json.batches/seed-1__rows-new.json',{'synthetic_new_work':True})
    worker_env(d,monkeypatch,1);d.r.worker(d.root,1)
    assert len(d.execs)==2
