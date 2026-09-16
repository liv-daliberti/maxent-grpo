"""Scratch-only final execution gates: no scheduler, model, grader, or real publication."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def audit():
    spec=importlib.util.spec_from_file_location('scratch_final_v4',ROOT/'artifacts/verify_modebench_scale_frozen_execution_v4_20260912.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def change(path,update):
    value=json.loads(Path(path).read_text());update(value);write(path,value)


def utc(seconds=0):
    return (datetime(2026,9,12,tzinfo=timezone.utc)+timedelta(seconds=seconds)).isoformat()


def scheduler(seconds=0):return utc(seconds).split('+')[0]


def accounting(a,stage='confirmation',job=99):
    rows=[]
    for index in range(10 if stage=='confirmation' else 7):
        start=7200+index*180;end=start+120
        row=[str(100+index),f'{job}_{index}','COMPLETED','0:0',scheduler(start),scheduler(end),
            'node202','6','60G','cpu=6,gres/gpu=2,gres/gpu:a5000=2,mem=60G,node=1',scheduler(3600)]
        if stage=='confirmation':
            for suffix in ('','.batch','.extern'):
                step=deepcopy(row);step[0]+=suffix;step[1]+=suffix
                if suffix:step[8]=''
                rows.append(step+['allcs','cs' if not suffix else ''])
        else:
            row[6]='node105'
            if index<5:row[2:4]=['FAILED','143:0' if index==2 else '1:0']
            elif index==5:row[2:4]=['NODE_FAIL','1:0']
            else:row[2:4]=['CANCELLED by 1','0:0'];row[4]='None';row[6]='None assigned';row[7]='0';row[9]=''
            rows.append(row)
    return {'schema':a.CAPTURE_SCHEMA,'stage':stage,'job_id':job,'command':a.capture_command(stage,job),
        'environment':{'TZ':'UTC'},'returncode':0,'stderr':'','observer_host':'spin.cs.princeton.edu',
        'captured_at_utc':utc(11000),'stdout':'\n'.join('|'.join(row) for row in rows)}


def test_exact_actual_confirmation_main_and_steps(audit):
    rows=audit.confirmation_rows(accounting(audit),99)
    assert len(rows)==10 and rows[9][0]=='109' and rows[0][11:]==['allcs','cs']


@pytest.mark.parametrize('old,new',[
    ('COMPLETED','FAILED'),('COMPLETED','NODE_FAIL'),('0:0','143:0'),('node202','node105'),
    ('node202','node205'),('|6|','|4|'),('|60G|','|59G|'),('mem=60G','mem=59G'),
    ('gres/gpu=2','gres/gpu=1'),('gres/gpu:a5000=2','gres/gpu:a6000=2'),
    ('cpu=6,','cpu=6,cpu=6,'),('|allcs|','|mltheory|'),('|allcs|cs','|cs|allcs'),
    ('100|99_0|','0|99_0|'),('100.batch|','123.batch|'),
    (scheduler(7200),scheduler(3500)),(scheduler(7320),scheduler(12000))])
def test_confirmation_rejects_wrong_actual_execution(audit,old,new):
    value=accounting(audit);value['stdout']=value['stdout'].replace(old,new)
    with pytest.raises(ValueError):audit.confirmation_rows(value,99)


@pytest.mark.parametrize('mutation',['missing','duplicate','extra','overlap','step_failure'])
def test_complete_unique_nonoverlapping_confirmation_required(audit,mutation):
    value=accounting(audit);rows=value['stdout'].splitlines()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows[-1]=rows[0]
    elif mutation=='extra':rows.append(rows[0])
    elif mutation=='step_failure':rows[1]=rows[1].replace('COMPLETED','FAILED')
    else:
        rows=[row.replace(scheduler(7380),scheduler(7250)) if '|99_1' in row else row for row in rows]
    value['stdout']='\n'.join(rows)
    with pytest.raises(ValueError):audit.confirmation_rows(value,99)


@pytest.mark.parametrize('key,value',[('environment',{}),('command',['sacct']),('returncode',1),
    ('stderr','warning'),('observer_host',''),('job_id',98),('captured_at_utc','2026-09-12T04:00:00')])
def test_capture_identity_and_utc_required(audit,key,value):
    record=accounting(audit);record[key]=value
    with pytest.raises(ValueError):audit.confirmation_rows(record,99)


@pytest.fixture
def stage_factory(audit,tmp_path,monkeypatch):
    a=audit
    for name,value in [('COMPOSITE',tmp_path/'composite'),('EVIDENCE',tmp_path/'evidence'),('RELEASE',tmp_path/'release')]:
        monkeypatch.setattr(a,name,value);value.mkdir()
    def make(stage):
        phase='dev' if stage=='revision_development' else 'eval';count=7 if phase=='dev' else 10
        job=a.DEVELOPMENT_ARRAY if phase=='dev' else 99
        path=a.COMPOSITE/stage/'plan.json';outer=path.parent/'frozen_transport'
        manifest=tmp_path/'view.json';write(manifest,{})
        cells=[];proofs={};receipts=[]
        domains=['countdown','graph_coloring','python','mathir','pantry']
        sources={'level4':{},'level5':{}}
        for i in range(count):
            level='level4' if i<5 else 'level5';domain=domains[i%5];source=tmp_path/'sources'/f'{level}_{domain}'
            write(source/'protocol.json',{'fixture':i})
            cell={'id':f'{level}_{domain}','level':level,'domain':domain,'phase':phase,
                'source_kind':'domain_revision_v1','source_root':str(source),'model_label':'7b' if i<5 else '14b',
                'command':['/literal/python','unchanged_evaluator','--resume'],
                'tasks':str(path.parent/'tasks'/f'{i}.json')}
            receipt=tmp_path/'outputs'/f'{stage}_{i}.json';receipts.append(receipt)
            write(receipt,{'identity':{'batch_size':8}})
            write(cell['tasks'],[{'output':str(receipt),'batch_size':8}])
            write(source/'canonical_confirmation.json',{'fixture':'existing original grader audit'})
            cells.append(cell);sources[level][domain]={k:cell[k] for k in ('source_kind','source_root')}
        for level in sources:write(a.RELEASE/level/'source_manifest.json',{'sources':sources[level]})
        plan={'phase':phase,'concurrency':1,'cells':cells,'hardware':{'nodes':'node105'},
            'dependency_ids':[31253118,123456], 'scientific_inputs_sha256':{str(manifest):a.sha(manifest)},
            'runtime_profile':{'thread_environment':{'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}},
            'submit_command':['sbatch','--account=mltheory','--partition=mltheory','--nodelist=node105','--mem=60G',str(path.parent/'worker.slurm'),str(path)]}
        write(path,plan);write(path.parent/'plan.sha256.json',{'sha256':a.sha(path)})
        if phase=='dev':monkeypatch.setattr(a,'DEVELOPMENT_PLAN_SHA',a.sha(path))
        candidate={'actual_authority':'wash' if phase=='dev' else 'spin'}
        tx={'authority':candidate,'view_manifest':str(manifest),'prepared_at_utc':utc(2400),
            'canonical_submit_command':plan['submit_command'],
            'effective_submit_command':['sbatch','--account=allcs','--partition=cs','--mem=60G',str(outer/'worker.slurm'),str(path)],
            'inputs_sha256':{str(manifest):a.sha(manifest)}}
        write(outer/'plan.json',tx);write(outer/'plan.sha256.json',{'sha256':a.sha(outer/'plan.json')})
        write(path.parent/'submission_intent.json',{'at':utc(3000),'status':'submission_attempt_started','command':plan['submit_command'],'plan_sha256':a.sha(path)})
        write(outer/'submission_intent.json',{'at_utc':utc(3599),'status':'effective_submission_attempt_started',
            'transport_plan_sha256':a.sha(outer/'plan.json'),'canonical_intent_sha256':a.sha(path.parent/'submission_intent.json'),
            'effective_command':tx['effective_submit_command'],'canonical_command':plan['submit_command']})
        write(outer/'submission_result.json',{'at_utc':utc(3601),'returncode':0,'stdout':str(job)+'\n','stderr':'',
            'transport_plan_sha256':a.sha(outer/'plan.json'),'intent_sha256':a.sha(outer/'submission_intent.json')})
        write(path.parent/'submission_result.json',{'at':utc(3602),'status':'submitted','returncode':0,'stdout':str(job)+'\n','stderr':'',
            'array_job_id':job,'cells':[{'array_index':i,'cell':cell['id']} for i,cell in enumerate(cells)]})
        terminal=accounting(a,stage,job);write(a.EVIDENCE/(stage+'_terminal_accounting.json'),terminal)
        raw=[line.split('|') for line in terminal['stdout'].splitlines()]
        dispatch=[];calls=[]
        for i,cell in enumerate(cells):
            if phase=='dev' and i<5:
                row=raw[i];proof={'stage_plan_sha256':a.sha(path),'literal_evaluator_command':cell['command'],
                    'completed_receipts':4,'scheduler_success':False,'scientific_outputs_complete':True,'exit_cause':'unknown',
                    'job_id_raw':row[0],'job_id':row[1],'exit_code':row[3],'end_utc':row[5],'files_sha256':{}}
                directory=path.parent/'execution_reconciliations'/str(i)
                write(directory/'reconciliation.json',proof);write(directory/'terminal_accounting.json',{'stdout':'|'.join(row)})
                proofs[i]=proof
            if phase=='dev':continue
            env={'SLURM_JOB_ID':str(100+i),'SLURM_ARRAY_JOB_ID':str(job),'SLURM_ARRAY_TASK_ID':str(i),
                'SLURMD_NODENAME':'node202','SLURM_JOB_ACCOUNT':'allcs','SLURM_JOB_PARTITION':'cs',
                'SLURM_MEM_PER_NODE':'61440','SLURM_CPUS_PER_TASK':'6','VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'XFORMERS',
                **plan['runtime_profile']['thread_environment']}
            write(outer/'runtime'/f'{i}.json',{'at_utc':utc(7205+i*180),'status':'validated_before_unchanged_scientific_worker_exec',
                'transport_schema':'modebench_scale_frozen_stage_transport_v4','effective_account':'allcs','effective_partition':'cs',
                'canonical_node_constraint':'node105','allowed_actual_nodes':list(a.ALLOWED_NODES),
                'array_job_id':job,'array_index':i,'stage_plan_sha256':a.sha(path),'transport_plan_sha256':a.sha(outer/'plan.json'),
                'view_manifest_sha256':a.sha(manifest),'effective_submission_result_sha256':a.sha(outer/'submission_result.json'),
                'canonical_submission_result_sha256':a.sha(path.parent/'submission_result.json'),'evaluator_command':cell['command'],
                'hostname':'node202.cs.princeton.edu','visible_gpu_names':['NVIDIA RTX A5000']*2,'vllm_version':'0.8.4','environment':env})
            write(path.parent/'runtime'/f'{i}.json',{'at':utc(7210+i*180),'array_job_id':job,'cell':cell['id'],
                'plan_sha256':a.sha(path),'command':cell['command'],'hostname':'node202.cs.princeton.edu'})
        def checker(version):
            def check(stage,index):dispatch.append((version,stage,index));return deepcopy(proofs[index])
            return SimpleNamespace(verify_reconciliation=check)
        evaluator=object()
        def complete(modules,plan_,cell,task,tier,protocol,terminal):
            assert modules.evaluator is evaluator and plan_==plan and tier==0 and task['batch_size']==8
            assert terminal['worker_started_at_utc']==a.read(path.parent/'runtime'/f"{cells.index(cell)}.json")['at']
            calls.append(('complete',cell['id']));return {'fixture':'summary'},[],[Path(task['output'])]
        def canonical(cell,task,summary,protocol):
            calls.append(('canonical',cell['id']))
            return {'path':str(Path(cell['source_root'])/'canonical_confirmation.json'),'difficulty_matched':True,'original_grader_replayed_attempts':4096}
        helper=SimpleNamespace(completed_task=complete,canonical_confirmation=canonical)
        transport=SimpleNamespace(SCHEMA='modebench_scale_frozen_stage_transport_v4',verify_transport=lambda launcher,p:(a.read(p),a.read(outer/'plan.json')))
        recovered={'array_job_id':123456,'original_execution':{'array_job_id':a.DEVELOPMENT_ARRAY},
            'original_terminal_rows_by_index':{str(i):row for i,row in enumerate(raw)}}
        return SimpleNamespace(a=a,path=path,outer=outer,plan=plan,tx=tx,candidate=candidate,transport=transport,
            launcher=SimpleNamespace(evaluator=lambda:evaluator),helpers={1:checker(1),2:checker(2)},helper=helper,
            recovered=recovered,proofs=proofs,dispatch=dispatch,calls=calls,receipts=receipts,terminal=terminal)
    return make


def run_dev(c):return c.a.validate_development(c.launcher,c.transport,c.candidate,c.helpers,c.recovered)
def run_confirmation(c):return c.a.validate_confirmation(c.launcher,c.transport,c.candidate,c.helper,c.recovered)


def test_development_preserves_five_failed_certificates_partial_and_unstarted(stage_factory):
    c=stage_factory('revision_development');result,pins=run_dev(c)
    assert result['scheduler_success'] is False and result['original_partial_state']=='NODE_FAIL'
    assert result['original_unstarted_start']=='None' and result['original_runtime6_exists'] is False
    assert c.dispatch==[(1,'revision_development',i) for i in range(2)]+[(2,'revision_development',i) for i in range(2,5)]
    assert result['individually_reconciled_cells'][2]['exit_code']=='143:0'
    assert len(result['individually_reconciled_cells'])==5 and len(pins)>10


@pytest.mark.parametrize('mutation',['wrong_row','wrong_array','fake_runtime','fake_outer_runtime','wrong_cell_proof','rewrite143','synthetic_completed'])
def test_development_rejects_rewritten_history_or_invented_execution(stage_factory,mutation):
    c=stage_factory('revision_development')
    if mutation=='wrong_row':c.recovered['original_terminal_rows_by_index']['5'][0]='9999'
    elif mutation=='wrong_array':c.recovered['original_execution']['array_job_id']=3
    elif mutation in ('fake_runtime','fake_outer_runtime'):
        write((c.outer if mutation=='fake_outer_runtime' else c.path.parent)/'runtime/6.json',{'invented':True})
    elif mutation=='wrong_cell_proof':c.proofs[2]['scheduler_success']=True
    else:
        p=c.a.EVIDENCE/'revision_development_terminal_accounting.json'
        change(p,lambda v:v.update(stdout=v['stdout'].replace('143:0','1:0') if mutation=='rewrite143' else v['stdout'].replace('NODE_FAIL','COMPLETED')))
    with pytest.raises(ValueError):run_dev(c)


def test_confirmation_all_ten_exact_helpers_and_fresh_grader_certificates(stage_factory):
    c=stage_factory('confirmation');result,pins=run_confirmation(c)
    assert result['original_grader_attempts']==40960 and result['account']=='allcs' and result['partition']=='cs'
    assert len(c.calls)==20 and [kind for kind,_ in c.calls]==['complete','canonical']*10
    assert set(map(str,c.receipts))<=set(pins)


@pytest.mark.parametrize('field,value',[('effective_account','cs'),('effective_partition','allcs'),
    ('transport_schema','modebench_scale_frozen_stage_transport_v3'),('allowed_actual_nodes',['node105']),
    ('canonical_node_constraint','node202'),('evaluator_command',['changed']),('array_index',1),
    ('hostname','node105.cs.princeton.edu'),('vllm_version','0.8.5'),('visible_gpu_names',['A5000']),
    ('stage_plan_sha256','wrong'),('effective_submission_result_sha256','wrong'),('at_utc',utc(7100))])
def test_confirmation_runtime_identity_cannot_drift(stage_factory,field,value):
    c=stage_factory('confirmation');change(c.outer/'runtime/0.json',lambda v:v.update({field:value}))
    with pytest.raises(ValueError):run_confirmation(c)


@pytest.mark.parametrize('field,value',[('SLURM_JOB_ACCOUNT','mltheory'),('SLURM_JOB_PARTITION','all'),
    ('SLURM_JOB_ID',''),('SLURM_JOB_ID','999'),('SLURM_CPUS_PER_TASK','4'),('SLURM_MEM_PER_NODE','30720'),
    ('SLURMD_NODENAME','node105'),('VLLM_USE_V1','1'),('OMP_NUM_THREADS','8')])
def test_actual_environment_is_checked_against_accounting(stage_factory,field,value):
    c=stage_factory('confirmation');change(c.outer/'runtime/0.json',lambda v:v['environment'].update({field:value}))
    with pytest.raises(ValueError):run_confirmation(c)


@pytest.mark.parametrize('mutation',['batch_size','difficulty','attempts','native_checks','dependency','source'])
def test_confirmation_preserves_existing_full_native_and_admission_gates(stage_factory,mutation):
    c=stage_factory('confirmation')
    if mutation=='batch_size':change(c.receipts[0],lambda v:v['identity'].update(batch_size=4))
    elif mutation in ('difficulty','attempts'):
        original=c.helper.canonical_confirmation
        def changed(*args):
            value=original(*args);value['difficulty_matched' if mutation=='difficulty' else 'original_grader_replayed_attempts']=False if mutation=='difficulty' else 4095
            return value
        c.helper.canonical_confirmation=changed
    elif mutation=='native_checks':
        def fail(*args):raise ValueError('sealed native check failed')
        c.helper.completed_task=fail
    elif mutation=='dependency':change(c.path,lambda v:v.update(dependency_ids=[31253118]))
    else:change(c.a.RELEASE/'level5/source_manifest.json',lambda v:v['sources']['pantry'].update(source_root='/wrong'))
    with pytest.raises(ValueError):run_confirmation(c)


@pytest.mark.parametrize('target,field,value',[
    ('canonical_intent','command',['changed']),('canonical_intent','plan_sha256','wrong'),
    ('effective_intent','effective_command',['changed']),('effective_intent','canonical_intent_sha256','wrong'),
    ('effective_result','intent_sha256','wrong'),('effective_result','stdout','100\n'),
    ('canonical_result','array_job_id',True),('canonical_result','cells',[]),
    ('canonical_result','at',utc(100)),('effective_result','returncode',143)])
def test_submission_ledgers_are_exact_and_chronological(stage_factory,target,field,value):
    c=stage_factory('confirmation');base=c.outer if target.startswith('effective') else c.path.parent
    p=base/('submission_intent.json' if target.endswith('intent') else 'submission_result.json')
    change(p,lambda v:v.update({field:value}))
    with pytest.raises(ValueError):c.a.submission(c.path,c.plan,c.tx)


def test_duplicate_ambiguous_submission_requires_explicit_reconciliation(stage_factory):
    c=stage_factory('confirmation');write(c.outer/'submission_ambiguous.json',{})
    with pytest.raises(ValueError,match='ambiguous'):run_confirmation(c)


@pytest.fixture
def authority_factory(audit,tmp_path,monkeypatch):
    a=audit;original=tmp_path/'original_recovery';monkeypatch.setattr(a,'ORIGINAL_RECOVERY',original)
    recovery={'schema':'modebench_scale_frozen_recovery_execution_reconciliation_v2',
        'status':'verified_scientific_outputs_with_failed_execution','scheduler_success':False,
        'scientific_outputs_complete':True,'exit_cause':'unknown','audit_completed_at_utc':utc(-60)}
    write(original/'execution_reconciliation.json',recovery)
    def make(predecessor=False):
        root=tmp_path/('wash' if predecessor else 'spin');root.mkdir()
        host=('wash' if predecessor else 'spin')+'.cs.princeton.edu';python=tmp_path/'python';source=tmp_path/(host+'.py')
        manifest=tmp_path/'view.json';write(manifest,{})
        activation={'scope':'reviewed actual authority','created_at_utc':utc(0),'view_manifest':str(manifest),'reviewed_bundle_sha256':'bundle','inputs_sha256':{}}
        write(root/'activation.json',activation);write(root/'activation.sha256.json',{'sha256':a.sha(root/'activation.json')})
        command=[str(python),'-B',str(source),'watch','--root',str(root),'--interval','60']
        runtime={'schema':'modebench_scale_wash_authority_runtime_v2' if predecessor else 'modebench_scale_spin_authority_runtime_v1',
            'host':host,'authority_path':str(root/'activation.json'),'authority_sha256':a.sha(root/'activation.json'),
            'owner_command':command,'owner_pid':123,'owner_start_ticks':'456','uid':1000,'fence_fd':8,
            'at_utc':utc(4),'lock_path':str(tmp_path/'controller.lock'),'lock_inode':999,'lock_device':5}
        write(root/'outer_runtime.json',runtime)
        record={'host':host,'owner_pid':123,'owner_start_ticks':'456','fence_fd':8}
        driver=SimpleNamespace(HOST=host,PYTHON=python,SOURCE=source,NEUTRAL_SHA='neutral',
            authority_record=lambda *args:deepcopy(record),validate_completion=lambda value:value,
            reconciliation_command=lambda manifest:['read-only-recovery',manifest],
            proof_command=lambda manifest,bundle:['read-only-proof',manifest,bundle],
            guest_command=lambda root,activation,index,fd:['/literal/runner','exec','--',str(python),'-B',str(source),'guest-sweep','--root',str(root),'--number',str(index),'--fence-fd',str(fd)])
        def probe(second):
            return {'command':[str(python),'-B',str(source),'host-probe','--owner-pid','123'],'returncode':0,
                'observation':{'at_utc':utc(second),'host':host,'uid':1000,'canonical_sha256':'neutral',
                    'lock_path':runtime['lock_path'],'lock_inode':999,'lock_device':5,
                    'old_soak_os_state' if predecessor else 'predecessor_os_state':'unverified',
                    'local_controllers':[{'pid':123,'start_ticks':'456','command':command}]}}
        start={'at_utc':utc(3),'activation_sha256':a.sha(root/'activation.json'),'scope':activation['scope'],'host_probe':probe(1)}
        if predecessor:
            proof={'command':driver.reconciliation_command(str(manifest)),'returncode':0,
                'observation':{**recovery,'certificate_path':str(original/'execution_reconciliation.json'),
                    'certificate_sha256':a.sha(original/'execution_reconciliation.json')}}
            write(root/'preparation_reconciliation_verification.json',proof);start['reconciliation_verification']=proof
            write(root/'recovery_completion.json',{'at_utc':utc(2)})
        else:
            proof={'command':driver.proof_command(str(manifest),'bundle'),'returncode':0,'observation':{'proof':'actual full existing closure'}}
            write(root/'revised_recovery_completion.json',{'proof_verification':proof});start['proof_verification']=proof
        write(root/'start_intent.json',start)
        count=266 if predecessor else 2
        final={'jobs':[{'job':'31254520_5','state':'NODE_FAIL'}],'status':'execution_failed'} if predecessor else {'status':'admitted','release':'fixture'}
        for i in range(1,count+1):
            directory=root/'sweeps'/f'{i:06d}';second=10+(i-1)*6
            write(directory/'before.json',probe(second))
            argv=driver.guest_command(root,activation,i,8)
            write(directory/'intent.json',{'at_utc':utc(second+1),'command':argv,'authority':record,'before_sha256':a.sha(directory/'before.json')})
            guest_argv=argv[argv.index('--')+1:];guest_argv.insert(1,'-B')
            write(directory/'guest_runtime.json',{'at_utc':utc(second+2),'host':host,'authority':record,'fence_fd':8,
                'uid':1000,'pid':10000+i,'start_ticks':str(20000+i),'state':'R','command':guest_argv,
                'before_sha256':a.sha(directory/'before.json')})
            write(directory/'result.json',{'at_utc':utc(second+3),'authority':record,
                'guest_runtime_sha256':a.sha(directory/'guest_runtime.json'),'result':final if i==count else {'status':'waiting'}})
            write(directory/'exit.json',{'at_utc':utc(second+4),'returncode':0});write(directory/'after.json',probe(second+5))
        write(root/'terminal.json',{'at_utc':'2026-09-12T13:29:23.067411+00:00' if predecessor else utc(30),
            'last_sweep':count,'result':final})
        transport_calls=[]
        def pins(candidate):assert candidate==record;transport_calls.append(candidate);return {str(manifest):a.sha(manifest)}
        transport=SimpleNamespace(authority_pins=pins)
        return SimpleNamespace(a=a,root=root,driver=driver,transport=transport,activation=activation,
            recovery=recovery,predecessor=predecessor,transport_calls=transport_calls,record=record)
    return make


def run_authority(c):
    return c.a.authority_ledger(c.root,c.driver,c.transport,c.activation,predecessor=c.predecessor,recovery=c.recovery)


@pytest.mark.parametrize('predecessor',[True,False])
def test_full_actual_authority_ledger_and_one_static_transport_closure(authority_factory,predecessor):
    c=authority_factory(predecessor);summary,pins,record=run_authority(c)
    assert summary['sweeps']==(266 if predecessor else 2)
    assert summary['terminal_status']==('execution_failed' if predecessor else 'admitted')
    assert summary['os_state']=='unverified' and record==c.record and len(c.transport_calls)==1
    assert str(c.root/'sweeps'/('000266' if predecessor else '000002')/'exit.json') in pins


@pytest.mark.parametrize('predecessor',[True,False])
@pytest.mark.parametrize('mutation',['missing','rewritten_terminal','owner','nonzero','single_b','extra_b','fence','host','state','time','os_claim','other_controller'])
def test_authority_rejects_historical_rewrite_or_broken_sweep(authority_factory,predecessor,mutation):
    c=authority_factory(predecessor);directory=c.root/'sweeps/000001'
    if mutation=='missing':(directory/'exit.json').unlink()
    elif mutation=='rewritten_terminal':change(c.root/'terminal.json',lambda v:v.update(result={'status':'admitted' if predecessor else 'execution_failed'}))
    elif mutation=='owner':change(directory/'result.json',lambda v:v['authority'].update(owner_pid=999))
    elif mutation=='nonzero':change(directory/'exit.json',lambda v:v.update(returncode=143))
    elif mutation in ('single_b','extra_b','fence','state','time'):
        def mutate(v):
            if mutation=='single_b':v['command'].pop(1)
            elif mutation=='extra_b':v['command'].insert(1,'-B')
            else:v.update({'fence_fd':9} if mutation=='fence' else {'state':None} if mutation=='state' else {'at_utc':utc(0)})
        change(directory/'guest_runtime.json',mutate)
        change(directory/'result.json',lambda v:v.update(guest_runtime_sha256=c.a.sha(directory/'guest_runtime.json')))
    else:
        def mutate(v):
            values={'host':'wash.cs.princeton.edu' if not predecessor else 'spin.cs.princeton.edu'} if mutation=='host' else {
                'old_soak_os_state' if predecessor else 'predecessor_os_state':'dead'} if mutation=='os_claim' else {'local_controllers':[{'pid':999}]}
            v['observation'].update(values)
        change(directory/'after.json',mutate)
    with pytest.raises((ValueError,FileNotFoundError)):run_authority(c)


def test_failed_spin_with_published_guest_result_is_not_automatically_reconciled(authority_factory):
    c=authority_factory();write(c.root/'failure.json',{'exit_code':143,'saved_result':'preserved'})
    with pytest.raises(ValueError,match='separate explicit reconciliation'):run_authority(c)
    assert (c.root/'sweeps/000002/result.json').exists()


def test_new_authority_requires_recovery_audit_before_activation(authority_factory):
    c=authority_factory();c.recovery['audit_completed_at_utc']=utc(1)
    with pytest.raises(ValueError,match='preceded'):run_authority(c)


@pytest.fixture
def final_case(audit,tmp_path,monkeypatch):
    a=audit
    for key in ('RELEASE','ORIGINAL_RECOVERY','RECOVERY','OLD_AUTHORITY','AUTHORITY'):
        path=tmp_path/key;path.mkdir();monkeypatch.setattr(a,key,path)
    original={'schema':'modebench_scale_frozen_recovery_execution_reconciliation_v2',
        'status':'verified_scientific_outputs_with_failed_execution','scheduler_success':False,
        'scientific_outputs_complete':True,'exit_cause':'unknown','files_sha256':{},
        'recovery_execution':{'job_id':31253118,'state':'FAILED','exit_code':'1:0'}}
    recovered={'schema':'modebench_scale_revised_recovery_execution_reconciliation_v1',
        'status':'verified_scientific_outputs_with_execution_recovery','array_job_id':123456,
        'scheduler_success':False,'recovery_scheduler_success':True,'scientific_outputs_complete':True,
        'completed_receipts':8,'completed_batches':864,'attempts_validated':54400,'files_sha256':{},
        'original_execution':{'array_job_id':31254520,'partial_cell':{'state':'NODE_FAIL'},'unstarted_cell':{'start':'None'}},
        'recovery_executions':[{'state':'COMPLETED','account':'allcs','partition':'cs'}],
        'retired_recovery_execution':{'array_job_id':31258973,'status':'cancelled_before_any_worker_execution'}}
    write(a.ORIGINAL_RECOVERY/'execution_reconciliation.json',original);write(a.RECOVERY/'execution_reconciliation.json',recovered)
    write(a.OLD_AUTHORITY/'activation.json',{});write(a.OLD_AUTHORITY/'terminal.json',{'at_utc':utc(0)})
    activation={'created_at_utc':utc(1)};calls=[]
    expected={str(a.RECOVERY_AUDITOR):'fixture_sha'}
    monkeypatch.setattr(a,'requirements',lambda:({'files_sha256':expected},{}))
    def schedule(*args):
        calls.append(('legacy_schedule',args));return {'initial_submit_array_spec':'0-9%1','per_cell_memory_overrides':{}},{},{}
    def release(*args):calls.append(('legacy_release',args));return {'level4':{},'level5':{}},{}
    legacy=SimpleNamespace(validate_schedule=schedule,validate_release=release)
    driver=SimpleNamespace(predecessor_inputs=lambda guest:{},verify=lambda root,guest:activation)
    objects={a.LEGACY:legacy,a.ORIGINAL_RECOVERY_AUDITOR:SimpleNamespace(verify_reconciliation=lambda:original),
        a.RECOVERY_AUDITOR:SimpleNamespace(verify_existing=lambda root:recovered),a.DRIVER:driver,
        a.OLD_DRIVER:object(),a.TRANSPORT:object(),a.OLD_TRANSPORT:object(),a.LAUNCHER:object(),
        **{p:object() for p in a.CELL_HELPERS.values()}}
    monkeypatch.setattr(a,'module',lambda path,*args:objects[path])
    def authority(root,*args,predecessor=False,**kwargs):
        calls.append(('authority',predecessor));return {'terminal_status':'execution_failed' if predecessor else 'admitted'},{},{'old':predecessor}
    def dev(*args):calls.append(('development',args[2]));return {'scheduler_success':False},{}
    def confirmation(*args):calls.append(('confirmation',args[2]));return {'scheduler_success':True},{}
    monkeypatch.setattr(a,'authority_ledger',authority);monkeypatch.setattr(a,'validate_development',dev);monkeypatch.setattr(a,'validate_confirmation',confirmation)
    return SimpleNamespace(a=a,original=original,recovered=recovered,legacy=legacy,calls=calls,activation=activation)


def test_unchanged_legacy_gates_then_complete_actual_chain_before_publication(final_case):
    c=final_case;result=c.a.verify()
    assert result['schema']==c.a.SCHEMA and result['status']=='verified' and result['difficulty_matched'] is True
    assert c.calls[:2]==[('legacy_schedule',(c.a.SCHEDULE,c.a.RELEASE)),('legacy_release',(c.a.RELEASE,))]
    assert c.calls[2:]==[('authority',True),('authority',False),('development',{'old':True}),('confirmation',{'old':False})]
    assert result['predecessor_authority']['terminal_status']=='execution_failed'
    assert result['spin_authority']['terminal_status']=='admitted'
    assert result['revised_execution_recovery']['scheduler_success'] is False
    assert result['revised_execution_recovery']['retired_recovery_execution']['array_job_id']==31258973
    destination=c.a.RELEASE/'execution_provenance.json';assert not destination.exists()
    assert c.a.verify(publish=True)==result and c.a.verify(publish=True)==result


@pytest.mark.parametrize('gate',['legacy_schedule','legacy_release','recovery','old_authority','spin_authority','development','confirmation','missing_level'])
def test_no_publication_before_every_required_gate(final_case,monkeypatch,gate):
    c=final_case
    def fail(*args,**kwargs):raise ValueError('required gate failed')
    if gate.startswith('legacy_'):setattr(c.legacy,'validate_'+gate.split('_')[1],fail)
    elif gate=='recovery':c.recovered['scientific_outputs_complete']=False;write(c.a.RECOVERY/'execution_reconciliation.json',c.recovered)
    elif gate in ('old_authority','spin_authority'):
        original=c.a.authority_ledger
        def check(root,*args,**kwargs):
            if kwargs.get('predecessor',False)==(gate=='old_authority'):fail()
            return original(root,*args,**kwargs)
        monkeypatch.setattr(c.a,'authority_ledger',check)
    elif gate=='missing_level':c.legacy.validate_release=lambda *args:({'level4':{}},{})
    else:monkeypatch.setattr(c.a,'validate_'+gate,fail)
    with pytest.raises(ValueError):c.a.verify(publish=True)
    assert not (c.a.RELEASE/'execution_provenance.json').exists()


@pytest.mark.parametrize('field,value',[('scheduler_success',True),('recovery_scheduler_success',False),
    ('attempts_validated',54399),('completed_receipts',7),('completed_batches',863)])
def test_recovery_chain_status_and_complete_original_grader_proof_required(final_case,field,value):
    c=final_case;c.recovered[field]=value;write(c.a.RECOVERY/'execution_reconciliation.json',c.recovered)
    with pytest.raises(ValueError,match='eight complete'):c.a.verify(publish=True)


def test_existing_different_final_is_preserved(final_case):
    c=final_case;destination=c.a.RELEASE/'execution_provenance.json';write(destination,{'status':'verified'})
    with pytest.raises(ValueError,match='never overwrite'):c.a.verify(publish=True)
    assert c.a.read(destination)=={'status':'verified'}


def test_failed_terminal_capture_creates_no_record(stage_factory,monkeypatch):
    c=stage_factory('confirmation');p=c.a.EVIDENCE/'confirmation_terminal_accounting.json';p.unlink()
    monkeypatch.setattr(c.a,'requirements',lambda:({'files_sha256':{}},{}))
    monkeypatch.setattr(c.a.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stderr='',stdout=c.terminal['stdout'].replace('COMPLETED','RUNNING')))
    with pytest.raises(ValueError):c.a.capture_accounting('confirmation')
    assert not p.exists()


def test_invalid_capture_stage_never_reaches_scheduler(audit,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('must not invoke a subprocess')
    monkeypatch.setattr(audit.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='registered stage'):audit.capture_accounting('../other')


@pytest.fixture
def requirements_case(audit,tmp_path,monkeypatch):
    a=audit
    for key in ('SOURCE','TEST_SOURCE','REVIEW','BASE_SOURCE','OLD_REQUIREMENTS','RECOVERY_AUDITOR'):
        p=tmp_path/(key+'.json');write(p,{'fixture':key});monkeypatch.setattr(a,key,p)
    for key in ('RELEASE','OLD_AUTHORITY','AUTHORITY','RECOVERY','COMPOSITE','EVIDENCE'):
        p=tmp_path/key;p.mkdir();monkeypatch.setattr(a,key,p)
    monkeypatch.setattr(a,'REQUIREMENTS',a.EVIDENCE/'requirements.json')
    monkeypatch.setattr(a,'OLD_REQUIREMENTS_SHA',a.sha(a.OLD_REQUIREMENTS))
    monkeypatch.setattr(a,'SEALED',{a.BASE_SOURCE:a.sha(a.BASE_SOURCE),a.OLD_REQUIREMENTS:a.OLD_REQUIREMENTS_SHA})
    mandatory=[a.SOURCE,a.TEST_SOURCE,a.REVIEW,a.RECOVERY_AUDITOR,a.RECOVERY/'execution_reconciliation.json',
        a.AUTHORITY/'activation.json',a.AUTHORITY/'activation.sha256.json',*a.SEALED]
    directory=a.COMPOSITE/'revision_development'
    mandatory += [base/name for base in (directory,directory/'frozen_transport')
        for name in ('plan.json','plan.sha256.json','submission_intent.json','submission_result.json')]
    mandatory += [directory/'execution_reconciliations'/str(i)/'reconciliation.json' for i in range(5)]
    for p in mandatory:
        if not p.exists():write(p,{'fixture':str(p)})
    write(directory/'submission_result.json',{'array_job_id':a.DEVELOPMENT_ARRAY})
    monkeypatch.setattr(a,'DEVELOPMENT_PLAN_SHA',a.sha(directory/'plan.json'))
    dependency=tmp_path/'nested_reviewed_dependency.py';dependency.write_text('reviewed')
    reviewed={str(p):a.sha(p) for p in (a.SOURCE,a.TEST_SOURCE,a.BASE_SOURCE,a.RECOVERY_AUDITOR,dependency)}
    write(a.REVIEW,{'schema':'modebench_scale_frozen_execution_v4_independent_review_v1','status':'reviewed','files_sha256':reviewed})
    value={'schema':'modebench_scale_frozen_execution_requirements_v4','status':'registered_before_final_release',
        'destination':str(a.RELEASE/'execution_provenance.json'),'required_schema':a.SCHEMA,'required_status':'verified',
        'predecessor_authority_root':str(a.OLD_AUTHORITY),'authority_root':str(a.AUTHORITY),'recovery_root':str(a.RECOVERY),
        'development_array_job_id':a.DEVELOPMENT_ARRAY,'development_plan_sha256':a.DEVELOPMENT_PLAN_SHA,
        'confirmation_account':'allcs','confirmation_partition':'cs','confirmation_allowed_nodes':list(a.ALLOWED_NODES),
        'supersedes_requirements':str(a.OLD_REQUIREMENTS),'supersedes_requirements_sha256':a.OLD_REQUIREMENTS_SHA,
        'failure_policy':'exact_historical_certificates_and_reviewed_cs_recovery_no_future_waiver',
        'files_sha256':{str(p):a.sha(p) for p in [*mandatory,dependency]}}
    write(a.REQUIREMENTS,value)
    return SimpleNamespace(a=a,value=value,dependency=dependency)


def test_explicit_requirements_pin_reviewed_closure_and_leave_previous_record(requirements_case):
    c=requirements_case;before=c.a.OLD_REQUIREMENTS.read_bytes();value,pins=c.a.requirements()
    assert value==c.value and pins[str(c.dependency)]==c.a.sha(c.dependency)
    assert pins[str(c.a.REQUIREMENTS)]==c.a.sha(c.a.REQUIREMENTS) and c.a.OLD_REQUIREMENTS.read_bytes()==before


@pytest.mark.parametrize('mutation',['missing_source','missing_certificate','missing_nested','mismatch_nested','draft_review','wrong_schema','self_cycle','requirements_cycle','previous_route','changed_sealed'])
def test_requirements_reject_incomplete_or_unreviewed_closure(requirements_case,mutation):
    c=requirements_case;a=c.a
    if mutation.startswith('missing_'):
        target={'missing_source':a.SOURCE,'missing_certificate':a.RECOVERY/'execution_reconciliation.json','missing_nested':c.dependency}[mutation]
        change(a.REQUIREMENTS,lambda v:v['files_sha256'].pop(str(target)))
    elif mutation in ('draft_review','wrong_schema','self_cycle','requirements_cycle','mismatch_nested'):
        def update(v):
            if mutation=='draft_review':v['status']='draft'
            elif mutation=='wrong_schema':v['schema']='another_review'
            else:v['files_sha256'][str(a.REVIEW if mutation=='self_cycle' else a.REQUIREMENTS if mutation=='requirements_cycle' else c.dependency)]='0'*64
        change(a.REVIEW,update);change(a.REQUIREMENTS,lambda v:v['files_sha256'].update({str(a.REVIEW):a.sha(a.REVIEW)}))
    elif mutation=='previous_route':change(a.REQUIREMENTS,lambda v:v.update(confirmation_account='mltheory',confirmation_partition='all'))
    else:
        a.BASE_SOURCE.write_text('changed');change(a.REQUIREMENTS,lambda v:v['files_sha256'].update({str(a.BASE_SOURCE):a.sha(a.BASE_SOURCE)}))
    with pytest.raises(ValueError):a.requirements()


def test_final_authority_composes_real_sealed_spin_driver_and_v4_transport(audit,tmp_path,monkeypatch):
    """Exercise final ledger through real static transport -> real driver.verify.

    Reuse the SHA-pinned scratch fixture, preparing only its temporary authority;
    all grader/scheduler/controller calls are fixture stubs that forbid processes.
    """
    fixture_path=ROOT/'tests/test_modebench_scale_spin_authority.py'
    assert audit.sha(fixture_path)=='004c4f4fa82f947ad00f05cf41e2da490fd68363d15e013d5253f37354e116fe'
    spec=importlib.util.spec_from_file_location('sealed_spin_scratch_fixture',fixture_path)
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    f=fixtures.fixture.__wrapped__(tmp_path,monkeypatch);fixtures.prepare(f)
    d=f.a;activation=d.read(f.root/'activation.json')
    start_time=audit.moment(activation['created_at_utc'])
    def t(seconds):return (start_time+timedelta(seconds=seconds)).isoformat()
    runtime={'schema':'modebench_scale_spin_authority_runtime_v1','host':d.HOST,'uid':activation['uid'],
        'authority_path':str(f.root/'activation.json'),'authority_sha256':d.sha(f.root/'activation.json'),
        'owner_pid':1234567,'owner_start_ticks':'7654321',
        'owner_command':[str(d.PYTHON),'-B',str(d.SOURCE),'watch','--root',str(f.root),'--interval','60'],
        'lock_path':str(d.LOCK),'lock_inode':d.LOCK_INODE,'lock_device':d.LOCK.stat().st_dev,'fence_fd':8,'at_utc':t(3),
        **{key:activation[key] for key in ('revised_recovery_job_id','revised_recovery_completion_path','revised_recovery_completion_sha256')}}
    write(f.root/'outer_runtime.json',runtime)
    record=d.authority_record(f.root,activation,runtime,8)
    def probe(second):
        return {'command':[str(d.PYTHON),'-B',str(d.SOURCE),'host-probe','--owner-pid',str(runtime['owner_pid'])],
            'returncode':0,'observation':{'at_utc':t(second),'host':d.HOST,'uid':runtime['uid'],'canonical_sha256':d.NEUTRAL_SHA,
                'lock_path':runtime['lock_path'],'lock_inode':runtime['lock_inode'],'lock_device':runtime['lock_device'],
                'predecessor_os_state':'unverified','local_controllers':[{'pid':runtime['owner_pid'],
                    'start_ticks':runtime['owner_start_ticks'],'command':runtime['owner_command']}]}}
    proof=d.read(f.root/'revised_recovery_completion.json')['proof_verification']
    write(f.root/'start_intent.json',{'at_utc':t(2),'activation_sha256':d.sha(f.root/'activation.json'),
        'scope':activation['scope'],'host_probe':probe(1),'proof_verification':proof})
    directory=f.root/'sweeps/000001';write(directory/'before.json',probe(4))
    argv=d.guest_command(f.root,activation,1,8)
    write(directory/'intent.json',{'at_utc':t(5),'command':argv,'authority':record,'before_sha256':d.sha(directory/'before.json')})
    guest_argv=argv[argv.index('--')+1:];guest_argv.insert(1,'-B')
    write(directory/'guest_runtime.json',{'at_utc':t(6),'authority':record,'host':d.HOST,'uid':runtime['uid'],
        'fence_fd':8,'pid':1234568,'start_ticks':'7654322','state':'R','command':guest_argv,'before_sha256':d.sha(directory/'before.json')})
    result={'status':'admitted','scratch_only':True}
    write(directory/'result.json',{'at_utc':t(7),'authority':record,'guest_runtime_sha256':d.sha(directory/'guest_runtime.json'),'result':result})
    write(directory/'exit.json',{'at_utc':t(8),'returncode':0});write(directory/'after.json',probe(9))
    write(f.root/'terminal.json',{'at_utc':t(10),'last_sweep':1,'result':result})
    transport=fixtures.load(d.TRANSPORT,'real_v4_transport_in_final_ledger')
    monkeypatch.setattr(transport,'AUTHORITY_ROOT',f.root);monkeypatch.setattr(transport,'ARTIFACTS',d.ARTIFACTS)
    monkeypatch.setattr(transport,'module',lambda path,expected,name:d if path==d.SOURCE and expected==d.sha(d.SOURCE) else None)
    monkeypatch.setattr(transport.socket,'gethostname',lambda:'node202.ionic.cs.princeton.edu')
    f.canonical.write_text('old')
    try:
        summary,pins,saved=audit.authority_ledger(f.root,d,transport,activation,recovery={'audit_completed_at_utc':t(-1)})
        assert summary['terminal_status']=='admitted' and saved==record
        assert pins[str(d.SOURCE)]==d.sha(d.SOURCE) and pins[str(d.TRANSPORT)]==d.sha(d.TRANSPORT)
        assert f.calls['sweeps']==[] and f.calls['process']==[]
    finally:f.canonical.write_text('neutral')
