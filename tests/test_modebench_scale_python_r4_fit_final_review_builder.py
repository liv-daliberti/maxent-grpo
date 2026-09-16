"""Synthetic publication gates only; actual auditor/fitter imports never run."""
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


@pytest.fixture
def builder():
    spec=importlib.util.spec_from_file_location('scratch_r4_fit_review_builder',ROOT/'artifacts/build_modebench_scale_python_r4_fit_final_review_20260913.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def records(builder,tmp_path):
    b=builder;m=SimpleNamespace(PYTHON=tmp_path/'python',RUNNER=tmp_path/'runner.py',VIEW=tmp_path/'manifest.json',
        AUDITOR=tmp_path/'auditor.py',AUDITOR_REVIEW=tmp_path/'auditor_review.json',RECOVERY=tmp_path/'development')
    for path in (m.PYTHON,m.RUNNER,m.VIEW,m.AUDITOR,m.AUDITOR_REVIEW):path.write_text('{}')
    a=SimpleNamespace(engine=m,AUDIT_ROOT=tmp_path/'audit',READONLY_ROOT=tmp_path/'readonly',AUDIT_RECORDER=tmp_path/'recorder.py',
        HOST='soak.cs.princeton.edu',UID=363432,contract=SimpleNamespace(moment=datetime.fromisoformat))
    a.AUDIT_RECORDER.write_text('no native actions')
    proof={'schema':'actual_r4_audit','status':'complete_existing_audit','array_job_id':31261726,
        'files_sha256':{str(m.AUDITOR):b.sha(m.AUDITOR)},'audit_completed_at_utc':'2026-09-13T16:00:02+00:00'}
    certificate_sha='c'*64;terminal_sha='d'*64
    for index,(action,root) in enumerate([('audit',a.AUDIT_ROOT),('verify',a.READONLY_ROOT)]):
        start=f'2026-09-13T16:00:0{index*3}+00:00';finish=f'2026-09-13T16:00:0{index*3+2}+00:00'
        command=b.audit_command(a,action,terminal_sha)
        intent={'schema':b.OUTER_SCHEMA,'action':action,'actual_host':a.HOST,'uid':a.UID,'pid':100+index,
            'started_at_utc':start,'command':command,'script_sha256':b.sha(a.AUDIT_RECORDER),
            'auditor_review_sha256':b.sha(m.AUDITOR_REVIEW),'terminal_sha256':terminal_sha}
        write(root/'intent.json',intent)
        write(root/'process.json',{'actual_host':a.HOST,'uid':a.UID,'pid':100+index,'child_pid':200+index,'at_utc':start})
        summary={k:proof[k] for k in ('schema','status','array_job_id')};summary['files']=len(proof['files_sha256'])
        (root/'stdout.txt').write_text(json.dumps(summary)+'\n');(root/'stderr.txt').write_text('')
        exited={**intent,'finished_at_utc':finish,'returncode':0,'postcheck_error':None,'certificate_present':True,
            'certificate_sha256':certificate_sha,'new_grader_invocations':24704 if action=='audit' else 0,
            **{field:b.sha(root/name) for field,name in [('intent_sha256','intent.json'),('process_sha256','process.json'),('stdout_sha256','stdout.txt'),('stderr_sha256','stderr.txt')]}}
        write(root/'exit.json',exited)
    def run():return b.outer_records(a,proof,certificate_sha,terminal_sha,b.sha(a.AUDIT_ROOT/'exit.json'),b.sha(a.READONLY_ROOT/'exit.json'))
    return SimpleNamespace(b=b,a=a,proof=proof,run=run)


def test_real_record_contract_requires_complete_separate_audit_and_readonly_zero(records):
    x=records;pins=x.run();assert len(pins)==10 and all(x.b.sha(p)==d for p,d in pins.items())


@pytest.mark.parametrize('action,code',[('audit',143),('audit',-15),('audit',1),('verify',143),('verify',1),('verify',True)])
def test_cpu_nonzero_and_noninteger_exits_are_never_normalized(records,action,code):
    x=records;root=x.a.AUDIT_ROOT if action=='audit' else x.a.READONLY_ROOT
    value=x.b.read(root/'exit.json');value['returncode']=code;write(root/'exit.json',value)
    with pytest.raises(ValueError,match='audit0'):x.run()


@pytest.mark.parametrize('kind',['certificate_hash','missing_certificate','wrong_host','wrong_uid','changed_argv','terminal_hash','recorder_hash','review_hash','postcheck','readonly_grader'])
def test_rehashed_outer_records_cannot_relink_wrong_observation(records,kind):
    x=records;root=x.a.READONLY_ROOT;intent=x.b.read(root/'intent.json');exited=x.b.read(root/'exit.json')
    if kind=='certificate_hash':exited['certificate_sha256']='e'*64
    elif kind=='missing_certificate':exited['certificate_present']=False
    elif kind=='wrong_host':intent['actual_host']=exited['actual_host']='spin.cs.princeton.edu'
    elif kind=='wrong_uid':intent['uid']=exited['uid']=0
    elif kind=='changed_argv':intent['command']=exited['command']=['relative','verify']
    elif kind=='terminal_hash':intent['terminal_sha256']='e'*64
    elif kind=='recorder_hash':intent['script_sha256']='e'*64
    elif kind=='review_hash':intent['auditor_review_sha256']='e'*64
    elif kind=='postcheck':exited['postcheck_error']='failed'
    else:exited['new_grader_invocations']=1
    write(root/'intent.json',intent);exited['intent_sha256']=x.b.sha(root/'intent.json');write(root/'exit.json',exited)
    with pytest.raises(ValueError):x.run()


@pytest.mark.parametrize('kind',['during_readonly','before_audit','missing_record','symlink_record','wrong_stdout'])
def test_chronology_and_actual_record_completeness_are_required(records,kind):
    x=records
    if kind=='during_readonly':x.proof['audit_completed_at_utc']='2026-09-13T16:00:04+00:00'
    elif kind=='before_audit':x.proof['audit_completed_at_utc']='2026-09-13T15:59:59+00:00'
    elif kind=='missing_record':(x.a.READONLY_ROOT/'process.json').unlink()
    elif kind=='symlink_record':
        path=x.a.READONLY_ROOT/'process.json';copy=path.parent/'alias.json';copy.write_bytes(path.read_bytes());path.unlink();path.symlink_to(copy)
    else:
        path=x.a.READONLY_ROOT/'stdout.txt';path.write_text(json.dumps({'schema':'wrong','status':'complete','array_job_id':31261726,'files':1}))
        value=x.b.read(x.a.READONLY_ROOT/'exit.json');value['stdout_sha256']=x.b.sha(path);write(x.a.READONLY_ROOT/'exit.json',value)
    with pytest.raises(ValueError):x.run()


@pytest.mark.parametrize('kind',['review','state','recipe','dataset','confirmation'])
def test_future_review_fit_and_heldout_outputs_are_not_accepted_as_evidence(builder,tmp_path,kind):
    b=builder;m=SimpleNamespace(REVIEW=tmp_path/'review.json',STATE=tmp_path/'fit',REVISION_ROOT=tmp_path/'r4')
    path={'review':m.REVIEW,'state':m.STATE/'runtime.json','recipe':m.REVISION_ROOT/'level4/recipes/python_factors.json',
        'dataset':m.REVISION_ROOT/'level4/dataset/python_factors/train/data.arrow',
        'confirmation':m.REVISION_ROOT/'level4/results/confirmation/python_factors.json.batches/0.json'}[kind]
    with pytest.raises(ValueError,match='future'):b.no_future_pins(m,{str(path):'a'*64})


def test_conflicting_hashes_and_nonabsolute_paths_are_not_silently_normalized(builder,tmp_path):
    p={str(tmp_path/'input'):'a'*64}
    with pytest.raises(ValueError,match='conflicting'):builder.merge(p,{str(tmp_path/'input'):'b'*64})
    with pytest.raises(ValueError,match='absolute'):builder.merge({}, {'relative':'a'*64})


def test_builder_import_and_assembly_contain_no_native_execution_entrypoints(builder):
    tree=ast.parse(builder.SOURCE.read_text())
    forbidden={'fit_domain','fit_registered','collect_inputs','verify_existing','audit','reconcile','register','materialize_pools','launch_inputs','freeze_dataset'}
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&forbidden
    assert 'atomic_new' in calls and 'reviewed' in calls
