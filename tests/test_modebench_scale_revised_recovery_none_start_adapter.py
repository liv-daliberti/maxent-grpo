"""Scratch-only observed-None adapter tests. Never invoke a real scheduler/model."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


base_tests=load(ROOT/'tests/test_modebench_scale_revised_node_failure_recovery.py','scratch_base_recovery_tests')
c=base_tests.c


@pytest.fixture
def a(c,monkeypatch):
    base_tests.prepare(c)
    original_read=c.r.run_read
    def actual_none(command):
        value=original_read(command)
        if command[0]=='sacct' and '--name='+c.job_name not in command and c.phase=='terminal':
            rows=[line.split('|') for line in value['stdout'].splitlines() if line.strip()]
            for row in rows:
                if row[1]=='31254520_6': row[4]='None'
            value['stdout']='\n'.join('|'.join(row) for row in rows)+'\n'
        return value
    monkeypatch.setattr(c.r,'run_read',actual_none)
    with pytest.raises(ValueError,match='allocation has started'): c.r.cancel_pending(c.root)
    assert len(c.calls)==2 and not (c.root/'cancellation.json').exists()
    module=load(ROOT/'artifacts/modebench_scale_revised_recovery_none_start_adapter_20260912.py','scratch_none_adapter')
    for key,value in {'ROOT':c.base,'SOURCE':c.file('artifacts/adapter.py'),'BASE':c.r.SOURCE,'BASE_SHA':c.r.sha(c.r.SOURCE),
        'RECOVERY':c.root,'PLAN_SHA':c.r.sha(c.root/'plan.json'),'CAPTURE':c.root/'cancelled_accounting_observation.json'}.items():
        monkeypatch.setattr(module,key,value)
    command=['sacct','-X','--array','-j','31254520','-n','-P','--format='+c.r.FIELDS]
    record=actual_none(command)
    row=next(line.split('|') for line in record['stdout'].splitlines() if line.split('|')[1]=='31254520_6')
    record.update(schema='modebench_scale_revised_pending_cancellation_observation_v1',plan_sha256=module.PLAN_SHA,
        original_runtime6_absent=True,original_outputs6_absent=True,actual_cancelled_row=row,
        captured_at_utc='2026-09-12T16:00:00+00:00',mutation_files_sha256={str(c.root/(kind+'_'+event+'.json')):
            c.r.sha(c.root/(kind+'_'+event+'.json')) for kind in ('hold','cancel') for event in ('intent','result')})
    module.CAPTURE.write_text(json.dumps(record)); monkeypatch.setattr(module,'CAPTURE_SHA',module.sha(module.CAPTURE))
    monkeypatch.setattr(module,'base_module',lambda:c.r)
    return module


def amend(a,c):
    a.prepare(c.root)
    return a.verify(c.root)


def reconciled(a,c):
    amend(a,c); return a.reconcile_cancellation(c.root)


def submitted(a,c):
    reconciled(a,c); return a.submit(c.root)


def test_prepared_amendment_preserves_base_plan_worker_and_mutations(a,c):
    paths=[c.r.SOURCE,c.root/'plan.json',c.root/'worker.slurm',c.root/'hold_intent.json',c.root/'hold_result.json',
           c.root/'cancel_intent.json',c.root/'cancel_result.json']
    before={p:p.read_bytes() for p in paths}
    tx=amend(a,c)
    assert all(p.read_bytes()==data for p,data in before.items())
    assert tx['canonical_plan_sha256']==a.PLAN_SHA
    assert tx['effective_submit_command'][:-1]==tx['canonical_submit_command'][:-1]
    assert tx['effective_submit_command'][-1]==str(c.root/'none_start_transport/worker.slurm')
    assert str(a.SOURCE) in (c.root/'none_start_transport/worker.slurm').read_text()
    assert len(c.calls)==2


def test_literal_none_is_preserved_with_both_observer_identities(a,c):
    amend(a,c); snapshot=a.snapshot(c.r,terminal=True)
    assert snapshot['rows_by_original_index']['6'][4]=='None'
    assert '|None|' in snapshot['sacct']['stdout']
    assert snapshot['none_start_amendment']['captured_observer_host']=='spin.cs.princeton.edu'
    assert snapshot['none_start_amendment']['transport_plan_sha256']==a.sha(c.root/'none_start_transport/plan.json')
    assert c.calls==[['scontrol','hold','31254520_6'],['scancel','31254520_6']]


def test_reconcile_already_successful_cancel_without_repeating_any_mutation(a,c):
    proof=reconciled(a,c)
    assert proof['status']=='reconciled_cancelled_only_unstarted_original_cell_6'
    assert proof['terminal']['rows_by_original_index']['6'][4]=='None'
    c.r.cancellation_gate(c.root)
    assert len(c.calls)==2
    with pytest.raises(ValueError,match='already recorded'): a.reconcile_cancellation(c.root)
    assert len(c.calls)==2


@pytest.mark.parametrize('change',['None_to_Unknown','started','cpu','node','exit','older_end','raw_id','other_cell'])
def test_narrow_gate_rejects_any_change_to_captured_actual_accounting(a,c,monkeypatch,change):
    amend(a,c); original_read=c.r.run_read
    def altered(command):
        value=original_read(command)
        if command[0]=='sacct':
            rows=[line.split('|') for line in value['stdout'].splitlines() if line.strip()]
            row=next(r for r in rows if r[1]=='31254520_6')
            field,new={'None_to_Unknown':(4,'Unknown'),'started':(4,'2026-09-12T15:00:00'),
                'cpu':(7,'6'),'node':(6,'node202'),'exit':(3,'1:0'),'older_end':(5,'2026-09-12T14:00:00'),
                'raw_id':(0,'999')}.get(change,(0,'0'))
            if change=='other_cell': rows[0][3]='9:0'
            else: row[field]=new
            value['stdout']='\n'.join('|'.join(r) for r in rows)+'\n'
        return value
    monkeypatch.setattr(c.r,'run_read',altered)
    with pytest.raises(ValueError,match='differs from exact registered'): a.snapshot(c.r,terminal=True)


def test_gate_refuses_live_original_array_and_pending_mode(a,c):
    amend(a,c)
    with pytest.raises(ValueError,match='terminal observation only'): a.snapshot(c.r,pending=True)
    c.phase='pending'
    with pytest.raises(ValueError): a.snapshot(c.r,terminal=True)
    assert len(c.calls)==2


@pytest.mark.parametrize('which',['capture','adapter','base','original_plan','original_worker','effective_worker','cancel_result'])
def test_changed_pins_refuse_before_new_submission(a,c,which):
    reconciled(a,c)
    path={'capture':a.CAPTURE,'adapter':a.SOURCE,'base':a.BASE,'original_plan':c.root/'plan.json',
        'original_worker':c.root/'worker.slurm','effective_worker':c.root/'none_start_transport/worker.slurm',
        'cancel_result':c.root/'cancel_result.json'}[which]
    path.chmod(0o644); path.write_text('changed')
    with pytest.raises((ValueError,KeyError)): a.submit(c.root)
    assert len(c.calls)==2 and not (c.root/'submission_intent.json').exists()


def test_submission_dispatches_only_explicit_effective_worker_once(a,c):
    submitted(a,c)
    tx=a.verify(c.root); canonical=c.r.read(c.root/'submission_result.json')
    effective=c.r.read(c.root/'none_start_transport/submission_result.json')
    assert c.calls[-1]==tx['effective_submit_command']
    assert canonical['command']==tx['canonical_submit_command']
    assert all(canonical[k]==effective[k] for k in ('returncode','stdout','stderr'))
    assert a.verify_submission(c.root)['submission']['array_job_id']==900
    with pytest.raises(ValueError): a.submit(c.root)
    assert sum(command[0]=='sbatch' for command in c.calls)==1


def test_private_facade_does_not_modify_shared_subprocess_run(a,c):
    reconciled(a,c); previous=subprocess.run
    a.install(c.r,c.root,submission=True)
    assert subprocess.run is previous and c.r.subprocess.run is not previous
    with pytest.raises(ValueError,match='no hold/cancel'): c.r.subprocess.run(['scancel','31254520_6'])
    assert len(c.calls)==2


@pytest.mark.parametrize('failure',['timeout','nonzero','malformed'])
def test_ambiguous_effective_dispatch_never_resubmits(a,c,failure):
    reconciled(a,c)
    if failure=='timeout': c.mutation_error='sbatch'
    elif failure=='nonzero': c.mutation_nonzero='sbatch'
    else: c.stdout='uncertain'
    with pytest.raises((ValueError,subprocess.TimeoutExpired)): a.submit(c.root)
    with pytest.raises(ValueError): a.submit(c.root)
    assert sum(command[0]=='sbatch' for command in c.calls)==1
    assert (c.root/'none_start_transport/submission_intent.json').exists()


@pytest.mark.parametrize('field,value',[('effective_command',['sbatch','fake']),('stdout','999\n'),('returncode',1),
                                       ('intent_sha256','0'*64)])
def test_worker_ledger_rejects_effective_result_tampering(a,c,field,value):
    submitted(a,c); path=c.root/'none_start_transport/submission_result.json'
    record=c.r.read(path); record[field]=value; base_tests.rewrite(path,record)
    with pytest.raises(ValueError): a.verify_submission(c.root)


def test_adapter_worker_records_amendment_and_invokes_exact_inherited_science(a,c,monkeypatch):
    submitted(a,c); base_tests.setup_worker_env(c,monkeypatch)
    seen=[]; monkeypatch.setattr(c.r.os,'execv',lambda exe,argv:seen.append((exe,argv)))
    a.worker(c.root,0)
    outer=c.r.read(c.root/'none_start_transport/runtime/0.json')
    inner=c.r.read(c.root/'runtime/0.json')
    assert outer['hostname']==inner['hostname']=='node202.ionic.cs.princeton.edu'
    assert outer['original_index']==inner['original_index']==5
    assert outer['original_terminal']['rows_by_original_index']['6'][4]=='None'
    assert inner['original_terminal']['none_start_amendment']['policy']==a.POLICY
    assert seen==[(c.plan['cells'][5]['command'][0],c.plan['cells'][5]['command'])]
    assert c.r.read(c.r.ORIGINAL_PLAN.parent/'runtime/5.json')['array_job_id']==31254520
    with pytest.raises(ValueError,match='already attempted'): a.worker(c.root,0)
    assert len(seen)==1 and len(c.calls)==3


def test_effective_result_allows_identity_reconciliation_without_second_sbatch(a,c):
    submitted(a,c); (c.root/'submission_identity.json').unlink()
    c.recovery_accounting=base_tests.accounting(c)
    result=a.reconcile_submission(c.root)
    assert result['array_job_id']==900 and result['status']=='reconciled_from_accounting'
    assert sum(command[0]=='sbatch' for command in c.calls)==1


def test_unresolved_effective_dispatch_requires_explicit_evidence_not_a_new_submission(a,c):
    reconciled(a,c); c.mutation_error='sbatch'
    with pytest.raises(ValueError): a.submit(c.root)
    with pytest.raises(ValueError,match='effective dispatch outcome unresolved'): a.reconcile_submission(c.root)
    assert not (c.root/'submission_identity.json').exists()
    assert sum(command[0]=='sbatch' for command in c.calls)==1
