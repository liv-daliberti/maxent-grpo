"""Prevent duplicate GPU submissions and premature expansion of the fresh panel."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import launch_modebench_fresh_concentration as m


@pytest.fixture
def campaign(tmp_path):
    base = tmp_path / 'campaign'
    (base / 'slurm').mkdir(parents=True)
    (base / 'slurm' / 'collect.sh').write_text('#!/bin/sh\nexit 0\n')
    tasks = []
    for i in range(4):
        model = tmp_path / 'models' / str(i)
        model.mkdir(parents=True)
        files = []
        for name, raw in [('config.json', b'{}'), ('model.safetensors', b'weights')]:
            p = model / name
            p.write_bytes(raw)
            files.append({'name': name, 'bytes': len(raw), 'sha256': m.digest(p)})
        task = {'task_id': f'task-{i}', 'checkpoint_stage': 'initial' if i < 2 else 'terminal',
                'model_path': str(model), 'files': files, 'domain': 'graph_coloring' if i % 2 == 0 else 'pantry_plan'}
        tasks.append(task)
        if i >= 2:
            receipt = tmp_path / 'var/cache/modebench_fresh_concentration_20260912/receipts' / f'task-{i}.json'
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({'model_path': str(model), 'files': files, 'domain': 'graph_coloring' if i % 2 == 0 else 'pantry_plan'}))
    plan = {'tasks': tasks, 'output_root': str(base / 'results'), 'seed_namespace': 'launcher-test', 'draw_labels': list(range(8)), 'batch_size': 128}
    (base / 'plan.json').write_text(json.dumps(plan))
    with patch.object(m, 'ROOT', tmp_path), patch.object(m.collector, 'validate_plan', return_value={t['task_id']: [{'problem': 'fixed problem'}] for t in tasks}):
        yield base, plan


def scheduler_reply(command, **kwargs):
    if command[0] == 'sbatch':
        return SimpleNamespace(stdout='12345\n', stderr='', returncode=0)
    if command[0] == 'scontrol':
        return SimpleNamespace(stdout='JobId=12345 JobState=PENDING\n', stderr='', returncode=0)
    if command[0] == 'squeue':
        return SimpleNamespace(stdout='', stderr='', returncode=0)
    raise AssertionError(f'unexpected process: {command}')


@pytest.mark.parametrize('indices', [[0, 0], [], [-1], [4], [True], ['0']])
def test_indices_are_explicit_unique_integer_slots(campaign, indices):
    _, plan = campaign
    with pytest.raises(ValueError, match='indices'):
        m.validate_indices(indices, plan)


def test_ready_receipt_cannot_bind_different_weights(campaign):
    base, plan = campaign
    task = plan['tasks'][2]
    receipt = m.ROOT / 'var/cache/modebench_fresh_concentration_20260912/receipts/task-2.json'
    value = json.loads(receipt.read_text())
    value['files'][1]['sha256'] = '0' * 64
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='receipt differs'):
        m.assert_ready(task)


def test_ready_receipt_cannot_bind_another_model_directory(campaign):
    _, plan = campaign
    task = copy.deepcopy(plan['tasks'][2])
    task['model_path'] = plan['tasks'][3]['model_path']
    with pytest.raises(ValueError, match='receipt differs'):
        m.assert_ready(task)


@pytest.mark.parametrize('completed_indices', [[], [0], [1]])
def test_full_panel_waits_for_both_interface_completions(campaign, completed_indices):
    base, plan = campaign
    for index in completed_indices:
        path = Path(plan['output_root']) / plan['tasks'][index]['task_id'] / 'result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'status': 'complete'}))
    with patch.object(m, 'authenticate_task', return_value={}), patch.object(m.subprocess, 'run', side_effect=scheduler_reply) as run:
        with pytest.raises(ValueError):
            m.submit([2, 3], 'full_panel', base)
        assert not any(call.args[0][0] == 'sbatch' for call in run.call_args_list)


def test_stale_complete_results_do_not_open_full_panel_gate(campaign):
    base, plan = campaign
    for index in (0, 1):
        path = Path(plan['output_root']) / plan['tasks'][index]['task_id'] / 'result.json'
        path.parent.mkdir(parents=True)
        (path.parent / 'run.json').write_text(json.dumps({'identity_sha256': '0' * 64, 'identity': {}}))
        (path.parent / 'runtime.json').write_text('{}')
        path.write_text(json.dumps({'status': 'complete', 'task_id': 'another-task',
                                    'identity_sha256': '0' * 64, 'runtime_sha256': '1' * 64}))
    with patch.object(m.subprocess, 'run', side_effect=scheduler_reply) as run:
        with pytest.raises(ValueError):
            m.submit([2, 3], 'full_panel', base)
        assert not any(call.args[0][0] == 'sbatch' for call in run.call_args_list)


def test_existing_submission_receipt_blocks_duplicate_array(campaign):
    base, _ = campaign
    directory = base / 'slurm' / 'submissions'
    directory.mkdir()
    (directory / '99.json').write_text(json.dumps({'job_id': '99', 'task_indices': [0, 1], 'intent_id': 'known'}))
    with patch.object(m.subprocess, 'run', side_effect=scheduler_reply) as run:
        with pytest.raises(ValueError, match='already|submission|overlap'):
            m.submit([0, 1], 'interface_validation', base)
        run.assert_not_called()


def test_unresolved_submission_intent_blocks_automatic_duplicate_retry(campaign):
    base, _ = campaign
    directory = base / 'slurm' / 'intents'
    directory.mkdir()
    (directory / 'ambiguous.json').write_text(json.dumps({
        'phase': 'interface_validation', 'task_indices': [0, 1],
        'plan_sha256': m.digest(base / 'plan.json'), 'command': ['sbatch']}))
    with patch.object(m.subprocess, 'run', side_effect=scheduler_reply) as run:
        with pytest.raises(ValueError, match='intent|ambiguous|unresolved'):
            m.submit([0, 1], 'interface_validation', base)
        run.assert_not_called()


def test_interface_validation_submits_only_the_two_selected_slots(campaign):
    base, _ = campaign
    with patch.object(m.subprocess, 'run', side_effect=scheduler_reply) as run:
        receipt = m.submit([1, 0], 'interface_validation', base)
    submissions = [c.args[0] for c in run.call_args_list if c.args[0][0] == 'sbatch']
    assert len(submissions) == 1
    assert '--array=0,1%2' in submissions[0]
    assert receipt['task_indices'] == [0, 1]
    assert receipt['max_concurrent_owned_gpus'] == 2
    assert receipt['plan_sha256'] == m.digest(base / 'plan.json')
    assert (base / 'slurm' / 'submissions' / '12345.json').is_file()


@pytest.fixture
def amended_campaign(campaign):
    base,plan=campaign
    prototype=copy.deepcopy(plan['tasks'][0])
    plan['tasks']=[{**copy.deepcopy(prototype),'task_id':f'task-{i}',
                    'model_scale':'falcon1b' if i>=50 else 'qwen05b'} for i in range(54)]
    (base/'plan.json').write_text(json.dumps(plan))
    submissions=base/'slurm/submissions';submissions.mkdir()
    old={'job_id':'99','task_indices':[2,50,51,52],'intent_id':'old',
         'plan_sha256':m.digest(base/'plan.json')}
    (submissions/'99.json').write_text(json.dumps(old))
    directory=base/'execution_amendments';directory.mkdir()
    probe=directory/'probe.json';probe.write_text(json.dumps({
        'gpu_names':['NVIDIA A100 80GB PCIe'],'shared_memory_per_block_optin_bytes':166912,'bf16_supported':True}))
    cancel=directory/'cancel.json';cancel.write_text(json.dumps({
        'cancel_exit':0,'affected':[{'job_id':'99_52','state':'PENDING'}]}))
    attempts=[]
    for i in (50,51):
        archive=directory/f'archive-{i}';archive.mkdir()
        raw=archive/'run.json';raw.write_text('{}')
        receipt=directory/f'archive-{i}.json';receipt.write_text(json.dumps({
            'job_id':'99','task_index':i,'task_id':f'task-{i}','no_committed_response_slots':True,
            'archive_path':str(archive),'files':[{'path':'run.json','sha256':m.digest(raw)}]}))
        attempts.append({'job_id':'99','task_index':i,'terminal_state':'FAILED',
                         'archive_receipt':{'path':str(receipt),'sha256':m.digest(receipt)}})
    attempts.append({'job_id':'99','task_index':52,'terminal_state':'CANCELLED','archive_receipt':None,
                     'state_evidence':{'cancellation_receipt':{'path':str(cancel),'sha256':m.digest(cancel)}}})
    a={'schema':m.AMENDMENT_SCHEMA,'plan_sha256':m.digest(base/'plan.json'),
       'collector_sha256':m.digest(Path(m.collector.__file__)),'affected_task_indices':[50,51,52,53],
       'gres':'gpu:a100:1','gpu_names':['NVIDIA A100 80GB PCIe'],
       'device_probe':{'path':str(probe),'sha256':m.digest(probe)},
       'superseded_attempts':attempts,'no_committed_response_slots':True}
    ap=directory/'falcon_a100.json';ap.write_text(json.dumps(a))
    for i in (0,1):
        complete_task(base,plan,i)
    with patch.object(m.collector,'validate_plan',return_value={t['task_id']:[] for t in plan['tasks']}):
        yield base,plan,a,old


def complete_task(base,plan,i,gpu='NVIDIA RTX A5000'):
    folder=Path(plan['output_root'])/plan['tasks'][i]['task_id'];folder.mkdir(parents=True,exist_ok=True)
    (folder/'result.json').write_text(json.dumps({'status':'complete','task_id':plan['tasks'][i]['task_id']}))
    (folder/'run.json').write_text(json.dumps({'identity':{'plan_sha256':m.digest(base/'plan.json')}}))
    runtime={'gpu_names':[gpu]}
    (folder/'runtime.json').write_text(json.dumps({'runtime':runtime,'runtime_sha256':m.collector.runtime_fingerprint(runtime)}))


def test_falcon_retry_is_exactly_bound_to_archived_attempt_and_a100(amended_campaign):
    base,plan,a,old=amended_campaign
    with patch.object(m,'authenticate_task',return_value={}), patch.object(m.subprocess,'run',side_effect=scheduler_reply):
        receipt=m.submit([50,51],'falcon_interface_validation',base)
        assert receipt['execution_amendment']['sha256']==m.digest(base/'execution_amendments/falcon_a100.json')
        assert '--gres=gpu:a100:1' in receipt['command']
        assert '--array=50,51%2' in receipt['command']
        with pytest.raises(ValueError,match='already'):
            m.submit([50,51],'falcon_interface_validation',base)


def test_amendment_does_not_excuse_unlisted_prior_attempt(amended_campaign):
    base,plan,a,old=amended_campaign
    a['superseded_attempts'].pop()
    (base/'execution_amendments/falcon_a100.json').write_text(json.dumps(a))
    with pytest.raises(ValueError,match='every old Falcon'):
        m.load_execution_amendment(base,plan,[old])


def test_archive_with_committed_batch_cannot_authorize_retry(amended_campaign):
    base,plan,a,old=amended_campaign
    (base/'execution_amendments/archive-50/batch_b000.json').write_text('{}')
    with pytest.raises(ValueError,match='committed|unlisted'):
        m.load_execution_amendment(base,plan,[old])


def test_failed_or_unproven_pending_cancel_cannot_authorize_retry(amended_campaign):
    base,plan,a,old=amended_campaign
    ref=a['superseded_attempts'][-1]['state_evidence']['cancellation_receipt']
    p=Path(ref['path']);p.write_text(json.dumps({'cancel_exit':1,'affected':[]}));ref['sha256']=m.digest(p)
    (base/'execution_amendments/falcon_a100.json').write_text(json.dumps(a))
    with pytest.raises(ValueError,match='cancellation evidence'):
        m.load_execution_amendment(base,plan,[old])


@pytest.mark.parametrize('indices',[[3,52],[50,52]])
def test_mixed_hardware_or_wrong_falcon_interface_is_rejected(amended_campaign,indices):
    base,_,_,_=amended_campaign
    with patch.object(m,'authenticate_task',return_value={}),patch.object(m.subprocess,'run',side_effect=scheduler_reply) as run:
        with pytest.raises(ValueError,match='GPU family|exactly tasks'):
            m.submit(indices,'falcon_interface_validation',base)
        assert not any(c.args[0][0]=='sbatch' for c in run.call_args_list)


def test_retry_requires_old_output_directory_to_be_archived(amended_campaign):
    base,plan,_,_=amended_campaign
    (Path(plan['output_root'])/'task-50').mkdir()
    with pytest.raises(ValueError,match='archived before retry'):
        m.submit([50,51],'falcon_interface_validation',base)


def test_falcon_broad_gate_authenticates_both_new_hardware_receipts(amended_campaign):
    base,plan,_,_=amended_campaign
    for i in (50,51):complete_task(base,plan,i)
    with patch.object(m,'authenticate_task',return_value={}),patch.object(m.subprocess,'run',side_effect=scheduler_reply) as run:
        with pytest.raises(ValueError,match='unexpected runtime hardware'):
            m.submit([53],'full_panel',base)
        assert not any(c.args[0][0]=='sbatch' for c in run.call_args_list)
    for i in (50,51):complete_task(base,plan,i,'NVIDIA A100 80GB PCIe')
    with patch.object(m,'authenticate_task',return_value={}),patch.object(m.subprocess,'run',side_effect=scheduler_reply):
        r=m.submit([53],'full_panel',base)
    assert '--array=53%8' in r['command']
    assert '--gres=gpu:a100:1' in r['command']


def test_previous_active_wave_still_blocks_falcon_recovery(amended_campaign):
    base,_,_,_=amended_campaign
    def active(command,**kwargs):
        if command[0]=='squeue':return SimpleNamespace(stdout='99_2 RUNNING\n',stderr='')
        return scheduler_reply(command,**kwargs)
    with patch.object(m,'authenticate_task',return_value={}),patch.object(m.subprocess,'run',side_effect=active) as run:
        with pytest.raises(ValueError,match='remain active'):
            m.submit([50,51],'falcon_interface_validation',base)
        assert not any(c.args[0][0]=='sbatch' for c in run.call_args_list)
